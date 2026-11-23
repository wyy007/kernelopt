"""任务执行层（本地子进程，无 shell）。

设计原则：
- 白名单：只允许这里登记的动作，参数逐个校验，绝不拼 shell 字符串；
- 单并发：GPU 是独占资源，同一时刻只跑一个任务（排队）；
- 可追溯：每个任务一份 JSON 元数据 + 一份实时日志，落在 runs/_jobs/ 下；
- 可中止：终止即对子进程发 SIGTERM，超时后 SIGKILL。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()


def _load_env_file(path: str = "/root/.optframe_env") -> None:
    """把远端凭据文件里的 export KEY=VALUE 读进环境（不覆盖已有值）。"""
    f = Path(path)
    try:
        if not f.exists():
            return
    except OSError:
        return
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                key, _, val = line[len("export "):].partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_env_file()
JOBS_DIR = REPO / "runs" / "_jobs"
CONTAINER = os.environ.get("OPTFRAME_CONTAINER", "sglang-debug")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class ParamError(ValueError):
    pass


def _python() -> str:
    """优先用仓库自带 venv（TraceLens / triton 都装在那里），否则退回当前解释器。"""
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _path_in_repo(value: str, *, must_exist: bool = True) -> str:
    p = Path(value)
    p = p.resolve() if p.is_absolute() else (REPO / value).resolve()
    if p != REPO and REPO not in p.parents:
        raise ParamError(f"路径必须在仓库内：{value}")
    if must_exist and not p.exists():
        raise ParamError(f"路径不存在：{value}")
    return str(p)


def _run_id(value: Any) -> str:
    rid = str(value or "").strip() or f"stage2_ui_{int(time.time())}"
    if not all(c.isalnum() or c in "_-" for c in rid):
        raise ParamError("run_id 只允许字母、数字、下划线、短横线")
    return rid


@dataclass
class Action:
    name: str
    title: str
    desc: str
    params: List[Dict[str, Any]]
    build: Callable[[Dict[str, Any]], List[str]]
    needs_gpu: bool = False
    danger: bool = False
    #: 视为"成功"的退出码（有的动作把"未 KEEP"编码成 1）
    ok_codes: tuple = (0,)
    timeout_s: int = 3600


def _build_analysis(p: Dict[str, Any]) -> List[str]:
    trace = _path_in_repo(p["trace"])
    arch = _path_in_repo(p.get("arch") or "arch/RTX_PRO_5000.json")
    run_id = _run_id(p.get("run_id"))
    argv = [_python(), "-m", "optframe.analysis.cli",
            "--trace", trace, "--arch", arch, "--workdir", f"runs/{run_id}",
            "--framework", str(p.get("framework") or "sglang"),
            "--model", str(p.get("model") or "qwen3-8b")]
    hints = str(p.get("shape_hints") or "").strip()
    if not hints:
        # 采集时自动生成的 hints 就在 trace 同目录，默认直接用
        auto = Path(trace).parent / "shape_hints.json"
        if auto.exists():
            hints = str(auto)
    if hints:
        argv += ["--shape-hints", _path_in_repo(hints)]
    return argv


_HINT_OPS = ("gptq_marlin_gemm", "fp8_blockwise_scaled_mm", "outplace_fused_experts")
def _latest_kernel_log() -> str:
    cands = sorted(
        (REPO / "runs").glob("*/kernel_api.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(cands[0]) if cands else ""


def _build_shape_hints(p: Dict[str, Any]) -> List[str]:
    """自动识别 kernel_api.log 里的已知核并生成 hints（无需选算子）。"""
    raw_log = str(p.get("log") or "").strip()
    log = _path_in_repo(raw_log) if raw_log else _latest_kernel_log()
    if not log or not Path(log).exists():
        raise ParamError("没有找到 kernel_api.log；请先勾选「记录 kernel API 日志」跑一次采集")
    out = str(p.get("out") or f"runs/shape_hints_{int(time.time())}.json").strip()
    if not out.startswith("runs/"):
        raise ParamError("out 必须以 runs/ 开头")
    return [_python(), "tools/auto_shape_hints.py",
            "--log", log, "--out", str(REPO / out)]


def _build_fusion_tasks(p: Dict[str, Any]) -> List[str]:
    report = _path_in_repo(p["report"])
    outdir = str(p.get("outdir") or f"runs/fusion_tasks_{int(time.time())}").strip()
    if not outdir.startswith("runs/"):
        raise ParamError("outdir 必须以 runs/ 开头")
    return [_python(), "-m", "optframe.inventory.fusion_tasks",
            "--report", report, "--outdir", outdir,
            "--arch", str(p.get("arch") or "sm_120"),
            "--dtype", str(p.get("dtype") or "bf16"),
            "--min-share", str(float(p.get("min_share") or 0.0))]


def _model_path(value: Any) -> str:
    """模型目录允许在 /model 下（不受仓库路径限制），但必须在白名单根内。"""
    p = Path(str(value or ""))
    if not p.is_absolute():
        raise ParamError("模型路径必须是绝对路径，例如 /model/Qwen3-8B")
    roots = [Path("/model").resolve()]
    rp = p.resolve()
    if not any(rp == r or r in rp.parents for r in roots):
        raise ParamError("模型路径必须在 /model 下")
    if not rp.exists():
        raise ParamError(f"模型目录不存在：{value}")
    return str(rp)


def _build_capture(p: Dict[str, Any]) -> List[str]:
    if not p.get("confirm"):
        raise ParamError("需要显式确认（勾选「我确认独占 GPU」）")
    model = _model_path(p.get("model") or "/model/Qwen3-8B")
    argv = [_python(), "-m", "optframe.webui.capture",
            "--model", model,
            "--concurrency", str(int(p.get("concurrency") or 8)),
            "--isl", str(int(p.get("isl") or 400)),
            "--seconds", str(float(p.get("seconds") or 30)),
            "--num-steps", str(int(p.get("num_steps") or 30)),
            "--mem-fraction", str(float(p.get("mem_fraction") or 0.55)),
            "--context-length", str(int(p.get("context_length") or 8192)),
            "--max-used-gb", str(float(p.get("max_used_gb") or 5.0))]
    if p.get("confirm"):
        argv += ["--force-free"]
    if p.get("run_id"):
        argv += ["--run-id", str(p["run_id"])]
    return argv


def _build_fusion_eval(p: Dict[str, Any]) -> List[str]:
    task = _path_in_repo(p["task"])
    if not (Path(task) / "task.toml").exists():
        raise ParamError("task 目录里没有 task.toml")
    gen = str(p.get("generator") or "fixture")
    if gen not in ("fixture", "llm"):
        raise ParamError("generator 只能是 fixture 或 llm")
    workdir = str(p.get("workdir") or f"runs/fusion_eval_{int(time.time())}")
    if not workdir.startswith("runs/"):
        raise ParamError("workdir 必须以 runs/ 开头")
    # 跑在容器里：宿主 venv 的 torch 版本与服务不同，基线会偏（实测 0.958 vs 1.10）
    argv = ["docker", "exec", "-w", str(REPO), "-e", f"PYTHONPATH={REPO}",
            CONTAINER, "python3", "-m", "optframe.tiers.fusion",
            "--task", task, "--generator", gen,
            "--n", str(int(p.get("n") or 1)),
            "--rounds", str(int(p.get("rounds") or 1)),
            "--repeat", str(int(p.get("repeat") or 20)),
            "--workdir", workdir]
    if gen == "fixture":
        fx = _path_in_repo(p.get("fixture_dir") or "fixtures/swiglu")
        argv += ["--fixture-dir", fx]
    else:
        if not os.environ.get("GPUSTACK_API_KEY"):
            raise ParamError("缺少 GPUSTACK_API_KEY（/root/.optframe_env 未加载）")
        argv += ["--model", str(p.get("model") or "ds-v4")]
    if p.get("baseline_module"):
        argv += ["--baseline-module", str(p["baseline_module"])]
    return argv


def _build_e2e(p: Dict[str, Any]) -> List[str]:
    if not p.get("confirm"):
        raise ParamError("需要显式确认（勾选「我确认独占 GPU」）")
    model = _model_path(p.get("model") or "/model/Qwen3-8B")
    order = str(p.get("order") or "alternate")
    if order not in ("alternate", "abba"):
        raise ParamError("order 只能是 alternate 或 abba")
    blocks = int(p.get("blocks") or 6)
    multiple = 4 if order == "abba" else 2
    if blocks < multiple or blocks % multiple:
        raise ParamError(f"{order} 顺序要求块数是 {multiple} 的倍数（当前 {blocks}）")
    min_gain = float(p.get("min_gain") or 1.015)
    if min_gain <= 1.0:
        raise ParamError("min_gain 必须 > 1.0；门限 ≤ 1 等于不设门限")
    argv = [_python(), "-m", "optframe.webui.e2e", "--model", model,
            "--blocks", str(blocks),
            "--seconds", str(float(p.get("seconds") or 20)),
            "--concurrency", str(int(p.get("concurrency") or 8)),
            "--prompt-chars", str(int(p.get("prompt_chars") or 12000)),
            "--max-tokens", str(int(p.get("max_tokens") or 8)),
            "--order", order,
            "--min-gain", f"{min_gain:g}"]
    if p.get("confirm"):
        argv += ["--force-free"]
    if p.get("run_id"):
        argv += ["--run-id", str(p["run_id"])]
    return argv


def _build_preflight(p: Dict[str, Any]) -> List[str]:
    return [_python(), "-m", "optframe.preflight"]


ACTIONS: Dict[str, Action] = {
    "preflight": Action(
        name="preflight", title="环境自检", needs_gpu=True, timeout_s=600,
        desc="检查 CUDA / 工具链 / arch 基线 / 任务一致性；换卡后先跑这个。",
        params=[], build=_build_preflight,
    ),
    "analysis": Action(
        name="analysis", title="分析 trace", needs_gpu=False, timeout_s=1800,
        desc="读 trace + arch.json，产出 bottlenecks.json / topk.csv / 摘要。只读 trace，不改代码。",
        params=[
            {"name": "trace", "label": "trace 路径", "type": "path",
             "placeholder": "runs/trace_8b_real/xxx.trace.json.gz"},
            {"name": "arch", "label": "硬件基线", "type": "path",
             "default": "arch/RTX_PRO_5000.json", "optional": True},
            {"name": "shape_hints", "label": "shape hints（可选）", "type": "path",
             "placeholder": "runs/shape_hints_xxx.json", "optional": True},
            {"name": "model", "label": "模型标签", "type": "text",
             "default": "qwen3-8b", "optional": True},
            {"name": "run_id", "label": "运行名", "type": "text",
             "placeholder": "留空自动生成", "optional": True},
        ],
        build=_build_analysis,
    ),
    "shape_hints": Action(
        name="shape_hints", title="生成 shape hints", needs_gpu=False, timeout_s=300,
        desc="自动扫描 kernel_api.log 里出现的已知核（marlin / fp8_blockwise_scaled_mm / outplace_fused_experts），"
             "逐个生成形状提示并合并成一个 JSON；不需要选算子。采集时勾「记录 kernel API 日志」即可。",
        params=[
            {"name": "log", "label": "kernel_api.log", "type": "path",
             "placeholder": "留空自动选最新一次采集的 kernel_api.log", "optional": True},
            {"name": "out", "label": "输出 JSON", "type": "text",
             "placeholder": "留空自动生成 runs/shape_hints_<ts>.json", "optional": True},
        ],
        build=_build_shape_hints,
    ),
    "fusion_eval": Action(
        name="fusion_eval", title="跑融合评测", needs_gpu=True, ok_codes=(0, 1), timeout_s=3600,
        desc="对一个 kind=fused 任务跑 best-of-N 与冷缓存 A/B，产出 KEEP/REVERT 与账本。"
             "fixture 模式用内置核验证链路，llm 模式调驱动模型生成候选。",
        params=[
            {"name": "task", "label": "融合任务目录", "type": "path",
             "placeholder": "runs/fusion_tasks_8b/fused__swiglu__8192x4096x24576"},
            {"name": "generator", "label": "生成器（fixture/llm）", "type": "text",
             "default": "fixture", "optional": True},
            {"name": "model", "label": "驱动模型（llm 时用）", "type": "text",
             "default": "ds-v4", "optional": True},
            {"name": "n", "label": "每轮候选数", "type": "number", "default": 1, "optional": True},
            {"name": "rounds", "label": "轮数", "type": "number", "default": 1, "optional": True},
            {"name": "repeat", "label": "A/B 配对数", "type": "number", "default": 20, "optional": True},
            {"name": "baseline_module", "label": "基线实现（可选）", "type": "text",
             "placeholder": "tools/swiglu_service_baseline.py:run", "optional": True},
        ],
        build=_build_fusion_eval,
    ),
    "e2e_ab": Action(
        name="e2e_ab", title="端到端 A/B", needs_gpu=True, danger=True,
        ok_codes=(0, 1), timeout_s=2400,
        desc="起带回灌的 8B → 交替块 A/B（关/开）→ 关服务。用于判定回灌有没有真实收益。"
             "会独占 GPU；INCONCLUSIVE 也会以退出码 1 结束（不是失败）。",
        params=[
            {"name": "model", "label": "模型", "type": "select", "source": "models",
             "default": "/model/Qwen3-8B"},
            {"name": "blocks", "label": "块数（偶数）", "type": "number", "default": 6, "optional": True},
            {"name": "seconds", "label": "每块秒数", "type": "number", "default": 20, "optional": True},
            {"name": "concurrency", "label": "并发", "type": "number", "default": 8, "optional": True},
            {"name": "prompt_chars", "label": "提示长度（字符）", "type": "number", "default": 12000, "optional": True},
            {"name": "max_tokens", "label": "每请求生成 tokens", "type": "number", "default": 8, "optional": True},
            {"name": "order", "label": "块顺序", "type": "select",
             "options": ["alternate", "abba"], "default": "alternate", "optional": True},
            {"name": "min_gain", "label": "最小可判收益", "type": "number",
             "default": 1.015, "optional": True},
            {"name": "run_id", "label": "运行名", "type": "text",
             "placeholder": "留空自动生成", "optional": True},
            {"name": "confirm", "label": "我确认独占 GPU", "type": "bool", "optional": False},
        ],
        build=_build_e2e,
    ),
    "capture": Action(
        name="capture", title="采集 trace", needs_gpu=True, danger=True, timeout_s=1800,
        desc="起 SGLang → 预热 → 采样 → 校验。自动处理：量化/MoE 模型会先短采 shape hints，"
             "profiler 始终记录张量形状（融合检测需要）；勾选「独占 GPU」会先停掉容器内的 "
             "sglang 服务（不动其它进程），结束后自动关闭。",
        params=[
            {"name": "model", "label": "模型", "type": "select", "source": "models",
             "default": "/model/Qwen3-8B"},
            {"name": "concurrency", "label": "并发", "type": "number", "default": 8, "optional": True},
            {"name": "isl", "label": "输入长度（词）", "type": "number", "default": 400, "optional": True},
            {"name": "seconds", "label": "打流秒数", "type": "number", "default": 30, "optional": True},
            {"name": "num_steps", "label": "采样步数", "type": "number", "default": 30, "optional": True},
            {"name": "mem_fraction", "label": "显存占比", "type": "number", "default": 0.55, "optional": True},
            {"name": "context_length", "label": "上下文长度", "type": "number", "default": 8192, "optional": True},
            {"name": "run_id", "label": "运行名", "type": "text",
             "placeholder": "留空自动生成", "optional": True},
            {"name": "confirm", "label": "我确认独占 GPU", "type": "bool", "optional": False},
        ],
        build=_build_capture,
    ),
    "fusion_tasks": Action(
        name="fusion_tasks", title="生成融合任务", needs_gpu=False, timeout_s=1200,
        desc="从瓶颈清单聚出 GEMM+gated-SiLU region，产出 kind=fused 的任务包。",
        params=[
            {"name": "report", "label": "bottlenecks.json", "type": "path",
             "placeholder": "runs/stage2_8b_real/bottlenecks.json"},
            {"name": "outdir", "label": "输出目录", "type": "text",
             "placeholder": "留空自动生成", "optional": True},
            {"name": "min_share", "label": "最小份额 %", "type": "number",
             "default": 0, "optional": True},
        ],
        build=_build_fusion_tasks,
    ),
}


@dataclass
class Job:
    id: str
    action: str
    params: Dict[str, Any]
    argv: List[str]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    returncode: Optional[int] = None
    error: str = ""

    @property
    def log_path(self) -> Path:
        return JOBS_DIR / f"{self.id}.log"

    @property
    def meta_path(self) -> Path:
        return JOBS_DIR / f"{self.id}.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "action": self.action, "status": self.status,
            "created_at": round(self.created_at, 3),
            "started_at": None if self.started_at is None else round(self.started_at, 3),
            "ended_at": None if self.ended_at is None else round(self.ended_at, 3),
            "returncode": self.returncode, "error": self.error,
            "argv_tail": " ".join(self.argv[-6:]),
            "log": str(self.log_path),
        }

    def save(self) -> None:
        try:
            self.meta_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        except OSError:
            pass


_jobs: Dict[str, Job] = {}
_queue: List[str] = []
_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_proc: Dict[str, subprocess.Popen] = {}


def _worker_loop() -> None:
    while True:
        with _lock:
            if not _queue:
                return
            jid = _queue.pop(0)
            job = _jobs.get(jid)
        if job is None:
            continue
        _run_job(job)


def _run_job(job: Job) -> None:
    action = ACTIONS[job.action]
    job.status = "running"
    job.started_at = time.time()
    job.save()
    try:
        with job.log_path.open("wb") as log:
            proc = subprocess.Popen(job.argv, cwd=str(REPO), stdout=log,
                                    stderr=subprocess.STDOUT, text=False,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            _proc[job.id] = proc
            try:
                rc = proc.wait(timeout=action.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -9
                job.error = f"超时（{action.timeout_s}s）"
            _proc.pop(job.id, None)
        job.returncode = rc
        ok = rc in action.ok_codes
        job.status = "done" if ok else ("canceled" if job.status == "canceled" else "failed")
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.ended_at = time.time()
        job.save()
        threading.Thread(target=_worker_loop, daemon=True).start()


def submit(action_name: str, params: Dict[str, Any]) -> Job:
    if action_name not in ACTIONS:
        raise ParamError(f"未知动作：{action_name}")
    argv = ACTIONS[action_name].build(params or {})
    job = Job(id=uuid.uuid4().hex[:12], action=action_name, params=params or {}, argv=argv)
    _jobs[job.id] = job
    job.save()
    global _worker
    with _lock:
        _queue.append(job.id)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()
    return job


def get(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def recent(limit: int = 20) -> List[Dict[str, Any]]:
    items = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return [j.to_dict() for j in items[:limit]]


def log_tail(job_id: str, limit: int = 200) -> str:
    job = _jobs.get(job_id)
    if job is None or not job.log_path.exists():
        return ""
    try:
        lines = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def cancel(job_id: str) -> bool:
    proc = _proc.get(job_id)
    if proc is None:
        return False
    job = _jobs.get(job_id)
    if job:
        job.status = "canceled"
        job.save()
    try:
        proc.send_signal(signal.SIGTERM)
        time.sleep(1.0)
        if proc.poll() is None:
            proc.kill()
    except OSError:
        return False
    return True
