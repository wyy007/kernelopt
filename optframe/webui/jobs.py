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
JOBS_DIR = REPO / "runs" / "_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class ParamError(ValueError):
    pass


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
    timeout_s: int = 3600


def _build_analysis(p: Dict[str, Any]) -> List[str]:
    trace = _path_in_repo(p["trace"])
    arch = _path_in_repo(p.get("arch") or "arch/RTX_PRO_5000.json")
    run_id = _run_id(p.get("run_id"))
    return [sys.executable, "-m", "optframe.analysis.cli",
            "--trace", trace, "--arch", arch, "--workdir", f"runs/{run_id}",
            "--framework", "sglang", "--model", str(p.get("model") or "qwen3-8b")]


def _build_fusion_tasks(p: Dict[str, Any]) -> List[str]:
    report = _path_in_repo(p["report"])
    outdir = str(p.get("outdir") or f"runs/fusion_tasks_{int(time.time())}").strip()
    if not outdir.startswith("runs/"):
        raise ParamError("outdir 必须以 runs/ 开头")
    return [sys.executable, "-m", "optframe.inventory.fusion_tasks",
            "--report", report, "--outdir", outdir,
            "--arch", str(p.get("arch") or "sm_120"),
            "--dtype", str(p.get("dtype") or "bf16"),
            "--min-share", str(float(p.get("min_share") or 0.0))]


def _build_preflight(p: Dict[str, Any]) -> List[str]:
    return [sys.executable, "-m", "optframe.preflight"]


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
            {"name": "run_id", "label": "运行名", "type": "text",
             "placeholder": "留空自动生成", "optional": True},
        ],
        build=_build_analysis,
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
        job.status = "done" if rc == 0 else ("canceled" if job.status == "canceled" else "failed")
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
