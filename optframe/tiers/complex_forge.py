"""复杂档适配器：外挂 ForgeKernel（CUDA / CuTe）。

四段契约（对应总体方案 §六 6.1）：
    prepare()    我们的 Task -> ForgeKernel 任务目录（tasks/<slug>/）
    run()        新建 run 并执行，独立进程 + 硬预算 + 超时
    collect()    从 .artifacts/<slug>_<id>/ 读取最优 kernel.cu
    reevaluate() 不采信 ForgeKernel 自测数字，回共享 evaluator 复测

隔离原则：ForgeKernel 失败/超时只影响本档，不阻塞简单档；因此全部走 subprocess。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.evaluator import Baseline, evaluate_candidate
from ..contracts.schema import Candidate, Result
from ..contracts.task import Task

#: 单输入算子才适合自动转成 ForgeKernel 任务（多输入需要 definition 里多个具名输入）
_SUPPORTED_INPUTS = 1


class ForgeError(RuntimeError):
    """prepare/run/collect 环节的硬失败。"""


@dataclass
class ForgeConfig:
    """ForgeKernel 运行参数。"""

    forge_root: Path = Path("/model/wyy/ka_eval/ForgeKernel")
    python: str = "/model/wyy/forge-venv/bin/python"
    #: 单次 run 的墙钟预算（小时），同时会折算成 subprocess 超时
    budget_hours: float = 0.2
    max_meta: int = 1
    timeout_margin_s: int = 300
    #: 传给 ForgeKernel 的额外环境变量（如 OPENAI_BASE_URL / OPENAI_MODEL）
    env: Dict[str, str] = field(default_factory=dict)
    #: 是否在 run 前执行 `forge new`
    create_run: bool = True

    def timeout_s(self) -> int:
        return int(self.budget_hours * 3600) + self.timeout_margin_s


@dataclass
class Prepared:
    task_dir: Path
    slug: str
    task_toml: Path

    def to_dict(self) -> Dict[str, Any]:
        return {"task_dir": str(self.task_dir), "slug": self.slug}


@dataclass
class ForgeRun:
    ok: bool
    returncode: int
    artifact_dir: Optional[Path]
    log_tail: str
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "elapsed_s": round(self.elapsed_s, 1),
            "log_tail": self.log_tail[-2000:],
        }


def _slug(task: Task) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", task.name.lower()).strip("_")
    return base or "op"


def _axes(shapes: List[List[int]]) -> Dict[str, int]:
    return {f"d{i}": int(dim) for i, dim in enumerate(shapes[0])}


def prepare(task: Task, workspace: str | Path, config: ForgeConfig) -> Prepared:
    """把 Task 翻译成 ForgeKernel 任务目录。"""
    if len(task.shapes) != _SUPPORTED_INPUTS:
        raise ForgeError(
            f"complex tier currently supports single-input ops, got {len(task.shapes)}"
        )

    workspace = Path(workspace)
    slug = _slug(task)
    task_dir = workspace / "tasks" / slug
    (task_dir / "reference").mkdir(parents=True, exist_ok=True)

    toml = "\n".join([
        "[task]",
        'language = "cuda"',
        f'arch     = "{task.arch}"',
        "",
        "[reference]",
        'run    = "reference/run.py:run"',
        'source = "reference"',
        "",
        "[target]",
        f"ratio = {task.target_ratio:.4f}",
        "",
        "[budget]",
        f"hours    = {config.budget_hours:g}",
        f"max_meta = {config.max_meta}",
        "",
    ])
    (task_dir / "task.toml").write_text(toml, encoding="utf-8")

    dims = "x".join(str(d) for d in task.shapes[0])
    (task_dir / "description.md").write_text(
        "\n".join([
            f"# {task.op}",
            "",
            f"Input: contiguous {task.dtype_name} tensor of shape [{dims}] on CUDA.",
            "Output: same shape, same dtype, same layout.",
            "",
            "Correctness is checked against the reference implementation; performance is",
            "measured on the declared shape.",
            "",
        ]),
        encoding="utf-8",
    )

    (task_dir / "shapes.json").write_text(
        json.dumps([{"name": "one", "axes": _axes(task.shapes)}], indent=2),
        encoding="utf-8",
    )

    reference_src = (task.dir / "reference.py").read_text(encoding="utf-8")
    (task_dir / "reference" / "original_reference.py").write_text(reference_src, encoding="utf-8")
    (task_dir / "reference" / "run.py").write_text(
        "\n".join([
            '"""ForgeKernel reference entry: delegates to the task\'s reference."""',
            "",
            "import importlib.util",
            "from pathlib import Path",
            "",
            f"_HERE = Path(__file__).resolve().parent",
            '_SPEC = importlib.util.spec_from_file_location("task_reference", _HERE / "original_reference.py")',
            "_MOD = importlib.util.module_from_spec(_SPEC)",
            "_SPEC.loader.exec_module(_MOD)",
            "",
            "",
            "def run(*args):",
            "    return _MOD.run(*args)",
            "",
        ]),
        encoding="utf-8",
    )

    return Prepared(task_dir=task_dir, slug=slug, task_toml=task_dir / "task.toml")


def _base_cmd(config: ForgeConfig) -> List[str]:
    return [config.python, "-m", "forgekernel"]


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    config: ForgeConfig,
    timeout_s: int,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("FK_CLAUDE_BIN", str(config.forge_root / "agent_bridge_cli.sh"))
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.update(config.env)
    return subprocess.run(
        cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
        timeout=timeout_s, check=False,
    )


def create_run(prepared: Prepared, config: ForgeConfig) -> Optional[Path]:
    """`forge new tasks/<slug>`，返回新建的 .artifacts 目录。"""
    before = set(_artifact_dirs(config.forge_root))
    proc = _run_cmd(
        _base_cmd(config) + ["new", f"tasks/{prepared.slug}"],
        cwd=config.forge_root,
        config=config,
        timeout_s=600,
    )
    after = set(_artifact_dirs(config.forge_root))
    new = sorted(after - before)
    if new:
        return Path(new[-1])
    if proc.returncode != 0:
        raise ForgeError(f"forge new failed ({proc.returncode}): {(proc.stderr or '')[-800:]}")
    return None


def _artifact_dirs(root: Path) -> List[str]:
    base = Path(root) / ".artifacts"
    if not base.exists():
        return []
    return [str(p) for p in base.iterdir() if p.is_dir()]


def run(
    prepared: Prepared,
    config: ForgeConfig,
    *,
    artifact_dir: Optional[Path] = None,
    stub_cmd: Optional[List[str]] = None,
) -> ForgeRun:
    """执行（或续跑）一次 ForgeKernel run。

    stub_cmd: 测试用，替代真实 forge 命令（仍走独立进程 + 超时）。
    """
    started = time.time()
    if artifact_dir is None and config.create_run and stub_cmd is None:
        artifact_dir = create_run(prepared, config)

    if stub_cmd is not None:
        cmd = list(stub_cmd)
        if artifact_dir is not None:
            cmd.append(str(artifact_dir))
        timeout_s = config.timeout_s()
    elif artifact_dir is not None:
        cmd = _base_cmd(config) + ["run", str(artifact_dir)]
        timeout_s = config.timeout_s()
    else:
        return ForgeRun(ok=False, returncode=-1, artifact_dir=None,
                        log_tail="no artifact dir and create_run disabled",
                        elapsed_s=time.time() - started)

    try:
        proc = _run_cmd(cmd, cwd=config.forge_root, config=config, timeout_s=timeout_s)
        rc = proc.returncode
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired:
        return ForgeRun(ok=False, returncode=-9, artifact_dir=artifact_dir,
                        log_tail=f"timed out after {timeout_s}s",
                        elapsed_s=time.time() - started)

    return ForgeRun(
        ok=rc in (0, 2),          # 2 = 未达目标，但 run 本身完成
        returncode=rc,
        artifact_dir=artifact_dir,
        log_tail=tail,
        elapsed_s=time.time() - started,
    )


def collect(artifact_dir: str | Path) -> Candidate:
    """从 run 产物里取最优 kernel。

    注意：`operators/<op>/kernel.cu` 是给学生填的**脚手架**（写死 TORCH_CHECK(false)），
    真正落地的是 `ws/kernel.cu`。所以优先取 ws，并显式排除脚手架。
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise ForgeError(f"artifact dir not found: {artifact_dir}")

    run_json = artifact_dir / "run.json"
    meta: Dict[str, Any] = {}
    if run_json.exists():
        meta = json.loads(run_json.read_text(encoding="utf-8"))

    operators = artifact_dir / "operators"
    best_ratio = None
    for op_dir in sorted(operators.glob("*")) if operators.exists() else []:
        if not op_dir.is_dir():
            continue
        ratio = _best_ratio(op_dir / "paired_verdict.jsonl")
        if ratio is not None and (best_ratio is None or ratio > best_ratio):
            best_ratio = ratio

    ws_cu = artifact_dir / "ws" / "kernel.cu"
    cu_path: Optional[Path] = None
    if ws_cu.exists() and not _is_scaffold(ws_cu.read_text(encoding="utf-8")):
        cu_path = ws_cu

    if cu_path is None:
        for op_dir in sorted(operators.glob("*")) if operators.exists() else []:
            candidate_cu = op_dir / "kernel.cu"
            if candidate_cu.exists() and not _is_scaffold(
                candidate_cu.read_text(encoding="utf-8")
            ):
                cu_path = candidate_cu
                break

    if cu_path is None:
        raise ForgeError(f"no real kernel.cu produced under {artifact_dir}")

    return Candidate(
        source=cu_path.read_text(encoding="utf-8"),
        language="cuda",
        generator="forgekernel",
        meta={
            "artifact_dir": str(artifact_dir),
            "kernel_path": str(cu_path),
            "run_id": meta.get("id"),
            "status": meta.get("status"),
            "target_met": meta.get("target_met"),
            "forge_ratio": best_ratio,
        },
    )


def _is_scaffold(source: str) -> bool:
    """识别 ForgeKernel 给学生留的空壳（还没有真正写入 kernel）。"""
    return "write the kernel" in source or (
        'TORCH_CHECK(false' in source and len(source) < 1200
    )


def _best_ratio(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = rec.get("median_ratio") or rec.get("ratio")
        if isinstance(value, (int, float)):
            best = float(value) if best is None else max(best, float(value))
    return best


def reevaluate(
    task: Task,
    candidate: Candidate,
    baseline: Baseline,
    inputs: List[Any],
    workdir: str | Path,
    *,
    warmup: int = 10,
    repeat: int = 50,
) -> Result:
    """用共享 evaluator 复测，忽略 ForgeKernel 自报的 ratio。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "collected_kernel.cu").write_text(candidate.source, encoding="utf-8")
    result = evaluate_candidate(
        task, candidate, baseline, inputs, workdir, warmup=warmup, repeat=repeat
    )
    result.artifacts.setdefault("forge_meta", candidate.meta)
    result.artifacts["forge_reported_ratio"] = candidate.meta.get("forge_ratio")
    return result
