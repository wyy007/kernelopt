"""Task 加载：task.toml -> 可执行任务对象。

任务目录由阶段3 生成（task.toml + reference.py + judge.py），本模块负责
把它变成"能造输入、能加载真值、能加载 judge"的对象，供 evaluator/各档位复用。
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

import torch

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int8": torch.int8,
    "fp8": torch.float8_e4m3fn,
}

_MODULE_COUNTER = itertools.count()


class TaskError(RuntimeError):
    """任务目录不合法或缺文件。"""


def dtype_of(name: str) -> torch.dtype:
    key = (name or "").lower().split("+")[0]
    if key not in _DTYPE_MAP:
        raise TaskError(f"unsupported dtype: {name!r}")
    return _DTYPE_MAP[key]


def load_module(path: Path, alias: str = "taskmod") -> ModuleType:
    """以唯一模块名加载 .py，避免同名覆盖。"""
    path = Path(path)
    if not path.exists():
        raise TaskError(f"module not found: {path}")
    name = f"{alias}_{next(_MODULE_COUNTER)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise TaskError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def resolve_callable(module: ModuleType, spec: str):
    """`reference.py:run` -> module.run"""
    if ":" not in spec:
        raise TaskError(f"callable spec must be 'file:attr', got {spec!r}")
    attr = spec.split(":", 1)[1]
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise TaskError(f"{spec} does not resolve to a callable")
    return fn


@dataclass
class Task:
    """一个算子优化任务。"""

    dir: Path
    name: str
    op: str
    arch: str
    dtype_name: str
    shapes: List[List[int]]
    target_ratio: float
    reference_spec: str = "reference.py:run"
    judge_spec: str = "judge.py:run"
    category: str = "other"
    allow_complex: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- 惰性加载 ----
    _reference_module: Optional[ModuleType] = None
    _judge_module: Optional[ModuleType] = None

    @property
    def dtype(self) -> torch.dtype:
        return dtype_of(self.dtype_name)

    @property
    def reference_module(self) -> ModuleType:
        if self._reference_module is None:
            self._reference_module = load_module(self.dir / "reference.py", "ref")
        return self._reference_module

    @property
    def judge_module(self) -> ModuleType:
        if self._judge_module is None:
            self._judge_module = load_module(self.dir / "judge.py", "judge")
        return self._judge_module

    def make_inputs(self, device: str = "cuda", seed: int = 0) -> List[torch.Tensor]:
        module = self.reference_module
        maker = getattr(module, "make_inputs", None)
        if callable(maker):
            return list(maker(device=device, dtype=self.dtype, seed=seed))
        # 兜底：按 shapes 造随机张量（reference 里没写 make_inputs 时）
        torch.manual_seed(seed)
        out = []
        for dims in self.shapes:
            if self.dtype.is_floating_point:
                out.append(torch.randn(*dims, device=device, dtype=self.dtype))
            else:
                out.append(torch.randint(0, 8, tuple(dims), device=device, dtype=self.dtype))
        return out

    def reference_run(self, inputs: Sequence[torch.Tensor]):
        fn = resolve_callable(self.reference_module, self.reference_spec)
        return fn(*inputs)

    def judge_check(self, ref_out, cand_out) -> Dict[str, Any]:
        fn = resolve_callable(self.judge_module, self.judge_spec)
        return fn(ref_out, cand_out, dtype=self.dtype_name)


def load_task(task_dir: str | Path) -> Task:
    """读取任务目录下的 task.toml。"""
    task_dir = Path(task_dir)
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        raise TaskError(f"task.toml not found in {task_dir}")

    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    block = data.get("task") or {}
    if not block:
        raise TaskError("task.toml missing [task] section")

    for required in ("name", "op", "arch", "dtype", "shapes"):
        if required not in block:
            raise TaskError(f"task.toml missing task.{required}")

    return Task(
        dir=task_dir,
        name=str(block["name"]),
        op=str(block["op"]),
        arch=str(block["arch"]),
        dtype_name=str(block["dtype"]),
        shapes=[list(dims) for dims in block["shapes"]],
        target_ratio=float(block.get("target_ratio", 1.0)),
        reference_spec=str(block.get("reference", "reference.py:run")),
        judge_spec=str(block.get("judge", "judge.py:run")),
        category=str(block.get("category", "other")),
        allow_complex=bool((data.get("escalation") or {}).get("allow_complex", False)),
        raw=data,
    )
