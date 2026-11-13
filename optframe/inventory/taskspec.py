"""Task 数据契约与 TOML 序列化。

为什么自己写 TOML 输出而不引第三方库：
目标环境里 `tomli_w` 不一定在，而 task.toml 的结构是我们自己定的、非常窄，
手写序列化比加一个依赖更可控；读回来用标准库 `tomllib` 校验，能立刻发现格式错误。
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: 硬件 -> CUDA arch（用于 kernel 生成时的编译目标）
ARCH_BY_HARDWARE = {
    "L20": "sm89",
    "L4": "sm89",
    "L40": "sm89",
    "L40S": "sm89",
    "A100": "sm80",
    "A800": "sm80",
    "A10": "sm86",
    "H100": "sm90",
    "H200": "sm90",
    "H800": "sm90",
    "B200": "sm100",
    "B300": "sm100",
}

_SAFE_NAME = re.compile(r"[^a-z0-9_]+")


def arch_for_hardware(hardware: Optional[str], fallback: str = "sm89") -> str:
    """把硬件名映射成 arch；未知硬件给 fallback 并在 TaskSpec 里留痕。"""
    if not hardware:
        return fallback
    key = hardware.strip().upper().replace("NVIDIA", "").strip()
    for name, arch in ARCH_BY_HARDWARE.items():
        if key.startswith(name.upper()):
            return arch
    return fallback


def parse_shapes(shape_field: str) -> List[List[int]]:
    """把 TraceLens 的 `((2048, 2048), (2048,))` 解析成 [[2048,2048],[2048]]。

    空 tuple（标量/参数占位）会被丢弃；解析失败返回空列表。
    """
    if not shape_field:
        return []
    try:
        parsed = ast.literal_eval(shape_field)
    except (ValueError, SyntaxError):
        return []

    def collect(node: Any, out: List[List[int]]) -> None:
        if isinstance(node, (list, tuple)):
            if not node:
                return
            if all(isinstance(v, int) for v in node):
                out.append(list(node))
                return
            for item in node:
                collect(item, out)

    out: List[List[int]] = []
    collect(parsed, out)
    return out


def sanitize_name(op: str) -> str:
    """`aten::silu` -> `silu`；保证只含 [a-z0-9_]。"""
    name = op.split("::")[-1].lower()
    name = name.lstrip("_")
    name = _SAFE_NAME.sub("_", name).strip("_")
    if not name:
        name = "op"
    if name[0].isdigit():
        name = f"op_{name}"
    return name


def shape_digest(shapes: Sequence[Sequence[int]]) -> str:
    """形状指纹，用于同 op 不同 shape 的任务目录名。"""
    text = ";".join("x".join(str(d) for d in dims) for dims in shapes)
    if not text:
        return "noshape"
    if len(text) <= 40:
        return _SAFE_NAME.sub("_", text)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_toml_value(v) for v in value)
        return f"[{inner}]"
    raise TypeError(f"unsupported toml value: {type(value)!r}")


@dataclass
class TaskSpec:
    """一个算子优化任务的完整描述（对应方案 §四 的输出）。"""

    name: str
    op: str
    arch: str
    dtype: str
    shapes: List[List[int]]
    language: str = "triton"
    target_ratio: float = 1.0
    reference: str = "reference.py:run"
    judge: str = "judge.py:run"
    category: str = "other"
    action: str = "tune"
    baseline: str = "torch"
    #: microbench 实测的 SOL，仅用于记录，不参与验收
    sol_pct: Optional[float] = None
    pct_e2e: Optional[float] = None
    #: 是否允许升级到复杂档（CUDA/CuTe）
    allow_complex: bool = False
    escalation_reasons: List[str] = field(default_factory=list)
    #: 运行期升级规则（由 tiers 在执行完 best-of-N 后判定）
    runtime_rule: str = "best_of_3_lt_1.05"
    #: 生成时的附加元信息（来源 run、形状解析告警等）
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "arch": self.arch,
            "dtype": self.dtype,
            "shapes": self.shapes,
            "language": self.language,
            "target_ratio": round(self.target_ratio, 4),
            "reference": self.reference,
            "judge": self.judge,
            "category": self.category,
            "action": self.action,
            "baseline": self.baseline,
            "sol_pct": None if self.sol_pct is None else round(self.sol_pct, 2),
            "pct_e2e": None if self.pct_e2e is None else round(self.pct_e2e, 3),
            "allow_complex": self.allow_complex,
            "escalation_reasons": self.escalation_reasons,
            "runtime_rule": self.runtime_rule,
            "meta": self.meta,
        }

    def to_toml(self) -> str:
        """渲染 task.toml（结构与双档方案 §4.1 对齐）。"""
        lines = [
            "[task]",
            f"name = {_toml_value(self.name)}",
            f"op = {_toml_value(self.op)}",
            f"language = {_toml_value(self.language)}",
            f"arch = {_toml_value(self.arch)}",
            f"dtype = {_toml_value(self.dtype)}",
            f"category = {_toml_value(self.category)}",
            f"action = {_toml_value(self.action)}",
            f"baseline = {_toml_value(self.baseline)}",
            f"target_ratio = {_toml_value(float(self.target_ratio))}",
            f"reference = {_toml_value(self.reference)}",
            f"judge = {_toml_value(self.judge)}",
        ]
        # shapes: 数组的数组，逐行展开便于人工审阅
        shape_lines = ",\n  ".join(_toml_value(dims) for dims in self.shapes)
        lines.append(f"shapes = [\n  {shape_lines},\n]")

        lines.extend(["", "[timing]", 'window = "device"', "warmup = 10", "repeat = 50"])

        lines.extend([
            "",
            "[escalation]",
            f"allow_complex = {_toml_value(self.allow_complex)}",
            f"reasons = {_toml_value(self.escalation_reasons)}",
            f"runtime_rule = {_toml_value(self.runtime_rule)}",
        ])

        if self.pct_e2e is not None or self.sol_pct is not None:
            lines.extend(["", "[trace]"])
            if self.pct_e2e is not None:
                lines.append(f"pct_e2e = {_toml_value(float(self.pct_e2e))}")
            if self.sol_pct is not None:
                lines.append(f"sol_pct = {_toml_value(float(self.sol_pct))}")

        return "\n".join(lines) + "\n"
