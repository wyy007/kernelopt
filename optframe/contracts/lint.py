"""候选源码的静态检查（上 GPU 之前）。

设计动机（2026-11-16，参考 Ave 的 dense process reward 思路）：
    现在的反馈只有"跑完才知道错"——评测一次要几十秒，错误信息还是一句异常。
    这里把能在编译期判定的问题提前拦下来，并且给出**可定位的见证**
    （哪一行、哪个 tile、为什么会越界），直接进重试 prompt。

与 `source_guard.py` 的分工：
    - source_guard：反作弊 + 最小结构（不许调 torch 计算、必须有 run/@triton.jit）
    - lint：形状/掩码/累加精度/tile 合法性/SMEM 预算，面向"会静默错或静默慢"的问题

判定原则：errors 阻断评测；warnings 只记录，不阻断（写进 ledger 便于事后归因）。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import os

from .source_guard import detect_disallowed_torch


def smem_limit_bytes() -> int:
    """当前生效的共享内存上限（可被 OPTFRAME_SMEM_LIMIT 覆盖）。"""
    try:
        override = int(os.environ.get("OPTFRAME_SMEM_LIMIT", "0") or 0)
    except ValueError:
        override = 0
    return override or SMEM_LIMIT_BYTES

#: 单 block 可用共享内存上限的**保守默认值**。
#: 2026-11-16 实测：RTX PRO 5000 上 Triton 报 "Hardware limit: 101376"，比硬件 opt-in 的
#: 227KB 小得多（Triton 自己扣掉了预留），所以默认按 96KB 卡，避免"lint 通过、运行时 OOM"。
#: 需要放宽时用环境变量 OPTFRAME_SMEM_LIMIT 覆盖。
SMEM_LIMIT_BYTES = 96 * 1024
#: 允许的 tile 尺寸范围（Triton 实测可用区间）
BLOCK_MIN, BLOCK_MAX = 16, 8192
VALID_WARPS = {1, 2, 4, 8, 16, 32}


@dataclass
class Violation:
    code: str
    message: str
    line: Optional[int] = None
    witness: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"     # error | warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "witness": self.witness,
            "severity": self.severity,
        }

    def render(self) -> str:
        head = f"[{self.code}]" + (f" 第 {self.line} 行" if self.line else "")
        lines = [f"{head}: {self.message}"]
        if self.witness:
            pairs = ", ".join(f"{k}={v}" for k, v in self.witness.items())
            lines.append(f"  见证: {pairs}")
        return "\n".join(lines)


@dataclass
class LintReport:
    errors: List[Violation] = field(default_factory=list)
    warnings: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [v.to_dict() for v in self.errors],
            "warnings": [v.to_dict() for v in self.warnings],
        }


# --------------------------------------------------------------------------- #
# 解析辅助
# --------------------------------------------------------------------------- #

_BLOCK_RE = re.compile(r"\bBLOCK(?:_[MNK])?\s*[:=]\s*(\d+)")
_WARPS_RE = re.compile(r"num_warps\s*=\s*(\d+)")
_STAGES_RE = re.compile(r"num_stages\s*=\s*(\d+)")


def _parse_consts(code: str) -> Dict[str, int]:
    consts: Dict[str, int] = {}
    for m in _BLOCK_RE.finditer(code):
        pass  # 尺寸由调用方按名字取，这里只收集 warps/stages
    m = _WARPS_RE.search(code)
    if m:
        consts["num_warps"] = int(m.group(1))
    m = _STAGES_RE.search(code)
    if m:
        consts["num_stages"] = int(m.group(1))
    return consts


def _tile_dims(code: str) -> Dict[str, int]:
    """取 BLOCK_M / BLOCK_N / BLOCK_K / BLOCK（取最后一次赋值）。"""
    out: Dict[str, int] = {}
    for name, value in re.findall(r"\b(BLOCK_[MNK]|BLOCK)\s*[:=]\s*(\d+)", code):
        out[name] = int(value)
    return out


def _has_run(tree: ast.Module) -> Optional[ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            return node  # type: ignore[return-value]
    return None


# --------------------------------------------------------------------------- #
# 各项检查
# --------------------------------------------------------------------------- #

def _check_signature(code: str, tree: ast.Module, n_inputs: Optional[int],
                     report: LintReport) -> None:
    fn = _has_run(tree)
    if fn is None:
        report.errors.append(Violation(
            code="run_missing",
            message="缺少模块级函数 `run(*args)`；评测入口找不到调用点",
        ))
        return

    args = fn.args
    has_varargs = args.vararg is not None
    positional = len(args.posonlyargs) + len(args.args)
    if not has_varargs and n_inputs is not None and positional != n_inputs:
        report.errors.append(Violation(
            code="run_arity",
            message=f"`run` 接受 {positional} 个位置参数，但该任务输入 {n_inputs} 个张量",
            line=fn.lineno,
            witness={"run_params": positional, "task_inputs": n_inputs},
        ))

    has_return = any(isinstance(n, ast.Return) for n in ast.walk(fn))
    if not has_return:
        report.errors.append(Violation(
            code="run_no_return",
            message="`run` 没有任何 return，评测拿不到输出张量",
            line=fn.lineno,
        ))


def _check_imports(code: str, report: LintReport) -> None:
    if "import torch" not in code:
        report.errors.append(Violation(
            code="import_torch_missing",
            message="缺少 `import torch`；候选必须自己分配输出张量",
        ))
    if "import triton" not in code:
        report.errors.append(Violation(
            code="import_triton_missing",
            message="缺少 `import triton`",
        ))
    if "triton.language as tl" not in code and "import triton.language" not in code:
        report.errors.append(Violation(
            code="import_tl_missing",
            message="缺少 `import triton.language as tl`；kernel 内部无法使用 tl.*",
        ))


def _check_masks(code: str, shapes: Optional[Sequence[Sequence[int]]],
                 tiles: Dict[str, int], report: LintReport) -> None:
    """形状不整除又没有 mask → 必然越界（静默读脏或 crash）。"""
    if not shapes:
        return
    unmasked = "mask=" not in code
    if not unmasked:
        return
    block_m = tiles.get("BLOCK_M") or tiles.get("BLOCK") or 0
    block_n = tiles.get("BLOCK_N") or 0
    for shape in shapes:
        if len(shape) != 2:
            continue
        m, n = int(shape[0]), int(shape[1])
        for dim, block, axis in ((m, block_m, "M"), (n, block_n, "N")):
            if block and dim % block:
                report.errors.append(Violation(
                    code="mask_missing",
                    message=(
                        f"kernel 里没有 `mask=`，而 {axis}={dim} 不是 "
                        f"BLOCK_{axis}={block} 的整数倍，最后一块会越界读/写"
                    ),
                    witness={
                        f"{axis}": dim,
                        f"BLOCK_{axis}": block,
                        "余数": dim % block,
                        "越界行区间": f"[{dim - dim % block}, {dim + block - dim % block})",
                    },
                ))
                return


def _check_accumulator(code: str, dtype: Optional[str], report: LintReport) -> None:
    if "tl.dot(" not in code and "tl.dotc(" not in code:
        return
    if "tl.float32" in code:
        return
    report.warnings.append(Violation(
        code="acc_dtype_unclear",
        message="使用了 tl.dot 但没有显式 `tl.zeros(..., dtype=tl.float32)` 累加；fp16/bf16 累加会静默掉精度",
        severity="warning",
        witness={"dtype": dtype or "unknown"},
    ))


def _check_tiles(tiles: Dict[str, int], consts: Dict[str, int], report: LintReport) -> None:
    for name, value in tiles.items():
        if value < BLOCK_MIN or value > BLOCK_MAX:
            report.errors.append(Violation(
                code="block_out_of_range",
                message=f"{name}={value} 超出可用范围 [{BLOCK_MIN}, {BLOCK_MAX}]",
                witness={name: value},
            ))
        elif value & (value - 1):
            report.warnings.append(Violation(
                code="block_not_power_of_two",
                message=f"{name}={value} 不是 2 的幂，Triton 会退化或编译失败",
                severity="warning",
                witness={name: value},
            ))

    warps = consts.get("num_warps")
    if warps is not None and warps not in VALID_WARPS:
        report.errors.append(Violation(
            code="warps_invalid",
            message=f"num_warps={warps} 不在 {sorted(VALID_WARPS)} 之内",
            witness={"num_warps": warps},
        ))


def _check_smem(tiles: Dict[str, int], consts: Dict[str, int], report: LintReport) -> None:
    bm, bn, bk = tiles.get("BLOCK_M"), tiles.get("BLOCK_N"), tiles.get("BLOCK_K")
    if not (bm and bn and bk):
        return
    stages = consts.get("num_stages", 3)
    # 双缓冲 A/B tile，bf16 按 2 字节估算（fp8 会更小，属于保守估计）
    per_stage = (bm * bk + bk * bn) * 2
    total = per_stage * max(1, stages)
    limit = smem_limit_bytes()
    if total > limit:
        report.errors.append(Violation(
            code="smem_overflow",
            message=(
                f"估算共享内存 {total / 1024:.0f} KB（BLOCK_M={bm}, BLOCK_N={bn}, "
                f"BLOCK_K={bk}, stages={stages}）超过上限 {limit / 1024:.0f} KB；"
                "调小 BLOCK 或 num_stages"
            ),
            witness={"per_stage_kb": round(per_stage / 1024, 1), "stages": stages},
        ))


def _check_torch_cheat(code: str, report: LintReport) -> None:
    reason = detect_disallowed_torch(code)
    if reason:
        report.errors.append(Violation(
            code="torch_compute",
            message=reason,
        ))


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def lint_source(
    code: Optional[str],
    *,
    shapes: Optional[Sequence[Sequence[int]]] = None,
    dtype: Optional[str] = None,
    language: str = "triton",
) -> LintReport:
    """静态检查一个候选。errors 非空则不应进入 GPU 评测。"""
    report = LintReport()
    if not code or not code.strip():
        report.errors.append(Violation(code="empty_source", message="候选源码为空"))
        return report

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report.errors.append(Violation(
            code="syntax_error",
            message=f"Python 语法错误: {exc.msg}",
            line=exc.lineno,
        ))
        return report

    if language != "triton":
        return report

    n_inputs = len(shapes) if shapes else None
    _check_imports(code, report)
    _check_signature(code, tree, n_inputs, report)

    if not any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            (getattr((d.func if isinstance(d, ast.Call) else d), "attr", None)
             or getattr((d.func if isinstance(d, ast.Call) else d), "id", None)) == "jit"
            for d in n.decorator_list
        )
        for n in ast.walk(tree)
    ):
        report.errors.append(Violation(
            code="jit_missing",
            message="没有找到 @triton.jit kernel",
        ))

    tiles = _tile_dims(code)
    consts = _parse_consts(code)
    _check_masks(code, shapes, tiles, report)
    _check_accumulator(code, dtype, report)
    _check_tiles(tiles, consts, report)
    _check_smem(tiles, consts, report)
    _check_torch_cheat(code, report)
    return report


def format_violations(report: LintReport, limit: int = 6) -> str:
    """把 violations 渲染成可直接塞进重试 prompt 的文本。"""
    items = report.errors + report.warnings
    if not items:
        return ""
    head = f"静态检查发现 {len(report.errors)} 个阻断问题、{len(report.warnings)} 个警告："
    body = "\n".join(v.render() for v in items[:limit])
    return f"{head}\n{body}"
