"""按算子生成 reference.py / judge.py。

硬约束：**只支持能明确写出语义的算子**。遇到不认识的 op 一律拒绝生成，
绝不"猜一个差不多的 PyTorch 调用"——错误 reference 会让正确性校验整体失效，
比不生成任务坏得多。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

#: 归一化后的 op 名 -> 参考实现模板
@dataclass(frozen=True)
class OpTemplate:
    n_inputs: int
    body: str
    imports: Sequence[str] = ()
    note: str = ""


OP_TEMPLATES: Dict[str, OpTemplate] = {
    # ---- 一元元素级
    "silu": OpTemplate(1, "return F.silu(x)", ("torch.nn.functional as F",)),
    "gelu": OpTemplate(1, "return F.gelu(x)", ("torch.nn.functional as F",)),
    "relu": OpTemplate(1, "return F.relu(x)", ("torch.nn.functional as F",)),
    "sigmoid": OpTemplate(1, "return torch.sigmoid(x)"),
    "tanh": OpTemplate(1, "return torch.tanh(x)"),
    "exp": OpTemplate(1, "return torch.exp(x)"),
    "neg": OpTemplate(1, "return torch.neg(x)"),
    # ---- 二元元素级
    "mul": OpTemplate(2, "return x * y"),
    "add": OpTemplate(2, "return x + y"),
    "sub": OpTemplate(2, "return x - y"),
    "div": OpTemplate(2, "return x / y"),
    "maximum": OpTemplate(2, "return torch.maximum(x, y)"),
    # ---- 矩阵
    "mm": OpTemplate(2, "return torch.mm(x, y)"),
    "bmm": OpTemplate(2, "return torch.bmm(x, y)"),
    "matmul": OpTemplate(2, "return torch.matmul(x, y)"),
    # ---- 归一化 / 归约
    "softmax": OpTemplate(1, "return F.softmax(x, dim=-1)", ("torch.nn.functional as F",)),
    "log_softmax": OpTemplate(1, "return F.log_softmax(x, dim=-1)", ("torch.nn.functional as F",)),
    "layer_norm": OpTemplate(
        1,
        "return F.layer_norm(x, (x.shape[-1],))",
        ("torch.nn.functional as F",),
        note="无 affine 参数；带 weight/bias 的场景需另建任务",
    ),
    "rms_norm": OpTemplate(
        1,
        "variance = x.pow(2).mean(dim=-1, keepdim=True)\n"
        "return x * torch.rsqrt(variance + 1e-6)",
        (),
        note="eps=1e-6 与主流实现一致",
    ),
    "sum": OpTemplate(1, "return torch.sum(x)"),
}


def normalize_op(op: str) -> str:
    """`aten::_softmax` -> `softmax`。"""
    return op.split("::")[-1].lower().lstrip("_")


def lookup(op: str) -> Optional[OpTemplate]:
    return OP_TEMPLATES.get(normalize_op(op))


REFERENCE_HEADER = '''"""自动生成的 PyTorch 参考实现（真值）。

算子: {op}
形状: {shapes}
dtype: {dtype}

由 optframe.inventory.trace_to_tasks 生成，请勿手改；重新生成会覆盖。
用途：作为正确性 judge 的真值基准，不参与性能验收。
"""

import torch
{extra_imports}

DTYPE = torch.{torch_dtype}
SHAPES = {shapes_repr}
OP = {op_repr}
NOTE = {note_repr}


def _rand(shape, device, dtype):
    """按 dtype 生成可复现输入；fp8 先按 bf16 生成再转换。"""
    if not dtype.is_floating_point:
        return torch.randint(0, 8, tuple(shape), device=device, dtype=dtype)
    itemsize = getattr(dtype, "itemsize", 2)
    if itemsize == 1:
        base = torch.randn(tuple(shape), device=device, dtype=torch.bfloat16)
        return base.to(dtype)
    return torch.randn(tuple(shape), device=device, dtype=dtype)


def make_inputs(device="cuda", dtype=None, seed: int = 0):
    """按 SHAPES 顺序造输入张量。"""
    dtype = dtype or DTYPE
    torch.manual_seed(seed)
    return [_rand(shape, device, dtype) for shape in SHAPES]


def run(*args):
{body}
'''

JUDGE_HEADER = '''"""自动生成的正确性 judge。

比较 reference 与 candidate 的输出，按 dtype 选择容差。
返回 dict，供共享 evaluator 直接消费。

由 optframe.inventory.trace_to_tasks 生成，请勿手改。
"""

import torch

DEFAULT_DTYPE = {dtype_repr}

#: dtype -> (atol, rtol)
TOLERANCES = {{
    "fp64": (1e-9, 1e-9),
    "fp32": (1e-5, 1e-5),
    "fp16": (1e-3, 1e-3),
    "bf16": (2e-2, 2e-2),
    "int8": (0.0, 0.0),
}}


def _tol(dtype: str):
    return TOLERANCES.get(dtype, (1e-3, 1e-3))


def _as_list(value):
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def run(ref_out, cand_out, dtype=None, atol=None, rtol=None):
    """逐输出比较，任一不过即 ok=False。"""
    dtype = dtype or DEFAULT_DTYPE
    default_atol, default_rtol = _tol(dtype)
    atol = default_atol if atol is None else float(atol)
    rtol = default_rtol if rtol is None else float(rtol)

    refs, cands = _as_list(ref_out), _as_list(cand_out)
    if len(refs) != len(cands):
        return {{
            "ok": False,
            "error": f"output count mismatch: ref={{len(refs)}} cand={{len(cands)}}",
            "atol": atol,
            "rtol": rtol,
        }}

    max_abs_err = 0.0
    max_rel_err = 0.0
    for ref, cand in zip(refs, cands):
        if not torch.is_tensor(ref) or not torch.is_tensor(cand):
            if ref != cand:
                return {{"ok": False, "error": "non-tensor output mismatch",
                        "atol": atol, "rtol": rtol}}
            continue
        if ref.shape != cand.shape:
            return {{
                "ok": False,
                "error": f"shape mismatch: ref={{tuple(ref.shape)}} cand={{tuple(cand.shape)}}",
                "atol": atol,
                "rtol": rtol,
            }}
        ref_f = ref.detach().float()
        cand_f = cand.detach().float()
        diff = (ref_f - cand_f).abs()
        max_abs_err = max(max_abs_err, float(diff.max().item()) if diff.numel() else 0.0)
        denom = ref_f.abs().clamp_min(1e-12)
        if diff.numel():
            max_rel_err = max(max_rel_err, float((diff / denom).max().item()))
        if not torch.allclose(cand_f, ref_f, atol=atol, rtol=rtol, equal_nan=True):
            return {{
                "ok": False,
                "error": "allclose failed",
                "atol": atol,
                "rtol": rtol,
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
            }}

    return {{
        "ok": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
    }}


def main():
    """自检：reference 与自身比较必须通过。"""
    import reference

    inputs = reference.make_inputs(device="cpu", dtype=torch.float32)
    out = reference.run(*inputs)
    result = run(out, out, dtype="fp32")
    print(result)
    assert result["ok"], result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: 统一 dtype 写法 -> torch dtype 名
_TORCH_DTYPE = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "int8": "int8",
    "fp8": "float8_e4m3fn",
}


def torch_dtype_name(dtype: str) -> str:
    key = (dtype or "bf16").lower()
    if key in _TORCH_DTYPE:
        return _TORCH_DTYPE[key]
    # 复合 dtype（如 bf16+scalar）取第一种
    first = key.split("+")[0]
    return _TORCH_DTYPE.get(first, "bfloat16")


#: 支持的计算 dtype（拒绝未知值，避免生成 "猜出来的" 参考实现）
SUPPORTED_DTYPES = frozenset(_TORCH_DTYPE)


def reference_source(op: str, shapes: Sequence[Sequence[int]], dtype: str) -> Optional[str]:
    """生成 reference.py 源码；不支持的算子返回 None。"""
    template = lookup(op)
    if template is None:
        return None
    if len(shapes) < template.n_inputs:
        return None

    body_lines = ["    " + line for line in template.body.splitlines()]
    extra = "\n".join(f"import {imp}" for imp in template.imports)
    arg_names = ["x", "y", "z", "w"][: template.n_inputs]
    prelude = "".join(
        f"    {name} = args[{idx}]\n" for idx, name in enumerate(arg_names)
    )

    return REFERENCE_HEADER.format(
        op=op,
        shapes="; ".join("x".join(str(d) for d in dims) for dims in shapes),
        dtype=dtype,
        extra_imports=extra,
        torch_dtype=torch_dtype_name(dtype),
        shapes_repr=repr([list(dims) for dims in shapes]),
        op_repr=repr(op),
        note_repr=repr(template.note),
        body=prelude + "\n".join(body_lines),
    )


def judge_source(dtype: str) -> str:
    return JUDGE_HEADER.format(dtype_repr=repr(dtype))
