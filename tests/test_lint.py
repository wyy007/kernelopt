"""静态检查（lint）与见证式反馈的单元测试。"""

import pytest

from optframe.contracts.lint import format_violations, lint_source
from optframe.inventory.codegen import judge_source

GOOD = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, (x * x).to(tl.bfloat16), mask=mask)


def run(*args):
    x = args[0]
    out = torch.empty_like(x)
    n = x.numel()
    _k[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
'''

NO_MASK = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def run(*args):
    a, b = args[0], args[1]
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    _k[(1,)](a, b, c, a.shape[0], b.shape[1], a.shape[1], BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    return c
'''


def test_good_candidate_passes():
    report = lint_source(GOOD, shapes=[[128]], dtype="bf16")
    assert report.ok, report.to_dict()


def test_missing_run_is_reported():
    report = lint_source("import torch\nimport triton\n", shapes=[[64]])
    codes = {v.code for v in report.errors}
    assert "run_missing" in codes
    assert "jit_missing" in codes


def test_unmasked_non_divisible_shape_blocks_evaluation():
    report = lint_source(NO_MASK, shapes=[[7835, 12288], [12288, 4096]], dtype="bf16")
    assert not report.ok
    v = next(v for v in report.errors if v.code == "mask_missing")
    assert v.witness["M"] == 7835
    assert v.witness["BLOCK_M"] == 128
    assert v.witness["余数"] == 27
    # 渲染文本要能直接进重试 prompt
    text = format_violations(report)
    assert "mask_missing" in text and "见证" in text


def test_smem_overflow_blocks():
    code = NO_MASK.replace("BLOCK_M=128", "BLOCK_M=256").replace("BLOCK_N=128", "BLOCK_N=256").replace(
        "BLOCK_K=32", "BLOCK_K=128")
    report = lint_source(code, shapes=[[256, 4096], [4096, 4096]], dtype="bf16")
    assert any(v.code == "smem_overflow" for v in report.errors)


def test_invalid_warps_blocks():
    code = GOOD.replace("BLOCK=1024)", "BLOCK=1024, num_warps=7)")
    report = lint_source(code, shapes=[[128]], dtype="bf16")
    assert any(v.code == "warps_invalid" for v in report.errors)


def test_torch_cheat_is_rejected():
    code = GOOD.replace("out = torch.empty_like(x)", "out = torch.mm(x, x)")
    report = lint_source(code, shapes=[[128]], dtype="bf16")
    assert any(v.code == "torch_compute" for v in report.errors)


def test_accumulator_warning_without_blocking():
    code = NO_MASK.replace("dtype=tl.float32", "dtype=tl.bfloat16")
    code = code.replace("acc = tl.zeros", "acc = tl.dot(a_tile, b_tile)\n    acc = tl.zeros")
    report = lint_source(code, shapes=[[128, 128], [128, 128]], dtype="bf16")
    assert any(v.code == "acc_dtype_unclear" for v in report.warnings)


def test_judge_reports_witness_index():
    """judge 失败时要给出坐标与两侧数值，而不是一句 allclose failed。"""
    import torch

    src = judge_source("bf16")
    ns = {}
    exec(src, ns)  # noqa: S102 - 测试用，源码由我们自己的模板生成
    ref = torch.zeros(4, 4, dtype=torch.bfloat16)
    cand = ref.clone()
    cand[2, 3] = 5.0
    out = ns["run"](ref, cand)
    assert out["ok"] is False
    assert out["index"] == [2, 3]
    assert "index (2, 3)" in out["error"]
