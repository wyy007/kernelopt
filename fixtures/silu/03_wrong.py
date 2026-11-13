"""Triton SiLU 候选（fixture 3）：数学写错，用于验证正确性拦截。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.sigmoid(x) + 1.0
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


def run(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _silu_kernel[grid](x, out, n_elements, BLOCK=BLOCK, num_warps=4)
    return out
