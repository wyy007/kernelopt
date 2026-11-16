"""服务同款的 SwiGLU 基线（预分配缓冲）。

为什么需要它：
    直接用 `torch.mm(a, b)` 做基线时，每次调用要分配一个 (M, 2H) 中间结果；
    而 cuBLASLt 的算法/workspace 选择依赖进程内的 allocator 状态，
    导致同一份基线在不同进程里绝对耗时差 13%（2026-11-16 实测 8.09ms vs 9.27ms），
    比被测效应（~10%）还大。

    真实 serving 不会每次 forward 都 malloc 这么大一块（引擎会预分配 / 复用池），
    所以"预分配 + out=" 才是服务同款形态，也是唯一可跨进程比较的口径。

用法：
    python -m optframe.tiers.fusion --task <fused_task> \
        --baseline-module tools/swiglu_service_baseline.py:run

自测：
    python tools/swiglu_service_baseline.py --m 8192
"""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

#: (m, two_h, dtype, device) -> (full, out)
_BUFFERS: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _act_and_mul_kernel(X, OUT, n_rows, HALF_N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n_rows * HALF_N
    mask = offs < total
    row = offs // HALF_N
    col = offs % HALF_N
    g = tl.load(X + row * (2 * HALF_N) + col, mask=mask, other=0.0)
    u = tl.load(X + row * (2 * HALF_N) + col + HALF_N, mask=mask, other=0.0)
    g32 = g.to(tl.float32)
    out = g32 * tl.sigmoid(g32) * u.to(tl.float32)
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


def _buffers(m: int, two_h: int, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (m, two_h, dtype, str(device))
    bufs = _BUFFERS.get(key)
    if bufs is None:
        full = torch.empty(m, two_h, dtype=dtype, device=device)
        out = torch.empty(m, two_h // 2, dtype=dtype, device=device)
        _BUFFERS[key] = (full, out)
        bufs = (full, out)
    return bufs


def clear_cache() -> None:
    """释放缓存（测试或换形状时用）。"""
    _BUFFERS.clear()


def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """服务同款：预分配的 GEMM + 单个 act_and_mul kernel。"""
    m, _k = x.shape
    two_h = w.shape[1]
    half = two_h // 2
    full, out = _buffers(m, two_h, x.dtype, x.device)

    torch.mm(x, w, out=full)
    grid = (triton.cdiv(m * half, 1024),)
    _act_and_mul_kernel[grid](full, out, m, half, BLOCK=1024)
    return out


def _reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    full = torch.mm(x, w)
    half = full.shape[1] // 2
    return torch.nn.functional.silu(full[:, :half]) * full[:, half:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--half", type=int, default=12288)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("requires CUDA")
        return 2

    dt = torch.bfloat16
    x = torch.randn(args.m, args.k, device="cuda", dtype=dt)
    w = torch.randn(args.k, 2 * args.half, device="cuda", dtype=dt)

    got = run(x, w)
    ref = _reference(x, w)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"shape M={args.m} K={args.k} half={args.half}")
    print(f"correctness max_abs_err={err:.3f}")
    print(f"buffer reused: {run(x, w).data_ptr() == got.data_ptr()}")

    flush = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")

    def bench(fn, iters=args.iters):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(iters):
            flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            xs.append(s.elapsed_time(e))
        return statistics.median(xs)

    t_alloc = bench(lambda: _reference(x, w))
    t_pre = bench(lambda: run(x, w))
    print(f"allocating reference = {t_alloc:.3f} ms")
    print(f"preallocated baseline = {t_pre:.3f} ms  ({t_alloc / t_pre:.3f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
