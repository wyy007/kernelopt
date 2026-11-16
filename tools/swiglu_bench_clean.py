#!/usr/bin/env python3
"""无分配噪音的 SwiGLU 对照：所有缓冲区预分配，计时区间内只有 kernel。

三次度：
  mm_only        : torch.mm(a, b, out=full)
  act_only       : act_and_mul(full) -> out
  mm_plus_act    : 两者顺序发射（服务同款路径）
  fused          : 单 kernel（双累加器）

ratio = mm_plus_act / fused
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch

sys.path.insert(0, "/model/wyy/optframe")

from tools.fused_gemm_act import _act_and_mul_kernel, fused_swiglu  # noqa: E402
import triton  # noqa: E402

DT = torch.bfloat16
PCK = 1231.5


def timed(fn, flush, warmup=8, iters=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--half", type=int, default=12288)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    M, K, H = args.m, args.k, args.half
    a = torch.randn(M, K, device="cuda", dtype=DT)
    b = torch.randn(K, 2 * H, device="cuda", dtype=DT)
    full = torch.empty(M, 2 * H, device="cuda", dtype=DT)
    out = torch.empty(M, H, device="cuda", dtype=DT)
    flush = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(M * H, 1024),)

    def mm_only():
        torch.mm(a, b, out=full)

    def act_only():
        _act_and_mul_kernel[grid](full, out, M, H, BLOCK=1024)

    def mm_plus_act():
        torch.mm(a, b, out=full)
        _act_and_mul_kernel[grid](full, out, M, H, BLOCK=1024)

    def fused():
        fused_swiglu(a, b, block_m=256, block_n=64, block_k=64, warps=8, stages=3)

    # 交错 A/B（同进程、同缓冲、逐对交替），消掉时钟漂移
    for _ in range(8):
        mm_plus_act(); fused()
    torch.cuda.synchronize()
    pairs = []
    for i in range(args.iters):
        if i % 2 == 0:
            a1, a2 = mm_plus_act, fused
        else:
            a1, a2 = fused, mm_plus_act
        flush.zero_(); s1, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        s1.record(); a1(); e1.record(); torch.cuda.synchronize()
        flush.zero_(); s2, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        s2.record(); a2(); e2.record(); torch.cuda.synchronize()
        t1, t2 = s1.elapsed_time(e1), s2.elapsed_time(e2)
        base, cand = (t1, t2) if a1 is mm_plus_act else (t2, t1)
        pairs.append(base / cand)

    t_mm = timed(mm_only, flush, iters=args.iters)
    t_act = timed(act_only, flush, iters=args.iters)
    t_ma = timed(mm_plus_act, flush, iters=args.iters)
    t_f = timed(fused, flush, iters=args.iters)

    print(f"shape M={M} K={K} half={H} | iters={args.iters} | 预分配，无分配开销")
    print(f"mm_only     = {t_mm:.3f} ms")
    print(f"act_only    = {t_act:.3f} ms")
    print(f"mm+act      = {t_ma:.3f} ms   (期望 ≈ {t_mm + t_act:.3f})")
    print(f"fused       = {t_f:.3f} ms")
    print(f"ratio (mm+act)/fused = {t_ma / t_f:.4f}")
    print(f"fused TFLOPS = {2 * M * K * 2 * H / (t_f / 1e3) / 1e12:.1f}")
    pairs.sort()
    print(f"交错配对 ratio 中位 = {statistics.median(pairs):.4f} "
          f"CI95=[{pairs[int(len(pairs)*0.025)]:.4f}, {pairs[int(len(pairs)*0.975)]:.4f}] n={len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
