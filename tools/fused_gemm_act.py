#!/usr/bin/env python3
"""GEMM + SwiGLU 融合核（单文件实验）：判定"融合链值不值得做"的最小单元。

对照两种实现：
  baseline：torch.mm(A, B) 产出 (M, 2H)，再做 silu(gate) * up（两步，等价服务里的
            GEMM + act_and_mul 两个 kernel）
  fused   ：一个 Triton kernel，同时算 gate/up 两个 tile，输出 silu(gate)*up

形状取 8B 真实 trace 里的 gate_up 形状：M=2162, K=4096, N=24576（half=12288）。

用法：
  python3 tools/fused_gemm_act.py --m 2162 --k 4096 --half 12288 --pairs 40
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_gemm_kernel(
    A, B, C,
    M, HALF_N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(HALF_N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_gate = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    b_up = B + rk[:, None] * stride_bk + (rn[None, :] + HALF_N) * stride_bn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_rem), other=0.0)
        g = tl.load(b_gate, mask=(rk[:, None] < k_rem) & (rn[None, :] < HALF_N), other=0.0)
        u = tl.load(b_up, mask=(rk[:, None] < k_rem) & (rn[None, :] < HALF_N), other=0.0)
        acc_g += tl.dot(a, g)
        acc_u += tl.dot(a, u)
        a_ptrs += BLOCK_K * stride_ak
        b_gate += BLOCK_K * stride_bk
        b_up += BLOCK_K * stride_bk

    out = acc_g * tl.sigmoid(acc_g) * acc_u

    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < HALF_N))


def fused_swiglu(a: torch.Tensor, b: torch.Tensor, *, block_m=128, block_n=128,
                 block_k=64, group_m=8, warps=8, stages=3) -> torch.Tensor:
    M, K = a.shape
    half_n = b.shape[1] // 2
    c = torch.empty((M, half_n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(half_n, block_n),)
    _swiglu_gemm_kernel[grid](
        a, b, c,
        M, half_n, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=warps, num_stages=stages,
    )
    return c


def baseline_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """服务里的做法：GEMM 出 (M, 2H)，再 act_and_mul。"""
    full = torch.mm(a, b)
    gate, up = full[:, : full.shape[1] // 2], full[:, full.shape[1] // 2 :]
    return gate * torch.sigmoid(gate) * up


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


def baseline_mm_act(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """服务同款：cuBLAS GEMM + 单个 act_and_mul kernel。"""
    full = torch.mm(a, b)
    M, two_h = full.shape
    half = two_h // 2
    out = torch.empty((M, half), device=full.device, dtype=full.dtype)
    grid = (triton.cdiv(M * half, 1024),)
    _act_and_mul_kernel[grid](full, out, M, half, BLOCK=1024)
    return out


_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")
    _FLUSH.zero_()


def _time_once(fn):
    _flush()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2162)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--half", type=int, default=12288)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    args = ap.parse_args()

    torch.cuda.init()
    random.seed(0)
    dev, dt = "cuda", torch.bfloat16
    M, K, H = args.m, args.k, args.half

    a = torch.randn(M, K, device=dev, dtype=dt)
    b = torch.randn(K, 2 * H, device=dev, dtype=dt)

    cfg = dict(block_m=args.block_m, block_n=args.block_n, block_k=args.block_k,
               warps=args.warps, stages=args.stages)
    ref = baseline_swiglu(a, b)
    ref2 = baseline_mm_act(a, b)
    cand = fused_swiglu(a, b, **cfg)
    err = (cand.float() - ref.float()).abs().max().item()
    rel = err / (ref.float().abs().max().item() + 1e-6)

    for _ in range(args.warmup):
        baseline_swiglu(a, b); baseline_mm_act(a, b); fused_swiglu(a, b, **cfg)

    t, t2, f = [], [], []
    for i in range(args.pairs):
        order = [("t", baseline_swiglu), ("t2", baseline_mm_act),
                 ("f", lambda x, y: fused_swiglu(x, y, **cfg))]
        if i % 2:
            order.reverse()
        for tag, fn in order:
            v = _time_once(lambda fn=fn: fn(a, b))
            (t if tag == "t" else t2 if tag == "t2" else f).append(v)

    def stats(base):
        ratios = sorted(x / y for x, y in zip(base, f))
        return (statistics.median(ratios),
                ratios[int(len(ratios) * 0.025)], ratios[int(len(ratios) * 0.975)])

    med, lo, hi = stats(t)
    med2, lo2, hi2 = stats(t2)
    flops = 2 * M * K * 2 * H

    print(f"shape M={M} K={K} N={2*H} (half={H})")
    print(f"correctness: max_abs_err={err:.3f} rel={rel:.2e}")
    print(f"baseline A (mm+3拍 eager act) = {statistics.median(t):.3f} ms")
    print(f"baseline B (mm+单 act kernel)  = {statistics.median(t2):.3f} ms")
    print(f"fused                          = {statistics.median(f):.3f} ms")
    print(f"ratio vs A = {med:.4f}  CI95 = [{lo:.4f}, {hi:.4f}]")
    print(f"ratio vs B = {med2:.4f}  CI95 = [{lo2:.4f}, {hi2:.4f}]  n={len(f)}")
    print(f"fused TFLOPS = {flops / (statistics.median(f) / 1e3) / 1e12:.1f}")
    print("JSON " + json.dumps({
        "shape": [M, K, 2 * H],
        "max_abs_err": err,
        "baseline_ms": round(statistics.median(t), 4),
        "fused_ms": round(statistics.median(f), 4),
        "ratio_vs_eager_act": round(med, 4),
        "ci_vs_eager_act": [round(lo, 4), round(hi, 4)],
        "ratio_vs_single_act_kernel": round(med2, 4),
        "ci_vs_single_act_kernel": [round(lo2, 4), round(hi2, 4)],
        "pairs": len(f),
        "cache_policy": "cold",
        "timer": "cuda_event",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
