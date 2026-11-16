#!/usr/bin/env python3
"""同进程、锁频下对比三条路径，隔离"harness 差异"与"分配差异"。

A. 分配 vs 分配（融合档等价路径）
B. 预分配 vs 预分配（干净对照等价路径）
C. 预分配基线 vs 分配候选（原 clean bench 的不对称设置）
"""
from __future__ import annotations

import importlib.util
import statistics
import sys

import torch
import triton

sys.path.insert(0, "/model/wyy/optframe")
from tools.fused_gemm_act import baseline_mm_act, _act_and_mul_kernel  # noqa: E402

M, K, H = 8192, 4096, 12288
DT = torch.bfloat16
ITERS = 30

a = torch.randn(M, K, device="cuda", dtype=DT)
b = torch.randn(K, 2 * H, device="cuda", dtype=DT)
full = torch.empty(M, 2 * H, device="cuda", dtype=DT)
out = torch.empty(M, H, device="cuda", dtype=DT)
flush = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")

spec = importlib.util.spec_from_file_location("fx", "/model/wyy/optframe/fixtures/swiglu/01_good.py")
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
grid = (triton.cdiv(M * H, 1024),)


def baseline_pre():          # 预分配基线
    torch.mm(a, b, out=full)
    _act_and_mul_kernel[grid](full, out, M, H, BLOCK=1024)
    return out


def fused_alloc():           # fixture 候选（内部 allocate）
    return fx.run(a, b)


def fused_pre():             # 预分配候选：直接调 fixture 的 kernel
    fx._swiglu_kernel[(triton.cdiv(M, 256) * triton.cdiv(H, 64),)](
        a, b, out, M, H, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=3,
    )
    return out


def burn(n=25):
    for _ in range(n):
        torch.mm(a, b, out=full)
    torch.cuda.synchronize()


def pair(ref, cand, iters=ITERS):
    for _ in range(10):
        ref(); cand()
    torch.cuda.synchronize()
    rs, cs = [], []
    for i in range(iters):
        first_ref = (i % 2 == 0)
        order = ((ref, cand) if first_ref else (cand, ref))
        vals = []
        for fn in order:
            flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            vals.append(s.elapsed_time(e))
        r, c = (vals[0], vals[1]) if first_ref else (vals[1], vals[0])
        rs.append(r); cs.append(c)
    ratios = sorted(r / c for r, c in zip(rs, cs))
    return statistics.median(rs), statistics.median(cs), statistics.median(ratios), ratios


def report(name, ref, cand):
    r, c, m, ratios = pair(ref, cand)
    lo, hi = ratios[int(len(ratios) * .025)], ratios[int(len(ratios) * .975)]
    print(f"{name:38s} ref={r:7.3f} cand={c:7.3f} ratio={m:.4f} CI=[{lo:.4f},{hi:.4f}]")
    return m


def main() -> int:
    burn()
    print("A 分配 vs 分配（融合档路径）      ", end="")
    report("", lambda: baseline_mm_act(a, b), fused_alloc)
    burn()
    print("B 预分配 vs 预分配（干净路径）    ", end="")
    report("", baseline_pre, fused_pre)
    burn()
    print("C 预分配基线 vs 分配候选（旧口径）", end="")
    report("", baseline_pre, fused_alloc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
