"""device 计时（双档方案 §4 共享合同）。

口径必须统一，否则两个档位的数字不可比：
- 同一 GPU、同一输入、同一 dtype；
- 每次迭代前 flush L2（否则 8MB 级算子会被缓存放大到"超过 DRAM 峰值"）；
- 取中位数，抑制 GC/调度抖动；
- warmup 不计入。
"""

from __future__ import annotations

import random
from statistics import median
from typing import Callable, List, Optional, Sequence, Tuple

import torch

#: L2 flush 缓冲区（一次分配，复用）。256MB 足以覆盖 L20 的 48MB L2。
_FLUSH_BUFFER: Optional[torch.Tensor] = None


def _flush_cache() -> None:
    global _FLUSH_BUFFER
    if _FLUSH_BUFFER is None:
        _FLUSH_BUFFER = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")
    _FLUSH_BUFFER.zero_()


def time_ms(
    fn: Callable,
    *,
    warmup: int = 10,
    repeat: int = 50,
    flush_cache: bool = True,
) -> float:
    """返回单次调用的中位耗时（毫秒）。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device timing")

    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(max(repeat, 1)):
        if flush_cache:
            _flush_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    return float(median(samples))


def time_pair_samples(
    fn_ref: Callable,
    fn_cand: Callable,
    *,
    warmup: int = 10,
    repeat: int = 50,
    flush_cache: bool = True,
) -> Tuple[List[float], List[float]]:
    """交错 A/B 采样，返回 (ref_samples, cand_samples)。

    共享 GPU 上两次独立计时会因争用差异得到 1.00x / 1.03x 这种不一致结果；
    交错采样让两边经历同一段负载，配对样本才有统计意义。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device timing")

    for _ in range(max(warmup, 0)):
        fn_ref()
        fn_cand()
    torch.cuda.synchronize()

    ref_samples: List[float] = []
    cand_samples: List[float] = []
    for _ in range(max(repeat, 1)):
        if flush_cache:
            _flush_cache()
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        fn_ref()
        e0.record()
        torch.cuda.synchronize()
        ref_samples.append(s0.elapsed_time(e0))

        if flush_cache:
            _flush_cache()
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s1.record()
        fn_cand()
        e1.record()
        torch.cuda.synchronize()
        cand_samples.append(s1.elapsed_time(e1))

    return ref_samples, cand_samples


def time_pair(
    fn_ref: Callable,
    fn_cand: Callable,
    *,
    warmup: int = 10,
    repeat: int = 50,
    flush_cache: bool = True,
) -> tuple[float, float]:
    """交错 A/B 计时，返回两侧中位数 (ref_ms, cand_ms)。"""
    ref_samples, cand_samples = time_pair_samples(
        fn_ref, fn_cand, warmup=warmup, repeat=repeat, flush_cache=flush_cache
    )
    return float(median(ref_samples)), float(median(cand_samples))


def bootstrap_median_ci(
    samples: Sequence[float],
    *,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """中位数的 bootstrap 置信区间（返回 (low, high)）。"""
    if not samples:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(samples)
    meds: List[float] = []
    for _ in range(max(iters, 1)):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        meds.append(median(resample))
    meds.sort()
    lo_idx = max(int(len(meds) * (alpha / 2.0)) - 1, 0)
    hi_idx = min(int(len(meds) * (1.0 - alpha / 2.0)), len(meds) - 1)
    return float(meds[lo_idx]), float(meds[hi_idx])


def summarize(samples: List[float]) -> dict:
    """给账本用的统计摘要。"""
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "median_ms": float(median(ordered)),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p90_ms": ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)],
    }
