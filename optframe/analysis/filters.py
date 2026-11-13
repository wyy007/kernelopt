"""瓶颈过滤与动作路由（总体方案 §三 过滤规则 + §五 5.4 路由表）。

三条硬规则：
1. 系统通信/高度手工优化内核直接排除，不送自动优化；
2. Amdahl：端到端占比低于阈值直接丢弃；
3. 证据不足（无 roofline）标 investigate，交人工，不自动改。

除此之外的动作路由是**规则化的**，不依赖大模型，保证同一份报表每次跑出同样结果。
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .parse import OpRow
from .schema import Finding

#: 系统通信：NCCL / c10d / 显式 collective
_COLLECTIVE_RE = re.compile(
    r"(nccl|all_?reduce|all_?gather|reduce_?scatter|broadcast|sendrecv|"
    r"c10d|aten::all_|aten::broadcast|aten::send|aten::recv)",
    re.IGNORECASE,
)

#: 高度手工优化的内核：自动生成基本不可能超越，投入产出比极差
_HAND_OPTIMIZED_RE = re.compile(
    r"(flash_?mla|flashmla|deep_?ep|deepep|fmha.*mla|mla.*fmha)", re.IGNORECASE
)

#: 纯拷贝类事件，不是算子优化对象
_MEMCPY_RE = re.compile(r"(memcpy|memset)", re.IGNORECASE)

#: 融合收益最大的类别（elementwise/norm/softmax 串起来通常是访存打满的低效链）
_FUSION_CATEGORIES = {"elementwise", "other", "norm", "reduction"}

#: 计算密集且值得调 tile/num_warps 的类别
_TUNABLE_COMPUTE_CATEGORIES = {"gemm", "sdpa", "conv", "attention", "moe"}

_LOW_PRECISION = {"fp8", "int8", "fp4"}


def _category_key(category: str) -> str:
    return (category or "other").strip().lower()


def _is_collective(row: OpRow) -> bool:
    return bool(_COLLECTIVE_RE.search(row.name) or _COLLECTIVE_RE.search(row.kernel or ""))


def _is_hand_optimized(row: OpRow) -> bool:
    return bool(
        _HAND_OPTIMIZED_RE.search(row.name) or _HAND_OPTIMIZED_RE.search(row.kernel or "")
    )


def _is_memcpy(row: OpRow) -> bool:
    """只有"算子本身是拷贝"才排除。

    TraceLens 有时会把 `aten::mm` 挂到一个 Memset kernel 上（GEMM kernel 未被链接），
    若按 kernel 名判断，会把真实的大头 GEMM 误杀（实测 18.2% 的那条就是这样）。
    """
    return bool(_MEMCPY_RE.search(row.name))


def _route_action(row: OpRow, bound: str) -> tuple[str, str]:
    """返回 (recommended_action, suggested_owner)。"""
    category = _category_key(row.category)
    dtype = (row.dtype or "").lower()

    if bound == "unknown":
        return "investigate", "kernel"

    if bound == "latency" or bound == "launch":
        # launch 开销靠融合/批处理解决，归 serving
        return ("fuse" if category in _FUSION_CATEGORIES else "backend"), "serving"

    if bound == "comm":
        return "investigate", "serving"

    if bound == "compute":
        # 低精度算子在 compute bound 下优先换后端（Cutlass/CuTe）而不是调参
        if dtype in _LOW_PRECISION and category in _TUNABLE_COMPUTE_CATEGORIES:
            return "tune", "kernel"
        if dtype in {"fp32", "fp64"}:
            # 精度迁移收益最大，但需要模型侧允许，故单列
            return "precision", "kernel"
        if category in _TUNABLE_COMPUTE_CATEGORIES:
            return "tune", "kernel"
        return "backend", "kernel"

    if bound == "memory":
        if category in _FUSION_CATEGORIES:
            return "fuse", "framework"
        return "tune", "kernel"

    return "investigate", "kernel"


def classify(
    row: OpRow,
    pct_e2e: float,
    *,
    min_pct_e2e: float = 2.0,
    near_roofline_pct: float = 85.0,
    sol_clamp: float = 100.0,
) -> Finding:
    """把一行报表映射为 Finding，并决定是否进入自动优化。"""
    from .parse import normalize_bound

    bound = normalize_bound(row.bound_raw)
    sol = row.sol_pct_raw
    notes = []

    # roofline 超过 100% 说明模型高估了数据量/算量（常见于 elementwise 的读写字节数估算），
    # 不能当成"超过硬件峰值"，也不能当成"已贴顶"——直接丢弃该达成率，退回启发式判定。
    if sol is not None and sol > sol_clamp:
        notes.append(
            f"sol_pct_raw={sol:.1f} 超过 {sol_clamp:.0f}%，roofline 投影不可信，已丢弃该达成率"
        )
        sol = None
    if sol is not None and sol <= 0:
        sol = None

    action, owner = _route_action(row, bound)

    if sol is not None and bound != "unknown":
        confidence = "derived"
    elif bound == "unknown":
        confidence = "inconclusive" if row.gpu_time_us <= 0 else "heuristic"
    else:
        confidence = "heuristic"

    if sol is not None:
        estimated_gain = pct_e2e * (1.0 - sol / 100.0)
    elif bound != "unknown":
        # 没有可信达成率时用保守系数估算（0.3），并留痕说明这是估算值
        estimated_gain = pct_e2e * 0.3
        notes.append("无可用 SOL，收益为保守估算（pct_e2e × 0.3）")
    else:
        estimated_gain = 0.0

    finding = Finding(
        op=row.name,
        kernel=row.kernel,
        shape=row.shape,
        dtype=row.dtype,
        category=row.category,
        gpu_time_ms=row.gpu_time_us / 1000.0,
        pct_e2e=pct_e2e,
        count=row.count,
        mean_kernel_us=row.mean_kernel_us,
        bound_type=bound,
        sol_pct=sol,
        sol_pct_raw=row.sol_pct_raw,
        tflops=row.tflops,
        confidence=confidence,
        recommended_action=action,
        suggested_owner=owner,
        eligible=False,
        estimated_gain_pct=estimated_gain,
        impact_score=estimated_gain,
        evidence={
            "trace": None,  # 由 export 统一回填
            "report_csv": "unified_perf_summary.csv",
            "ncu": None,
            "notes": notes,
        },
    )

    # ---- 排除判定（顺序即优先级）----
    if row.gpu_time_us <= 0:
        finding.excluded_reason = "no_gpu_time"
    elif _is_memcpy(row):
        finding.excluded_reason = "system_memcpy"
    elif _is_collective(row):
        finding.excluded_reason = "system_collective"
    elif _is_hand_optimized(row):
        finding.excluded_reason = "hand_optimized_kernel"
    elif pct_e2e < min_pct_e2e:
        finding.excluded_reason = "below_amdahl_threshold"
    elif bound == "unknown" or action == "investigate":
        finding.excluded_reason = "insufficient_evidence"
    elif sol is not None and sol >= near_roofline_pct:
        finding.excluded_reason = "near_roofline"
        # 已经贴顶，收益按 0 计，避免误排优先级
        finding.estimated_gain_pct = 0.0
        finding.impact_score = 0.0

    finding.eligible = finding.excluded_reason is None
    return finding


def dedupe_and_sort(findings: Iterable[Finding]) -> list[Finding]:
    """按 (op, shape, dtype) 去重（保留 GPU 时间最大的），并按影响分排序。"""
    best: dict[tuple[str, str, str], Finding] = {}
    for f in findings:
        key = (f.op, f.shape, f.dtype)
        cur = best.get(key)
        if cur is None or f.gpu_time_ms > cur.gpu_time_ms:
            best[key] = f
    return sorted(best.values(), key=lambda x: (x.impact_score, x.pct_e2e), reverse=True)
