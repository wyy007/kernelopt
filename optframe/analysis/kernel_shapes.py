"""按 kernel 名称模式补充形状/算力信息，让自定义 op 也能参与 roofline。

背景（2026-11-10）：
    SGLang 的量化 GEMM 是自定义 op（`void sglang::device::marlin::Marlin<...>`），
    trace 里只有 kernel 名和耗时，没有 shape，TraceLens 算不出 FLOPs → bound=unknown。
    但 SGLang 自带 `SGLANG_KERNEL_API_LOGLEVEL=3` 的日志会打印每次调用的 size_m/n/k。
    本模块把那份日志变成"形状提示"，按 pattern 匹配到 finding 上，补算 FLOPs/TFLOPS/SOL。

置信度：这是**推导**（derived），不是实测计数器；证据里标明来源与"加权平均"这一近似。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import Finding


@dataclass
class ShapeHint:
    """一个 kernel 模式的形状统计。"""

    match: str
    avg_flops: float          # 每次调用的平均 FLOPs（按调用次数加权）
    shapes: List[List[int]]   # [M, N, K]
    calls: int
    source: str = ""
    avg_bytes: float = 0.0    # 每次调用的平均访存量（权重按量化位宽估算）
    m_max: int = 0            # 该模式的 M 上界（decode 通常 <= 4）
    m_min: int = 0            # 该模式的 M 下界（prefill 通常 >= 8）
    tag: str = ""             # decode / prefill，仅用于留痕

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "avg_flops": round(self.avg_flops, 1),
            "avg_bytes": round(self.avg_bytes, 1),
            "m_max": self.m_max,
            "m_min": self.m_min,
            "tag": self.tag,
            "calls": self.calls,
            "shapes": self.shapes,
            "source": self.source,
        }


def load_hints(path: str | Path) -> List[ShapeHint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[ShapeHint] = []
    for item in data.get("patterns", []):
        out.append(
            ShapeHint(
                match=str(item["match"]),
                avg_flops=float(item["avg_flops"]),
                shapes=[list(map(int, s)) for s in item.get("shapes", [])],
                calls=int(item.get("calls", 0)),
                source=str(item.get("source", "")),
                avg_bytes=float(item.get("avg_bytes", 0.0)),
                m_max=int(item.get("m_max", 0)),
                m_min=int(item.get("m_min", 0)),
                tag=str(item.get("tag", "")),
            )
        )
    return out


def _match(hint: ShapeHint, finding: Finding) -> bool:
    pattern = re.compile(hint.match, re.IGNORECASE)
    return bool(pattern.search(finding.op) or pattern.search(finding.kernel or ""))


def enrich(
    findings: List[Finding],
    hints: List[ShapeHint],
    *,
    peak_tflops: float,
    peak_bw_gbs: float = 0.0,
    near_roofline_pct: float = 85.0,
    min_pct_e2e: float = 2.0,
    allow_eligible: bool = False,
) -> List[Finding]:
    """给 bound=unknown 的 finding 补算 FLOPs 与 SOL（就地修改并返回）。"""
    if peak_tflops <= 0:
        return findings

    for finding in findings:
        if finding.bound_type != "unknown":
            continue
        candidates = [h for h in hints if _match(h, finding)]
        if not candidates or finding.count <= 0:
            continue
        per_call_ms = finding.gpu_time_ms / finding.count
        # decode 的每次调用很短（几十 µs），prefill 通常 > 0.3ms
        prefer_prefill = per_call_ms >= 0.3
        hint = next(
            (h for h in candidates if (h.m_min >= 8) == prefer_prefill), candidates[0]
        )
        if hint.avg_flops <= 0:
            continue
        if finding.gpu_time_ms <= 0:
            continue

        seconds = finding.gpu_time_ms / 1000.0
        flops = hint.avg_flops * finding.count
        tflops = flops / seconds / 1e12 if seconds > 0 else 0.0
        bytes_moved = hint.avg_bytes * finding.count
        gbs = bytes_moved / seconds / 1e9 if seconds > 0 else 0.0

        # 算术强度 vs 机器平衡点：低于平衡点就是访存受限，拿算力峰值算 SOL 会严重低估
        ai = (flops / bytes_moved) if bytes_moved > 0 else float("inf")
        crossover = (
            peak_tflops * 1e12 / (peak_bw_gbs * 1e9) if peak_bw_gbs > 0 else float("inf")
        )
        memory_bound = peak_bw_gbs > 0 and ai < crossover

        if memory_bound:
            sol = min(gbs / peak_bw_gbs * 100.0, 100.0)
            bound = "memory"
            detail = f"{gbs:.0f} GB/s / {peak_bw_gbs:.0f} GB/s（算术强度 {ai:.1f} < 平衡点 {crossover:.1f}）"
        else:
            sol = min(tflops / peak_tflops * 100.0, 100.0)
            bound = "compute"
            detail = f"{tflops:.1f} TFLOPS / {peak_tflops:.1f} TFLOPS"

        # 小 batch（M<=4）的 GEMM 是延迟受限，拿带宽/算力峰值算 SOL 会给出"还有 90% 空间"的假象
        if 0 < hint.m_max <= 4:
            bound = "latency"

        finding.bound_type = bound
        finding.sol_pct = sol
        finding.sol_pct_raw = sol
        finding.tflops = tflops
        finding.confidence = "derived"
        finding.category = "GEMM"
        finding.evidence.setdefault("notes", []).append(
            f"shape 来自 kernel API 日志（{hint.source or 'n/a'}）："
            f"{len(hint.shapes)} 种形状，{hint.calls} 次调用，"
            f"形状组={hint.tag or 'n/a'}(M {hint.m_min}-{hint.m_max})，每次 {per_call_ms*1000:.0f}µs；"
            f"按调用次数加权平均 {hint.avg_flops/1e9:.2f} GFLOP/call、"
            f"{hint.avg_bytes/1e6:.1f} MB/call；SOL 按 {detail}；"
            f"shapes/bytes 为推导值，非硬件计数器"
        )

        # profiler 窗口里的绝对时间不可信（本次请求从 0.69s 涨到 30s），
        # SOL 只作诊断，默认不据此排任务
        if not allow_eligible:
            finding.recommended_action = "investigate"
            finding.eligible = False
            finding.excluded_reason = "profiled_timing"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
            finding.evidence.setdefault("notes", []).append(
                "时间取自 profiler 窗口，存在放大；需未开 profiler 的计时才能作为优化依据"
            )
        elif bound == "latency":
            finding.recommended_action = "investigate"
            finding.eligible = False
            finding.excluded_reason = "latency_bound_small_batch"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
            finding.evidence.setdefault("notes", []).append(
                f"M<={hint.m_max}：decode 小 batch 属延迟受限，roofline SOL 仅供参考，不作为优化目标"
            )
        elif sol >= near_roofline_pct:
            finding.recommended_action = "tune"
            finding.eligible = False
            finding.excluded_reason = "near_roofline"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
        elif finding.pct_e2e < min_pct_e2e:
            finding.recommended_action = "tune"
            finding.eligible = False
            finding.excluded_reason = "below_amdahl_threshold"
            finding.estimated_gain_pct = 0.0
            finding.impact_score = 0.0
        else:
            finding.recommended_action = "tune"
            finding.suggested_owner = "kernel"
            finding.eligible = True
            finding.excluded_reason = None
            gain = finding.pct_e2e * (1.0 - sol / 100.0)
            finding.estimated_gain_pct = gain
            finding.impact_score = gain

    return findings
