"""bottlenecks.json 的数据契约、序列化与校验。

设计原则（对应总体方案 §三 步骤3 和 §五 5.2 置信度规则）：
- 字段名与 `doc/TraceLens-采集分析瓶颈输出方案-20261105.md` 的 5.1 schema 对齐；
- 置信度严格分级，**没有 arch 支撑不允许出现 compute/memory 判定**；
- 每条 finding 必须带 trace 证据，否则不允许输出；
- 校验逻辑自带实现，不依赖 jsonschema（目标环境里该包是坏的）。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

BOUND_TYPES = ("compute", "memory", "latency", "launch", "comm", "unknown")
CONFIDENCES = ("measured", "derived", "heuristic", "inconclusive")
ACTIONS = ("tune", "fuse", "backend", "precision", "investigate")
OWNERS = ("kernel", "framework", "serving")
PHASES = ("prefill", "decode", "mixed", "unknown")

#: 排除原因 -> 人可读说明
EXCLUDE_REASONS = {
    "system_collective": "系统通信算子（NCCL/collective），不进入算子级自动优化",
    "system_memcpy": "纯拷贝/memset 事件，不是算子优化对象",
    "hand_optimized_kernel": "高度手工优化的内核（FlashMLA/DeepEP 等），自动生成难以超越",
    "below_amdahl_threshold": "端到端占比低于 Amdahl 阈值，收益不足以抵消集成成本",
    "insufficient_evidence": "证据不足（无 roofline 或无法定位 bound），交人工排查",
    "near_roofline": "已达 roofline 85% 以上，剩余空间小于噪声门限",
    "latency_bound_small_batch": "decode 小 batch 属延迟受限，roofline SOL 不可比，不自动优化",
    "profiled_timing": "时间来自 profiler 窗口（有放大），需未开 profiler 的计时才能作为优化依据",
    "no_gpu_time": "无有效 GPU 时间统计",
}


def utc_now_iso() -> str:
    """返回带时区的 ISO8601 时间戳（UTC）。"""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Finding:
    """单条瓶颈。`eligible=False` 表示保留在报告中但不进入自动优化。"""

    op: str
    shape: str
    dtype: str
    gpu_time_ms: float
    pct_e2e: float
    bound_type: str = "unknown"
    sol_pct: Optional[float] = None
    sol_pct_raw: Optional[float] = None
    tflops: Optional[float] = None
    category: str = "other"
    kernel: Optional[str] = None
    count: int = 0
    mean_kernel_us: Optional[float] = None
    confidence: str = "heuristic"
    recommended_action: str = "investigate"
    suggested_owner: str = "kernel"
    eligible: bool = False
    excluded_reason: Optional[str] = None
    estimated_gain_pct: float = 0.0
    impact_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成 JSON 友好的 dict（剔除 None 之外的私有字段）。"""
        return {
            "op": self.op,
            "kernel": self.kernel,
            "shape": self.shape,
            "dtype": self.dtype,
            "category": self.category,
            "gpu_time_ms": round(self.gpu_time_ms, 6),
            "pct_e2e": round(self.pct_e2e, 3),
            "count": self.count,
            "mean_kernel_us": (
                round(self.mean_kernel_us, 3) if self.mean_kernel_us is not None else None
            ),
            "bound_type": self.bound_type,
            "sol_pct": round(self.sol_pct, 2) if self.sol_pct is not None else None,
            "sol_pct_raw": (
                round(self.sol_pct_raw, 2) if self.sol_pct_raw is not None else None
            ),
            "tflops": round(self.tflops, 3) if self.tflops is not None else None,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "suggested_owner": self.suggested_owner,
            "eligible": self.eligible,
            "excluded_reason": self.excluded_reason,
            "estimated_gain_pct": round(self.estimated_gain_pct, 3),
            "impact_score": round(self.impact_score, 3),
            "evidence": self.evidence,
        }


@dataclass
class Report:
    """bottlenecks.json 的顶层结构。"""

    run_id: str
    source: Dict[str, Any]
    target: Dict[str, Any]
    summary: Dict[str, Any]
    findings: List[Finding]
    params: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    generated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "source": self.source,
            "target": self.target,
            "params": self.params,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate(payload: Dict[str, Any]) -> List[str]:
    """按内置规则校验 bottlenecks.json，返回错误列表（空表示通过）。

    校验项对应总体方案 §五 5.2：
    - 顶层必填字段齐全；
    - 每条 finding 必须有 trace 证据；
    - bound_type/sol_pct 的组合必须自洽（unknown 不得带 sol_pct）；
    - confidence 取值合法，且 heuristic 不得声称 roofline 结论。
    """
    errors: List[str] = []

    for key in ("schema_version", "run_id", "generated_at", "source", "target",
                "summary", "findings"):
        if key not in payload:
            errors.append(f"missing top-level field: {key}")
    if errors:
        return errors

    if not isinstance(payload["findings"], list) or not payload["findings"]:
        errors.append("findings must be a non-empty list")
        return errors

    src = payload.get("source") or {}
    if not src.get("trace"):
        errors.append("source.trace is required (每条结论必须可回溯到 trace)")

    trace_val = src.get("trace")

    for idx, f in enumerate(payload["findings"]):
        where = f"findings[{idx}]"
        for key in ("op", "shape", "dtype", "gpu_time_ms", "pct_e2e", "bound_type",
                    "confidence", "recommended_action", "suggested_owner", "evidence"):
            if key not in f:
                errors.append(f"{where}.{key} is required")

        if f.get("bound_type") not in BOUND_TYPES:
            errors.append(f"{where}.bound_type invalid: {f.get('bound_type')!r}")
        if f.get("confidence") not in CONFIDENCES:
            errors.append(f"{where}.confidence invalid: {f.get('confidence')!r}")
        if f.get("recommended_action") not in ACTIONS:
            errors.append(f"{where}.recommended_action invalid")
        if f.get("suggested_owner") not in OWNERS:
            errors.append(f"{where}.suggested_owner invalid")

        ev = f.get("evidence") or {}
        if not ev.get("trace") and not trace_val:
            errors.append(f"{where}.evidence.trace is required")

        if f.get("bound_type") == "unknown" and f.get("sol_pct") is not None:
            errors.append(
                f"{where}: bound_type=unknown 不允许带 sol_pct（无 arch 支撑的 roofline 结论不可信）"
            )

        if f.get("confidence") == "measured":
            ncu = ev.get("ncu") or {}
            if not ncu:
                errors.append(
                    f"{where}: confidence=measured 但没有 ncu 硬件证据（总体方案禁止）"
                )

        if f.get("confidence") == "heuristic" and f.get("sol_pct") is not None:
            errors.append(f"{where}: heuristic 置信度不应带精确 sol_pct")

        if f.get("eligible") and f.get("recommended_action") == "investigate":
            errors.append(f"{where}: investigate 任务不允许标记为可自动优化")

        pct = f.get("pct_e2e")
        if not isinstance(pct, (int, float)) or pct < 0 or pct > 100.5:
            errors.append(f"{where}.pct_e2e out of range: {pct!r}")

    return errors


def summarize(report: Report) -> str:
    """生成一行简报，便于 CLI 输出。"""
    s = report.summary
    return (
        f"run={report.run_id} findings={len(report.findings)} "
        f"eligible={s.get('candidates', 0)} top={s.get('top_bottleneck')} "
        f"gpu_busy={s.get('gpu_busy_pct')}% idle={s.get('idle_pct')}%"
    )
