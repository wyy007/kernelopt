"""阶段3：Amdahl 过滤。

阶段2 只做了"占比够不够大"的粗筛；阶段3 必须回答一个更实际的问题：
**这个算子就算优化到目标 SOL，端到端到底能快多少？**

模型：
    设端到端 GPU 忙碌时间 T，算子耗时 t = p·T，
    该算子加速 s 倍后：T' = T − t + t/s
    端到端加速比 = T/T' = 1 / (1 − p·(1 − 1/s))
    p → 1 时上界 1/(1−p)，即 Amdahl 上限。

注意：本实现按**可加性**处理多个任务（GPU 时间串行累加），
不做重叠假设，避免把并行收益算进去导致虚高。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 阶段2 里允许进入优化的动作
OPTIMIZABLE_ACTIONS = ("tune", "fuse")

#: 拒绝原因（与报告里的字符串保持一致）
REASON_STAGE2_EXCLUDED = "stage2_excluded"
REASON_ACTION_NOT_OPTIMIZABLE = "action_not_optimizable"
REASON_BELOW_NOISE_FLOOR = "below_noise_floor"
REASON_BELOW_E2E_GAIN = "below_e2e_gain_threshold"
REASON_BUDGET = "budget_exceeded"
REASON_DIMINISHING = "diminishing_returns"


@dataclass
class AmdahlParams:
    """过滤参数（全部可在 CLI 覆盖，默认值有明确依据）。"""

    #: 目标 SOL：算子在 roofline 上的目标达成率
    target_sol_pct: float = 85.0
    #: 无 roofline 时的保守加速比假设
    default_speedup_tune: float = 1.20
    default_speedup_fuse: float = 1.30
    #: 小于该加速比视为噪声（方案里 KEEP 的 2% 噪声门限）
    noise_floor_speedup: float = 1.02
    #: 单任务端到端收益下限（%）
    min_e2e_gain_pct: float = 0.5
    #: 边际收益下限（%），低于此值认为后续任务不值得做
    min_marginal_gain_pct: float = 0.1
    #: 单个任务的目标加速比上限：SOL 很低时 target 会被算成几十倍，不现实
    max_target_speedup: float = 4.0
    #: 本轮最多生成多少个任务
    max_tasks: int = 10
    #: 累计端到端收益上限（%），超过即停止，避免在同一瓶颈上反复投入
    max_total_gain_pct: float = 35.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_sol_pct": self.target_sol_pct,
            "default_speedup_tune": self.default_speedup_tune,
            "default_speedup_fuse": self.default_speedup_fuse,
            "noise_floor_speedup": self.noise_floor_speedup,
            "min_e2e_gain_pct": self.min_e2e_gain_pct,
            "min_marginal_gain_pct": self.min_marginal_gain_pct,
            "max_target_speedup": self.max_target_speedup,
            "max_tasks": self.max_tasks,
            "max_total_gain_pct": self.max_total_gain_pct,
        }


@dataclass
class AcceptedTask:
    """通过 Amdahl 过滤、可以进入 task.toml 生成的任务。"""

    op: str
    shape: str
    dtype: str
    action: str
    bound_type: str
    category: str
    pct_e2e: float
    gpu_time_ms: float
    sol_pct: Optional[float]
    target_speedup: float
    speedup_basis: str          # roofline | default_tune | default_fuse
    e2e_gain_pct: float         # 单独做这一个任务时的端到端收益
    marginal_gain_pct: float    # 计入前面任务后的边际收益
    cumulative_gain_pct: float  # 计入本身后的累计收益
    projected_op_ms: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "shape": self.shape,
            "dtype": self.dtype,
            "action": self.action,
            "bound_type": self.bound_type,
            "category": self.category,
            "pct_e2e": round(self.pct_e2e, 3),
            "gpu_time_ms": round(self.gpu_time_ms, 4),
            "sol_pct": None if self.sol_pct is None else round(self.sol_pct, 2),
            "target_speedup": round(self.target_speedup, 4),
            "speedup_basis": self.speedup_basis,
            "e2e_gain_pct": round(self.e2e_gain_pct, 4),
            "marginal_gain_pct": round(self.marginal_gain_pct, 4),
            "cumulative_gain_pct": round(self.cumulative_gain_pct, 4),
            "projected_op_ms": round(self.projected_op_ms, 4),
            "rank": self.rank,
            # task.toml 直接消费的字段
            "target_ratio": round(self.target_speedup, 4),
        }


@dataclass
class RejectedTask:
    op: str
    shape: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "shape": self.shape,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class AmdahlResult:
    tasks: List[AcceptedTask]
    rejected: List[RejectedTask]
    total_gpu_ms_before: float
    total_gpu_ms_after: float

    @property
    def e2e_speedup(self) -> float:
        if self.total_gpu_ms_after <= 0:
            return 1.0
        return self.total_gpu_ms_before / self.total_gpu_ms_after

    @property
    def total_gain_pct(self) -> float:
        if self.total_gpu_ms_before <= 0:
            return 0.0
        return (1.0 - self.total_gpu_ms_after / self.total_gpu_ms_before) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "tasks": len(self.tasks),
                "rejected": len(self.rejected),
                "total_gpu_ms_before": round(self.total_gpu_ms_before, 4),
                "total_gpu_ms_after": round(self.total_gpu_ms_after, 4),
                "e2e_speedup": round(self.e2e_speedup, 4),
                "total_gain_pct": round(self.total_gain_pct, 4),
            },
            "tasks": [t.to_dict() for t in self.tasks],
            "rejected": [r.to_dict() for r in self.rejected],
        }


# ---------------------------------------------------------------- 纯函数部分

def e2e_gain_pct(pct_e2e: float, speedup: float) -> float:
    """单算子加速 speedup 倍后，端到端收益百分比。

    p 为占比（0-100）。speedup<=1 返回 0。
    """
    if speedup <= 1.0 or pct_e2e <= 0:
        return 0.0
    p = min(pct_e2e, 100.0) / 100.0
    new_time_ratio = 1.0 - p + p / speedup
    return (1.0 - new_time_ratio) * 100.0


def amdahl_ceiling_speedup(pct_e2e: float) -> float:
    """Amdahl 上限：该算子无限加速时的端到端加速比。"""
    p = min(max(pct_e2e, 0.0), 100.0) / 100.0
    if p >= 1.0:
        return math.inf
    return 1.0 / (1.0 - p)


def target_speedup_for(finding: Dict[str, Any], params: AmdahlParams) -> Tuple[float, str]:
    """根据 SOL 推算可达到的 kernel 加速比，并给出依据。"""
    sol = finding.get("sol_pct")
    action = finding.get("recommended_action", "")

    if isinstance(sol, (int, float)) and sol > 0:
        if sol >= params.target_sol_pct:
            return 1.0, "roofline"
        raw = params.target_sol_pct / float(sol)
        if raw > params.max_target_speedup:
            return params.max_target_speedup, "roofline_capped"
        return raw, "roofline"

    if action == "fuse":
        return params.default_speedup_fuse, "default_fuse"
    return params.default_speedup_tune, "default_tune"


# ---------------------------------------------------------------- 主流程

def select(
    bottlenecks: Dict[str, Any],
    params: Optional[AmdahlParams] = None,
) -> AmdahlResult:
    """读取 bottlenecks.json（dict），做 Amdahl 过滤并给出累计预测。"""
    params = params or AmdahlParams()
    findings: Iterable[Dict[str, Any]] = bottlenecks.get("findings") or []
    total_ms = float((bottlenecks.get("summary") or {}).get("total_gpu_ms") or 0.0)

    if total_ms <= 0:
        total_ms = sum(float(f.get("gpu_time_ms") or 0.0) for f in findings)
    if total_ms <= 0:
        raise ValueError("bottlenecks.json 缺少有效 GPU 时间，无法做 Amdahl 分析")

    rejected: List[RejectedTask] = []
    candidates: List[Tuple[float, Dict[str, Any], float, str]] = []

    for finding in findings:
        op = str(finding.get("op", "?"))
        shape = str(finding.get("shape", ""))
        action = str(finding.get("recommended_action", ""))
        pct = float(finding.get("pct_e2e") or 0.0)

        if not finding.get("eligible", False):
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_STAGE2_EXCLUDED,
                detail={"stage2_reason": finding.get("excluded_reason")},
            ))
            continue
        if action not in OPTIMIZABLE_ACTIONS:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_ACTION_NOT_OPTIMIZABLE,
                detail={"action": action},
            ))
            continue

        speedup, basis = target_speedup_for(finding, params)
        if speedup < params.noise_floor_speedup:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_BELOW_NOISE_FLOOR,
                detail={
                    "target_speedup": round(speedup, 4),
                    "noise_floor": params.noise_floor_speedup,
                    "sol_pct": finding.get("sol_pct"),
                },
            ))
            continue

        gain = e2e_gain_pct(pct, speedup)
        if gain < params.min_e2e_gain_pct:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_BELOW_E2E_GAIN,
                detail={
                    "e2e_gain_pct": round(gain, 4),
                    "threshold": params.min_e2e_gain_pct,
                },
            ))
            continue

        candidates.append((gain, finding, speedup, basis))

    # 按单任务收益降序，保证先做最划算的
    candidates.sort(key=lambda item: item[0], reverse=True)

    accepted: List[AcceptedTask] = []
    remaining_ms = total_ms
    cumulative = 0.0

    for gain, finding, speedup, basis in candidates:
        op = str(finding.get("op", "?"))
        shape = str(finding.get("shape", ""))
        pct = float(finding.get("pct_e2e") or 0.0)
        op_ms = float(finding.get("gpu_time_ms") or 0.0)

        if len(accepted) >= params.max_tasks:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_BUDGET,
                detail={"max_tasks": params.max_tasks},
            ))
            continue

        if cumulative >= params.max_total_gain_pct:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_BUDGET,
                detail={"max_total_gain_pct": params.max_total_gain_pct,
                        "cumulative_gain_pct": round(cumulative, 4)},
            ))
            continue

        saved_ms = op_ms * (1.0 - 1.0 / speedup)
        marginal = (saved_ms / total_ms) * 100.0 if total_ms > 0 else 0.0
        if marginal < params.min_marginal_gain_pct:
            rejected.append(RejectedTask(
                op=op, shape=shape, reason=REASON_DIMINISHING,
                detail={"marginal_gain_pct": round(marginal, 4),
                        "threshold": params.min_marginal_gain_pct},
            ))
            continue

        remaining_ms -= saved_ms
        cumulative = (1.0 - remaining_ms / total_ms) * 100.0

        accepted.append(AcceptedTask(
            op=op,
            shape=shape,
            dtype=str(finding.get("dtype", "unknown")),
            action=str(finding.get("recommended_action")),
            bound_type=str(finding.get("bound_type", "unknown")),
            category=str(finding.get("category", "other")),
            pct_e2e=pct,
            gpu_time_ms=op_ms,
            sol_pct=finding.get("sol_pct"),
            target_speedup=speedup,
            speedup_basis=basis,
            e2e_gain_pct=gain,
            marginal_gain_pct=marginal,
            cumulative_gain_pct=cumulative,
            projected_op_ms=(op_ms / speedup) if speedup > 0 else op_ms,
            rank=len(accepted) + 1,
        ))

    return AmdahlResult(
        tasks=accepted,
        rejected=rejected,
        total_gpu_ms_before=total_ms,
        total_gpu_ms_after=max(remaining_ms, 0.0),
    )
