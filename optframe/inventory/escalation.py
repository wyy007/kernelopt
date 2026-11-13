"""档位升级判定（总体方案 §四 步骤4）。

四条规则里有一条**只能在运行期判定**（简单档 best-of-3 提升不足 1.05×），
静态阶段不可能知道 best-of-3 的结果。所以这里明确拆成两部分：

- `decide()`：静态判定，只看任务本身（类别、动作、SOL），决定 `allow_complex`；
- `should_escalate_after_best_of_three()`：运行期判定，由 tiers 在执行完
  best-of-3 后调用，命中则把任务提交复杂档。

不做"用 target_ratio 假装 best-of-3 结果"的代理判断 —— 那是把假设伪装成结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: 简单档 best-of-3 提升不足该倍数 -> 升级复杂档（运行期）
SIMPLE_TIER_MIN_RATIO = 1.05

#: 这些类别天然适合复杂档（长融合链、手工优化空间大）
COMPLEX_CATEGORIES = {"attention", "moe", "sdpa", "conv"}

#: 低于该 SOL 视为"简单档参数搜索空间不足"
LOW_SOL_PCT = 60.0

_RUNTIME_RULE = "best_of_3_lt_1.05"

_REASON_TEXT = {
    "attention_moe_or_fusion": "attention/MoE 或融合链任务，需要 CUDA/CuTe 级别的控制力",
    "sol_lt_60": "roofline 达成率低于 60%，简单档参数搜索空间不足",
    "trace_recommended": "trace 分析显式建议升级",
    "best_of_3_lt_1.05": "简单档 best-of-3 提升不足 1.05×（运行期判定）",
}


@dataclass
class EscalationDecision:
    allow_complex: bool
    reasons: List[str]
    runtime_rule: str = _RUNTIME_RULE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_complex": self.allow_complex,
            "reasons": self.reasons,
            "runtime_rule": self.runtime_rule,
            "reason_text": [_REASON_TEXT.get(r, r) for r in self.reasons],
        }


def decide(task: Dict[str, Any]) -> EscalationDecision:
    """静态判定：只看任务自身，决定是否**允许**升级复杂档。

    task 字段：action / category / sol_pct / escalate_hint。
    """
    reasons: List[str] = []

    category = str(task.get("category") or "").lower()
    action = str(task.get("action") or "").lower()
    if category in COMPLEX_CATEGORIES or action == "fuse":
        reasons.append("attention_moe_or_fusion")

    sol = task.get("sol_pct")
    if isinstance(sol, (int, float)) and 0 < float(sol) < LOW_SOL_PCT:
        reasons.append("sol_lt_60")

    if task.get("escalate_hint"):
        reasons.append("trace_recommended")

    return EscalationDecision(allow_complex=bool(reasons), reasons=reasons)


def should_escalate_after_best_of_three(
    best_ratio: Optional[float],
    *,
    attempts: int = 3,
    threshold: float = SIMPLE_TIER_MIN_RATIO,
) -> bool:
    """运行期规则：best-of-N 全部没超过 threshold 时返回 True。

    `best_ratio` 为 None（全部失败）同样视为打不动。
    """
    if best_ratio is None:
        return True
    return float(best_ratio) < threshold


def reason_text(reason: str) -> str:
    return _REASON_TEXT.get(reason, reason)
