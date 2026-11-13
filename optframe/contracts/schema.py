"""共享合同层的数据结构（双档方案 §4.2）。

所有档位只产出 Candidate，所有评测结果只由 Evaluator 产出 Result ——
生成器不允许自己报性能，这是整条链路可信度的前提。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Verdict(str, Enum):
    """账本里的最终判定。"""

    KEEP = "KEEP"
    REVERT = "REVERT"
    BASELINE = "BASELINE"   # 基线记录，不参与 KEEP/REVERT
    INVALID = "INVALID"     # 结构不合法，未执行


@dataclass
class Candidate:
    """生成器产出的一版候选实现。"""

    source: str
    language: str = "triton"          # triton | cuda
    generator: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.source.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "generator": self.generator,
            "sha": self.sha,
            "meta": self.meta,
        }


@dataclass
class Result:
    """一次候选评测的完整结果（统一口径）。"""

    correct: bool = False
    device_ms: Optional[float] = None
    ref_ms: Optional[float] = None
    ratio: Optional[float] = None
    #: 配对 bootstrap 95% 置信区间（交错采样的逐次比值）
    ratio_low: Optional[float] = None
    ratio_high: Optional[float] = None
    pairs: int = 0
    #: 计时口径：cold（每次冲 L2）/ warm（不冲）。见 contracts/README.md
    cache_policy: str = "cold"
    #: 计时来源：cuda_event（验收用）/ profiler（仅限占比分析）
    timer: str = "cuda_event"
    max_abs_err: Optional[float] = None
    status: str = "unknown"           # ok | incorrect | runtime_error | import_error | rejected
    error: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "device_ms": None if self.device_ms is None else round(self.device_ms, 6),
            "ref_ms": None if self.ref_ms is None else round(self.ref_ms, 6),
            "ratio": None if self.ratio is None else round(self.ratio, 4),
            "ratio_low": None if self.ratio_low is None else round(self.ratio_low, 4),
            "ratio_high": None if self.ratio_high is None else round(self.ratio_high, 4),
            "pairs": self.pairs,
            "cache_policy": self.cache_policy,
            "timer": self.timer,
            "max_abs_err": None if self.max_abs_err is None else float(self.max_abs_err),
            "status": self.status,
            "error": self.error,
            "artifacts": self.artifacts,
        }
