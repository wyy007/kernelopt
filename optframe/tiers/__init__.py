"""优化档位：简单档（Triton）/ 复杂档（外挂 ForgeKernel）。"""

from .simple import DEFAULT_NOISE_FLOOR, TierOutcome, run_simple_tier

__all__ = ["DEFAULT_NOISE_FLOOR", "TierOutcome", "run_simple_tier"]
