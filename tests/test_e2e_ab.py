"""e2e A/B 判定口径的测试（纯函数，不需要 GPU / 服务）。

口径见 optframe/contracts/README.md §9：
    KEEP          n ≥ 2 且 CI 下界 > min_gain
    REVERT        CI 上界 < 1.0
    INCONCLUSIVE  其余（带 reason）
"""

import math

from tools import e2e_ab


def test_default_min_gain_is_1_015():
    assert e2e_ab.DEFAULT_MIN_GAIN == 1.015


def test_below_noise_floor_is_inconclusive():
    # 2026-11-18 实测：CI [1.0050, 1.0054]，点估计 +0.5%，落在 1.5% 噪声里
    verdict, reason = e2e_ab.decide(1.0050, 1.0054, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "below_noise_floor"


def test_degenerate_ci_below_threshold_is_inconclusive():
    # 契约里点名的回归用例：CI 退化成一个点也必须走门限
    verdict, reason = e2e_ab.decide(1.005, 1.005, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "below_noise_floor"


def test_above_threshold_is_keep():
    verdict, reason = e2e_ab.decide(1.021, 1.034, min_gain=1.015, n_pairs=3)
    assert verdict == "KEEP"
    assert reason is None


def test_ci_spanning_threshold_is_inconclusive():
    verdict, reason = e2e_ab.decide(1.010, 1.020, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "ci_spans_threshold"


def test_clear_regression_is_revert():
    verdict, reason = e2e_ab.decide(0.970, 0.985, min_gain=1.015, n_pairs=3)
    assert verdict == "REVERT"
    assert reason == "slower_than_baseline"


def test_ci_spanning_baseline_is_inconclusive():
    # 下限低于基线、上限高于门限：区间太宽，方向不明
    verdict, reason = e2e_ab.decide(0.990, 1.030, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "ci_spans_baseline"


def test_ci_upper_below_threshold_is_below_noise():
    # 上限 1.004 连门限都没到：即便比基线略快也判不了，归 below_noise_floor
    verdict, reason = e2e_ab.decide(0.998, 1.004, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "below_noise_floor"


def test_single_pair_cannot_be_judged():
    verdict, reason = e2e_ab.decide(1.200, 1.300, min_gain=1.015, n_pairs=1)
    assert verdict == "INCONCLUSIVE"
    assert reason == "insufficient_pairs"


def test_nan_ci_is_no_pairs():
    verdict, reason = e2e_ab.decide(math.nan, math.nan, min_gain=1.015, n_pairs=3)
    assert verdict == "INCONCLUSIVE"
    assert reason == "no_pairs"


def test_summarize_pairs_carries_threshold_fields():
    # 三对全是 +0.5%：中位 1.005，bootstrap CI 退化成 [1.005, 1.005]
    pairs = [{"lat_ratio": 1.005, "rps_ratio": 1.005} for _ in range(3)]
    out = e2e_ab.summarize_pairs(pairs, min_gain=1.015)
    assert out["n_pairs"] == 3
    assert out["min_gain"] == 1.015
    assert out["noise_floor"] == 0.015
    assert out["lat_ratio_median"] == 1.005
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["verdict_reason"] == "below_noise_floor"


def test_summarize_pairs_empty():
    out = e2e_ab.summarize_pairs([], min_gain=1.015)
    assert out["n_pairs"] == 0
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["verdict_reason"] == "no_pairs"
