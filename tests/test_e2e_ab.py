"""e2e A/B 判定口径的测试（纯函数，不需要 GPU / 服务）。

口径见 optframe/contracts/README.md §9：
    KEEP          n ≥ 2 且 CI 下界 > min_gain
    REVERT        CI 上界 < 1.0
    INCONCLUSIVE  其余（带 reason）
"""

import math
import subprocess
import sys
from pathlib import Path

from tools import e2e_ab

ROOT = Path(__file__).resolve().parents[1]


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


def test_block_modes_alternate():
    assert e2e_ab.block_modes(6, "alternate") == ["off", "on", "off", "on", "off", "on"]


def test_block_modes_abba():
    assert e2e_ab.block_modes(8, "abba") == [
        "off", "on", "on", "off", "off", "on", "on", "off"]


def test_block_modes_rejects_bad_counts():
    for n in (6, 7, 0):
        try:
            e2e_ab.block_modes(n, "abba")
        except ValueError as exc:
            assert "倍数" in str(exc) or "blocks" in str(exc)
        else:
            raise AssertionError(f"abba 不该接受 {n} 块")


def test_pair_blocks_abba_alternates_position():
    # ABBA: off,on,on,off,off,on,on,off → 配对应为相邻块，且 on 块交替占第一位/第二位
    rows = [{"block": i, "mode": m} for i, m in enumerate(e2e_ab.block_modes(8, "abba"))]
    pairs = e2e_ab.pair_blocks(rows)
    assert [(o["block"], n["block"]) for o, n in pairs] == [(0, 1), (3, 2), (4, 5), (7, 6)]
    assert [n["block"] < o["block"] for o, n in pairs] == [False, True, False, True]


def test_pair_blocks_alternate_is_sequential():
    rows = [{"block": i, "mode": m} for i, m in enumerate(e2e_ab.block_modes(6, "alternate"))]
    assert [(o["block"], n["block"]) for o, n in e2e_ab.pair_blocks(rows)] == [(0, 1), (2, 3), (4, 5)]


def test_cli_rejects_uncapped_threshold():
    """--min-gain ≤ 1 等于不设门限，必须在跑任何块之前就被拒。"""
    for bad in ("1.0", "0.99"):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "e2e_ab.py"),
                            "--min-gain", bad],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 2, r.stderr
        assert "--min-gain" in r.stderr
