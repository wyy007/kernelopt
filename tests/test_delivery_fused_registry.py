"""交付层双注册表：融合优先、形状门槛、开关、统计（不需要 GPU）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delivery"))

from optframe_delivery import (
    configure, dispatch, match, match_fused, register, register_fused, reset, stats,
)


@pytest.fixture(autouse=True)
def _clean():
    reset()
    configure(True)
    yield
    reset()
    configure(False)


def fake_single(x):
    return "single"


def fake_fused(x):
    return "fused"


def test_single_registry_still_works():
    register(1024, 7168, fake_single, "mm", min_m=512)
    assert match(1024, 1024, 7168) is fake_single
    assert match(256, 1024, 7168) is None      # 低于 min_m


def test_fused_takes_precedence_over_single():
    register(4096, 24576, fake_single, "mm", min_m=512)
    register_fused("swiglu", 4096, 24576, fake_fused, min_m=512)

    mode, name, fn = dispatch(8192, 4096, 24576)
    assert mode == "fused" and fn is fake_fused


def test_fused_min_m_falls_back_to_single():
    register(4096, 24576, fake_single, "mm", min_m=64)
    register_fused("swiglu", 4096, 24576, fake_fused, min_m=512)

    # 小 M：融合被 min_m=512 挡掉，应落回单算子（单算子 min_m=64）
    mode, _, fn = dispatch(100, 4096, 24576)
    assert mode == "single" and fn is fake_single


def test_max_m_upper_bound():
    register_fused("swiglu", 4096, 24576, fake_fused, min_m=512, max_m=8192)
    assert match_fused(8192, 4096, 24576) is fake_fused
    assert match_fused(8193, 4096, 24576) is None


def test_pattern_filter():
    register_fused("swiglu", 4096, 24576, fake_fused, min_m=0)
    assert match_fused(1024, 4096, 24576, pattern="swiglu") is fake_fused
    assert match_fused(1024, 4096, 24576, pattern="other") is None


def test_flag_disables_both_tables():
    register(4096, 24576, fake_single, min_m=0)
    register_fused("swiglu", 4096, 24576, fake_fused, min_m=0)
    configure(False)
    assert dispatch(8192, 4096, 24576) is None


def test_stats_lists_both_tables():
    register(1024, 7168, fake_single, "mm")
    register_fused("swiglu", 4096, 24576, fake_fused, name="fs")
    s = stats()
    assert s["registered"][0]["k"] == 1024
    assert s["fused"][0]["pattern"] == "swiglu"
    assert s["fused"][0]["name"] == "fs"
