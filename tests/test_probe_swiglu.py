"""SiluAndMul 探针：纯逻辑单测（不需要 sglang/GPU）。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delivery"))

from optframe_delivery import probe_swiglu as probe


def setup_function(_):
    probe.reset_stats()
    probe._local.last_gemm = None


def test_pair_hit_when_gemm_matches():
    probe.record_gemm(8192, 4096, 24576)
    assert probe.pair_with_last(8192, 24576) == (8192, 4096, 24576)


def test_no_pair_when_m_mismatch():
    probe.record_gemm(8192, 4096, 24576)
    assert probe.pair_with_last(1024, 24576) is None


def test_no_pair_when_n_mismatch():
    probe.record_gemm(8192, 4096, 24576)
    assert probe.pair_with_last(8192, 12288) is None


def test_no_pair_when_stale():
    probe.record_gemm(8192, 4096, 24576)
    probe._local.last_gemm = (8192, 4096, 24576, time.monotonic() * 1000 - 100)
    assert probe.pair_with_last(8192, 24576) is None


def test_install_without_sglang_is_safe():
    # 仓库 venv 里没有 sglang，install 应安静返回 False 而不是抛错
    assert probe.install() in (True, False)
    assert isinstance(probe.stats(), dict)
