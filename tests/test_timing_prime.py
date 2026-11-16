"""分配器预热：两侧都应被预热，且不影响测量结果（需要 CUDA）。"""

import pytest
import torch

from optframe.contracts.timing import prime_allocator, time_pair_samples


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_prime_allocator_holds_then_releases():
    calls = {"n": 0}
    buf = torch.zeros(1024, device="cuda")

    def fn():
        calls["n"] += 1
        return buf + 1

    prime_allocator(fn, rounds=4)
    assert calls["n"] == 4


def test_pair_samples_prime_both_sides():
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    counts = {"ref": 0, "cand": 0}

    def ref():
        counts["ref"] += 1
        return torch.mm(a, b)

    def cand():
        counts["cand"] += 1
        out = torch.empty(256, 256, device="cuda")
        out.copy_(torch.mm(a, b))
        return out

    ref_s, cand_s = time_pair_samples(ref, cand, warmup=2, repeat=3, prime=True)

    # prime 3 + warmup 2 + repeat 3
    assert counts["ref"] == 8
    assert counts["cand"] == 8
    assert len(ref_s) == 3 and len(cand_s) == 3
    assert all(x > 0 for x in ref_s + cand_s)


def test_prime_can_be_disabled():
    a = torch.randn(128, 128, device="cuda")
    counts = {"n": 0}

    def fn():
        counts["n"] += 1
        return torch.mm(a, a)

    time_pair_samples(fn, fn, warmup=1, repeat=2, prime=False)
    assert counts["n"] == 2 * (1 + 2)  # 两侧各 warmup+repeat
