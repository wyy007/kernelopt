"""融合机会普查的单元测试（纯 CPU，不依赖 GPU 与 trace 文件）。"""

import pytest

from optframe.analysis.fusion import classify, role, survey


def k(name, ts, dur, stream=0):
    return {"name": name, "ts": float(ts), "dur": float(dur), "stream": stream, "cat": classify(name)}


def test_classify_and_role():
    assert classify("void cutlass::Kernel2<...>") == "gemm"
    assert classify("sglang::outplace_fused_experts->fused_moe_kernel") == "moe"
    assert classify("void sglang::act_and_mul_kernel<...>") == "elementwise"
    assert classify("fused_qkvzba_split_reshape_cat_contiguous_kernel") == "copy"
    assert role("gemm") == "big" and role("copy") == "small"
    assert role("other") == "other"


def test_epilogue_gemm_plus_activation():
    ks = [k("cutlass_gemm_kernel", 0, 100), k("sglang::act_and_mul_kernel", 101, 10)]
    res = survey(ks, max_gap=2.0, max_small=300)
    top = res["epilogue"][0]
    assert top["pattern"] == "gemm+elementwise"
    assert top["count"] == 1
    assert top["save_ms"] == pytest.approx(0.010)


def test_epilogue_respects_gap_threshold():
    ks = [k("cutlass_gemm_kernel", 0, 100), k("sglang::act_and_mul_kernel", 150, 10)]
    res = survey(ks, max_gap=2.0, max_small=300)
    assert res["epilogue"] == []


def test_chain_counts_multiple_small_ops():
    ks = [k("aten::copy_", 0, 10), k("act_kernel", 11, 10), k("reduce_kernel", 22, 10)]
    res = survey(ks, max_gap=2.0, max_small=300)
    top = res["chain"][0]
    assert top["count"] == 1
    assert top["members"] == 3
    # 三个各 10µs 合成一次 pass：省 2/3
    assert top["save_ms"] == pytest.approx(0.020)


def test_big_kernel_breaks_chain():
    ks = [k("aten::copy_", 0, 10), k("cutlass_gemm_kernel", 11, 50), k("act_kernel", 62, 10)]
    res = survey(ks, max_gap=2.0, max_small=300)
    # 两个小核被大核隔开，不应该连成一条链
    assert all(r["members"] < 3 for r in res["chain"])
