"""自动 shape hints：从合成 kernel API 日志识别已知核。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import auto_shape_hints as auto

MARK = "SGLang Kernel API Call: "


def _tensor(idx, shape):
    return (
        f"  arg[{idx}]=Tensor(\n"
        f"      shape=({shape})\n"
        f"      dtype=torch.bfloat16\n"
        f"    )\n"
    )


def _moe_block():
    return (
        f"{MARK}srt.layers.moe.moe_runner.triton_utils.fused_moe.outplace_fused_experts\n"
        "Positional input arguments:\n"
        + _tensor(0, "80, 2048")
        + _tensor(1, "256, 1024, 2048")
        + _tensor(2, "256, 2048, 512")
        + _tensor(3, "80, 8")
    )


def _fp8_block():
    return (
        f"{MARK}sglang.kernels.ops.gemm.fp8_blockwise_gemm.fp8_blockwise_scaled_mm\n"
        + _tensor(0, "80, 2048")
        + _tensor(1, "2048, 12288")
    )


def _marlin_block():
    return (
        f"{MARK}sglang.srt.layers.quantization.gptq_marlin.gptq_marlin_gemm\n"
        "size_m=80 size_n=4096 size_k=4096\n"
    )


def test_detect_ops_and_build_patterns():
    text = _moe_block() + _fp8_block() + _marlin_block()
    ops = auto.detect_ops(text)
    assert ops == ["outplace_fused_experts", "fp8_blockwise_scaled_mm", "gptq_marlin_gemm"]

    payload = auto.build_payload(text, "synthetic.log")
    matches = {p["match"]: p for p in payload["patterns"]}
    assert set(matches) == {"fused_moe", "fp8_blockwise_scaled_mm", "marlin"}

    moe = matches["fused_moe"]
    assert moe["calls"] == 1
    assert moe["m_min"] == 80 and moe["m_max"] == 80
    assert moe["avg_bytes"] > 0

    marlin = matches["marlin"]
    assert marlin["calls"] == 1
    assert marlin["shapes"] == [[80, 4096, 4096]]


def test_build_payload_skips_unknown_log():
    payload = auto.build_payload("nothing useful here", "x.log")
    assert payload["patterns"] == []


def test_main_writes_file(tmp_path):
    log = tmp_path / "kernel_api.log"
    log.write_text(_moe_block() + _fp8_block(), encoding="utf-8")
    out = tmp_path / "hints.json"
    rc = auto.main(["--log", str(log), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    import json

    data = json.loads(out.read_text())
    assert len(data["patterns"]) == 2
