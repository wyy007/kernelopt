"""控制台 e2e 动作的参数校验与 argv 构建（不需要 GPU）。"""

from __future__ import annotations

import pytest

from optframe.webui import jobs


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(jobs, "_python", lambda: "python3")
    monkeypatch.setattr(jobs, "_model_path", lambda value: str(value))
    monkeypatch.setattr(jobs, "_path_in_repo", lambda value, **kw: str(value))


def test_e2e_abba_passes_order_and_min_gain():
    argv = jobs._build_e2e({
        "confirm": True, "order": "abba", "blocks": 8,
        "min_gain": 1.02, "seconds": 30,
    })
    assert "--order" in argv and argv[argv.index("--order") + 1] == "abba"
    assert "--min-gain" in argv and argv[argv.index("--min-gain") + 1] == "1.02"
    assert argv[argv.index("--blocks") + 1] == "8"


def test_e2e_default_is_alternate_without_min_gain_flag():
    argv = jobs._build_e2e({"confirm": True})
    assert argv[argv.index("--order") + 1] == "alternate"
    assert argv[argv.index("--min-gain") + 1] == "1.015"


def test_e2e_requires_confirm():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"blocks": 6})


def test_e2e_rejects_bad_order():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "order": "random"})


def test_e2e_abba_requires_multiple_of_four_blocks():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "order": "abba", "blocks": 6})


def test_e2e_rejects_min_gain_at_or_below_one():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "min_gain": 1.0})


def test_e2e_confirm_adds_force_free_and_max_tokens():
    argv = jobs._build_e2e({"confirm": True, "max_tokens": 32})
    assert "--force-free" in argv
    assert argv[argv.index("--max-tokens") + 1] == "32"


def test_capture_confirm_frees_gpu_and_passes_model_params():
    argv = jobs._build_capture({
        "confirm": True, "mem_fraction": 0.65, "context_length": 4096,
        "kernel_api": True, "run_id": "trace_x",
    })
    assert "--force-free" in argv
    assert "--kernel-api" in argv
    assert argv[argv.index("--mem-fraction") + 1] == "0.65"
    assert argv[argv.index("--context-length") + 1] == "4096"
    assert argv[argv.index("--run-id") + 1] == "trace_x"


def test_capture_requires_confirm():
    with pytest.raises(jobs.ParamError):
        jobs._build_capture({})


def test_analysis_passes_shape_hints_and_model():
    argv = jobs._build_analysis({
        "trace": "runs/x/a.trace.json.gz",
        "shape_hints": "runs/x/hints.json",
        "model": "qwen3-35b",
    })
    assert argv[argv.index("--shape-hints") + 1] == "runs/x/hints.json"
    assert argv[argv.index("--model") + 1] == "qwen3-35b"
    assert argv[argv.index("--framework") + 1] == "sglang"


def test_shape_hints_builds_argv():
    argv = jobs._build_shape_hints({
        "log": "runs/c/kernel_api.log", "op": "outplace_fused_experts",
        "min_m": 16, "max_m": 0, "out": "runs/c/hints.json",
    })
    assert argv[1].endswith("tools/extract_kernel_shapes.py")
    assert argv[argv.index("--op") + 1] == "outplace_fused_experts"
    assert argv[argv.index("--min-m") + 1] == "16"
    assert argv[argv.index("--out") + 1].endswith("runs/c/hints.json")


def test_shape_hints_rejects_unknown_op():
    with pytest.raises(jobs.ParamError):
        jobs._build_shape_hints({"log": "runs/c/kernel_api.log", "op": "foo"})
