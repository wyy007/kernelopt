"""阶段3 任务生成（taskspec / codegen / escalation / trace_to_tasks）的测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

from optframe.inventory.escalation import decide, should_escalate_after_best_of_three
from optframe.inventory.taskspec import (
    TaskSpec,
    arch_for_hardware,
    parse_shapes,
    sanitize_name,
    shape_digest,
)
from optframe.inventory.codegen import judge_source, reference_source
from optframe.inventory.trace_to_tasks import build_spec, run


# ---------------------------------------------------------------- taskspec

def test_parse_shapes_handles_nested_tuples():
    assert parse_shapes("((2048, 2048), (2048,))") == [[2048, 2048], [2048]]
    assert parse_shapes("((2048, 2048),)") == [[2048, 2048]]
    assert parse_shapes("((2048, 2048), (), ())") == [[2048, 2048]]
    assert parse_shapes("") == []
    assert parse_shapes("garbage") == []


def test_sanitize_and_digest():
    assert sanitize_name("aten::silu") == "silu"
    assert sanitize_name("aten::_softmax") == "softmax"
    assert sanitize_name("aten::mul.Tensor") == "mul_tensor"
    assert sanitize_name("123") == "op_123"
    assert shape_digest([[2048, 2048]]) == "2048x2048"
    assert len(shape_digest([[8] * 30])) == 10


def test_arch_mapping():
    assert arch_for_hardware("L20") == "sm89"
    assert arch_for_hardware("NVIDIA H100") == "sm90"
    assert arch_for_hardware("B200") == "sm100"
    assert arch_for_hardware("UnknownCard") == "sm89"


def test_task_toml_roundtrip():
    spec = TaskSpec(
        name="silu__2048x2048",
        op="aten::silu",
        arch="sm89",
        dtype="bf16",
        shapes=[[2048, 2048]],
        target_ratio=1.3,
        allow_complex=True,
        escalation_reasons=["attention_moe_or_fusion"],
        pct_e2e=4.682,
    )
    text = spec.to_toml()
    data = tomllib.loads(text)

    assert data["task"]["name"] == "silu__2048x2048"
    assert data["task"]["arch"] == "sm89"
    assert data["task"]["shapes"] == [[2048, 2048]]
    assert data["task"]["target_ratio"] == pytest.approx(1.3)
    assert data["timing"]["warmup"] == 10
    assert data["escalation"]["allow_complex"] is True
    assert data["escalation"]["reasons"] == ["attention_moe_or_fusion"]
    assert data["escalation"]["runtime_rule"] == "best_of_3_lt_1.05"
    assert data["trace"]["pct_e2e"] == pytest.approx(4.682)


# ---------------------------------------------------------------- codegen

def test_reference_source_for_silu():
    src = reference_source("aten::silu", [[2048, 2048]], "bf16")
    assert src is not None
    assert "F.silu(x)" in src
    assert "torch.bfloat16" in src
    assert "[[2048, 2048]]" in src
    compile(src, "<ref>", "exec")  # 语法必须合法


def test_reference_source_for_broadcast_mul():
    src = reference_source("aten::mul", [[2048, 2048], [2048]], "bf16")
    assert src is not None
    assert "return x * y" in src
    assert "args[1]" in src
    compile(src, "<ref>", "exec")


def test_reference_source_unsupported_returns_none():
    assert reference_source("aten::custom_fancy_op", [[8, 8]], "bf16") is None
    assert reference_source("aten::mm", [[8, 8]], "bf16") is None  # 缺第二个输入


def test_judge_source_is_valid_python():
    src = judge_source("bf16")
    compile(src, "<judge>", "exec")
    assert "TOLERANCES" in src
    assert '"bf16": (2e-2, 2e-2)' in src


# ---------------------------------------------------------------- escalation

@pytest.mark.parametrize(
    "task,expected",
    [
        ({"target_speedup": 1.5, "action": "fuse"}, "attention_moe_or_fusion"),
        ({"target_speedup": 1.5, "action": "tune", "category": "attention"},
         "attention_moe_or_fusion"),
        ({"target_speedup": 1.5, "action": "tune", "sol_pct": 42.0}, "sol_lt_60"),
    ],
)
def test_escalation_reasons(task, expected):
    decision = decide(task)
    assert decision.allow_complex is True
    assert expected in decision.reasons


def test_escalation_runtime_rule_is_separate():
    """运行期规则不能靠 target_ratio 假装：小目标不等于“简单档打不动”。"""
    decision = decide({"target_speedup": 1.04, "action": "tune", "category": "gemm",
                       "sol_pct": 93.0})
    assert decision.allow_complex is False
    assert decision.runtime_rule == "best_of_3_lt_1.05"

    assert should_escalate_after_best_of_three(1.02) is True
    assert should_escalate_after_best_of_three(None) is True
    assert should_escalate_after_best_of_three(1.08) is False


def test_escalation_none_for_healthy_task():
    decision = decide({"target_speedup": 1.5, "action": "tune", "category": "gemm",
                       "sol_pct": 70.0})
    assert decision.allow_complex is False
    assert decision.reasons == []


# ---------------------------------------------------------------- 组装

def test_build_spec_skips_unsupported_and_missing_inputs():
    spec, _, reason = build_spec(
        {"op": "aten::mystery_op", "shape": "((8, 8),)", "dtype": "bf16"},
        hardware="L20", arch_override=None,
    )
    assert spec is None and reason == "unsupported_op"

    spec, _, reason = build_spec(
        {"op": "aten::mm", "shape": "((8, 8),)", "dtype": "bf16"},
        hardware="L20", arch_override=None,
    )
    assert spec is None and reason == "missing_inputs"


def test_build_spec_handles_compound_dtype(monkeypatch):
    # 屏蔽设备探测，验证映射表兜底路径
    monkeypatch.setattr(
        "optframe.contracts.cuda_loader.device_arch", lambda: None, raising=False
    )
    spec, comments, reason = build_spec(
        {"op": "aten::silu", "shape": "((8, 8),)", "dtype": "bf16+scalar",
         "action": "fuse", "target_speedup": 1.3, "target_ratio": 1.3},
        hardware="L20", arch_override=None,
    )
    assert reason is None
    assert spec.dtype == "bf16"
    assert any("复合 dtype" in c for c in comments)
    assert spec.arch == "sm89"


def test_build_spec_prefers_device_arch(monkeypatch):
    """有设备时优先用设备 arch，避免映射表过期。"""
    monkeypatch.setattr(
        "optframe.contracts.cuda_loader.device_arch", lambda: "sm120"
    )
    spec, comments, reason = build_spec(
        {"op": "aten::silu", "shape": "((8, 8),)", "dtype": "bf16",
         "action": "tune", "target_speedup": 1.2, "target_ratio": 1.2},
        hardware="RTX PRO 5000", arch_override=None,
    )
    assert reason is None
    assert spec.arch == "sm120"
    assert any("当前设备" in c for c in comments)


def test_build_spec_rejects_unknown_dtype():
    spec, _, reason = build_spec(
        {"op": "aten::silu", "shape": "((8, 8),)", "dtype": "bbox"},
        hardware="L20", arch_override=None,
    )
    assert spec is None and reason == "unsupported_dtype"


def test_end_to_end_generates_runnable_task(tmp_path, monkeypatch):
    # 固定成映射表路径，测试不随机器 GPU 变化
    monkeypatch.setattr(
        "optframe.contracts.cuda_loader.device_arch", lambda: None, raising=False
    )
    amdahl = {
        "schema_version": "1.0",
        "source_run_id": "unit_test",
        "target": {"hardware": "L20", "framework": "pytorch", "model": "smoke"},
        "tasks": [
            {
                "op": "aten::silu",
                "shape": "((64, 64),)",
                "dtype": "bf16",
                "action": "fuse",
                "bound_type": "memory",
                "category": "elementwise",
                "pct_e2e": 4.68,
                "target_speedup": 1.3,
                "target_ratio": 1.3,
                "speedup_basis": "default_fuse",
                "rank": 1,
            },
            {
                "op": "aten::mystery_op",
                "shape": "((8, 8),)",
                "dtype": "bf16",
                "action": "tune",
                "category": "other",
                "pct_e2e": 3.0,
                "target_speedup": 1.2,
            },
        ],
    }
    path = tmp_path / "amdahl.json"
    path.write_text(json.dumps(amdahl), encoding="utf-8")

    index = run(path, tmp_path / "tasks", do_self_check=True)

    assert len(index["tasks"]) == 1
    assert index["skipped"] == [
        {"op": "aten::mystery_op", "shape": "((8, 8),)", "reason": "unsupported_op"}
    ]
    entry = index["tasks"][0]
    assert entry["self_check"]["ok"] is True
    assert entry["allow_complex"] is True  # fuse -> 允许复杂档

    task_dir = Path(entry["dir"])
    for fname in ("task.toml", "reference.py", "judge.py", "notes.md"):
        assert (task_dir / fname).exists()

    data = tomllib.loads((task_dir / "task.toml").read_text())
    assert data["task"]["op"] == "aten::silu"
    assert data["task"]["arch"] == "sm89"


def test_generated_judge_detects_mismatch(tmp_path):
    amdahl = {
        "source_run_id": "t",
        "target": {"hardware": "L20"},
        "tasks": [{
            "op": "aten::silu", "shape": "((32, 32),)", "dtype": "fp32",
            "action": "tune", "category": "elementwise",
            "pct_e2e": 3.0, "target_speedup": 1.2, "target_ratio": 1.2,
        }],
    }
    path = tmp_path / "amdahl.json"
    path.write_text(json.dumps(amdahl), encoding="utf-8")
    index = run(path, tmp_path / "tasks", do_self_check=False)

    task_dir = Path(index["tasks"][0]["dir"])
    spec = importlib.util.spec_from_file_location("gen_judge", task_dir / "judge.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_judge"] = module
    spec.loader.exec_module(module)

    import torch
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4)
    assert module.run(a, a, dtype="fp32")["ok"] is True
    result = module.run(a, b, dtype="fp32")
    assert result["ok"] is False
    assert result["max_abs_err"] == pytest.approx(1.0)
