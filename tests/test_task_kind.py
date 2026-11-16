"""两种模式的 schema 兼容性测试：单算子默认零变化，融合字段可选。"""

from __future__ import annotations

from pathlib import Path

import pytest

from optframe.contracts.task import TaskError, load_task
from optframe.inventory.taskspec import TaskSpec

LEGACY_TOML = '''[task]
name = "mm__8x8"
op = "aten::mm"
language = "triton"
arch = "sm_120"
dtype = "bf16"
category = "GEMM"
action = "tune"
baseline = "torch"
target_ratio = 1.2
reference = "reference.py:run"
judge = "judge.py:run"
shapes = [
  [8, 8],
  [8, 8],
]

[timing]
window = "device"
warmup = 10
repeat = 50

[escalation]
allow_complex = false
reasons = []
runtime_rule = "best_of_3_lt_1.05"
'''

FUSED_TOML = LEGACY_TOML.replace(
    'judge = "judge.py:run"',
    'judge = "judge.py:run"\nkind = "fused"\nops = ["aten::mm", "sglang::act_and_mul"]\nregion = "r0"',
)


def _write(tmp_path: Path, text: str) -> Path:
    d = tmp_path / "task"
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(text, encoding="utf-8")
    return d


def test_legacy_task_defaults_to_single(tmp_path):
    task = load_task(_write(tmp_path, LEGACY_TOML))
    assert task.kind == "single"
    assert task.is_fused is False
    assert task.ops == []
    assert task.region is None


def test_fused_task_fields_are_parsed(tmp_path):
    task = load_task(_write(tmp_path, FUSED_TOML))
    assert task.kind == "fused"
    assert task.is_fused is True
    assert task.ops == ["aten::mm", "sglang::act_and_mul"]
    assert task.region == "r0"


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(TaskError):
        load_task(_write(tmp_path, LEGACY_TOML.replace('op = "aten::mm"', 'op = "aten::mm"\nkind = "hybrid"')))


def _spec(**kw) -> TaskSpec:
    base = dict(
        name="mm__8x8", op="aten::mm", arch="sm_120", dtype="bf16",
        shapes=[[8, 8], [8, 8]], target_ratio=1.2, category="GEMM",
    )
    base.update(kw)
    return TaskSpec(**base)


def test_single_mode_toml_is_byte_identical(tmp_path):
    """单算子模式的 toml 不能因为新增字段而改变（历史任务包要能原样重建）。"""
    out = _spec().to_toml()
    assert "kind" not in out and "ops" not in out and "region" not in out
    # 与手写基线逐字节比对（形状/顺序/空行都一致）
    assert out == LEGACY_TOML.replace('name = "mm__8x8"', 'name = "mm__8x8"')


def test_fused_toml_roundtrip(tmp_path):
    out = _spec(kind="fused", ops=["aten::mm", "sglang::act_and_mul"], region="r0").to_toml()
    assert 'kind = "fused"' in out
    assert "sglang::act_and_mul" in out
    assert 'region = "r0"' in out

    task = load_task(_write(tmp_path, out))
    assert task.kind == "fused"
    assert task.ops == ["aten::mm", "sglang::act_and_mul"]
    assert task.region == "r0"
