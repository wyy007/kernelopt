"""融合档编排测试（无 GPU）：拒绝单算子任务、收益归因、prompt 语义。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from optframe.contracts.evaluator import Baseline
from optframe.contracts.schema import Result
from optframe.generators.base import GenContext
from optframe.generators.kernelgen_local import build_prompt
from optframe.tiers.fusion import run_fusion_tier


class FakeFusedTask:
    def __init__(self, tmp: Path, *, kind="fused", allow_complex=False, coverage=34.18):
        self.dir = tmp
        self.name = "fused__swiglu__8192x4096x24576"
        self.kind = kind
        self.allow_complex = allow_complex
        self.target_ratio = 1.05
        self.op = "fused::swiglu"
        self.shapes = [[8192, 4096], [4096, 24576]]
        self.dtype_name = "bf16"
        self.ops = ["aten::mm", "sglang::_run_activation_inplace"]
        self.region = "swiglu_m8192_k4096_n24576"
        self.pct_e2e = coverage

    @property
    def is_fused(self) -> bool:
        return self.kind == "fused"

    @property
    def coverage_pct(self) -> float:
        return float(self.pct_e2e or 0.0)

    def make_inputs(self, device="cuda", seed=0):
        return []


class FakeGenerator:
    name = "fake"
    validator = None

    def propose(self, task, context, n=1):
        from optframe.contracts.schema import Candidate

        return [Candidate(source="A", language="triton", generator="fake") for _ in range(n)]


def _keep(ratio=1.10, low=1.09):
    return Result(correct=True, ratio=ratio, ratio_low=low, ratio_high=1.12,
                  device_ms=1.0, ref_ms=1.1, pairs=50, status="ok")


def test_rejects_single_op_task(tmp_path):
    task = FakeFusedTask(tmp_path, kind="single")
    with pytest.raises(ValueError, match="kind"):
        run_fusion_tier(task, FakeGenerator(), n=1, rounds=1, inputs=[],
                        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
                        evaluate_fn=lambda *a, **k: _keep(), lint=False)


def test_rejects_complex_escalation(tmp_path):
    task = FakeFusedTask(tmp_path, allow_complex=True)
    with pytest.raises(ValueError, match="简单档"):
        run_fusion_tier(task, FakeGenerator(), n=1, rounds=1, inputs=[],
                        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
                        evaluate_fn=lambda *a, **k: _keep(), lint=False)


def test_contribution_uses_coverage_and_ratio(tmp_path):
    task = FakeFusedTask(tmp_path, coverage=34.18)
    res = run_fusion_tier(
        task, FakeGenerator(), n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: _keep(1.10),
        workdir=tmp_path / "runs", lint=False,
    )
    assert res.outcome.verdict == "KEEP"
    assert res.coverage_pct == pytest.approx(34.18)
    # 34.18 * (1 - 1/1.10) = 3.1073
    assert res.contribution_pct == pytest.approx(3.1073, abs=1e-4)
    assert res.region == "swiglu_m8192_k4096_n24576"


def test_no_contribution_when_not_keep(tmp_path):
    task = FakeFusedTask(tmp_path)
    res = run_fusion_tier(
        task, FakeGenerator(), n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: Result(correct=True, ratio=0.95, ratio_low=0.90,
                                           status="ok"),
        workdir=tmp_path / "runs", lint=False,
    )
    assert res.outcome.verdict != "KEEP"
    assert res.contribution_pct == 0.0


def test_prompt_contains_fused_semantics():
    task = SimpleNamespace(
        op="fused::swiglu", arch="sm_120", dtype_name="bf16",
        shapes=[[8192, 4096], [4096, 24576]],
    )
    prompt = build_prompt(task, GenContext())
    assert "silu(gate)" in prompt
    assert "(M, H)" in prompt
    assert "禁止 torch.mm" in prompt
