"""简单档编排测试（用假生成器/假评测器，不需要 GPU）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from optframe.contracts.evaluator import Baseline
from optframe.contracts.schema import Candidate, Result
from optframe.generators.base import GenContext
from optframe.tiers.simple import run_simple_tier


class FakeTask:
    """只提供 tier 需要的字段，避免依赖真实 task.toml。"""

    def __init__(self, tmp: Path, allow_complex: bool = True):
        self.dir = tmp
        self.name = "fake__task"
        self.allow_complex = allow_complex
        self.target_ratio = 1.3
        self.op = "aten::silu"
        self.shapes = [[8, 8]]
        self.dtype_name = "bf16"

    def make_inputs(self, device="cuda", seed=0):
        return []


class FakeGenerator:
    name = "fake"

    def __init__(self, rounds_sources):
        self.rounds_sources = rounds_sources
        self.calls = 0

    def propose(self, task, context: GenContext, n=1):
        sources = self.rounds_sources[min(self.calls, len(self.rounds_sources) - 1)]
        self.calls += 1
        return [Candidate(source=s, generator="fake", meta={}) for s in sources[:n]]


def make_result(ratio, correct=True, ratio_low=None):
    return Result(
        correct=correct,
        device_ms=(1.0 / ratio) if ratio else None,
        ref_ms=1.0,
        ratio=ratio,
        ratio_low=ratio if ratio_low is None else ratio_low,
        pairs=50,
        status="ok" if correct else "incorrect",
    )


def test_keep_when_first_round_wins(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=2, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(1.25),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "KEEP"
    assert outcome.best_ratio == pytest.approx(1.25)
    assert len(outcome.rounds) == 1
    assert outcome.escalate is False


def test_revert_then_escalate(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"], ["B"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=2, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(0.9),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "REVERT"
    assert len(outcome.rounds) == 2
    assert outcome.escalate is True   # allow_complex + best<1.05


def test_retry_recovers_in_second_round(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"], ["B"]])
    calls = {"n": 0}

    def evaluate(*args, **kwargs):
        calls["n"] += 1
        return make_result(0.95 if calls["n"] == 1 else 1.4)

    outcome = run_simple_tier(
        task, gen, n=1, rounds=2, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=evaluate,
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "KEEP"
    assert len(outcome.rounds) == 2
    assert outcome.rounds[1].best_ratio == pytest.approx(1.4)


def test_incorrect_candidate_not_kept(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(None, correct=False),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "REVERT"
    assert outcome.best_ratio is None


def test_empty_candidate_recorded_as_invalid(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([[""]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(1.5),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "REVERT"

    from optframe.contracts.ledger import Ledger
    records = Ledger(Path(outcome.ledger_path), run_id="x").read()
    invalid = [r for r in records if r["verdict"] == "INVALID"]
    assert len(invalid) == 1
    assert "empty candidate" in invalid[0]["error"]


def test_noise_floor_blocks_tiny_gain(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(1.01),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "REVERT"   # 1.01 < 1.02 噪声门限


def test_ci_low_blocks_keep(tmp_path):
    """点估计 1.08 但置信区间下界 0.99 -> 不能 KEEP（实测在共享 GPU 上会翻盘）。"""
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(1.08, ratio_low=0.99),
        workdir=tmp_path / "runs",
    )
    assert outcome.verdict == "REVERT"


def test_ci_can_be_disabled_for_debug(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A"]])
    outcome = run_simple_tier(
        task, gen, n=1, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=1.0),
        evaluate_fn=lambda *a, **k: make_result(1.08, ratio_low=0.99),
        workdir=tmp_path / "runs",
        require_ci=False,
    )
    assert outcome.verdict == "KEEP"


def test_ledger_contains_baseline_and_candidates(tmp_path):
    task = FakeTask(tmp_path)
    gen = FakeGenerator([["A", "B"]])
    run_simple_tier(
        task, gen, n=2, rounds=1, inputs=[],
        baseline_fn=lambda *a, **k: Baseline(outputs=[], ms=2.0),
        evaluate_fn=lambda *a, **k: make_result(1.1),
        workdir=tmp_path / "runs",
    )
    from optframe.contracts.ledger import Ledger
    records = Ledger(tmp_path / "runs" / "ledger.jsonl", run_id="x").read()
    assert records[0]["stage"] == "baseline"
    assert records[0]["ref_ms"] == 2.0
    assert len([r for r in records if r["stage"] == "candidate"]) == 2
