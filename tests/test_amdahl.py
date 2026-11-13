"""阶段3 Amdahl 过滤的单元测试。"""

from __future__ import annotations

import math

import pytest

from optframe.inventory.amdahl import (
    AmdahlParams,
    amdahl_ceiling_speedup,
    e2e_gain_pct,
    select,
    target_speedup_for,
)


def F(op, pct, ms, action="tune", sol=None, eligible=True, shape="s1", dtype="bf16",
      reason=None, bound="compute"):
    return {
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "pct_e2e": pct,
        "gpu_time_ms": ms,
        "recommended_action": action,
        "sol_pct": sol,
        "eligible": eligible,
        "excluded_reason": reason,
        "bound_type": bound,
    }


def report(findings, total_ms=100.0):
    return {"findings": findings, "summary": {"total_gpu_ms": total_ms}, "run_id": "t"}


# ------------------------------------------------------------------ 数学部分

def test_e2e_gain_matches_definition():
    # p=0.5, s=2 -> T' = 0.5 + 0.25 = 0.75 -> 收益 25%
    assert e2e_gain_pct(50.0, 2.0) == pytest.approx(25.0)
    # p=0.2, s=4 -> T' = 0.8 + 0.05 = 0.85 -> 15%
    assert e2e_gain_pct(20.0, 4.0) == pytest.approx(15.0)
    assert e2e_gain_pct(10.0, 1.0) == 0.0


def test_amdahl_ceiling():
    assert e2e_gain_pct(50.0, math.inf) == pytest.approx(50.0)
    assert amdahl_ceiling_speedup(50.0) == pytest.approx(2.0)
    assert amdahl_ceiling_speedup(80.0) == pytest.approx(5.0)
    assert amdahl_ceiling_speedup(5.0) == pytest.approx(1.0526, rel=1e-3)


def test_target_speedup_uses_roofline_when_available():
    params = AmdahlParams(target_sol_pct=85.0)
    assert target_speedup_for({"sol_pct": 60.0, "recommended_action": "tune"}, params) == (
        pytest.approx(85.0 / 60.0), "roofline")
    # 已达目标 SOL -> 无需优化
    assert target_speedup_for({"sol_pct": 93.0, "recommended_action": "tune"}, params) == (
        pytest.approx(1.0), "roofline")
    # 无 roofline -> 保守假设
    assert target_speedup_for({"sol_pct": None, "recommended_action": "fuse"}, params) == (
        1.30, "default_fuse")


# ------------------------------------------------------------------ 过滤部分

def test_rejects_stage2_excluded_and_non_optimizable():
    data = report([
        F("aten::mm", 30.0, 30.0, eligible=False, reason="near_roofline"),
        F("aten::fp32_op", 20.0, 20.0, action="precision"),
    ])
    result = select(data)
    reasons = {r.op: r.reason for r in result.rejected}
    assert reasons["aten::mm"] == "stage2_excluded"
    assert reasons["aten::fp32_op"] == "action_not_optimizable"
    assert result.tasks == []


def test_rejects_noise_floor_and_small_gain():
    data = report([
        F("already_fast", 40.0, 40.0, sol=90.0),        # 目标 85 -> 1.0x -> 噪声
        F("tiny", 2.0, 2.0, action="fuse"),             # 2% * 0.23 = 0.46% < 0.5%
    ])
    result = select(data)
    reasons = {r.op: r.reason for r in result.rejected}
    assert reasons["already_fast"] == "below_noise_floor"
    assert reasons["tiny"] == "below_e2e_gain_threshold"


def test_selects_and_orders_by_gain_with_cumulative():
    data = report([
        F("a", 20.0, 20.0, sol=50.0),   # s=1.7 -> 收益 20*(1-1/1.7)=8.24%
        F("b", 30.0, 30.0, sol=40.0),   # s=2.125 -> 15.9%
        F("c", 5.0, 5.0, action="fuse"),  # 1.15%
    ])
    result = select(data)
    assert [t.op for t in result.tasks] == ["b", "a", "c"]
    assert result.tasks[0].rank == 1
    assert result.tasks[-1].cumulative_gain_pct == pytest.approx(result.total_gain_pct)
    # 相加性：累计节省 = 各任务节省之和
    saved = sum(t.gpu_time_ms * (1 - 1 / t.target_speedup) for t in result.tasks)
    assert result.total_gpu_ms_after == pytest.approx(100.0 - saved)


def test_budget_limits_tasks():
    findings = [
        F(f"op{i}", 10.0, 10.0, sol=50.0) for i in range(5)
    ]
    result = select(report(findings), AmdahlParams(max_tasks=2))
    assert len(result.tasks) == 2
    assert sum(1 for r in result.rejected if r.reason == "budget_exceeded") == 3


def test_diminishing_returns_cuts_tail():
    # 第一个任务吃掉大部分收益后，后续任务边际低于阈值
    findings = [
        F("huge", 80.0, 80.0, sol=10.0),   # s=8.5 -> 收益 ~70%
        F("small", 0.2, 0.2, action="fuse"),  # 0.046% -> 先被 min_e2e 拦
    ]
    result = select(report(findings), AmdahlParams(min_e2e_gain_pct=0.01,
                                                   min_marginal_gain_pct=0.05))
    assert result.tasks[0].op == "huge"
    assert any(r.op == "small" for r in result.rejected)


def test_total_gain_cap_stops_selection():
    findings = [F(f"op{i}", 30.0, 30.0, sol=30.0) for i in range(4)]
    result = select(
        report(findings),
        AmdahlParams(max_total_gain_pct=20.0, max_tasks=10, min_marginal_gain_pct=0.0),
    )
    assert result.total_gain_pct <= 20.0 + 1e-9 or len(result.tasks) < len(findings)
    assert any(r.reason == "budget_exceeded" for r in result.rejected)


def test_missing_gpu_time_raises():
    with pytest.raises(ValueError):
        select({"findings": [], "summary": {"total_gpu_ms": 0}})
