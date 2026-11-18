"""控制台数据层（report.py）的架构门禁与 SOL 差值测试。"""

from __future__ import annotations

import json

from optframe.webui import report


def _finding(op="aten::mm", shape="2048x2048", sol=40.0, pct=12.5):
    return {
        "op": op,
        "kernel": "k",
        "shape": shape,
        "category": "gemm",
        "bound_type": "compute",
        "sol_pct": sol,
        "pct_e2e": pct,
        "gpu_time_ms": 0.021,
        "recommended_action": "tune",
        "eligible": True,
    }


def _run(root, name, arch, sol=40.0, findings=None, hardware=None):
    d = root / name
    d.mkdir(parents=True)
    payload = {"findings": findings or [_finding(sol=sol)], "gpu_busy_pct": 91.0}
    if hardware:
        payload["target"] = {"hardware": hardware, "framework": "sglang", "model": "m"}
    (d / "bottlenecks.json").write_text(json.dumps(payload), encoding="utf-8")
    if arch is not None:
        rec = {"stage": "candidate", "verdict": "KEEP", "run_id": "t",
               "artifacts": {"arch": arch} if arch else {}}
        (d / "ledger.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return d


def test_arch_normalized(tmp_path):
    _run(tmp_path, "a", "sm_120")
    _run(tmp_path, "b", "SM89")
    assert report.load_run(tmp_path, "a")["archs"] == ["sm120"]
    assert report.load_run(tmp_path, "b")["archs"] == ["sm89"]


def test_scan_runs_carries_archs(tmp_path):
    _run(tmp_path, "a", "sm89")
    entry = next(e for e in report.scan_runs(tmp_path) if e["id"] == "a")
    assert entry["archs"] == ["sm89"]


def test_diff_same_arch_reports_sol_delta(tmp_path):
    _run(tmp_path, "new", "sm120", sol=44.0)
    _run(tmp_path, "old", "sm_120", sol=40.0)
    out = report.diff_runs(tmp_path, "new", "old")
    assert out["arch_mismatch"] is False
    assert out["rows"], "同架构对比应产出行"
    row = out["rows"][0]
    assert row["d_sol"] == 4.0
    assert row["cur_sol"] == 44.0 and row["prev_sol"] == 40.0


def test_diff_cross_arch_refused(tmp_path):
    _run(tmp_path, "new", "sm120")
    _run(tmp_path, "old", "sm89")
    out = report.diff_runs(tmp_path, "new", "old")
    assert out["arch_mismatch"] is True
    assert out["error"] == "arch mismatch"
    assert out["rows"] == []
    assert "sm120" in out["arch_note"] and "sm89" in out["arch_note"]


def test_diff_without_arch_still_compares(tmp_path):
    _run(tmp_path, "new", None)
    _run(tmp_path, "old", "sm89")
    out = report.diff_runs(tmp_path, "new", "old")
    assert out["arch_mismatch"] is False
    assert out["rows"], "一侧未记录架构时保持可对比"


def test_analysis_run_gets_arch_from_hardware(tmp_path):
    _run(tmp_path, "a", None, hardware="RTX PRO 5000")
    run = report.load_run(tmp_path, "a")
    assert run["archs"] == ["sm120"]
    assert run["arch_source"] == "target.hardware"
    entry = next(e for e in report.scan_runs(tmp_path) if e["id"] == "a")
    assert entry["archs"] == ["sm120"]


def test_unknown_hardware_stays_unlabeled(tmp_path):
    _run(tmp_path, "a", None, hardware="Blackwell")
    run = report.load_run(tmp_path, "a")
    assert run["archs"] == [] and run["arch_source"] == ""


def test_ledger_arch_wins_over_hardware(tmp_path):
    _run(tmp_path, "a", "sm89", hardware="RTX PRO 5000")
    run = report.load_run(tmp_path, "a")
    assert run["archs"] == ["sm89"]
    assert run["arch_source"] == "ledger"


def test_gate_fires_on_derived_arch(tmp_path):
    _run(tmp_path, "new", None, hardware="RTX PRO 5000")
    _run(tmp_path, "old", None, hardware="L20")
    out = report.diff_runs(tmp_path, "new", "old")
    assert out["arch_mismatch"] is True
    assert out["arch_note"] == "sm120 vs sm89"
