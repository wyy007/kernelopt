"""融合任务生成（swiglu）的单元测试：不需要 GPU。"""

from __future__ import annotations

import json

import pytest

from optframe.contracts.task import load_task
from optframe.inventory.fusion_tasks import build, detect_swiglu

REPORT = {
    "run_id": "stage2_test",
    "target": {"hardware": "RTX PRO 5000", "framework": "sglang", "model": "qwen3-8b"},
    "findings": [
        {
            "op": "aten::mm",
            "shape": "((8192, 4096), (4096, 24576))",
            "pct_e2e": 32.29,
            "gpu_time_ms": 3283.1,
        },
        {
            "op": "sglang::_run_activation_inplace",
            "shape": "((), (8192, 24576), (8192, 12288))",
            "pct_e2e": 1.89,
            "gpu_time_ms": 192.4,
        },
    ],
}


def test_detect_swiglu_pairs_mm_with_act():
    regions = detect_swiglu(REPORT["findings"])
    assert len(regions) == 1
    r = regions[0]
    assert r.pattern == "swiglu"
    assert r.shapes == [[8192, 4096], [4096, 24576]]
    assert r.pct_e2e == pytest.approx(34.18)
    assert r.region == "swiglu_m8192_k4096_n24576"
    assert len(r.ops) == 2


def test_detect_requires_matching_act_shape():
    findings = [REPORT["findings"][0]]  # 只有 mm，没有 act
    assert detect_swiglu(findings) == []

    # shape 不匹配（2H 不一致）也不能配对
    mm = REPORT["findings"][0]
    act = dict(REPORT["findings"][1])
    act["shape"] = "((), (4096, 24576), (4096, 12288))"
    assert detect_swiglu([mm, act]) == []


def test_min_share_filters_small_regions():
    assert detect_swiglu(REPORT["findings"], min_share=50.0) == []
    assert len(detect_swiglu(REPORT["findings"], min_share=30.0)) == 1


def test_build_writes_fused_task_and_reference_runs(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(REPORT), encoding="utf-8")

    outdir = tmp_path / "fusion_tasks"
    index = build(report_path, outdir, arch="sm_120", dtype="bf16")

    assert index["kind"] == "fused"
    assert index["total_pct_e2e"] == pytest.approx(34.18)
    assert len(index["tasks"]) == 1

    task = load_task(outdir / index["tasks"][0]["name"])
    assert task.kind == "fused"
    assert task.is_fused is True
    assert task.region == "swiglu_m8192_k4096_n24576"
    assert len(task.ops) == 2

    # 参考实现能在 CPU 上跑，输出形状为 (M, N/2)
    inputs = [__import__("torch").zeros(8, 4096), __import__("torch").zeros(4096, 24576)]
    out = task.reference_run(inputs)
    assert tuple(out.shape) == (8, 12288)


def test_index_records_coverage(tmp_path):
    report_path = tmp_path / "r.json"
    report_path.write_text(json.dumps(REPORT), encoding="utf-8")
    index = build(report_path, tmp_path / "out", arch="sm_120", dtype="bf16")
    assert "total_pct_e2e" in index
    assert index["tasks"][0]["ops"][0] == "aten::mm"
