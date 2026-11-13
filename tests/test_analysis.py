"""阶段2 的单元测试（不依赖 GPU / TraceLens，纯 CSV 夹具）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from optframe.analysis import ncu as ncu_mod
from optframe.analysis.export import build_report, write_json, write_topk_csv
from optframe.analysis.filters import classify
from optframe.analysis.parse import load_ops, load_timeline, normalize_bound, normalize_dtype
from optframe.analysis.schema import validate

HEADERS = [
    "name", "op category", "Input Dims", "Input type", "Roofline Bound",
    "Pct Roofline_mean", "TFLOPS/s_mean", "Data Moved (MB)",
    "Kernel Time (µs)_sum", "Kernel Time (µs)_mean", "operation_count",
    "Percentage (%)", "kernel_details_summary",
]


def _write_csv(path: Path, headers, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def make_csv_dir(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "csvs"
    _write_csv(
        csv_dir / "gpu_timeline.csv",
        ["type", "time ms", "percent"],
        [
            ["computation_time", 100.0, 50.0],
            ["exposed_comm_time", 2.0, 1.0],
            ["busy_time", 100.0, 50.0],
            ["idle_time", 100.0, 50.0],
            ["total_time", 200.0, 100.0],
        ],
    )
    rows = [
        ["aten::mm", "GEMM", "((2048, 2048), (2048, 2048))", "('c10::BFloat16',)",
         "COMPUTE_BOUND", 93.0, 104.9, 24.0, 60000.0, 2000.0, 30, 60.0,
         "[{'name': 'cutlass_gemm_kernel'}]"],
        ["aten::silu", "elementwise", "((2048, 2048),)", "('c10::BFloat16',)",
         "MEMORY_BOUND", 40.0, 0.42, 16.0, 20000.0, 666.0, 30, 20.0,
         "[{'name': 'vectorized_elementwise_kernel'}]"],
        ["aten::_softmax", "other", "((2048, 2048),)", "('c10::BFloat16',)",
         "", "", "", "", 10000.0, 333.0, 30, 10.0, ""],
        ["nccl_all_reduce", "communication", "", "", "COMM_BOUND", "", "", "",
         5000.0, 5000.0, 1, 5.0, "[{'name': 'ncclKernel_AllReduce'}]"],
        ["aten::mul", "elementwise", "((2048, 2048),)", "('c10::BFloat16',)",
         "MEMORY_BOUND", 250.0, 0.37, 16.0, 400.0, 13.0, 30, 0.4,
         "[{'name': 'elementwise_kernel'}]"],
        ["flash_mla_kernel", "attention", "", "", "COMPUTE_BOUND", 30.0, 10.0,
         8.0, 1500.0, 50.0, 30, 1.5, "[{'name': 'flash_mla_fwd'}]"],
    ]
    _write_csv(csv_dir / "unified_perf_summary.csv", HEADERS, rows)
    return csv_dir


def test_parse_timeline_and_ops(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    tl = load_timeline(csv_dir)
    assert tl.computation_ms == 100.0
    assert tl.gpu_busy_pct == pytest.approx(50.0)

    ops = load_ops(csv_dir)
    assert [o.name for o in ops][0] == "aten::mm"      # 按 GPU 时间降序
    assert ops[0].dtype == "bf16"
    assert ops[0].kernel == "cutlass_gemm_kernel"


def test_normalizers():
    assert normalize_dtype("('c10::BFloat16', 'Scalar', 'Scalar')") == "bf16+scalar"
    assert normalize_dtype("") == "unknown"
    assert normalize_bound("MEMORY_BOUND") == "memory"
    assert normalize_bound("COMPUTE_BOUND") == "compute"
    assert normalize_bound(None) == "unknown"


def test_classify_rules(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    ops = {o.name: o for o in load_ops(csv_dir)}

    near = classify(ops["aten::mm"], 60.0)
    assert near.bound_type == "compute"
    assert near.confidence == "derived"
    assert near.eligible is False and near.excluded_reason == "near_roofline"

    fused = classify(ops["aten::silu"], 20.0)
    assert fused.recommended_action == "fuse"
    assert fused.suggested_owner == "framework"
    assert fused.eligible is True

    collective = classify(ops["nccl_all_reduce"], 5.0)
    assert collective.excluded_reason == "system_collective"

    hand = classify(ops["flash_mla_kernel"], 1.5)
    assert hand.excluded_reason == "hand_optimized_kernel"

    unknown = classify(ops["aten::_softmax"], 10.0)
    assert unknown.bound_type == "unknown"
    assert unknown.recommended_action == "investigate"
    assert unknown.excluded_reason == "insufficient_evidence"
    assert unknown.sol_pct is None

    tiny = classify(ops["aten::mul"], 0.4)
    assert tiny.excluded_reason == "below_amdahl_threshold"


def test_sol_over_100_is_dropped(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    ops = {o.name: o for o in load_ops(csv_dir)}
    f = classify(ops["aten::mul"], 5.0)
    assert f.sol_pct is None          # 不可信的 roofline 达成率必须丢弃，不能当“已贴顶”
    assert f.sol_pct_raw == 250.0
    assert f.confidence == "heuristic"
    assert f.eligible is True         # 仍应作为候选，因为问题是测量不可信而非确实已优化到顶
    assert any("不可信" in note for note in f.evidence["notes"])


def test_build_report_and_validate(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    trace = tmp_path / "decode_steady_state.json.gz"
    trace.write_text("{}")
    arch = tmp_path / "L20.json"
    arch.write_text(json.dumps({"name": "L20"}))

    report = build_report(
        csv_dir, trace_path=trace, arch_path=arch, hardware="L20", run_id="t1"
    )
    payload = report.to_dict()
    assert validate(payload) == []
    assert payload["summary"]["candidates"] >= 1
    assert payload["source"]["phase_hint"] == "decode"

    out = write_json(report, tmp_path / "bottlenecks.json")
    reloaded = json.loads(out.read_text())
    assert reloaded["run_id"] == "t1"
    assert validate(reloaded) == []

    topk = write_topk_csv(report, tmp_path / "topk.csv")
    assert topk.exists() and len(topk.read_text().strip().splitlines()) >= 2


def test_validate_rejects_measured_without_ncu(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    report = build_report(csv_dir, trace_path=trace, arch_path=trace)
    payload = report.to_dict()
    payload["findings"][0]["confidence"] = "measured"
    errors = validate(payload)
    assert any("measured" in e for e in errors)


def test_validate_rejects_unknown_with_sol(tmp_path):
    csv_dir = make_csv_dir(tmp_path)
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    report = build_report(csv_dir, trace_path=trace, arch_path=trace)
    payload = report.to_dict()
    target = next(f for f in payload["findings"] if f["bound_type"] == "unknown")
    target["sol_pct"] = 42.0
    errors = validate(payload)
    assert any("unknown" in e for e in errors)


def test_ncu_matches_namespace_variant_with_preamble(tmp_path):
    """ncu 会丢命名空间并在 CSV 前打印 ==PROF== 行，必须仍能匹配上。"""
    ncu_csv = tmp_path / "ncu_real.csv"
    ncu_csv.write_text(
        "==PROF== Connected to process 1\n"
        '"ID","Kernel Name","Metric Name","Metric Value"\n'
        '"0","void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8>(Params)",'
        '"sm__throughput.avg.pct_of_peak_sustained_elapsed","89.03"\n'
        '"0","void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8>(Params)",'
        '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","8.50"\n',
        encoding="utf-8",
    )
    table = ncu_mod.load_ncu_csv(ncu_csv)
    assert len(table) == 1

    finding = ncu_mod.match_kernel(
        "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8>(cutlass_80::Params)",
        "aten::mm",
        table,
    )
    assert finding is not None
    assert "Kernel2" in finding


def test_ncu_long_csv_upgrades_confidence(tmp_path):
    ncu_csv = tmp_path / "ncu.csv"
    _write_csv(
        ncu_csv,
        ["Kernel Name", "Metric Name", "Metric Value"],
        [
            ["cutlass_gemm_kernel", ncu_mod.METRIC_SM, "78.5"],
            ["cutlass_gemm_kernel", ncu_mod.METRIC_DRAM, "41.2"],
            ["cutlass_gemm_kernel", ncu_mod.METRIC_OCCUPANCY, "62.0"],
        ],
    )
    table = ncu_mod.load_ncu_csv(ncu_csv)
    assert "cutlass_gemm_kernel" in table

    csv_dir = make_csv_dir(tmp_path)
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    report = build_report(
        csv_dir, trace_path=trace, arch_path=trace, ncu_csv=ncu_csv
    )
    mm = next(f for f in report.findings if f.op == "aten::mm")
    assert mm.confidence == "measured"
    assert mm.evidence["ncu"]["sm_throughput_pct"] == pytest.approx(78.5)
    assert validate(report.to_dict()) == []


def test_normalize_op_name_strips_launch_wrapper():
    from optframe.analysis.parse import normalize_op_name

    assert normalize_op_name("cudaLaunchKernelExC->void marlin::foo") == "void marlin::foo"
    assert normalize_op_name("cudaGraphLaunch->void Kernel2<int>") == "void Kernel2<int>"
    assert normalize_op_name("aten::mm") == "aten::mm"


def test_shape_hints_unlock_custom_gemm():
    """自定义 op 无 shape 时，用 kernel 日志提示补算 FLOPs/SOL。"""
    from optframe.analysis.kernel_shapes import ShapeHint, enrich
    from optframe.analysis.schema import Finding

    f = Finding(
        op="void sglang::device::marlin::Marlin<__nv_bfloat16, 1>",
        shape="", dtype="bf16", gpu_time_ms=10.0, pct_e2e=40.0,
        bound_type="unknown", confidence="inconclusive", count=100,
    )
    hint = ShapeHint(match="marlin", avg_flops=1e10, shapes=[[1, 4096, 4096]],
                     calls=100, source="kernel_api.log")

    enrich([f], [hint], peak_tflops=100.0, allow_eligible=True)

    # FLOPs = 1e10 * 100 = 1e12；10ms → 100 TFLOPS → SOL 100% → 视为贴顶
    assert f.bound_type == "compute"
    assert f.sol_pct == pytest.approx(100.0)
    assert f.excluded_reason == "near_roofline"
    assert f.confidence == "derived"
    assert any("kernel API" in n for n in f.evidence["notes"])


def test_shape_hints_marks_eligible_when_headroom_exists():
    from optframe.analysis.kernel_shapes import ShapeHint, enrich
    from optframe.analysis.schema import Finding

    f = Finding(
        op="marlin_gemm", shape="", dtype="bf16", gpu_time_ms=10.0, pct_e2e=40.0,
        bound_type="unknown", confidence="inconclusive", count=100,
    )
    # FLOPs = 1e10*100 = 1e12 over 10ms = 100 TFLOPS? 需要 40% SOL → 用 2.5e9
    hint = ShapeHint(match="marlin", avg_flops=1e9, shapes=[[1, 1024, 1024]], calls=100)
    enrich([f], [hint], peak_tflops=100.0, allow_eligible=True)

    # FLOPs=1e11, 10ms → 10 TFLOPS → SOL 10%
    assert f.sol_pct == pytest.approx(10.0)
    assert f.eligible is True
    assert f.estimated_gain_pct == pytest.approx(36.0)  # 40 * (1-0.1)


def test_profiled_sol_does_not_queue_task_by_default():
    """profiler 窗口的时间有放大，默认不允许据此排任务。"""
    from optframe.analysis.kernel_shapes import ShapeHint, enrich
    from optframe.analysis.schema import Finding

    f = Finding(op="marlin_gemm", shape="", dtype="bf16", gpu_time_ms=10.0,
                pct_e2e=40.0, bound_type="unknown", confidence="inconclusive", count=100)
    hint = ShapeHint(match="marlin", avg_flops=1e9, shapes=[[1, 1024, 1024]], calls=100)
    enrich([f], [hint], peak_tflops=100.0)

    assert f.eligible is False
    assert f.excluded_reason == "profiled_timing"
    assert f.sol_pct is not None
