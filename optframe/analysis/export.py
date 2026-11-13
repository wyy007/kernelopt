"""把 TraceLens 报表组装成机器可读的 bottlenecks.json。

对应总体方案 §三 步骤3 的输出契约：
    analysis.md（人读，由 agent 层产出）
    bottlenecks.json（机器读，本模块产出）
    topk.csv（快速扫读，本模块产出）
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import ncu as ncu_mod
from .filters import classify, dedupe_and_sort
from .parse import OpRow, Timeline, load_ops, load_timeline
from .schema import Finding, Report

_PHASE_PATTERNS = (
    ("decode", re.compile(r"decode", re.IGNORECASE)),
    ("prefill", re.compile(r"prefill|extend", re.IGNORECASE)),
    ("mixed", re.compile(r"mixed", re.IGNORECASE)),
)


def infer_phase(trace_name: str) -> str:
    """从 trace 文件名猜阶段（decode/prefill/mixed），猜不到给 unknown。"""
    for phase, pattern in _PHASE_PATTERNS:
        if pattern.search(trace_name):
            return phase
    return "unknown"


def make_run_id(prefix: str = "run") -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}"


def build_report(
    csv_dir: str | Path,
    *,
    trace_path: str | Path,
    arch_path: str | Path,
    hardware: str = "unknown",
    framework: str = "unknown",
    model: str = "unknown",
    tracelens_commit: Optional[str] = None,
    ncu_csv: Optional[str | Path] = None,
    shape_hints: Optional[str | Path] = None,
    peak_tflops: Optional[float] = None,
    trust_profiled_sol: bool = False,
    min_pct_e2e: float = 2.0,
    near_roofline_pct: float = 85.0,
    sol_clamp: float = 100.0,
    run_id: Optional[str] = None,
) -> Report:
    """读取报表 CSV，产出 Report（未写盘）。"""
    csv_dir = Path(csv_dir)
    timeline = load_timeline(csv_dir)
    rows = load_ops(csv_dir)

    total_gpu_ms = timeline.computation_ms
    if total_gpu_ms <= 0:
        total_gpu_ms = sum(r.gpu_time_us for r in rows) / 1000.0

    findings: List[Finding] = []
    for row in rows:
        gpu_ms = row.gpu_time_us / 1000.0
        pct_e2e = (gpu_ms / total_gpu_ms * 100.0) if total_gpu_ms > 0 else 0.0
        finding = classify(
            row,
            pct_e2e,
            min_pct_e2e=min_pct_e2e,
            near_roofline_pct=near_roofline_pct,
            sol_clamp=sol_clamp,
        )
        finding.evidence["trace"] = str(Path(trace_path).resolve())
        finding.evidence["report_csv_dir"] = str(csv_dir.resolve())

        if row.pct_reported is not None and abs(row.pct_reported - pct_e2e) > 1.0:
            finding.evidence.setdefault("notes", []).append(
                f"报表百分比 {row.pct_reported:.2f}% 与按 gpu_timeline 计算 "
                f"{pct_e2e:.2f}% 存在差异（分母口径不同）"
            )
        findings.append(finding)

    findings = dedupe_and_sort(findings)

    # 自定义 op（如 marlin）没有 shape，先按 kernel API 日志的提示补算 FLOPs/SOL
    if shape_hints:
        from .kernel_shapes import enrich, load_hints

        peak = peak_tflops
        peak_bw = 0.0
        if peak is None or True:
            try:
                arch_data = json.loads(Path(arch_path).read_text(encoding="utf-8"))
                if peak is None:
                    peak = float(
                        (arch_data.get("max_achievable_tflops") or {}).get("matrix_bf16")
                        or 0.0
                    )
                peak_bw = float(arch_data.get("mem_bw_gbps") or 0.0)
            except (OSError, json.JSONDecodeError, ValueError):
                peak = peak or 0.0
        try:
            hints = load_hints(shape_hints)
        except (OSError, json.JSONDecodeError, KeyError):
            hints = []
        if hints and peak:
            enrich(
                findings, hints, peak_tflops=peak, peak_bw_gbs=peak_bw,
                near_roofline_pct=near_roofline_pct,
                min_pct_e2e=min_pct_e2e,
                allow_eligible=trust_profiled_sol,
            )
            findings = dedupe_and_sort(findings)

    # ncu 证据最后应用：实测计数器优先于时间估算
    if ncu_csv:
        table = ncu_mod.load_ncu_csv(ncu_csv)
        for finding in findings:
            ncu_mod.apply_to_finding(
                finding, table,
                near_roofline_pct=near_roofline_pct,
                min_pct_e2e=min_pct_e2e,
            )
        findings = dedupe_and_sort(findings)

    eligible = [f for f in findings if f.eligible]

    top = eligible[0] if eligible else (findings[0] if findings else None)

    summary: Dict[str, Any] = {
        "total_gpu_ms": round(total_gpu_ms, 3),
        "gpu_busy_pct": round(timeline.gpu_busy_pct, 2),
        "idle_pct": round(timeline.idle_pct, 2),
        "exposed_comm_pct": round(timeline.exposed_comm_pct, 2),
        "top_bottleneck": top.op if top else None,
        "top_bottleneck_pct_e2e": round(top.pct_e2e, 2) if top else None,
        "candidates": len(eligible),
        "excluded": len(findings) - len(eligible),
        "gain_ceiling_pct": round(sum(f.estimated_gain_pct for f in eligible), 2),
        "by_action": _count_by(eligible, "recommended_action"),
        "by_bound": _count_by(eligible, "bound_type"),
    }

    source: Dict[str, Any] = {
        "trace": str(Path(trace_path).resolve()),
        "arch": str(Path(arch_path).resolve()),
        "perf_report_dir": str(csv_dir.parent.resolve()),
        "tracelens_commit": tracelens_commit,
        "phase_hint": infer_phase(Path(trace_path).name),
        "report_csv_dir": str(csv_dir.resolve()),
    }

    target = {"hardware": hardware, "framework": framework, "model": model}

    params = {
        "min_pct_e2e": min_pct_e2e,
        "near_roofline_pct": near_roofline_pct,
        "sol_clamp": sol_clamp,
        "ncu_used": bool(ncu_csv),
    }

    return Report(
        run_id=run_id or make_run_id(),
        source=source,
        target=target,
        summary=summary,
        findings=findings,
        params=params,
    )


def _count_by(findings: List[Finding], attr: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for f in findings:
        key = str(getattr(f, attr))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def write_json(report: Report, path: str | Path) -> Path:
    """写 bottlenecks.json（UTF-8，缩进 2）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def write_topk_csv(report: Report, path: str | Path, topk: int = 20) -> Path:
    """写 topk.csv，供人快速扫读。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "rank", "op", "kernel", "shape", "dtype", "category", "bound", "sol_pct",
        "gpu_time_ms", "pct_e2e", "action", "owner", "eligible", "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for idx, f in enumerate(report.findings[:topk], start=1):
            writer.writerow([
                idx,
                f.op,
                (f.kernel or "")[:80],
                f.shape,
                f.dtype,
                f.category,
                f.bound_type,
                "" if f.sol_pct is None else f"{f.sol_pct:.2f}",
                f"{f.gpu_time_ms:.4f}",
                f"{f.pct_e2e:.2f}",
                f.recommended_action,
                f.suggested_owner,
                "yes" if f.eligible else "no",
                f.excluded_reason or "",
            ])
    return path


def write_analysis_stub(report: Report, path: str | Path) -> Path:
    """生成一份确定性的人读摘要。

    注意：这不是方案里的 agent `analysis.md`（那份由 Analysis Orchestrator 产出），
    而是无大模型依赖的确定性版本，保证任何环境都能出报告。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report.summary
    lines = [
        "# 瓶颈分析摘要（确定性版）",
        "",
        f"- run_id: `{report.run_id}`",
        f"- 生成时间: {report.generated_at}",
        f"- trace: `{report.source['trace']}`",
        f"- arch: `{report.source['arch']}`",
        f"- 硬件: {report.target['hardware']} / 框架: {report.target['framework']}",
        f"- GPU 忙碌: {s['gpu_busy_pct']}% / 空闲: {s['idle_pct']}% / 暴露通信: {s['exposed_comm_pct']}%",
        f"- 候选算子: {s['candidates']} / 已排除: {s['excluded']} / 收益上限: {s['gain_ceiling_pct']}%",
        "",
        "## Top 瓶颈",
        "",
        "| # | 算子 | 类别 | bound | SOL% | 占比% | 动作 | 进入优化 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, f in enumerate(report.findings[:15], start=1):
        sol = "—" if f.sol_pct is None else f"{f.sol_pct:.1f}"
        lines.append(
            f"| {idx} | `{f.op}` | {f.category} | {f.bound_type} | {sol} | "
            f"{f.pct_e2e:.1f} | {f.recommended_action} | "
            f"{'✅' if f.eligible else '❌ ' + (f.excluded_reason or '')} |"
        )
    lines.extend([
        "",
        "## 排除说明",
        "",
    ])
    excluded = [f for f in report.findings if not f.eligible]
    if not excluded:
        lines.append("无。")
    else:
        for f in excluded:
            lines.append(f"- `{f.op}` ({f.pct_e2e:.1f}%)：{f.excluded_reason}")
    lines.append("")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
