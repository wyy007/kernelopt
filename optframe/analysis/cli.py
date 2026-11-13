"""阶段2 CLI：trace + arch.json -> bottlenecks.json。

用法：
    python -m optframe.analysis.cli \
        --trace runs/smoke/trace.json \
        --arch arch/L20_full.json \
        --workdir runs/smoke_stage2

仅复用已有报表（不重跑 TraceLens）：
    python -m optframe.analysis.cli --trace ... --arch ... \
        --csv-dir runs/smoke_check --workdir runs/smoke_stage2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .export import (
    build_report,
    infer_phase,
    make_run_id,
    write_analysis_stub,
    write_json,
    write_topk_csv,
)
from .schema import summarize, validate
from .tracelens import TraceLensError, run_perf_report, tracelens_commit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optframe.analysis.cli",
        description="TraceLens 报表 -> bottlenecks.json（阶段2）",
    )
    parser.add_argument("--trace", required=True, help="trace 文件（.json / .json.gz）")
    parser.add_argument("--arch", required=True, help="目标 GPU 的 arch.json（microbench 实测）")
    parser.add_argument("--out", default=None, help="bottlenecks.json 输出路径")
    parser.add_argument("--workdir", default=None, help="本次运行目录（默认 runs/<run_id>）")
    parser.add_argument("--csv-dir", default=None,
                        help="复用已有 TraceLens CSV 目录，跳过报表生成")
    parser.add_argument("--ncu", default=None, help="可选：ncu CSV，用于把置信度升到 measured")
    parser.add_argument("--shape-hints", default=None,
                        help="可选：kernel 形状提示 JSON（自定义 op 的 FLOPs/SOL 补算）")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="覆盖峰值 TFLOPS（默认读 arch.json 的 matrix_bf16）")
    parser.add_argument("--trust-profiled-sol", action="store_true",
                        help="允许用 profiler 窗口的 SOL 排任务（默认不允许，数字有放大）")
    parser.add_argument("--min-pct", type=float, default=2.0,
                        help="Amdahl 过滤阈值（端到端占比 %%），默认 2.0")
    parser.add_argument("--near-roofline", type=float, default=85.0,
                        help="达到该 SOL%% 视为已贴顶，默认 85")
    parser.add_argument("--topk", type=int, default=20, help="topk.csv 行数，默认 20")
    parser.add_argument("--hardware", default=None, help="硬件名（默认取 arch.json 的 name）")
    parser.add_argument("--framework", default="unknown")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--tracelens-src", default="/model/wyy/TraceLens-src",
                        help="TraceLens 源码目录（用于记录 commit）")
    parser.add_argument("--timeout", type=int, default=3600, help="TraceLens 超时秒数")
    return parser


def _arch_name(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return str(data.get("name") or "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    trace = Path(args.trace)
    arch = Path(args.arch)
    phase = infer_phase(trace.name)
    run_id = make_run_id(phase if phase != "unknown" else "stage2")
    workdir = Path(args.workdir) if args.workdir else Path("runs") / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else workdir / "bottlenecks.json"

    try:
        if args.csv_dir:
            csv_dir = Path(args.csv_dir)
            if not (csv_dir / "unified_perf_summary.csv").exists():
                raise FileNotFoundError(
                    f"{csv_dir} 里没有 unified_perf_summary.csv，无法复用"
                )
            print(f"[1/3] 复用报表: {csv_dir}")
        else:
            print(f"[1/3] 生成 TraceLens 报表 -> {workdir}")
            artifacts = run_perf_report(
                trace, arch, workdir, timeout_s=args.timeout
            )
            csv_dir = artifacts.csv_dir
            for line in artifacts.sanity_lines:
                print(f"      {line}")
            for warning in artifacts.warnings:
                print(f"      ⚠️  {warning}")
    except (TraceLensError, FileNotFoundError) as exc:
        print(f"[FAIL] 报表阶段失败: {exc}", file=sys.stderr)
        return 2

    try:
        report = build_report(
            csv_dir,
            trace_path=trace,
            arch_path=arch,
            hardware=args.hardware or _arch_name(arch),
            framework=args.framework,
            model=args.model,
            tracelens_commit=tracelens_commit(Path(args.tracelens_src)),
            ncu_csv=args.ncu,
            shape_hints=args.shape_hints,
            peak_tflops=args.peak_tflops,
            trust_profiled_sol=args.trust_profiled_sol,
            min_pct_e2e=args.min_pct,
            near_roofline_pct=args.near_roofline,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] 解析阶段失败: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    errors = validate(payload)
    if errors:
        print("[FAIL] schema 校验未通过：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 3

    write_json(report, out_path)
    write_topk_csv(report, workdir / "topk.csv", topk=args.topk)
    write_analysis_stub(report, workdir / "analysis_deterministic.md")

    print(f"[2/3] {summarize(report)}")
    print(f"[3/3] 产物:")
    print(f"      {out_path}")
    print(f"      {workdir / 'topk.csv'}")
    print(f"      {workdir / 'analysis_deterministic.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
