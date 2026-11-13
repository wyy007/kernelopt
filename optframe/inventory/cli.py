"""阶段3 Amdahl 过滤 CLI。

用法：
    python -m optframe.inventory.cli \
        --report runs/stage2_final/bottlenecks.json \
        --workdir runs/stage3_amdahl \
        --target-sol 85 --min-e2e-gain 0.5 --max-tasks 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .amdahl import AmdahlParams, AmdahlResult, select


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optframe.inventory.cli",
        description="bottlenecks.json -> Amdahl 过滤 -> amdahl.json（阶段3 第一步）",
    )
    p.add_argument("--report", required=True, help="阶段2 产出的 bottlenecks.json")
    p.add_argument("--out", default=None, help="输出 amdahl.json 路径")
    p.add_argument("--workdir", default=None, help="运行目录（默认 runs/stage3_amdahl）")
    p.add_argument("--target-sol", type=float, default=85.0, help="目标 SOL%%，默认 85")
    p.add_argument("--default-tune", type=float, default=1.20, help="无 roofline 时 tune 的保守加速比")
    p.add_argument("--default-fuse", type=float, default=1.30, help="无 roofline 时 fuse 的保守加速比")
    p.add_argument("--noise-floor", type=float, default=1.02, help="噪声门限（加速比），默认 1.02")
    p.add_argument("--min-e2e-gain", type=float, default=0.5, help="单任务端到端收益下限 %%，默认 0.5")
    p.add_argument("--min-marginal", type=float, default=0.1, help="边际收益下限 %%，默认 0.1")
    p.add_argument("--max-tasks", type=int, default=10, help="本轮最多任务数，默认 10")
    p.add_argument("--max-total-gain", type=float, default=35.0, help="累计收益上限 %%，默认 35")
    return p


def _render_markdown(result: AmdahlResult, params: AmdahlParams, report_path: str) -> str:
    lines = [
        "# 阶段3 Amdahl 过滤结果",
        "",
        f"- 输入：`{report_path}`",
        f"- 目标 SOL：{params.target_sol_pct}% ｜ 噪声门限：{params.noise_floor_speedup}×",
        f"- 单任务收益下限：{params.min_e2e_gain_pct}% ｜ 边际下限：{params.min_marginal_gain_pct}%",
        f"- 预算：最多 {params.max_tasks} 个任务 / 累计上限 {params.max_total_gain_pct}%",
        "",
        f"**结论：{len(result.tasks)} 个任务进入生成队列，"
        f"端到端预计 {result.e2e_speedup:.3f}×（−{result.total_gain_pct:.2f}%）**",
        "",
        "| # | 算子 | shape | 动作 | 占比% | 目标加速 | 依据 | 边际收益% | 累计收益% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in result.tasks:
        lines.append(
            f"| {t.rank} | `{t.op}` | `{t.shape}` | {t.action} | {t.pct_e2e:.2f} | "
            f"{t.target_speedup:.2f}× | {t.speedup_basis} | {t.marginal_gain_pct:.3f} | "
            f"{t.cumulative_gain_pct:.3f} |"
        )
    lines.extend(["", "## 被拒绝", ""])
    if not result.rejected:
        lines.append("无。")
    else:
        lines.append("| 算子 | shape | 原因 | 细节 |")
        lines.append("| --- | --- | --- | --- |")
        for r in result.rejected:
            detail = ", ".join(f"{k}={v}" for k, v in r.detail.items()) or "—"
            lines.append(f"| `{r.op}` | `{r.shape}` | {r.reason} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[FAIL] 找不到 {report_path}", file=sys.stderr)
        return 2

    try:
        with report_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] {report_path} 不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    params = AmdahlParams(
        target_sol_pct=args.target_sol,
        default_speedup_tune=args.default_tune,
        default_speedup_fuse=args.default_fuse,
        noise_floor_speedup=args.noise_floor,
        min_e2e_gain_pct=args.min_e2e_gain,
        min_marginal_gain_pct=args.min_marginal,
        max_tasks=args.max_tasks,
        max_total_gain_pct=args.max_total_gain,
    )

    try:
        result = select(payload, params)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir) if args.workdir else Path("runs") / "stage3_amdahl"
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else workdir / "amdahl.json"

    document = {
        "schema_version": "1.0",
        "source_report": str(report_path.resolve()),
        "source_run_id": payload.get("run_id"),
        "source_trace": (payload.get("source") or {}).get("trace"),
        "target": payload.get("target") or {},
        "params": params.to_dict(),
        **result.to_dict(),
    }
    out_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workdir / "amdahl.md").write_text(
        _render_markdown(result, params, str(report_path)), encoding="utf-8"
    )

    print(f"[1/2] 输入 {len(payload.get('findings') or [])} 条 → "
          f"接受 {len(result.tasks)} / 拒绝 {len(result.rejected)}")
    print(f"[2/2] 端到端预计 {result.e2e_speedup:.3f}×（−{result.total_gain_pct:.2f}%）")
    print(f"      产物: {out_path}")
    print(f"             {workdir / 'amdahl.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
