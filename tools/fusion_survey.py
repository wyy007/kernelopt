#!/usr/bin/env python3
"""融合机会普查 CLI：见 optframe/analysis/fusion.py 的说明。

用法：
  python3 tools/fusion_survey.py --trace runs/trace_8b/trace.json.gz \
      --out-dir runs/fusion_8b --max-gap 2.0 --max-small-us 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optframe.analysis.fusion import load_kernels, render_md, survey  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-gap", type=float, default=2.0, help="相邻 kernel 最大间隔（µs）")
    ap.add_argument("--max-small-us", type=float, default=300.0, help="算作小核的时长上限（µs）")
    ap.add_argument("--title", default="融合机会普查")
    args = ap.parse_args()

    path = Path(args.trace)
    kernels = load_kernels(path)
    res = survey(kernels, args.max_gap, args.max_small_us)
    res["trace"] = str(path)
    res["params"] = {"max_gap_us": args.max_gap, "max_small_us": args.max_small_us}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_survey.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    (out / "fusion_survey.md").write_text(render_md(res, args.title), encoding="utf-8")

    win = res["window_ms"] or 1.0
    epi = sum(r["save_ms"] for r in res["epilogue"][:5]) / win * 100
    chn = sum(r["save_ms"] for r in res["chain"][:5]) / win * 100
    print(f"trace={path.name} kernels={res['kernels']} busy={res['gpu_busy_pct']}%")
    print(f"epilogue Top5 = {epi:.2f}% window | chain Top5 = {chn:.2f}% window")
    print(f"-> {out/'fusion_survey.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
