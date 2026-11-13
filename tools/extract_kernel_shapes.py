#!/usr/bin/env python3
"""把 SGLang 的 kernel API 日志转成分析层用的 shape hints。

用法：
  python3 tools/extract_kernel_shapes.py \
      --log runs/kernel_api.log \
      --op gptq_marlin_gemm \
      --match "marlin" \
      --out runs/shape_hints.json

日志来自 SGLang 启动时的
  SGLANG_KERNEL_API_LOGLEVEL=3
  函数每次调用的 size_m / size_n / size_k 都会打印。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

CALL_MARK = "Kernel API Call: "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--op", default="gptq_marlin_gemm")
    ap.add_argument("--match", default="marlin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight-bits", type=int, default=4, help="量化权重位宽")
    ap.add_argument("--max-m", type=int, default=0, help="只统计 M<=该值的形状（decode 用 1）")
    ap.add_argument("--min-m", type=int, default=0, help="只统计 M>=该值的形状（prefill 用 8）")
    ap.add_argument("--tag", default="all", help="该组形状的名字（decode/prefill）")
    args = ap.parse_args()

    text = Path(args.log).read_text(errors="ignore")
    blocks = [
        block for block in text.split(CALL_MARK)[1:]
        if args.op in block.splitlines()[0]
    ]

    counts: collections.Counter = collections.Counter()
    for block in blocks:
        m = re.search(r"size_m=(\d+)", block)
        n = re.search(r"size_n=(\d+)", block)
        k = re.search(r"size_k=(\d+)", block)
        if m and n and k:
            shape = (int(m.group(1)), int(n.group(1)), int(k.group(1)))
            if args.max_m and shape[0] > args.max_m:
                continue
            if args.min_m and shape[0] < args.min_m:
                continue
            counts[shape] += 1

    bits = args.weight_bits
    total_calls = sum(counts.values())
    total_flops = sum(2 * m * n * k * c for (m, n, k), c in counts.items())
    # 权重按量化位宽、激活按 bf16 估算访存量
    total_bytes = sum(
        (n * k * bits / 8 + (m * k + m * n) * 2) * c
        for (m, n, k), c in counts.items()
    )
    avg_flops = total_flops / total_calls if total_calls else 0.0
    avg_bytes = total_bytes / total_calls if total_calls else 0.0

    payload = {
        "source": str(args.log),
        "patterns": [{
            "match": args.match,
            "avg_flops": round(avg_flops, 1),
            "avg_bytes": round(avg_bytes, 1),
            "calls": total_calls,
            "m_max": max((m for (m, _, _) in counts), default=0),
            "m_min": min((m for (m, _, _) in counts), default=0),
            "tag": args.tag,
            "shapes": [[m, n, k] for (m, n, k), _ in counts.most_common()],
            "shape_counts": [
                {"shape": [m, n, k], "calls": c}
                for (m, n, k), c in counts.most_common()
            ],
        }],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"op={args.op} calls={total_calls} distinct={len(counts)} "
          f"avg={avg_flops/1e9:.3f} GFLOP/call avg_bytes={avg_bytes/1e6:.2f} MB "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
