#!/usr/bin/env python3
"""自动从 kernel_api.log 生成 shape hints（不需要选算子）。

扫描日志里出现的已知核，逐个聚合形状，输出一个含多条 pattern 的 JSON，
analysis --shape-hints 可直接消费。

用法：
    python3 tools/auto_shape_hints.py --log runs/<capture>/kernel_api.log \
        --out runs/<capture>/shape_hints.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_kernel_shapes as ex  # noqa: E402

MATCH = {
    "gptq_marlin_gemm": "marlin",
    "fp8_blockwise_scaled_mm": "fp8_blockwise_scaled_mm",
    "outplace_fused_experts": "fused_moe",
}
BITS = {"gptq_marlin_gemm": 4, "fp8_blockwise_scaled_mm": 8, "outplace_fused_experts": 8}


def detect_ops(text: str) -> list[str]:
    ops = []
    if "outplace_fused_experts" in text:
        ops.append("outplace_fused_experts")
    if "fp8_blockwise_scaled_mm" in text:
        ops.append("fp8_blockwise_scaled_mm")
    if "gptq_marlin_gemm" in text or ("marlin" in text and "size_m=" in text):
        ops.append("gptq_marlin_gemm")
    return ops


def _aggregate(records: list, *, match: str, tag: str) -> dict | None:
    if not records:
        return None
    calls = len(records)
    avg_flops = sum(r[0] for r in records) / calls
    avg_bytes = sum(r[1] for r in records) / calls
    ms = [r[2][0] for r in records]
    uniq = collections.Counter(r[2] for r in records)
    return {
        "match": match,
        "avg_flops": round(avg_flops, 1),
        "avg_bytes": round(avg_bytes, 1),
        "calls": calls,
        "m_max": max(ms),
        "m_min": min(ms),
        "tag": tag,
        "shapes": [list(s) for s, _ in uniq.most_common()],
        "shape_counts": [{"shape": list(s), "calls": c} for s, c in uniq.most_common()],
    }


def _marlin_records(blocks: list, bits: int, min_m: int, max_m: int) -> list:
    out = []
    for block in blocks:
        m = re.search(r"size_m=(\d+)", block)
        n = re.search(r"size_n=(\d+)", block)
        k = re.search(r"size_k=(\d+)", block)
        if not (m and n and k):
            continue
        M, N, K = int(m.group(1)), int(n.group(1)), int(k.group(1))
        if max_m and M > max_m:
            continue
        if min_m and M < min_m:
            continue
        flops = 2.0 * M * N * K
        byts = N * K * bits / 8.0 + (M * K + M * N) * 2.0
        out.append((flops, byts, (M, N, K)))
    return out


def build_payload(text: str, source: str, *, min_m: int = 0, max_m: int = 0, tag: str = "auto") -> dict:
    patterns = []
    for op in detect_ops(text):
        args = SimpleNamespace(
            log=source, op=op, match=MATCH[op], weight_bits=BITS[op],
            min_m=min_m, max_m=max_m, tag=tag,
        )
        blocks = [b for b in text.split(ex.CALL_MARK)[1:] if op in b.splitlines()[0]]
        if not blocks:
            continue
        if op == "outplace_fused_experts":
            records = ex._moe_records(args, blocks)
        elif op == "fp8_blockwise_scaled_mm":
            records = ex._fp8_gemm_records(args, blocks)
        else:
            records = _marlin_records(blocks, BITS[op], min_m, max_m)
        pattern = _aggregate(records, match=MATCH[op], tag=tag)
        if pattern:
            patterns.append(pattern)
    return {"source": source, "patterns": patterns}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-m", type=int, default=0)
    ap.add_argument("--max-m", type=int, default=0)
    ap.add_argument("--tag", default="auto")
    args = ap.parse_args(argv)

    text = Path(args.log).read_text(errors="ignore")
    payload = build_payload(text, str(args.log), min_m=args.min_m, max_m=args.max_m, tag=args.tag)
    if not payload["patterns"]:
        print(f"未在 {args.log} 里识别到已知核（marlin / fp8_blockwise_scaled_mm / outplace_fused_experts）")
        return 1
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in payload["patterns"]:
        print(f"  {p['match']}: calls={p['calls']} shapes={len(p['shapes'])} "
              f"avg={p['avg_flops'] / 1e9:.3f} GFLOP/call avg_bytes={p['avg_bytes'] / 1e6:.2f} MB")
    print(f"-> {args.out}（{len(payload['patterns'])} 个 pattern）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
