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
TENSOR_MARK = re.compile(r"arg\[(\d+)\]=\s*Tensor\(")
SHAPE_RE = re.compile(r"shape=\(([0-9,\s]+)\)")


# ---- 通用张量参数解析 -------------------------------------------------------

def parse_tensor_shapes(block: str) -> dict:
    """从一段日志里抽出 {arg 下标: 形状元组}。

    SGLang 的日志形如：
        arg[0]=Tensor(
            shape=(80, 2048)
            dtype=torch.bfloat16
    """
    shapes: dict = {}
    cur = None
    for line in block.splitlines():
        m = TENSOR_MARK.search(line)
        if m:
            cur = int(m.group(1))
            # 形状可能写在同一行
            s = SHAPE_RE.search(line)
            if s:
                shapes[cur] = tuple(int(x) for x in s.group(1).split(",") if x.strip())
                cur = None
            continue
        if cur is not None:
            s = SHAPE_RE.search(line)
            if s:
                shapes[cur] = tuple(int(x) for x in s.group(1).split(",") if x.strip())
                cur = None
    return shapes


def _emit(args, records: list, source: str) -> int:
    """把 [(flops, bytes, shape)] 记录聚合成 hints JSON。"""
    if not records:
        print(f"op={args.op} 未解析到任何调用块，请检查日志与 --op 名称")
        return 1
    calls = len(records)
    avg_flops = sum(r[0] for r in records) / calls
    avg_bytes = sum(r[1] for r in records) / calls
    ms = [r[2][0] for r in records]
    uniq = collections.Counter(r[2] for r in records)
    payload = {
        "source": source,
        "patterns": [{
            "match": args.match,
            "avg_flops": round(avg_flops, 1),
            "avg_bytes": round(avg_bytes, 1),
            "calls": calls,
            "m_max": max(ms),
            "m_min": min(ms),
            "tag": args.tag,
            "shapes": [list(s) for s, _ in uniq.most_common()],
            "shape_counts": [
                {"shape": list(s), "calls": c} for s, c in uniq.most_common()
            ],
        }],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"op={args.op} calls={calls} distinct={len(uniq)} "
          f"avg={avg_flops/1e9:.3f} GFLOP/call avg_bytes={avg_bytes/1e6:.2f} MB "
          f"M=[{min(ms)},{max(ms)}] -> {args.out}")
    return 0


def _keep_m(args, m: int) -> bool:
    if args.max_m and m > args.max_m:
        return False
    if args.min_m and m < args.min_m:
        return False
    return True


def _moe_records(args, blocks: list) -> list:
    """outplace_fused_experts：按 (tokens, topk, experts, hidden, inter) 推 FLOPs/访存。

    FLOPs   = 2 * (M*topk) * 3 * inter * hidden      # gate+up+down（swiglu）
    权重访存 = min(E, M*topk) * 3 * inter * hidden * bits/8
    激活访存 = (M*hidden 输入 + M*hidden 输出) * 2 (bf16)
    说明：只读被触发的 expert 切片，M*topk 覆盖不满 E 时按覆盖数计，属保守估计。
    """
    bits = args.weight_bits
    out = []
    for block in blocks:
        s = parse_tensor_shapes(block)
        hidden_states = s.get(0)
        w1 = s.get(1)
        w2 = s.get(2)
        topk_ids = s.get(3)
        if not hidden_states or len(hidden_states) != 2 or not w2 or len(w2) != 3:
            continue
        M, hidden = hidden_states
        if not _keep_m(args, M):
            continue
        experts, _, inter = w2
        topk = topk_ids[1] if topk_ids and len(topk_ids) > 1 else 1
        pairs = max(1, M * topk)
        active_experts = min(experts, pairs)
        flops = 2.0 * pairs * 3.0 * inter * hidden
        w_bytes = active_experts * 3.0 * inter * hidden * bits / 8.0
        act_bytes = (M * hidden + M * hidden) * 2.0
        out.append((flops, w_bytes + act_bytes, (M, 3 * inter, hidden)))
    return out


def _fp8_gemm_records(args, blocks: list) -> list:
    """fp8_blockwise_scaled_mm：从 A(M,K) 与 B(K,N) 直接取 M/N/K。"""
    bits = args.weight_bits
    out = []
    for block in blocks:
        s = parse_tensor_shapes(block)
        a, b = s.get(0), s.get(1)
        if not a or not b or len(a) != 2 or len(b) != 2:
            continue
        M, K = a
        if not _keep_m(args, M):
            continue
        if b[0] == K:
            N = b[1]
        elif b[1] == K:
            N = b[0]
        else:
            continue
        flops = 2.0 * M * N * K
        w_bytes = N * K * bits / 8.0
        act_bytes = (M * K + M * N) * 2.0
        out.append((flops, w_bytes + act_bytes, (M, N, K)))
    return out


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

    if args.op == "outplace_fused_experts":
        return _emit(args, _moe_records(args, blocks), str(args.log))
    if args.op == "fp8_blockwise_scaled_mm":
        return _emit(args, _fp8_gemm_records(args, blocks), str(args.log))

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
