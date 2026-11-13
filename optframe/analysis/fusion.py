"""融合机会普查：从真实 trace 里找相邻可融合的 kernel，估算上限收益。

不改任何生产代码，只回答一个问题：**如果把能融的都融了，e2e 最多能省多少？**

做法：
1. 取 trace 里所有 kernel 事件，按 (stream, ts) 排序；
2. 相邻两个 kernel 间隔 < `--max-gap` 视为"连续下发"（有融合可能）；
3. 按名字粗分类（gemm/moe/attn/conv/gdn/elementwise/copy/reduce）；
4. 两种融合模式：
   - epilogue：大核（gemm/moe/attn）+ 紧随的小 elementwise/copy → 小核可并入大核的 epilogue；
   - chain：连续多个小 elementwise/copy → 可合成一次 pass。
5. 估算收益（**上限口径**）：epilogue 记小核时长；chain 记 n 个小核总时长的 (n-1)/n。
   假设：融合后额外访存归零、无额外启动开销。

输出：`fusion_survey.json` + `fusion_survey.md`。

"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

BIG = "big"       # 计算密集，适合当融合的宿主
SMALL = "small"   # 访存密集，适合被吸收


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("fused_moe", "fused_experts", "moe_sum", "moe_fused_gate")):
        return "moe"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "mm_kernel", "matmul", "splitkreduce")):
        return "gemm"
    if any(k in n for k in ("attention", "flash", "paged_kv", "prefill")):
        return "attn"
    if "conv1d" in n:
        return "conv"
    if any(k in n for k in ("gated_delta", "gdn", "recurrent")):
        return "gdn"
    if any(k in n for k in ("copy", "reshape", "cat", "transpose", "split", "memset", "fill")):
        return "copy"
    if any(k in n for k in ("norm", "silu", "gelu", "sigmoid", "tanh", "activation",
                            "elementwise", "act_and_mul", "act_kernel", "and_mul")):
        return "elementwise"
    if any(k in n for k in ("reduce", "sum_", "softmax")):
        return "reduce"
    return "other"


def role(cat: str) -> str:
    if cat in ("gemm", "moe", "attn", "gdn", "conv"):
        return BIG
    if cat in ("elementwise", "copy", "reduce"):
        return SMALL
    return "other"


def load_kernels(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    out = []
    for e in data.get("traceEvents", []):
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        dur = e.get("dur") or 0
        if dur <= 0:
            continue
        args = e.get("args") or {}
        stream = args.get("stream")
        out.append({
            "name": e.get("name") or "",
            "ts": float(e.get("ts") or 0),
            "dur": float(dur),
            "stream": int(stream) if stream is not None else 0,
            "cat": classify(e.get("name") or ""),
        })
    out.sort(key=lambda k: (k["stream"], k["ts"]))
    return out


def survey(kernels: List[dict], max_gap: float, max_small: float) -> Dict:
    if not kernels:
        return {"kernels": 0, "window_ms": 0.0, "patterns": [], "epilogue": [], "chain": []}

    start = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    window_ms = (end - start) / 1000.0
    total_ms = sum(k["dur"] for k in kernels) / 1000.0

    epi = defaultdict(lambda: {"pairs": 0, "save_ms": 0.0, "examples": set()})
    chain = defaultdict(lambda: {"runs": 0, "members": 0, "save_ms": 0.0, "examples": set()})

    by_stream = defaultdict(list)
    for k in kernels:
        by_stream[k["stream"]].append(k)

    for stream, ks in by_stream.items():
        i = 0
        n = len(ks)
        while i < n:
            cur = ks[i]
            # epilogue：大核 + 紧随小核
            if role(cur["cat"]) == BIG and i + 1 < n and role(ks[i + 1]["cat"]) == SMALL:
                nxt = ks[i + 1]
                gap = nxt["ts"] - (cur["ts"] + cur["dur"])
                if gap <= max_gap and nxt["dur"] <= max_small * 1000:
                    key = f"{cur['cat']}+{nxt['cat']}"
                    epi[key]["pairs"] += 1
                    epi[key]["save_ms"] += nxt["dur"] / 1000.0
                    if len(epi[key]["examples"]) < 3:
                        epi[key]["examples"].add(nxt["name"][:60])
            # chain：连续小核（允许中间夹 0 个大核）
            if role(cur["cat"]) == SMALL:
                j = i
                run = []
                while j < n and role(ks[j]["cat"]) == SMALL and ks[j]["dur"] <= max_small * 1000:
                    if run and ks[j]["ts"] - (ks[j - 1]["ts"] + ks[j - 1]["dur"]) > max_gap:
                        break
                    run.append(ks[j])
                    j += 1
                if len(run) >= 2:
                    key = "→".join(sorted({r["cat"] for r in run}))
                    dur_sum = sum(r["dur"] for r in run) / 1000.0
                    chain[key]["runs"] += 1
                    chain[key]["members"] += len(run)
                    chain[key]["save_ms"] += dur_sum * (len(run) - 1) / len(run)
                    for r in run[:3]:
                        if len(chain[key]["examples"]) < 3:
                            chain[key]["examples"].add(r["name"][:60])
                    i = j
                    continue
            i += 1

    def dump(d, label):
        rows = []
        for key, v in d.items():
            rows.append({
                "pattern": key,
                "count": v.get("pairs", v.get("runs", 0)),
                "members": v.get("members", v.get("pairs", 0)),
                "save_ms": round(v["save_ms"], 3),
                "save_pct_window": round(v["save_ms"] / window_ms * 100, 3) if window_ms else 0.0,
                "examples": sorted(v["examples"]),
            })
        rows.sort(key=lambda r: -r["save_ms"])
        return rows

    return {
        "kernels": len(kernels),
        "window_ms": round(window_ms, 3),
        "kernel_time_ms": round(total_ms, 3),
        "gpu_busy_pct": round(total_ms / window_ms * 100, 2) if window_ms else 0.0,
        "epilogue": dump(epi, "epilogue"),
        "chain": dump(chain, "chain"),
    }


def render_md(res: Dict, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- kernel 数：{res['kernels']}")
    lines.append(f"- 窗口时长：{res['window_ms']} ms ｜ kernel 总时长：{res['kernel_time_ms']} ms ｜ GPU 忙碌：{res['gpu_busy_pct']}%")
    lines.append("")
    lines.append("> 收益为**上限口径**：假设融合后额外访存归零、无启动开销。真实收益通常只有 30–60%。")
    lines.append("")
    for name, label in (("epilogue", "① Epilogue 融合（大核 + 紧随小核）"), ("chain", "② Elementwise 链融合")):
        rows = res[name]
        lines.append(f"## {label}")
        lines.append("")
        if not rows:
            lines.append("无")
            lines.append("")
            continue
        lines.append("| 模式 | 次数 | 涉及算子 | 可省时间 (ms) | 占窗口 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in rows[:10]:
            lines.append(f"| `{r['pattern']}` | {r['count']} | {r['members']} | {r['save_ms']} | {r['save_pct_window']}% |")
        lines.append("")
        top = sum(r["save_ms"] for r in rows[:5])
        lines.append(f"Top5 合计：**{top:.3f} ms**（{top / res['window_ms'] * 100:.2f}% 窗口）")
        lines.append("")
    return "\n".join(lines)


