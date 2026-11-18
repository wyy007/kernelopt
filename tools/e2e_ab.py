#!/usr/bin/env python3
"""端到端 A/B：融合回灌开/关的交替块测量。

设计要点（都来自前几次踩坑）：
1. **随机前缀**：同一 prompt 会被 SGLang radix cache 命中，M 掉到 decode 量级，
   `min_m=512` 直接把融合挡掉——每请求注入随机 nonce 强制 cache miss。
2. **交替块**：off → on → off → on ...，同分钟窗口，消掉时钟/温度漂移。
3. **开关落地确认**：flag 文件每秒最多读一次，切换后等 `--settle` 秒再打流。
4. **命中校验**：on 块的融合命中数必须 > 0，否则这一块作废（否则等于在测同一个实现）。
5. 统计：每个块记 p50/p90/吞吐；按 (off_i, on_i) 配对算比值，中位数 + bootstrap CI。
6. **判定口径**（`optframe/contracts/README.md` §9）：延迟比必须超过噪声门限才算 KEEP，
   只看 CI 下界 > 1.0 会把 ±1.5% 的漂移当成收益：

   ```
   KEEP          ⇔ n ≥ 2 且 CI 下界 > min_gain      # min_gain = 1 + noise_floor，默认 1.015
   REVERT        ⇔ CI 上界 < 1.0                     # 确定比基线慢
   INCONCLUSIVE  ⇔ 其余（带 reason）
   ```

用法：
    python tools/e2e_ab.py --url http://127.0.0.1:40012/v1 --model /model/Qwen3-8B \
        --flag /model/wyy/optframe/runs/fuse.flag --hits /model/wyy/optframe/runs/fuse_hits.log \
        --blocks 8 --seconds 20 --concurrency 4 --prompt-chars 3000 --min-gain 1.015
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import string
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

_VOCAB = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
          "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()

#: e2e 最小可判收益。依据：交替块实测块间漂移 1.5%–2.0%；
#: n=3 时 bootstrap CI 半宽会缩到 0.4% 以下，不能拿 CI 半宽当门限。
#: 口径写在 optframe/contracts/README.md §9。
DEFAULT_MIN_GAIN = 1.015


def _prompt(chars: int, rng: random.Random) -> str:
    nonce = "".join(rng.choice(string.ascii_letters) for _ in range(24)) + uuid_safe(rng)
    body = " ".join(rng.choice(_VOCAB) for _ in range(max(1, chars // 6)))
    return f"[req {nonce}] {body}\nSummarize the text above in one sentence."


def uuid_safe(rng: random.Random) -> str:
    return f"{rng.getrandbits(32):08x}"


def one_request(url: str, model: str, chars: int, max_tokens: int,
                timeout: float, rng: random.Random) -> float:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(chars, rng)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return time.time() - t0


def worker(url: str, model: str, chars: int, max_tokens: int, deadline: float,
           lat: List[float], seed: int) -> None:
    rng = random.Random(seed)
    while time.time() < deadline:
        try:
            lat.append(one_request(url, model, chars, max_tokens, 300, rng))
        except Exception:  # noqa: BLE001
            lat.append(-1.0)


def block(url: str, model: str, *, seconds: float, concurrency: int,
          chars: int, max_tokens: int) -> Dict[str, float]:
    deadline = time.time() + seconds
    lat: List[float] = []
    threads = [threading.Thread(target=worker, args=(url, model, chars, max_tokens,
                                                     deadline, lat, 1000 + i))
               for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    ok = sorted(x for x in lat if x > 0)
    if not ok:
        return {"n": 0, "p50": float("nan"), "p90": float("nan"), "rps": 0.0, "elapsed": elapsed}
    return {
        "n": len(ok),
        "p50": statistics.median(ok),
        "p90": ok[min(len(ok) - 1, int(len(ok) * 0.9))],
        "rps": len(ok) / elapsed,
        "elapsed": elapsed,
        "fail": len(lat) - len(ok),
    }


def set_flag(path: Path, value: int) -> None:
    path.write_text(f"{value}\n", encoding="utf-8")


def count_hits(path: Path, prefix: str = "fused:") -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                   if line.startswith(prefix))
    except OSError:
        return 0


def bootstrap_median_ci(values: List[float], iters: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> tuple:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    meds = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    return meds[int(iters * alpha / 2)], meds[int(iters * (1 - alpha / 2))]


def decide(ci_lo: Optional[float], ci_hi: Optional[float],
           min_gain: float = DEFAULT_MIN_GAIN, n_pairs: int = 2) -> tuple:
    """按契约判定 e2e A/B，返回 (verdict, reason)。

    规则（contracts/README.md §9）：
      KEEP          n ≥ 2 且 CI 下界 > min_gain
      REVERT        CI 上界 < 1.0（确定比基线慢）
      INCONCLUSIVE  其余；reason ∈ insufficient_pairs / no_pairs /
                    below_noise_floor / ci_spans_threshold / ci_spans_baseline

    低于门限的差值不判 KEEP——e2e 的噪声量级就是 1.5% 上下，
    点估计好看不代表方向可信。
    """
    if n_pairs < 2:
        return ("INCONCLUSIVE", "insufficient_pairs")
    if ci_lo is None or ci_hi is None or math.isnan(ci_lo) or math.isnan(ci_hi):
        return ("INCONCLUSIVE", "no_pairs")
    if ci_lo > min_gain:
        return ("KEEP", None)
    if ci_hi < 1.0:
        return ("REVERT", "slower_than_baseline")
    if ci_hi <= min_gain:
        return ("INCONCLUSIVE", "below_noise_floor")
    if ci_lo >= 1.0:
        return ("INCONCLUSIVE", "ci_spans_threshold")
    return ("INCONCLUSIVE", "ci_spans_baseline")


def summarize_pairs(pairs: List[Dict[str, float]],
                    min_gain: float = DEFAULT_MIN_GAIN) -> Dict[str, object]:
    """配对比值 → 中位数、CI95、判定与门限字段（写进 e2e_ab.json 的那一份）。"""
    out: Dict[str, object] = {
        "n_pairs": len(pairs),
        "min_gain": min_gain,
        "noise_floor": round(min_gain - 1.0, 6),
    }
    lat = [p["lat_ratio"] for p in pairs]
    if not lat:
        out.update({"verdict": "INCONCLUSIVE", "verdict_reason": "no_pairs"})
        return out
    lo, hi = bootstrap_median_ci(lat)
    verdict, reason = decide(lo, hi, min_gain=min_gain, n_pairs=len(pairs))
    out.update({
        "lat_ratio_median": statistics.median(lat),
        "lat_ratio_ci": [lo, hi],
        "rps_ratio_median": statistics.median(p["rps_ratio"] for p in pairs),
        "verdict": verdict,
        "verdict_reason": reason,
    })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:40012/v1")
    ap.add_argument("--model", default="/model/Qwen3-8B")
    ap.add_argument("--flag", default="/model/wyy/optframe/runs/fuse.flag")
    ap.add_argument("--hits", default="/model/wyy/optframe/runs/fuse_hits.log")
    ap.add_argument("--blocks", type=int, default=8, help="总块数（off/on 交替）")
    ap.add_argument("--seconds", type=float, default=20.0, help="每块打流时长")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-chars", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--settle", type=float, default=2.0, help="切开关后的等待秒数")
    ap.add_argument("--flush", action="store_true", help="每块开始前调用 /flush_cache")
    ap.add_argument("--warmup-block", action="store_true", help="先跑一个丢弃的热身块")
    ap.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN,
                    help="最小可判收益（延迟比），默认 1.015；KEEP 要求 CI 下界 > 该值")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.min_gain <= 1.0:
        ap.error(f"--min-gain 必须 > 1.0（当前 {args.min_gain}）；门限 ≤ 1 等于不设门限")

    flag = Path(args.flag)
    hits = Path(args.hits)

    def flush_cache() -> None:
        base = args.url.rsplit("/v1", 1)[0]
        try:
            req = urllib.request.Request(base + "/flush_cache", data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
        except Exception:  # noqa: BLE001
            pass

    if args.warmup_block:
        set_flag(flag, 0)
        time.sleep(args.settle)
        block(args.url, args.model, seconds=min(10.0, args.seconds),
              concurrency=args.concurrency, chars=args.prompt_chars,
              max_tokens=args.max_tokens)
        print("warmup block done (discarded)", flush=True)

    rows = []
    for i in range(args.blocks):
        mode = "off" if i % 2 == 0 else "on"
        set_flag(flag, 0 if mode == "off" else 1)
        time.sleep(args.settle)
        if args.flush:
            flush_cache()
            time.sleep(0.5)
        before = count_hits(hits)
        r = block(args.url, args.model, seconds=args.seconds,
                  concurrency=args.concurrency, chars=args.prompt_chars,
                  max_tokens=args.max_tokens)
        after = count_hits(hits)
        r.update({"block": i, "mode": mode, "fused_hits": after - before})
        rows.append(r)
        print(f"block {i} [{mode:3s}] n={r['n']:4d} p50={r['p50']*1000:7.1f}ms "
              f"p90={r['p90']*1000:8.1f}ms rps={r['rps']:5.2f} hits={r['fused_hits']}",
              flush=True)

    offs = [r for r in rows if r["mode"] == "off"]
    ons = [r for r in rows if r["mode"] == "on"]
    valid_ons = [r for r in ons if r["fused_hits"] > 0]
    print(f"\non 块命中校验: {len(valid_ons)}/{len(ons)} 有效")

    pairs = []
    for i in range(min(len(offs), len(ons))):
        o, n = offs[i], ons[i]
        if n["fused_hits"] <= 0 or o["p50"] <= 0 or n["p50"] <= 0:
            continue
        pairs.append({"lat_ratio": o["p50"] / n["p50"], "rps_ratio": n["rps"] / o["rps"]})

    summary: Dict[str, object] = {"blocks": rows, "pairs": pairs}
    summary.update(summarize_pairs(pairs, args.min_gain))
    if pairs:
        lo, hi = summary["lat_ratio_ci"]
        print(f"延迟比 (off/on) 中位 = {summary['lat_ratio_median']:.4f} CI95=[{lo:.4f}, {hi:.4f}]")
        print(f"吞吐比 (on/off) 中位 = {summary['rps_ratio_median']:.4f}")
        reason = summary["verdict_reason"]
        print(f"判定: {summary['verdict']}" + (f"（{reason}）" if reason else "")
              + f"  min_gain={args.min_gain:.3f} noise_floor={args.min_gain - 1.0:.3f} n={summary['n_pairs']}")
    else:
        print("没有有效配对（on 块无融合命中或数据缺失）→ INCONCLUSIVE(no_pairs)")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
