"""端到端 A/B（独占 GPU）：起带回灌的 8B → 交替块 A/B → 关服务。

与 capture 的区别：它不采样，而是**对比开关前后的请求延迟**，用来判定回灌有没有收益。
前提：GPU 空闲（不杀未知进程）、机器上存在 tools/e2e_ab.py。

用法：
    python -m optframe.webui.e2e --model /model/Qwen3-8B --blocks 6 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .capture import CONTAINER, PORT, gpu_used_gb, sh, wait_health

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model/Qwen3-8B")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt-chars", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--min-gain", type=float, default=1.015,
                    help="最小可判收益（延迟比），透传给 tools/e2e_ab.py")
    ap.add_argument("--max-used-gb", type=float, default=5.0)
    args = ap.parse_args(argv)

    run_id = args.run_id or f"e2e_ui_{int(time.time())}"
    if not all(c.isalnum() or c in "_-" for c in run_id):
        print("[FAIL] run_id 只允许字母数字下划线")
        return 2
    out_dir = REPO / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    flag = out_dir / "fuse.flag"
    hits = out_dir / "hits.log"
    flag.write_text("0\n")
    hits.write_text("")
    service_log = out_dir / "service.log"
    result_json = out_dir / "e2e_ab.json"

    used = gpu_used_gb()
    print(f"[1/5] GPU 已用 {used:.1f} GB（阈值 {args.max_used_gb}）")
    if used < 0 or used > args.max_used_gb:
        print(f"[FAIL] GPU 不空闲（{used:.1f} GB），请先停掉其他服务")
        return 2

    env = (f"PYTHONPATH={REPO}/delivery OPTFRAME_BACKFILL=1 "
           f"OPTFRAME_FLAG={flag} OPTFRAME_HIT_LOG={hits} "
           f"nohup python3 -m sglang.launch_server --model-path {args.model} "
           f"--host 0.0.0.0 --port {PORT} --tp-size 1 --context-length 8192 "
           f"--mem-fraction-static 0.55 --dtype bfloat16 --disable-cuda-graph "
           f"> {service_log} 2>&1 &")
    print("[2/5] 起带回灌的服务…")
    sh(["docker", "exec", "-d", CONTAINER, "bash", "-lc", env], timeout=60)

    started = False
    try:
        if not wait_health(300):
            print("[FAIL] 服务未就绪，见", service_log)
            return 3
        started = True
        print("[3/5] 服务就绪，跑交替块 A/B…")
        cmd = [sys.executable, str(REPO / "tools" / "e2e_ab.py"),
               "--url", f"http://127.0.0.1:{PORT}/v1", "--model", args.model,
               "--flag", str(flag), "--hits", str(hits),
               "--blocks", str(args.blocks), "--seconds", str(args.seconds),
               "--concurrency", str(args.concurrency),
               "--prompt-chars", str(args.prompt_chars),
               "--max-tokens", str(args.max_tokens),
               "--min-gain", str(args.min_gain),
               "--flush", "--warmup-block", "--out", str(result_json)]
        rc = subprocess.call(cmd, cwd=str(REPO))
        if not result_json.exists():
            print("[FAIL] e2e_ab 未产出结果，rc=", rc)
            return 4
        data = json.loads(result_json.read_text(encoding="utf-8"))
        lat = data.get("lat_ratio_median")
        ci = data.get("lat_ratio_ci") or [None, None]
        # 判定只认 tools/e2e_ab.py 按契约算出的结果，这里不再自己定标准
        verdict = data.get("verdict") or "INCONCLUSIVE"
        reason = data.get("verdict_reason")
        summary: Dict[str, Any] = {
            "run_id": run_id, "model": args.model, "verdict": verdict,
            "verdict_reason": reason,
            "min_gain": data.get("min_gain", args.min_gain),
            "noise_floor": data.get("noise_floor", round(args.min_gain - 1.0, 6)),
            "n_pairs": data.get("n_pairs", len(data.get("pairs") or [])),
            "lat_ratio_median": lat, "lat_ratio_ci": ci,
            "rps_ratio_median": data.get("rps_ratio_median"),
            "params": vars(args), "result": str(result_json),
            "created_at": round(time.time(), 3),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print("[4/5] 结果:", lat, ci, "→", verdict + (f"（{reason}）" if reason else ""))
        print("JSON " + json.dumps(summary, ensure_ascii=False))
        # 与动作描述一致：非 KEEP 以退出码 1 结束（jobs 的 ok_codes 已把 1 当成功）
        return 0 if verdict == "KEEP" else 1
    finally:
        if started:
            print("[5/5] 关闭服务")
            sh(["docker", "exec", CONTAINER, "bash", "-lc",
                "pkill -f sglang.launch_server || true"], timeout=60)
            time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main())
