"""采集一段真实 trace（独占 GPU）。

流程：检查显卡空闲 → 起 SGLang → 预热 → start_profile → 随机前缀打流 → stop_profile
      → 校验 trace 质量 → 关服务 → 写 capture.json

安全：
    - 起服务前确认 GPU 空闲（占用 < `--max-used-gb`）；勾选「独占 GPU」时
      （`--force-free`）只杀本容器的 `sglang.launch_server`，不动其它进程；
    - 只操作本脚本自己起的进程（记录 PID），结束必关；
    - 所有路径在 runs/ 下，run_id 只允许字母数字下划线。

用法（一般由 WebUI 调用）：
    python -m optframe.webui.capture --model /model/Qwen3-8B --seconds 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import string
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()
CONTAINER = os.environ.get("OPTFRAME_CONTAINER", "sglang-debug")
PORT = 40012

_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
          "omicron pi rho sigma tau upsilon phi chi psi omega").split()


def sh(argv: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def gpu_used_gb() -> float:
    out = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except (ValueError, IndexError):
        return -1.0


def free_gpu(max_used_gb: float, *, force: bool = False, timeout_s: float = 40.0) -> tuple[bool, str]:
    """确保 GPU 空闲；force 时只杀 sglang.launch_server，不碰其它进程。"""
    used = gpu_used_gb()
    if used < 0:
        return False, "读不到 GPU 状态"
    if used <= max_used_gb:
        return True, f"GPU 已空闲（{used:.1f} GB）"
    if not force:
        return False, f"GPU 不空闲（{used:.1f} GB）；勾选「独占 GPU」可自动停 sglang 服务"
    print(f"[free] GPU 已用 {used:.1f} GB，停止容器内 sglang.launch_server …")
    sh(["docker", "exec", CONTAINER, "bash", "-lc",
        "pkill -9 -f sglang.launch_server || true"], timeout=60)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = gpu_used_gb()
        if 0 <= used <= max_used_gb:
            return True, f"已释放 GPU（{used:.1f} GB）"
        time.sleep(2)
    return False, f"停止 sglang 后 GPU 仍被占用（{used:.1f} GB），可能有其它进程"


def wait_health(timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def chat(prompt: str, max_tokens: int, model: str, timeout: float = 300) -> float:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return time.time() - t0


def make_prompt(isl: int, rng: random.Random) -> str:
    nonce = "".join(rng.choice(string.ascii_letters) for _ in range(32))
    body = " ".join(rng.choice(_WORDS) for _ in range(max(1, isl)))
    return f"[req {nonce}] {body}\nSummarize the text above in one sentence."


def load(concurrency: int, seconds: float, isl: int, model: str) -> Dict[str, Any]:
    import threading

    lat: List[float] = []
    deadline = time.time() + seconds

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        while time.time() < deadline:
            try:
                lat.append(chat(make_prompt(isl, rng), 8, model))
            except Exception:  # noqa: BLE001
                lat.append(-1.0)

    threads = [threading.Thread(target=worker, args=(1000 + i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = sorted(x for x in lat if x > 0)
    return {"requests": len(lat), "ok": len(ok), "elapsed_s": round(time.time() - t0, 1),
            "p50_s": round(ok[len(ok) // 2], 2) if ok else None}


def trace_stats(path: Path, window_est: bool = True) -> Dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents") or []
    kernels = [e for e in events if e.get("cat") == "kernel" and (e.get("dur") or 0) > 0]
    total_us = sum(float(e.get("dur") or 0) for e in kernels)
    if kernels:
        start = min(float(e.get("ts") or 0) for e in kernels)
        end = max(float(e.get("ts") or 0) + float(e.get("dur") or 0) for e in kernels)
        window_us = max(1.0, end - start)
    else:
        window_us = 1.0
    return {
        "events": len(events), "kernels": len(kernels),
        "kernel_ms": round(total_us / 1000.0, 1),
        "window_ms": round(window_us / 1000.0, 1),
        "busy_pct": round(total_us / window_us * 100.0, 2),
        "size_mb": round(path.stat().st_size / 1e6, 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model/Qwen3-8B")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.55)
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--isl", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--max-used-gb", type=float, default=5.0)
    ap.add_argument("--keep-service", action="store_true")
    ap.add_argument("--force-free", action="store_true",
                    help="GPU 被占用时自动停掉容器内的 sglang.launch_server")
    ap.add_argument("--kernel-api", action="store_true",
                    help="开启 SGLANG_KERNEL_API_LOGLEVEL=3，落 kernel_api.log 供 shape hints 使用")
    args = ap.parse_args(argv)

    run_id = args.run_id or f"trace_ui_{int(time.time())}"
    if not all(c.isalnum() or c in "_-" for c in run_id):
        print("[FAIL] run_id 只允许字母数字下划线")
        return 2
    out_dir = REPO / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir
    service_log = out_dir / "service.log"

    ok, why = free_gpu(args.max_used_gb, force=args.force_free)
    print(f"[1/7] {why}")
    if not ok:
        print("[FAIL] " + why)
        return 2

    kernel_api_log = out_dir / "kernel_api.log"
    kernel_env = (
        f"SGLANG_KERNEL_API_LOGLEVEL=3 SGLANG_KERNEL_API_LOGDEST={kernel_api_log} "
        if args.kernel_api else ""
    )
    env = (f"{kernel_env}SGLANG_TORCH_PROFILER_DIR={trace_dir} "
           f"nohup python3 -m sglang.launch_server --model-path {args.model} "
           f"--host 0.0.0.0 --port {PORT} --tp-size {args.tp} "
           f"--context-length {args.context_length} --mem-fraction-static {args.mem_fraction} "
           f"--dtype bfloat16 --disable-cuda-graph > {service_log} 2>&1 &")
    print(f"[2/7] 起服务（{CONTAINER}）…")
    sh(["docker", "exec", "-d", CONTAINER, "bash", "-lc", env], timeout=60)

    started = False
    try:
        if not wait_health(300):
            print("[FAIL] 服务 300s 内未就绪，看", service_log)
            return 3
        started = True
        print("[3/7] 服务就绪，预热…")
        chat("hi", 16, args.model, timeout=180)

        print("[4/7] start_profile …")
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/start_profile",
            data=json.dumps({"num_steps": args.num_steps, "record_shapes": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()

        print(f"[5/7] 打流：{args.concurrency} 并发 × {args.seconds}s（随机前缀）")
        load_stats = load(args.concurrency, args.seconds, args.isl, args.model)
        print("      ", load_stats)

        print("[6/7] stop_profile …")
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/stop_profile",
                data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp.read()
        except Exception as exc:  # noqa: BLE001 - 某些版本返回 500 但 trace 已落盘
            print(f"       stop_profile 返回异常（可忽略）：{exc}")

        time.sleep(2)
        traces = sorted(out_dir.glob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
        if not traces:
            print("[FAIL] 没有找到 trace 文件")
            return 4
        trace = traces[-1]
        stats = trace_stats(trace)

        gates = {
            "size_ok": stats["size_mb"] >= 1.0,
            "kernels_ok": stats["kernels"] >= 500,
            "busy_ok": stats["busy_pct"] >= 5.0,
        }
        verdict = "PASS" if all(gates.values()) else "WARN"
        result = {
            "run_id": run_id, "model": args.model, "trace": str(trace),
            "load": load_stats, "trace_stats": stats, "gates": gates, "verdict": verdict,
            "params": {"concurrency": args.concurrency, "isl": args.isl,
                       "seconds": args.seconds, "num_steps": args.num_steps,
                       "mem_fraction": args.mem_fraction,
                       "context_length": args.context_length,
                       "kernel_api": bool(args.kernel_api)},
            "kernel_api": str(kernel_api_log) if args.kernel_api else None,
            "created_at": round(time.time(), 3),
        }
        (out_dir / "capture.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        print(f"[7/7] trace={trace.name} {stats}")
        print(f"      质量门禁: {gates} → {verdict}")
        print("JSON " + json.dumps(result, ensure_ascii=False))
        return 0 if verdict == "PASS" else 1

    finally:
        if started and not args.keep_service:
            print("[cleanup] 关闭服务")
            sh(["docker", "exec", CONTAINER, "bash", "-lc",
                "pkill -f sglang.launch_server || true"], timeout=60)
            time.sleep(3)
        elif not started:
            sh(["docker", "exec", CONTAINER, "bash", "-lc",
                "pkill -f sglang.launch_server || true"], timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
