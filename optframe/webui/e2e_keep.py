#!/usr/bin/env python3
"""端到端直跑：选中自动流水线的 KEEP 候选 → 临时 overlay 注入 → 8B ABBA。

用法（一般由 WebUI 调用）：
    python3 -m optframe.webui.e2e_keep --run stage2_xxx --task fused__swiglu__3508x4096x24576
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from . import e2e as e2e_mod

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()
DEFAULT_MIN_M = 512


def _find_task(pipeline: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for task in (pipeline.get("fused") or {}).get("tasks") or []:
        if task.get("name") == name:
            return task
    return None


def _candidate_source(task: Dict[str, Any]) -> Optional[Path]:
    ev = task.get("eval") or {}
    src = ev.get("source")
    if src and Path(src).is_file():
        return Path(src)
    workdir = ev.get("workdir")
    if workdir and Path(workdir).is_dir():
        cands = sorted(Path(workdir).glob("cand_*.py"))
        if cands:
            return cands[-1]
    return None


def _task_shape(task_dir: Path) -> tuple[str, int, int, int]:
    data = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    task = data.get("task") or {}
    op = str(task.get("op") or "")
    pattern = op.split("::", 1)[1] if "::" in op else "swiglu"
    shapes = task.get("shapes") or []
    if len(shapes) < 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        raise ValueError("task.toml 缺少 shapes（需要 [[M,K],[K,N]]）")
    k = int(shapes[0][1])
    n = int(shapes[1][1])
    return pattern, k, n, DEFAULT_MIN_M


def build_overlay(run_dir: Path, task: Dict[str, Any], candidate: Path) -> tuple[Path, Dict[str, Any]]:
    """复制 delivery 并把候选注册进去；返回 (overlay 目录, 注册信息)。"""
    pattern, k, n, min_m = _task_shape(Path(task["dir"]))
    stamp = int(time.time())
    overlay = run_dir / f"e2e_delivery_{stamp}"
    shutil.copytree(REPO / "delivery", overlay)
    extra = overlay / "optframe_delivery" / "extra_kernel.py"
    shutil.copyfile(candidate, extra)

    block = f"""

# ---- e2e_keep: 注册本次 KEEP 候选（临时 overlay，不改仓库注册表）----
try:
    from optframe_delivery import register_fused
    from optframe_delivery.extra_kernel import run as _extra_run

    register_fused({pattern!r}, {k}, {n}, _extra_run, "keep:{task['name']}", min_m={min_m})
except Exception:  # noqa: BLE001 - 注册失败时回落到仓库默认注册
    pass
"""
    with (overlay / "sitecustomize.py").open("a", encoding="utf-8") as fh:
        fh.write(block)
    return overlay, {"pattern": pattern, "k": k, "n": n, "min_m": min_m}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="分析 run（含 pipeline_report.json）")
    ap.add_argument("--task", required=True, help="KEEP 的融合任务名")
    ap.add_argument("--model", default="/model/Qwen3-8B")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=1.015)
    ap.add_argument("--order", choices=("alternate", "abba"), default="abba")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--force-free", action="store_true")
    args = ap.parse_args(argv)

    if not all(c.isalnum() or c in "_-" for c in args.run):
        print("[FAIL] run 只允许字母数字下划线")
        return 2
    run_dir = REPO / "runs" / args.run
    pipeline_path = run_dir / "pipeline_report.json"
    if not pipeline_path.is_file():
        print(f"[FAIL] 找不到 {pipeline_path}")
        return 2
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    task = _find_task(pipeline, args.task)
    if task is None:
        print(f"[FAIL] pipeline 里没有任务 {args.task}")
        return 2
    candidate = _candidate_source(task)
    if candidate is None:
        print("[FAIL] 找不到该任务的候选文件（eval 目录下没有 cand_*.py）")
        return 2

    print(f"[1/3] KEEP 任务 {args.task}，候选 {candidate.name}")
    overlay, reg = build_overlay(run_dir, task, candidate)
    print(f"[2/3] overlay: {overlay}  注册 {reg['pattern']} K={reg['k']} N={reg['n']} min_m={reg['min_m']}")
    run_id = args.run_id or f"e2e_keep_{int(time.time())}"
    e2e_argv = [
        "--model", args.model,
        "--blocks", str(args.blocks),
        "--seconds", str(args.seconds),
        "--concurrency", str(args.concurrency),
        "--min-gain", str(args.min_gain),
        "--order", args.order,
        "--run-id", run_id,
        "--delivery-dir", str(overlay),
    ]
    if args.force_free:
        e2e_argv.append("--force-free")
    print("[3/3] 起服务跑 ABBA …")
    return e2e_mod.main(e2e_argv)


if __name__ == "__main__":
    raise SystemExit(main())
