#!/usr/bin/env python3
"""分析产物 → 自动生成任务（融合 + 单算子 Top-K）→ 自动评测 → 汇总报告。

由分析任务完成后自动触发，也可单独运行：
    python3 -m optframe.webui.autopipe --report runs/<id>/bottlenecks.json --top-k 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()
CONTAINER = os.environ.get("OPTFRAME_CONTAINER", "sglang-debug")
EXPERT_RE = re.compile(r"(moe|expert|outplace_fused|fused_moe)", re.IGNORECASE)


def _run(cmd: List[str], timeout: int = 1800, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd or REPO), capture_output=True, text=True, timeout=timeout
    )


def _arch_from_report(report: Dict[str, Any]) -> str:
    target = report.get("target") or {}
    if target.get("arch"):
        return str(target["arch"])
    hw = str(target.get("hardware") or "")
    mapping = {"RTX PRO 5000": "sm120", "L20": "sm89", "H100": "sm90", "H200": "sm90"}
    for name, arch in mapping.items():
        if hw.startswith(name):
            return arch
    return "sm120"


def _task_meta(task_toml: Path) -> Dict[str, Any]:
    text = task_toml.read_text(encoding="utf-8", errors="ignore")

    def grab(key: str) -> str:
        m = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', text, re.MULTILINE)
        return m.group(1) if m else ""

    pct = re.search(r"^\s*pct_e2e\s*=\s*([0-9.]+)", text, re.MULTILINE)
    return {
        "name": grab("name") or task_toml.parent.name,
        "op": grab("op"),
        "kind": grab("kind") or "single",
        "pct_e2e": float(pct.group(1)) if pct else 0.0,
        "dir": str(task_toml.parent),
    }


def select_single_tasks(outdir: Path, top_k: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 pct_e2e 取 Top-K，排除专家/MoE 算子。返回 (kept, skipped)。"""
    tasks = [_task_meta(p) for p in sorted(outdir.glob("*/task.toml"))]
    ranked = sorted(tasks, key=lambda t: -t["pct_e2e"])
    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for t in ranked:
        if EXPERT_RE.search(t["op"] or "") or EXPERT_RE.search(t["name"]):
            skipped.append({**t, "reason": "expert/moe 算子"})
            continue
        if len(kept) < top_k:
            kept.append(t)
        else:
            skipped.append({**t, "reason": f"超出 top-{top_k}"})
    return kept, skipped


def _ledger_last(workdir: Path) -> Optional[Dict[str, Any]]:
    ledger = workdir / "ledger.jsonl"
    if not ledger.exists():
        return None
    last = None
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("stage") == "candidate":
                last = rec
    except OSError:
        return None
    return last


def _eval_in_container(module: str, extra: List[str], workdir: Path, timeout: int = 1800) -> Dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    env_args = ["-w", str(REPO), "-e", f"PYTHONPATH={REPO}"]
    if os.environ.get("GPUSTACK_API_KEY"):
        # `-e VAR` 从 docker 客户端环境取值，避免把密钥写进 argv
        env_args += ["-e", "GPUSTACK_API_KEY"]
    cmd = ["docker", "exec", *env_args, CONTAINER,
           "python3", "-m", module, *extra, "--workdir", str(workdir)]
    try:
        proc = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s", "workdir": str(workdir)}
    rec = _ledger_last(workdir) or {}
    out = {
        "verdict": rec.get("verdict"),
        "ratio": rec.get("ratio"),
        "ci": [rec.get("ratio_low"), rec.get("ratio_high")],
        "cache_policy": rec.get("cache_policy"),
        "workdir": str(workdir),
        "ledger": str(workdir / "ledger.jsonl"),
        "rc": proc.returncode,
    }
    if proc.returncode not in (0, 1):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        out["error"] = " | ".join(tail) or f"rc={proc.returncode}"
    return out


def eval_fused(task_dir: Path, workdir: Path) -> Dict[str, Any]:
    return _eval_in_container(
        "optframe.tiers.fusion",
        ["--task", str(task_dir), "--generator", "fixture",
         "--fixture-dir", str(REPO / "fixtures" / "swiglu"),
         "--n", "1", "--rounds", "1", "--repeat", "20"],
        workdir,
    )


def eval_single(task: Dict[str, Any], workdir: Path, *, allow_llm: bool = False,
                llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    op = (task.get("op") or "").lower()
    fixture = REPO / "fixtures" / "silu"
    has_key = bool(os.environ.get("GPUSTACK_API_KEY"))
    if "silu" in op and fixture.exists():
        return _eval_in_container(
            "optframe.tiers.simple_cli",
            ["--task", task["dir"], "--generator", "fixture",
             "--fixture-dir", str(fixture), "--n", "1", "--rounds", "1", "--repeat", "20"],
            workdir,
        )
    if allow_llm and has_key:
        conf = llm or {}
        return _eval_in_container(
            "optframe.tiers.simple_cli",
            ["--task", task["dir"], "--generator", "llm",
             "--model", str(conf.get("model") or "ds-v4"),
             "--base-url", str(conf.get("base_url") or "http://10.51.198.60:10152/v1-openai"),
             "--n", str(int(conf.get("n") or 1)),
             "--rounds", str(int(conf.get("rounds") or 1)),
             "--repeat", str(int(conf.get("repeat") or 20))],
            workdir,
        )
    if not allow_llm:
        return {"skipped": "无匹配 fixture；自动流水线默认不调 LLM（加 --single-llm 可开）",
                "workdir": str(workdir)}
    return {"skipped": "没有匹配的 fixture，且未配置 GPUSTACK_API_KEY", "workdir": str(workdir)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="bottlenecks.json")
    ap.add_argument("--top-k", type=int, default=5, help="单算子任务最多评测几个")
    ap.add_argument("--single-llm", action="store_true",
                    help="单算子没有 fixture 时用 LLM 生成候选（慢，默认关）")
    ap.add_argument("--llm-model", default="ds-v4")
    ap.add_argument("--llm-base-url", default="http://10.51.198.60:10152/v1-openai")
    ap.add_argument("--llm-n", type=int, default=1)
    ap.add_argument("--llm-rounds", type=int, default=1)
    ap.add_argument("--llm-repeat", type=int, default=20)
    ap.add_argument("--out", default=None, help="pipeline_report.json 输出路径")
    args = ap.parse_args(argv)

    report_path = (REPO / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    if not report_path.exists():
        print(f"[FAIL] report not found: {report_path}")
        return 2
    base = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arch = _arch_from_report(report)
    started = time.time()

    out: Dict[str, Any] = {
        "report": str(report_path),
        "arch": arch,
        "started_at": round(started, 3),
        "fused": {"outdir": None, "tasks": []},
        "single": {"outdir": None, "tasks": [], "skipped": []},
    }

    # 1) 融合任务
    from optframe.inventory import fusion_tasks as ft

    fused_dir = base / "tasks_fused"
    try:
        idx = ft.build(str(report_path), str(fused_dir), arch=arch, dtype="bf16",
                       pattern="swiglu", min_share=0.0)
        out["fused"]["outdir"] = str(fused_dir)
        for entry in idx.get("tasks") or []:
            out["fused"]["tasks"].append({
                "name": entry["name"],
                "dir": entry["dir"],
                "coverage_pct": entry.get("pct_e2e"),
                "eval": eval_fused(Path(entry["dir"]), base / f"eval_fused__{entry['name']}"),
            })
        print(f"[fused] 生成 {len(out['fused']['tasks'])} 个任务")
    except Exception as exc:  # noqa: BLE001
        out["fused"]["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[fused] 失败：{exc}")

    # 2) 单算子任务
    single_dir = base / "tasks_single"
    try:
        amdahl_dir = base / "amdahl"
        p1 = _run([sys.executable, "-m", "optframe.inventory.cli",
                   "--report", str(report_path), "--workdir", str(amdahl_dir),
                   # 自动流水线按“耗时 Top-K”取任务，不用收益阈值把候选提前筛掉
                   "--min-e2e-gain", "0.05", "--min-marginal", "0.01",
                   "--noise-floor", "1.0", "--max-tasks", "20"], timeout=1200)
        if p1.returncode != 0:
            raise RuntimeError((p1.stderr or p1.stdout or "").strip().splitlines()[-3:])
        p2 = _run([sys.executable, "-m", "optframe.inventory.trace_to_tasks",
                   "--amdahl", str(amdahl_dir / "amdahl.json"),
                   "--outdir", str(single_dir), "--arch", arch], timeout=1200)
        if p2.returncode != 0:
            raise RuntimeError((p2.stderr or p2.stdout or "").strip().splitlines()[-3:])
        kept, skipped = select_single_tasks(single_dir, args.top_k)
        out["single"]["outdir"] = str(single_dir)
        out["single"]["skipped"] = skipped
        for task in kept:
            out["single"]["tasks"].append({
                **task,
                "eval": eval_single(task, base / f"eval_single__{task['name']}",
                                    allow_llm=bool(args.single_llm),
                                    llm={"model": args.llm_model, "base_url": args.llm_base_url,
                                         "n": args.llm_n, "rounds": args.llm_rounds,
                                         "repeat": args.llm_repeat}),
            })
        print(f"[single] 生成并选中 {len(out['single']['tasks'])} 个（跳过 {len(skipped)}）")
    except Exception as exc:  # noqa: BLE001
        out["single"]["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[single] 失败：{exc}")

    out["ended_at"] = round(time.time(), 3)
    fused_keep = sum(1 for t in out["fused"]["tasks"] if (t.get("eval") or {}).get("verdict") == "KEEP")
    single_keep = sum(1 for t in out["single"]["tasks"] if (t.get("eval") or {}).get("verdict") == "KEEP")
    out["summary"] = {"fused_total": len(out["fused"]["tasks"]), "fused_keep": fused_keep,
                      "single_total": len(out["single"]["tasks"]), "single_keep": single_keep}

    target = Path(args.out) if args.out else base / "pipeline_report.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pipeline: {target}  fused {fused_keep}/{len(out['fused']['tasks'])} KEEP, "
          f"single {single_keep}/{len(out['single']['tasks'])} KEEP")
    print("JSON " + json.dumps(out["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
