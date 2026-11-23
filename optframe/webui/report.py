"""从 runs/ 读取运行数据（只读，不含任何服务端代码）。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..inventory.taskspec import arch_for_hardware_exact

_ARTIFACTS = [
    ("analysis", "bottlenecks.json"),
    ("capture", "capture.json"),
    ("tasks", "index.json"),
    ("eval", "ledger.jsonl"),
    ("e2e", "e2e_ab.json"),
]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_artifact(run_dir: Path, name: str) -> Optional[Path]:
    direct = run_dir / name
    if direct.is_file():
        return direct
    for cand in sorted(run_dir.glob(f"*{name}")):
        if cand.is_file():
            return cand
    return None


def _norm_arch(value: Any) -> Optional[str]:
    """架构名归一化：`sm_120` / `SM120` / `sm120` 都算同一个。"""
    if not value:
        return None
    text = str(value).strip().lower().replace("-", "").replace("_", "")
    return text or None


def _rec_arch(rec: Dict[str, Any]) -> Optional[str]:
    return _norm_arch((rec.get("artifacts") or {}).get("arch"))


def _ledger_archs(path: Path) -> List[str]:
    """只扫带 arch 的行，避免为大账本付全量解析成本。"""
    found = set()
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"arch"' not in line:
                    continue
                try:
                    arch = _rec_arch(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if arch:
                    found.add(arch)
    except OSError:
        return []
    return sorted(found)


def _analysis_arch(data: Dict[str, Any]) -> tuple[Optional[str], str]:
    """优先取阶段2实测写入的 target.arch；否则按 hardware 精确映射。"""
    target = data.get("target") or {}
    arch = _norm_arch(target.get("arch"))
    if arch:
        return arch, "analysis"
    arch = _norm_arch(arch_for_hardware_exact(target.get("hardware")))
    if arch:
        return arch, "target.hardware"
    return None, ""


def _summary(data: Dict[str, Any]) -> Dict[str, Any]:
    findings = data.get("findings") or []
    eligible = [f for f in findings if f.get("eligible")]
    top = max((f.get("pct_e2e") or 0.0) for f in findings) if findings else 0.0
    return {
        "findings": len(findings),
        "eligible": len(eligible),
        "top_pct": round(top, 3),
        "gpu_busy": data.get("gpu_busy_pct"),
        "coverage": round(sum(f.get("pct_e2e") or 0.0 for f in eligible), 3),
        "run_id": data.get("run_id"),
        "target": data.get("target"),
    }


def _run_model(run_dir: Path, payload: Dict[str, Any]) -> tuple[str, str]:
    """尽力还原这条 run 对应的模型：分析产物 > 采集记录 > e2e 记录 > service.log。"""
    target = ((payload.get("summary") or {}).get("target") or {})
    if target.get("model"):
        return str(target["model"]), "analysis"
    cap = payload.get("capture") or {}
    if cap.get("model"):
        return str(cap["model"]), "capture"
    e2e = payload.get("e2e") or {}
    if e2e.get("model"):
        return str(e2e["model"]), "e2e"
    log = _find_artifact(run_dir, "service.log")
    if log is not None:
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")[:40000]
        except OSError:
            text = ""
        for match in re.finditer(r"--model-path\s+(\S+)", text):
            cand = match.group(1).strip("'\"")
            # 跳过 argparse 帮助里的占位符（如 `<model>`）
            if not cand.startswith("/") or any(ch in cand for ch in "<>{}$"):
                continue
            return cand, "service.log"
    return "", ""


def scan_runs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        kinds: List[str] = []
        for kind, name in _ARTIFACTS:
            if _find_artifact(d, name) is not None:
                kinds.append(kind)
        if "e2e" not in kinds and list(d.glob("e2e_ab*.json")):
            kinds.append("e2e")
        if not kinds:
            continue
        entry: Dict[str, Any] = {"id": d.name, "mtime": int(d.stat().st_mtime),
                                 "kinds": kinds, "summary": {}, "archs": [],
                                 "arch_source": ""}
        ledger = _find_artifact(d, "ledger.jsonl")
        if ledger is not None:
            entry["archs"] = _ledger_archs(ledger)
            if entry["archs"]:
                entry["arch_source"] = "ledger"
        bn = _find_artifact(d, "bottlenecks.json")
        if bn is not None:
            data = _read_json(bn) or {}
            entry["summary"] = _summary(data)
            if not entry["archs"]:
                arch, arch_source = _analysis_arch(data)
                if arch:
                    entry["archs"] = [arch]
                    entry["arch_source"] = arch_source
        out.append(entry)
    return out


def delete_run(root: Path, run_id: str) -> tuple[bool, str]:
    """把一条 run 移到 runs/_trash/（可恢复），拒绝下划线目录、符号链接与越界路径。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(run_id or "")):
        return False, "run_id 不合法（只允许字母/数字/点/下划线/短横线，且不能以 _ 开头）"
    root = root.resolve()
    d = root / run_id
    if d.is_symlink():
        return False, "拒绝删除符号链接"
    d = d.resolve()
    if d.parent != root or not d.is_dir():
        return False, "运行不存在或不在 runs/ 下"
    trash = root / "_trash"
    trash.mkdir(exist_ok=True)
    dest = trash / f"{run_id}-{int(time.time())}"
    d.rename(dest)
    return True, str(dest)


def list_tasks(root: Path) -> List[Dict[str, Any]]:
    """列出 runs/<run>/<task>/task.toml 的任务包（供页面下拉选择）。"""
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for run_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        for task_toml in sorted(run_dir.glob("*/task.toml")):
            try:
                text = task_toml.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            name = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
            kind = re.search(r'^\s*kind\s*=\s*"([^"]+)"', text, re.MULTILINE)
            out.append(
                {
                    "path": str(task_toml.parent),
                    "name": name.group(1) if name else task_toml.parent.name,
                    "kind": kind.group(1) if kind else "single",
                    "run": run_dir.name,
                }
            )
            if len(out) >= 500:
                return out
    return out


def load_run(root: Path, run_id: str) -> Optional[Dict[str, Any]]:
    d = root / run_id
    if not d.is_dir():
        return None

    payload: Dict[str, Any] = {"id": run_id, "path": str(d.resolve()),
                               "mtime": int(d.stat().st_mtime),
                               "files": {}, "summary": {}, "rows": [],
                               "ledger": [], "e2e": None, "tasks": None,
                               "archs": [], "arch_source": ""}
    hw_arch: Optional[str] = None
    hw_arch_source = ""

    bn = _find_artifact(d, "bottlenecks.json")
    if bn is not None:
        data = _read_json(bn) or {}
        hw_arch, hw_arch_source = _analysis_arch(data)
        payload["files"]["bottlenecks"] = str(bn)
        payload["summary"] = _summary(data)
        rows = []
        for i, f in enumerate(sorted(data.get("findings") or [],
                                     key=lambda x: -(x.get("pct_e2e") or 0)), 1):
            rows.append({
                "rank": i,
                "op": f.get("op"),
                "kernel": f.get("kernel"),
                "shape": f.get("shape"),
                "category": f.get("category"),
                "bound": f.get("bound_type"),
                "sol": f.get("sol_pct"),
                "pct": f.get("pct_e2e"),
                "ms": f.get("gpu_time_ms"),
                "action": f.get("recommended_action"),
                "eligible": bool(f.get("eligible")),
                "reason": f.get("excluded_reason"),
            })
        payload["rows"] = rows

    ledger = _find_artifact(d, "ledger.jsonl")
    if ledger is not None:
        payload["files"]["ledger"] = str(ledger)
        recs = []
        try:
            with ledger.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        except OSError:
            pass
        payload["ledger"] = recs[-50:]
        payload["archs"] = sorted({a for a in (_rec_arch(x) for x in recs) if a})
        if payload["archs"]:
            payload["arch_source"] = "ledger"

    if not payload["archs"] and hw_arch:
        payload["archs"] = [hw_arch]
        payload["arch_source"] = hw_arch_source

    e2e = _find_artifact(d, "e2e_ab.json")
    if e2e is None:
        cands = sorted(d.glob("e2e_ab*.json"))
        e2e = cands[-1] if cands else None
    if e2e is not None:
        payload["files"]["e2e"] = str(e2e)
        payload["e2e"] = _read_json(e2e)

    cap = _find_artifact(d, "capture.json")
    if cap is not None:
        payload["files"]["capture"] = str(cap)
        payload["capture"] = _read_json(cap)

    tasks = _find_artifact(d, "index.json")
    if tasks is not None:
        payload["files"]["tasks"] = str(tasks)
        payload["tasks"] = _read_json(tasks)

    model, model_source = _run_model(d, payload)
    payload["model"] = model
    payload["model_source"] = model_source

    return payload


def _rows_by_key(run: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for r in run.get("rows") or []:
        ms = r.get("ms")
        if r.get("op") and ms:
            out[(r["op"], r.get("shape"))] = {"ms": float(ms), "sol": r.get("sol")}
    return out


def _arch_note(cur_archs: List[str], prev_archs: List[str]) -> str:
    return f"{'/'.join(cur_archs) or '未记录'} vs {'/'.join(prev_archs) or '未记录'}"


def diff_runs(root: Path, cur_id: str, prev_id: str) -> Dict[str, Any]:
    cur, prev = load_run(root, cur_id), load_run(root, prev_id)
    if not cur or not prev:
        return {"error": "run not found", "rows": []}
    cur_archs, prev_archs = cur.get("archs") or [], prev.get("archs") or []
    base = {"cur": cur_id, "prev": prev_id, "cur_archs": cur_archs,
            "prev_archs": prev_archs, "arch_note": _arch_note(cur_archs, prev_archs)}
    # 两侧都记了架构且没有交集时拒绝对比：不同卡的时间/SOL 不可比。
    if cur_archs and prev_archs and not (set(cur_archs) & set(prev_archs)):
        return dict(base, error="arch mismatch", arch_mismatch=True, rows=[])

    a, b = _rows_by_key(cur), _rows_by_key(prev)
    rows = []
    for key, now in a.items():
        old = b.get(key)
        if not old or not old.get("ms"):
            continue
        row: Dict[str, Any] = {"op": key[0], "shape": key[1],
                               "cur_ms": round(now["ms"], 6),
                               "prev_ms": round(old["ms"], 6),
                               "delta_pct": round((now["ms"] / old["ms"] - 1.0) * 100.0, 2)}
        if now.get("sol") is not None and old.get("sol") is not None:
            row["cur_sol"] = now["sol"]
            row["prev_sol"] = old["sol"]
            row["d_sol"] = round(float(now["sol"]) - float(old["sol"]), 2)
        rows.append(row)
    rows.sort(key=lambda x: x["delta_pct"])
    return dict(base, arch_mismatch=False, rows=rows)
