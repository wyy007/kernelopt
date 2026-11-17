"""从 runs/ 读取运行数据（只读，不含任何服务端代码）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def scan_runs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
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
                                 "kinds": kinds, "summary": {}}
        bn = _find_artifact(d, "bottlenecks.json")
        if bn is not None:
            entry["summary"] = _summary(_read_json(bn) or {})
        out.append(entry)
    return out


def load_run(root: Path, run_id: str) -> Optional[Dict[str, Any]]:
    d = root / run_id
    if not d.is_dir():
        return None

    payload: Dict[str, Any] = {"id": run_id, "mtime": int(d.stat().st_mtime),
                               "files": {}, "summary": {}, "rows": [],
                               "ledger": [], "e2e": None, "tasks": None}

    bn = _find_artifact(d, "bottlenecks.json")
    if bn is not None:
        data = _read_json(bn) or {}
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

    return payload


def _rows_by_key(run: Dict[str, Any]) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    for r in run.get("rows") or []:
        ms = r.get("ms")
        if r.get("op") and ms:
            out[(r["op"], r.get("shape"))] = float(ms)
    return out


def diff_runs(root: Path, cur_id: str, prev_id: str) -> Dict[str, Any]:
    cur, prev = load_run(root, cur_id), load_run(root, prev_id)
    if not cur or not prev:
        return {"error": "run not found", "rows": []}
    a, b = _rows_by_key(cur), _rows_by_key(prev)
    rows = []
    for key, ms in a.items():
        old = b.get(key)
        if not old:
            continue
        rows.append({"op": key[0], "shape": key[1],
                     "cur_ms": round(ms, 6), "prev_ms": round(old, 6),
                     "delta_pct": round((ms / old - 1.0) * 100.0, 2)})
    rows.sort(key=lambda x: x["delta_pct"])
    return {"cur": cur_id, "prev": prev_id, "rows": rows}
