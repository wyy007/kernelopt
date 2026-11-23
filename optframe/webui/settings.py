"""控制台的持久化配置（目前是驱动模型）。

存放：runs/_settings.json（不进 run 列表）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

REPO = Path(os.environ.get("OPTFRAME_REPO", "/model/wyy/optframe")).resolve()
SETTINGS_PATH = REPO / "runs" / "_settings.json"

DEFAULTS: Dict[str, Any] = {
    "llm_enabled": True,
    "llm_model": "ds-v4",
    "llm_base_url": "http://10.51.198.60:10152/v1-openai",
    "llm_n": 1,
    "llm_rounds": 1,
    "llm_repeat": 20,
}


def load_settings(path: Path | None = None) -> Dict[str, Any]:
    p = path or SETTINGS_PATH
    data = dict(DEFAULTS)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return data
    if isinstance(raw, dict):
        for key in DEFAULTS:
            if key in raw:
                data[key] = raw[key]
    return data


def save_settings(payload: Dict[str, Any], path: Path | None = None) -> Dict[str, Any]:
    current = load_settings(path)
    for key in DEFAULTS:
        if key in payload:
            current[key] = payload[key]
    current["llm_enabled"] = bool(current["llm_enabled"])
    current["llm_model"] = str(current["llm_model"] or "ds-v4")
    current["llm_base_url"] = str(current["llm_base_url"] or DEFAULTS["llm_base_url"])
    for key in ("llm_n", "llm_rounds", "llm_repeat"):
        current[key] = max(1, int(current[key] or 1))
    p = path or SETTINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current
