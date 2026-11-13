"""optframe 交付包：把 KEEP 的 kernel 注册成可按形状命中的实现。

只做三件事：
    registry.match(M, K, N) -> callable | None
    registry.configure(enabled)       # 运行期开关
    registry.stats()                  # 命中计数，供 A/B 核对
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_HIT_LOG = Path(os.environ.get("OPTFRAME_HIT_LOG", "/tmp/optframe_hits.log"))
_enabled = os.environ.get("OPTFRAME_BACKFILL", "0") == "1"
#: 运行期开关：存在该文件且内容为 1 则启用（每秒最多读一次，开销可忽略）
_FLAG = Path(os.environ.get("OPTFRAME_FLAG", "/tmp/optframe_backfill.flag"))
_flag_cache = {"t": 0.0, "v": None}


def _enabled_now() -> bool:
    now = time.monotonic()
    if _flag_cache["v"] is not None and now - _flag_cache["t"] < 1.0:
        return bool(_flag_cache["v"])
    try:
        value = _FLAG.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        value = _enabled
    _flag_cache["t"] = now
    _flag_cache["v"] = value
    return bool(value)
_registry: Dict[Tuple[int, int], Tuple[int, Callable]] = {}
_hits: Dict[str, int] = {}


def register(k: int, n: int, fn: Callable, name: str = "", min_m: int = 64) -> None:
    """按 K/N 注册，M 只要 >= min_m 就适用（kernel 本身对 M/N/K 都有掩码）。"""
    _registry[(k, n)] = (min_m, fn)
    _hits.setdefault(name or f"K{k}N{n}", 0)


def match(m: int, k: int, n: int) -> Optional[Callable]:
    if not _enabled_now():
        return None
    entry = _registry.get((k, n))
    if entry is None:
        return None
    min_m, fn = entry
    return fn if m >= min_m else None


def note_hit(key: str) -> None:
    _hits[key] = _hits.get(key, 0) + 1
    try:
        with _HIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(key + "\n")
    except OSError:
        pass


def configure(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)


def stats() -> Dict[str, Any]:
    return {"enabled": _enabled,
            "registered": [{"k": k, "n": n, "min_m": v[0]} for (k, n), v in _registry.items()],
            "hits": dict(_hits)}


# ---- 具体 kernel 的注册（导入即注册，但只有开关打开才生效）----
try:  # pragma: no cover - 交付期代码
    from .kernel_mm_5104x1024_7168 import run as _mm_run

    # 该 kernel 的 tile 是在 M≈5104（prefill）上定的；decode 小 batch 不启用
    register(1024, 7168, _mm_run, "mm_5104x1024x7168", min_m=512)
    register(1024, 5120, _mm_run, "mm_5104x1024x5120", min_m=512)
except Exception:  # noqa: BLE001
    pass
