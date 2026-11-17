"""optframe 交付包：把 KEEP 的 kernel 注册成可按形状/模式命中的实现。

两张注册表（见 docs/modes.md §5.3）：
    单算子 registry  : register(K, N, fn, min_m)      → match(M, K, N)
    融合   fused     : register_fused(pattern, K, N, fn, min_m, max_m)
                                                       → match_fused(M, K, N)
    dispatch(M, K, N) 统一调度：**先融合、后单算子**，命中即返回，避免重复计入。

其余能力不变：
    configure(enabled)  # 运行期开关（flag 文件）
    stats()             # 两张表的注册项与命中计数
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
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


# --------------------------------------------------------------------------- #
# 单算子表
# --------------------------------------------------------------------------- #

_registry: Dict[Tuple[int, int], Tuple[int, Callable]] = {}
_hits: Dict[str, int] = {}


def register(k: int, n: int, fn: Callable, name: str = "", min_m: int = 64) -> None:
    """按 K/N 注册单算子 kernel，M >= min_m 时适用。"""
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


# --------------------------------------------------------------------------- #
# 融合表
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FusedEntry:
    pattern: str
    k: int
    n: int
    fn: Callable
    name: str
    min_m: int
    max_m: Optional[int]

    def applies(self, m: int) -> bool:
        if m < self.min_m:
            return False
        if self.max_m is not None and m > self.max_m:
            return False
        return True


_fused: Dict[Tuple[str, int, int], FusedEntry] = {}


def register_fused(
    pattern: str,
    k: int,
    n: int,
    fn: Callable,
    name: str = "",
    min_m: int = 512,
    max_m: Optional[int] = None,
) -> None:
    """注册融合 kernel。

    形状白名单由 (pattern, K, N) 三个键固定；M 由 min_m/max_m 限定区间——
    decode 小 batch 必须被 min_m 挡住（实测 M=16 时融合为 0.70x）。
    """
    _fused[(pattern, k, n)] = FusedEntry(
        pattern=pattern, k=k, n=n, fn=fn,
        name=name or f"fused:{pattern}:K{k}N{n}",
        min_m=int(min_m), max_m=None if max_m is None else int(max_m),
    )
    entry = _fused[(pattern, k, n)]
    _hits.setdefault(entry.name, 0)


def match_fused(m: int, k: int, n: int, pattern: Optional[str] = None) -> Optional[Callable]:
    """命中融合 kernel；pattern 为空时按注册顺序找第一个匹配项。"""
    if not _enabled_now():
        return None
    if pattern is not None:
        entry = _fused.get((pattern, k, n))
        return entry.fn if entry and entry.applies(m) else None
    for entry in _fused.values():
        if entry.k == k and entry.n == n and entry.applies(m):
            return entry.fn
    return None


# --------------------------------------------------------------------------- #
# 统一调度
# --------------------------------------------------------------------------- #

def dispatch(m: int, k: int, n: int,
             pattern: Optional[str] = None) -> Optional[Tuple[str, str, Callable]]:
    """返回 (mode, name, callable)：先融合、后单算子；都不命中返回 None。

    mode ∈ {"fused", "single"}，调用方用它写命中日志，保证收益归因不重复。
    """
    if not _enabled_now():
        return None

    if pattern is not None:
        entry = _fused.get((pattern, k, n))
        if entry is not None and entry.applies(m):
            return ("fused", entry.name, entry.fn)
    else:
        for entry in _fused.values():
            if entry.k == k and entry.n == n and entry.applies(m):
                return ("fused", entry.name, entry.fn)

    single = _registry.get((k, n))
    if single is not None and m >= single[0]:
        return ("single", f"K{k}N{n}", single[1])
    return None


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
    _flag_cache["v"] = None


def reset() -> None:
    """清空两张表与命中计数（测试用）。"""
    _registry.clear()
    _fused.clear()
    _hits.clear()


def stats() -> Dict[str, Any]:
    return {
        "enabled": _enabled,
        "registered": [{"k": k, "n": n, "min_m": v[0]} for (k, n), v in _registry.items()],
        "fused": [
            {"pattern": e.pattern, "k": e.k, "n": e.n,
             "min_m": e.min_m, "max_m": e.max_m, "name": e.name}
            for e in _fused.values()
        ],
        "hits": dict(_hits),
    }


# ---- 具体 kernel 的注册（导入即注册，但只有开关打开才生效）----
try:  # pragma: no cover - 交付期代码
    from .kernel_mm_5104x1024_7168 import run as _mm_run

    # 该 kernel 的 tile 是在 M≈5104（prefill）上定的；decode 小 batch 不启用
    register(1024, 7168, _mm_run, "mm_5104x1024x7168", min_m=512)
    register(1024, 5120, _mm_run, "mm_5104x1024x5120", min_m=512)
except Exception:  # noqa: BLE001
    pass

try:  # pragma: no cover - 交付期代码
    from .kernel_swiglu_4096x24576 import run as _swiglu_run

    # 融合核：M=8192 上 ratio 1.1016；decode 小 batch 必须挡掉
    register_fused("swiglu", 4096, 24576, _swiglu_run,
                   "fused:swiglu:K4096N24576", min_m=512)
except Exception:  # noqa: BLE001
    pass
