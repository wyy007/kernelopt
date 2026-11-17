"""SiluAndMul 探针：只观测、不改数值。

目的（第 5 步前置）：
    融合回灌需要在 `SiluAndMul` 处知道"这个激活的输入来自哪个 GEMM（M, K, N）"。
    本探针先回答"能不能拿到"，不替换任何实现。

做法：
    1. 包装 `torch.mm` / `F.linear`，把最近一次 GEMM 的 (M, K, N) 记到线程本地；
    2. 包装 `SiluAndMul.forward`，检查紧邻的 GEMM 是否与输入形状匹配（M 相同、N == 2H）；
    3. 每次命中写一行 JSON 日志；统计 counts 可通过 stats() 读取。
    4. 原样返回 `orig(self, ...)` 的结果——数值路径完全不变。

开启：
    PYTHONPATH=<delivery 目录> OPTFRAME_PROBE=1 python -m sglang.launch_server ...
    日志默认 /tmp/optframe_probe.log，可用 OPTFRAME_PROBE_LOG 覆盖。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_LOG = Path(os.environ.get("OPTFRAME_PROBE_LOG", "/tmp/optframe_probe.log"))
#: 认为"紧邻"的时间窗（毫秒）
_WINDOW_MS = float(os.environ.get("OPTFRAME_PROBE_WINDOW_MS", "5"))

_local = threading.local()
_installed = False
_counts: Dict[str, int] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 记录与配对（纯逻辑，可单测）
# --------------------------------------------------------------------------- #

def record_gemm(m: int, k: int, n: int) -> None:
    _local.last_gemm = (int(m), int(k), int(n), time.monotonic() * 1000.0)


def last_gemm() -> Optional[Tuple[int, int, int, float]]:
    return getattr(_local, "last_gemm", None)


def pair_with_last(m: int, two_h: int) -> Optional[Tuple[int, int, int]]:
    """若紧邻的 GEMM 与激活形状匹配，返回 (M, K, N)，否则 None。"""
    g = last_gemm()
    if g is None:
        return None
    gm, gk, gn, ts = g
    if (time.monotonic() * 1000.0) - ts > _WINDOW_MS:
        return None
    if gm != m or gn != two_h:
        return None
    return gm, gk, gn


def _bump(key: str) -> None:
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1


def _log(record: Dict[str, Any]) -> None:
    try:
        with _LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 安装
# --------------------------------------------------------------------------- #

def install() -> bool:
    """安装探针；返回是否成功。重复调用安全。"""
    global _installed
    if _installed:
        return True

    import torch
    import torch.nn.functional as F

    try:
        from sglang.srt.layers.activation import SiluAndMul
    except Exception:  # noqa: BLE001 - 非 SGLang 环境直接放弃
        return False

    orig_mm = torch.mm
    orig_linear = F.linear

    def mm(a, b, *args, **kwargs):
        try:
            if a.dim() == 2 and b.dim() == 2:
                record_gemm(a.shape[0], a.shape[1], b.shape[1])
        except Exception:  # noqa: BLE001
            pass
        return orig_mm(a, b, *args, **kwargs)

    def linear(inp, weight, bias=None):
        try:
            if inp.dim() >= 2 and weight.dim() == 2:
                record_gemm(inp.shape[0], inp.shape[-1], weight.shape[0])
        except Exception:  # noqa: BLE001
            pass
        return orig_linear(inp, weight, bias)

    orig_forward = SiluAndMul.forward

    def forward(self, *a, backend=None, **kw):
        out = orig_forward(self, *a, backend=backend, **kw)
        try:
            x = a[0] if a else None
            if isinstance(x, torch.Tensor) and x.dim() == 2:
                m, two_h = int(x.shape[0]), int(x.shape[1])
                hit = pair_with_last(m, two_h)
                rec = {
                    "ts": round(time.time(), 3),
                    "m": m,
                    "two_h": two_h,
                    "gemm": None if hit is None else {"m": hit[0], "k": hit[1], "n": hit[2]},
                    "paired": hit is not None,
                }
                _bump("silu_calls")
                _bump("paired" if hit else "unpaired")
                if hit is None:
                    g = last_gemm()
                    if g is not None:
                        rec["last_gemm"] = {"m": g[0], "k": g[1], "n": g[2],
                                            "age_ms": round(time.monotonic() * 1000 - g[3], 2)}
                _log(rec)
        except Exception:  # noqa: BLE001 - 探针绝不干扰前向
            pass
        return out

    torch.mm = mm
    F.linear = linear
    SiluAndMul.forward = forward
    _installed = True
    return True


def stats() -> Dict[str, Any]:
    with _lock:
        return dict(_counts)


def reset_stats() -> None:
    with _lock:
        _counts.clear()
