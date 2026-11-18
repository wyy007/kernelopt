"""MoE 路由探针：统计每个 expert 的 token 数（只观测，不改数值）。

目的（35B MoE 方向第 2 步）：
    判断“按 expert 分组 token”是否有增量。若一次窗口内平均 token/expert
    明显大于 1，说明同一 expert 的权重被重复读取，分组后可以把
    805MB 的重复读取压到“每个活跃 expert 一次”。

实现：
    包装 `sglang.srt.layers.moe.topk.select_experts`（路由的唯一出口，
    与后端无关），每 N 次调用采样一次返回的 topk_ids，累计 expert 直方图，
    每 M 次调用原子写 JSON。

开启：
    PYTHONPATH=<delivery 目录> OPTFRAME_ROUTE_PROBE=1 python -m sglang.launch_server ...
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

_LOG = Path(os.environ.get("OPTFRAME_ROUTE_LOG", "/tmp/optframe_route.json"))
_SAMPLE_EVERY = int(os.environ.get("OPTFRAME_ROUTE_SAMPLE", "20"))
_DUMP_EVERY = int(os.environ.get("OPTFRAME_ROUTE_DUMP_EVERY", "50"))
_EXPERTS = int(os.environ.get("OPTFRAME_ROUTE_EXPERTS", "256"))

_installed = False
_lock = threading.Lock()

_state: Dict[str, Any] = {
    "calls": 0,
    "sampled": 0,
    "skipped": 0,
    "last_out_type": "",
    "last_error": "",
    "tokens": 0,
    "m_hist": {},
    "expert_hist": [0] * _EXPERTS,
    "active_per_sample": [],
    "top1_per_sample": [],
    "samples": [],
}


def reset_state() -> None:
    """清空统计（供单测/重新采样用）。"""
    with _lock:
        _state.update(
            {
                "calls": 0,
                "sampled": 0,
                "skipped": 0,
                "last_out_type": "",
                "last_error": "",
                "tokens": 0,
                "m_hist": {},
                "expert_hist": [0] * _EXPERTS,
                "active_per_sample": [],
                "top1_per_sample": [],
                "samples": [],
            }
        )


def _dump_locked() -> None:
    active = _state["active_per_sample"]
    summary = {
        "calls": _state["calls"],
        "sampled": _state["sampled"],
        "skipped": _state["skipped"],
        "last_out_type": _state["last_out_type"],
        "last_error": _state["last_error"],
        "tokens": _state["tokens"],
        "m_hist": dict(_state["m_hist"]),
        "expert_hist": list(_state["expert_hist"]),
        "active_per_sample": list(active),
        "top1_per_sample": list(_state["top1_per_sample"]),
        "samples": list(_state["samples"][-5000:]),
    }
    if active:
        summary["active_mean"] = round(sum(active) / len(active), 1)
        summary["active_max"] = max(active)
        summary["tokens_per_active_expert"] = round(
            (_state["tokens"] / max(1, _state["sampled"])) / (sum(active) / len(active)), 2
        )
    try:
        tmp = _LOG.with_name(_LOG.name + ".tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _LOG)
    except OSError:
        pass


def _sample(ids) -> None:
    try:
        cpu = ids.detach().to("cpu").reshape(-1).long()
        vals = cpu.tolist()
        counts = [0] * _EXPERTS
        for v in vals:
            if 0 <= v < _EXPERTS:
                counts[v] += 1
        active = sum(1 for c in counts if c > 0)
        top1 = max(counts) if counts else 0
        m = int(ids.shape[0]) if ids.dim() >= 1 else 0
        with _lock:
            hist = _state["expert_hist"]
            for i, c in enumerate(counts):
                hist[i] += c
            _state["sampled"] += 1
            _state["tokens"] += len(vals)
            _state["active_per_sample"].append(active)
            _state["top1_per_sample"].append(top1)
            _state["samples"].append([m, active, top1, len(vals)])
            mh = _state["m_hist"]
            mh[str(m)] = mh.get(str(m), 0) + 1
            _dump_locked()
    except Exception as exc:  # noqa: BLE001 - 探针失败必须放行
        with _lock:
            _state["last_error"] = f"{type(exc).__name__}: {exc}"
        _dump_locked()


def install() -> bool:
    global _installed
    if _installed:
        return True
    try:
        from sglang.srt.layers.moe import topk as topk_mod
    except Exception:  # noqa: BLE001 - 非 SGLang 环境直接放弃
        return False

    orig = topk_mod.select_experts
    if getattr(orig, "_optframe_route", False):
        _installed = True
        return True

    def wrapper(*args, **kwargs):
        with _lock:
            _state["calls"] += 1
            n = _state["calls"]
        out = orig(*args, **kwargs)
        try:
            import torch as _torch

            ids = getattr(out, "topk_ids", None)
            if ids is None and isinstance(out, (tuple, list)) and len(out) >= 2:
                ids = out[1]
            if n % _SAMPLE_EVERY == 0 or n % _DUMP_EVERY == 0:
                with _lock:
                    _state["last_out_type"] = type(out).__name__
            if isinstance(ids, _torch.Tensor):
                if n % _SAMPLE_EVERY == 0:
                    _sample(ids)
                elif n % _DUMP_EVERY == 0:
                    with _lock:
                        _dump_locked()
            else:
                with _lock:
                    _state["skipped"] += 1
        except Exception:  # noqa: BLE001
            pass
        return out

    wrapper._optframe_route = True
    topk_mod.select_experts = wrapper
    _installed = True
    print("[optframe] route probe installed (select_experts)", flush=True)
    return True
