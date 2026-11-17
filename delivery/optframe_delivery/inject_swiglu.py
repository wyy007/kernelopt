"""8B（dense）SwiGLU 融合回灌。

注入点与 5W1H：
    在 `F.linear` 处拦截 gate_up 投影（weight 形状 (2H, K)，无 bias），
    直接调用融合 kernel 得到 `silu(gate)*up`（形状 (M, H)），并给结果打标记；
    随后的 `SiluAndMul.forward` 看到标记就原样返回（激活已在融合核里算过）。

为什么不在 SiluAndMul 处拦：
    那时 GEMM 已经算完，输入张量已丢，只能拿到输出（M, 2H）——无法再做融合。
    探针实测（2026-11-17）也证实：8B 上 99.9% 的 SiluAndMul 前一跳是 F.linear。

安全边界（任一不满足即走原实现）：
    - OPTFRAME_BACKFILL=1 且 flag 文件为 1（复用 registry 的开关）
    - inp 为 2D、bf16、连续；weight 为 2D、bf16；bias is None
    - 形状命中融合表（pattern="swiglu"），且 M >= min_m
    - 任何异常一律回退原实现（try/except 包住整段）
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

#: 打在被融合过的张量上的标记属性名
MARK = "_optframe_swiglu_done"

_installed = False
_orig_linear = None
_orig_forward = None
_wt_cache: dict = {}
_counts = {"fused": 0, "passthrough": 0, "skipped": 0}


def mark(t: torch.Tensor) -> torch.Tensor:
    try:
        setattr(t, MARK, True)
    except Exception:  # noqa: BLE001 - 某些张量不允许设属性
        pass
    return t


def is_marked(t: Any) -> bool:
    return bool(getattr(t, MARK, False))


def fusion_shape(inp: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    """判断这次 linear 是否值得融合；返回 (M, K, 2H)。"""
    if bias is not None:
        return None
    if not (isinstance(inp, torch.Tensor) and isinstance(weight, torch.Tensor)):
        return None
    if inp.dim() != 2 or weight.dim() != 2:
        return None
    if inp.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    if not (inp.is_cuda and weight.is_cuda):
        return None
    if not inp.is_contiguous():
        return None
    m, k = int(inp.shape[0]), int(inp.shape[1])
    two_h, wk = int(weight.shape[0]), int(weight.shape[1])
    if wk != k or two_h % 2 != 0:
        return None
    return m, k, two_h


def _weight_t(weight: torch.Tensor) -> torch.Tensor:
    """(2H, K) -> (K, 2H) 连续布局，按 data_ptr+shape 缓存。"""
    key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
    cached = _wt_cache.get(key)
    if cached is None:
        cached = weight.t().contiguous()
        _wt_cache[key] = cached
    return cached


def install() -> bool:
    """安装回灌钩子；返回是否成功（非 SGLang 环境返回 False）。"""
    global _installed, _orig_linear, _orig_forward
    if _installed:
        return True

    import torch.nn.functional as F

    try:
        from sglang.srt.layers.activation import SiluAndMul
    except Exception:  # noqa: BLE001
        return False

    try:
        import optframe_delivery as registry
    except Exception:  # noqa: BLE001
        return False

    _orig_linear = F.linear
    _orig_forward = SiluAndMul.forward

    def linear(inp, weight, bias=None):
        try:
            if not registry._enabled_now():
                return _orig_linear(inp, weight, bias)
            shape = fusion_shape(inp, weight, bias)
            if shape is not None:
                m, k, two_h = shape
                fn = registry.match_fused(m, k, two_h, pattern="swiglu")
                if fn is not None:
                    wt = _weight_t(weight)
                    out = fn(inp, wt)
                    mark(out)
                    registry.note_hit(f"fused:swiglu:{m}x{k}x{two_h}")
                    _counts["fused"] += 1
                    return out
            _counts["passthrough"] += 1
        except Exception:  # noqa: BLE001 - 回灌失败必须落回原实现
            _counts["skipped"] += 1
        return _orig_linear(inp, weight, bias)

    def forward(self, *a, backend=None, **kw):
        x = a[0] if a else None
        if isinstance(x, torch.Tensor) and is_marked(x):
            # 融合核已经算过 silu(gate)*up，这里原样返回
            return x
        return _orig_forward(self, *a, backend=backend, **kw)

    F.linear = linear
    SiluAndMul.forward = forward
    _installed = True
    return True


def stats() -> dict:
    return dict(_counts)
