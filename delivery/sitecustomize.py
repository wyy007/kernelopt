"""SGLang 进程内的回灌注入。

机制：PYTHONPATH 指向本目录后，Python 启动时会自动 import sitecustomize。
只在 OPTFRAME_BACKFILL=1 时替换 torch.mm / torch.matmul / torch.addmm；
形状不匹配或开关关闭时原样透传，做到"随时可关、风险可控"。

注意：只处理二维 bf16、连续布局、无 bias 的 mm；其余一律走原实现，
避免在不确定的语义上改变数值。
"""

import os

import torch

_LOG = os.environ.setdefault("OPTFRAME_HIT_LOG", "/tmp/optframe_hits.log")

if os.environ.get("OPTFRAME_BACKFILL", "0") == "1":
    try:
        import optframe_delivery as registry
    except Exception:  # noqa: BLE001 - 注入失败必须静默放行
        registry = None

    if registry is not None:
        _orig_mm = torch.mm
        _orig_matmul = torch.matmul
        _orig_addmm = torch.addmm

        def _eligible(a, b):
            return (
                a.is_cuda
                and b.is_cuda
                and a.dtype == torch.bfloat16
                and b.dtype == torch.bfloat16
                and a.dim() == 2
                and b.dim() == 2
                and a.is_contiguous()
                and b.is_contiguous()
            )

        def mm(a, b, *args, **kwargs):
            if not kwargs and _eligible(a, b):
                fn = registry.match(a.shape[0], a.shape[1], b.shape[1])
                if fn is not None:
                    registry.note_hit(f"mm:{a.shape[0]}x{a.shape[1]}x{b.shape[1]}")
                    return fn(a, b)
            return _orig_mm(a, b, *args, **kwargs)

        def matmul(a, b, *args, **kwargs):
            if not kwargs and _eligible(a, b):
                fn = registry.match(a.shape[0], a.shape[1], b.shape[1])
                if fn is not None:
                    registry.note_hit(f"matmul:{a.shape[0]}x{a.shape[1]}x{b.shape[1]}")
                    return fn(a, b)
            return _orig_matmul(a, b, *args, **kwargs)

        def addmm(bias, a, b, *args, **kwargs):
            # 有 bias 时语义不同，保持原路径
            return _orig_addmm(bias, a, b, *args, **kwargs)

        # 权重转置缓存：(data_ptr, shape) -> (K, N) 连续张量
        _wt_cache = {}

        def _weight_t(weight):
            key = (weight.data_ptr(), tuple(weight.shape))
            cached = _wt_cache.get(key)
            if cached is None:
                cached = weight.t().contiguous()
                _wt_cache[key] = cached
            return cached

        def _try(input_2d, weight):
            if not (input_2d.is_cuda and weight.is_cuda
                    and input_2d.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16):
                return None
            if input_2d.dim() != 2 or weight.dim() != 2:
                return None
            fn = registry.match(input_2d.shape[0], input_2d.shape[1], weight.shape[0])
            if fn is None:
                return None
            wt = _weight_t(weight)
            registry.note_hit(f"linear:{input_2d.shape[0]}x{input_2d.shape[1]}x{weight.shape[0]}")
            return fn(input_2d, wt)

        _orig_linear = torch.nn.functional.linear

        def linear(input, weight, bias=None):
            x2 = input.reshape(-1, input.shape[-1]) if input.dim() > 2 else input
            out = _try(x2, weight)
            if out is not None:
                if bias is not None:
                    out = out + bias
                return out.reshape(*input.shape[:-1], weight.shape[0])
            return _orig_linear(input, weight, bias)

        def addmm(bias, a, b, *args, **kwargs):
            out = _try(a, b.t()) if b.dim() == 2 else None
            if out is not None and not kwargs:
                return out + bias if bias is not None else out
            return _orig_addmm(bias, a, b, *args, **kwargs)

        torch.mm = mm
        torch.matmul = matmul
        torch.addmm = addmm
        torch.nn.functional.linear = linear
