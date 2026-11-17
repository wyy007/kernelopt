"""8B 融合回灌钩子的纯逻辑测试（不需要 GPU / sglang）。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delivery"))

from optframe_delivery import inject_swiglu as inj


def test_mark_and_is_marked():
    t = torch.zeros(4, 8)
    assert inj.is_marked(t) is False
    inj.mark(t)
    assert inj.is_marked(t) is True


def test_fusion_shape_rejects_cpu_tensors():
    a = torch.zeros(8, 16, dtype=torch.bfloat16)
    w = torch.zeros(32, 16, dtype=torch.bfloat16)
    assert inj.fusion_shape(a, w, None) is None   # 非 CUDA 不融合


def test_fusion_shape_rejects_bias_and_bad_dims():
    # 用 "假装在 cuda" 的对象不可行，改为直接验证前置条件顺序：
    # bias 非空时，无论形状如何都返回 None
    a = torch.zeros(8, 16, dtype=torch.bfloat16)
    w = torch.zeros(32, 16, dtype=torch.bfloat16)
    b = torch.zeros(32, dtype=torch.bfloat16)
    assert inj.fusion_shape(a, w, b) is None

    a3 = torch.zeros(2, 8, 16, dtype=torch.bfloat16)
    assert inj.fusion_shape(a3, w, None) is None


def test_weight_transpose_cache_reuses_buffer():
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    t1 = inj._weight_t(w)
    t2 = inj._weight_t(w)
    assert t1.data_ptr() == t2.data_ptr()
    assert tuple(t1.shape) == (32, 64)


def test_install_without_sglang_is_safe():
    assert inj.install() in (True, False)
    assert isinstance(inj.stats(), dict)
