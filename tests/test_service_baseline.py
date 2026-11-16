"""服务同款 SwiGLU 基线模块的测试（需要 CUDA）。"""

import pytest
import torch

from tools import swiglu_service_baseline as base


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _inputs(m=256, k=512, half=768, dtype=torch.bfloat16):
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(k, 2 * half, device="cuda", dtype=dtype)
    return x, w


def test_output_shape_and_values():
    base.clear_cache()
    x, w = _inputs()
    out = base.run(x, w)
    assert tuple(out.shape) == (256, 768)

    full = torch.mm(x, w)
    expect = torch.nn.functional.silu(full[:, :768]) * full[:, 768:]
    assert torch.allclose(out.float(), expect.float(), atol=0.5, rtol=2e-2)


def test_buffers_are_reused():
    base.clear_cache()
    x, w = _inputs()
    a = base.run(x, w)
    b = base.run(x, w)
    assert a.data_ptr() == b.data_ptr(), "预分配缓冲应被复用"


def test_different_shape_gets_distinct_buffer():
    base.clear_cache()
    x1, w1 = _inputs(m=128)
    x2, w2 = _inputs(m=256)
    o1 = base.run(x1, w1)
    o2 = base.run(x2, w2)
    assert o1.data_ptr() != o2.data_ptr()
    assert tuple(o1.shape) == (128, 768)
    assert tuple(o2.shape) == (256, 768)
