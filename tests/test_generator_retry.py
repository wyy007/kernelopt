"""生成器的截断检测与同轮重试（不联网，全部用假 _chat）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from optframe.generators.base import GenContext
from optframe.generators.kernelgen_local import KernelgenLocal

CODE = '''```python
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x, mask=mask)


def run(*args):
    x = args[0]
    o = torch.empty_like(x)
    _k[(triton.cdiv(x.numel(), 1024),)](x, o, x.numel(), BLOCK=1024)
    return o
```'''


def make_task():
    return SimpleNamespace(
        op="aten::mm",
        arch="sm_120",
        dtype_name="bf16",
        shapes=[[8, 8], [8, 8]],
        name="fake__task",
    )


def make_gen(**kw):
    return KernelgenLocal(base_url="http://127.0.0.1:1", model="fake", api_key="k", **kw)


def usage(p=10, c=20):
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def test_truncated_response_retries_with_bigger_budget():
    gen = make_gen(max_tokens=1000, max_retries=1, truncation_boost=2.0)
    limits = []
    seq = [
        ("truncated...", usage(), "length"),
        (CODE, usage(), "stop"),
    ]

    def fake_chat(prompt, temperature, max_tokens=None):
        limits.append(max_tokens)
        return seq[len(limits) - 1]

    gen._chat = fake_chat  # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)

    assert len(cands) == 1
    c = cands[0]
    assert c.generator == "kernelgen_local"
    assert c.meta["attempts"] == 2
    assert limits[0] == 1000 and limits[1] == 2000      # 预算翻倍
    assert c.meta["usage"]["total_tokens"] == 60        # 两次 usage 累加


def test_no_code_retries_once_then_recovers():
    gen = make_gen(max_retries=1)
    seq = [("我没有代码", usage(), "stop"), (CODE, usage(), "stop")]
    calls = {"n": 0}

    def fake_chat(prompt, temperature, max_tokens=None):
        i = calls["n"]
        calls["n"] += 1
        return seq[i]

    gen._chat = fake_chat  # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)
    assert cands[0].source.startswith("import torch")
    assert cands[0].meta["attempts"] == 2


def test_persistent_no_code_is_marked_with_attempts():
    gen = make_gen(max_retries=1)

    def fake_chat(prompt, temperature, max_tokens=None):
        return ("只有解释，没有代码块", usage(), "stop")

    gen._chat = fake_chat  # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)
    c = cands[0]
    assert c.source == ""
    assert c.generator == "kernelgen_local:no_code"
    assert c.meta["attempts"] == 2


def test_network_error_then_success():
    gen = make_gen(max_retries=1)
    seq = [RuntimeError("boom"), (CODE, usage(), "stop")]
    calls = {"n": 0}

    def fake_chat(prompt, temperature, max_tokens=None):
        i = calls["n"]
        calls["n"] += 1
        if i == 0:
            raise __import__("urllib.error").error.URLError("temporary")
        return seq[1]

    gen._chat = fake_chat  # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)
    assert cands[0].source.startswith("import torch")
    assert cands[0].meta["attempts"] == 2


def test_zero_retries_keeps_single_attempt():
    gen = make_gen(max_retries=0)

    def fake_chat(prompt, temperature, max_tokens=None):
        return ("truncated", usage(), "length")

    gen._chat = fake_chat  # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)
    assert cands[0].meta["attempts"] == 1
    assert cands[0].generator == "kernelgen_local:no_code"


def test_validator_rejection_triggers_retry_with_witness():
    """lint 拒绝后，同轮带见证重试一次。"""
    gen = make_gen(max_retries=1)
    calls = {"n": 0, "prompts": []}

    bad = CODE.replace("mask = offs < n", "mask = offs < n + 1")  # 仅用于制造差异

    def fake_chat(prompt, temperature, max_tokens=None):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return (bad, usage(), "stop") if calls["n"] == 1 else (CODE, usage(), "stop")

    def validator(code):
        return "mask 覆盖不完整" if "offs < n + 1" in code else None

    gen._chat = fake_chat          # type: ignore[assignment]
    gen.validator = validator      # type: ignore[assignment]
    cands = gen.propose(make_task(), GenContext(), n=1)

    assert cands[0].meta["attempts"] == 2
    assert cands[0].meta["lint_rejected"] == 1
    assert "mask 覆盖不完整" in calls["prompts"][1]
    assert "offs < n + 1" not in cands[0].source
