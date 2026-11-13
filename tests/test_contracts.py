"""共享合同层测试：源码守卫、账本、任务加载、代码抽取。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optframe.contracts.ledger import Ledger
from optframe.contracts.schema import Candidate, Result, Verdict
from optframe.contracts.source_guard import check_candidate, detect_disallowed_torch
from optframe.contracts.task import TaskError, load_task
from optframe.generators.parsing import extract_code

GOOD = '''
import torch, triton, triton.language as tl

@triton.jit
def _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    v = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + off, (v * tl.sigmoid(v)).to(tl.bfloat16), mask=mask)

def run(x):
    out = torch.empty_like(x)
    _k[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out
'''


def test_good_candidate_passes():
    assert check_candidate(GOOD) is None


def test_missing_run_is_rejected():
    code = GOOD.replace("def run(x):", "def launch(x):")
    assert "run" in (check_candidate(code) or "")


def test_missing_jit_is_rejected():
    code = '''
import torch
def run(x):
    return torch.empty_like(x)
'''
    assert "jit" in (check_candidate(code) or "")


def test_torch_compute_is_rejected():
    code = '''
import torch, triton, triton.language as tl

@triton.jit
def _k(x_ptr):
    pass

def run(x):
    return torch.mm(x, x)
'''
    err = check_candidate(code)
    assert err and "torch.mm" in err


def test_torch_allocation_is_allowed():
    code = '''
import torch, triton, triton.language as tl

@triton.jit
def _k(x_ptr):
    pass

def run(x):
    out = torch.empty_like(x)
    return out
'''
    assert detect_disallowed_torch(code) is None


def test_ledger_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl", run_id="r1")
    ledger.append(stage="candidate", verdict=Verdict.KEEP.value,
                  task="t", generator="g", candidate_sha="abc",
                  payload={"ratio": 1.2})
    ledger.append(stage="candidate", verdict=Verdict.REVERT.value,
                  task="t", generator="g", candidate_sha="def",
                  payload={"ratio": 0.9})

    records = ledger.read()
    assert len(records) == 2
    assert records[0]["run_id"] == "r1"
    assert [r["candidate_sha"] for r in ledger.iter_verdicts("KEEP")] == ["abc"]
    # 行必须独立成 JSON（追加写不破坏前文）
    for line in (tmp_path / "l.jsonl").read_text().splitlines():
        json.loads(line)


def test_load_task_from_generated_layout(tmp_path):
    task_dir = tmp_path / "silu__8x8"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
[task]
name = "silu__8x8"
op = "aten::silu"
arch = "sm89"
dtype = "bf16"
category = "elementwise"
target_ratio = 1.3
reference = "reference.py:run"
judge = "judge.py:run"
shapes = [[8, 8]]

[escalation]
allow_complex = true
""",
        encoding="utf-8",
    )
    (task_dir / "reference.py").write_text(
        """
import torch
def make_inputs(device="cpu", dtype=None, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(8, 8, device=device, dtype=dtype)]
def run(x):
    return torch.nn.functional.silu(x)
""",
        encoding="utf-8",
    )
    (task_dir / "judge.py").write_text(
        """
def run(ref, cand, dtype=None):
    return {"ok": True, "max_abs_err": 0.0}
""",
        encoding="utf-8",
    )

    task = load_task(task_dir)
    assert task.name == "silu__8x8"
    assert task.dtype_name == "bf16"
    assert task.shapes == [[8, 8]]
    assert task.allow_complex is True

    inputs = task.make_inputs(device="cpu")
    assert len(inputs) == 1 and inputs[0].shape == (8, 8)
    assert task.judge_check(None, None)["ok"] is True


def test_load_task_missing_toml(tmp_path):
    with pytest.raises(TaskError):
        load_task(tmp_path)


def test_extract_code_prefers_run_block():
    text = """说明文字
```python
import torch
def helper():
    pass
```
再来一个
```python
import torch
def run(x):
    return x
```
"""
    code = extract_code(text, prefer="run")
    assert "def run" in code


def test_extract_code_from_bare_response():
    code = extract_code("import torch\ndef run(x):\n    return x\n")
    assert code.startswith("import torch")


def test_extract_code_none_when_no_code():
    assert extract_code("这是一个纯文字回答。") is None


def test_result_serialization():
    r = Result(correct=True, device_ms=0.1, ref_ms=0.2, ratio=2.0)
    data = r.to_dict()
    assert data["ratio"] == 2.0
    assert data["device_ms"] == 0.1


def test_candidate_sha_is_stable():
    a = Candidate(source="print(1)")
    b = Candidate(source="print(1)")
    assert a.sha == b.sha and len(a.sha) == 40


def test_bootstrap_ci_brackets_median():
    from optframe.contracts.timing import bootstrap_median_ci

    lo, hi = bootstrap_median_ci([1.0] * 20 + [2.0] * 20, iters=300)
    assert lo <= 1.5 <= hi

    lo2, hi2 = bootstrap_median_ci([0.9, 1.0, 1.1] * 10, iters=300)
    assert lo2 <= 1.05 <= hi2
    assert hi2 - lo2 < 0.2


def test_candidate_import_error_is_caught(tmp_path):
    """候选在 import 阶段抛错时必须归为 import_error，不能掀掉进程。"""
    from optframe.contracts.evaluator import Baseline, evaluate_candidate
    from optframe.contracts.schema import Candidate

    class _Task:
        arch = "sm120"

        def judge_check(self, ref, cand):
            return {"ok": True}

    source = """
import triton
import triton.language as tl
import definitely_not_a_real_module_xyz  # noqa: F401

@triton.jit
def _k():
    pass

def run(x):
    return x
"""
    result = evaluate_candidate(
        _Task(), Candidate(source=source), Baseline(outputs=[], ms=1.0), [], tmp_path
    )
    assert result.status == "import_error"
    assert "definitely_not_a_real_module_xyz" in result.error


def test_search_variants_rewrite_literals():
    from optframe.tiers.search import build_variants, has_config_knobs, rewrite_config

    src = """
@triton.jit
def _k(x, n, BLOCK: tl.constexpr):
    pass

def run(x):
    BLOCK = 1024
    _k[(1,)](x, 1, BLOCK=BLOCK, num_warps=4)
"""
    assert has_config_knobs(src) == (True, True)
    assert "BLOCK = 2048" in rewrite_config(src, 2048, None)
    assert "num_warps=8" in rewrite_config(src, None, 8)

    variants = build_variants(src, blocks=[512, 1024], warps=[4])
    assert len(variants) == 2
    assert {v.block for v in variants} == {512, 1024}


def test_search_variants_without_knobs_is_single():
    from optframe.tiers.search import build_variants

    variants = build_variants("def run(x):\n    return x\n")
    assert len(variants) == 1
