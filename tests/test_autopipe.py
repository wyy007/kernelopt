"""自动流水线：单算子任务筛选（不需要 GPU）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optframe.webui import autopipe


def _task(root, name, op, pct, kind="single"):
    d = root / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(
        f'[task]\nname = "{name}"\nop = "{op}"\nkind = "{kind}"\n\n[trace]\npct_e2e = {pct}\n',
        encoding="utf-8",
    )
    return d


def test_select_single_tasks_topk_and_excludes_experts(tmp_path):
    _task(tmp_path, "mm_big", "aten::mm", 30.0)
    _task(tmp_path, "mm_small", "aten::mm", 5.0)
    _task(tmp_path, "moe", "fused_moe_kernel", 40.0)
    _task(tmp_path, "expert_gemm", "sglang::experts_gemm", 35.0)
    _task(tmp_path, "mul", "aten::mul", 10.0)

    kept, skipped = autopipe.select_single_tasks(tmp_path, top_k=2)
    assert [t["name"] for t in kept] == ["mm_big", "mul"]
    reasons = {t["name"]: t["reason"] for t in skipped}
    assert "expert" in reasons["moe"] or "moe" in reasons["moe"]
    assert reasons["mm_small"] == "超出 top-2"


def test_select_single_tasks_empty_dir(tmp_path):
    kept, skipped = autopipe.select_single_tasks(tmp_path, top_k=3)
    assert kept == [] and skipped == []
