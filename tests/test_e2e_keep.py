"""端到端直跑：KEEP 选择与 overlay 注入包（不需要 GPU）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optframe.webui import e2e_keep


def _task_dir(root: Path) -> Path:
    d = root / "task"
    d.mkdir(parents=True)
    (d / "task.toml").write_text(
        '[task]\nname = "fused__swiglu__3508x4096x24576"\nop = "fused::swiglu"\n'
        "shapes = [[3508, 4096], [4096, 24576]]\n",
        encoding="utf-8",
    )
    return d


def test_task_shape_from_toml(tmp_path):
    d = _task_dir(tmp_path)
    pattern, k, n, min_m = e2e_keep._task_shape(d)
    assert pattern == "swiglu"
    assert (k, n, min_m) == (4096, 24576, 512)


def test_build_overlay_registers_candidate(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    delivery = repo / "delivery"
    (delivery / "optframe_delivery").mkdir(parents=True)
    (delivery / "optframe_delivery" / "__init__.py").write_text("", encoding="utf-8")
    (delivery / "sitecustomize.py").write_text("# base\n", encoding="utf-8")
    monkeypatch.setattr(e2e_keep, "REPO", repo)

    task_dir = _task_dir(tmp_path)
    candidate = tmp_path / "cand_abc.py"
    candidate.write_text("def run(x, w):\n    return x\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    overlay, reg = e2e_keep.build_overlay(
        run_dir, {"name": "fused__swiglu__3508x4096x24576", "dir": str(task_dir)}, candidate
    )
    assert (overlay / "optframe_delivery" / "extra_kernel.py").read_text() == candidate.read_text()
    site = (overlay / "sitecustomize.py").read_text()
    assert "register_fused('swiglu', 4096, 24576" in site
    assert reg == {"pattern": "swiglu", "k": 4096, "n": 24576, "min_m": 512}


def test_candidate_source_falls_back_to_eval_dir(tmp_path):
    workdir = tmp_path / "eval"
    workdir.mkdir()
    (workdir / "cand_deadbeef.py").write_text("x = 1\n", encoding="utf-8")
    task = {"eval": {"verdict": "KEEP", "workdir": str(workdir)}}
    assert e2e_keep._candidate_source(task).name == "cand_deadbeef.py"
