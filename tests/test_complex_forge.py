"""复杂档适配器测试（prepare / run 隔离 / collect / CUDA 源码守卫）。

真正编译 CUDA 的集成测试不在 pytest 里跑（一次 60s+），由命令行在目标机执行。
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from optframe.contracts.cuda_loader import parse_declared_flags, validate_cuda_source
from optframe.tiers.complex_forge import (
    ForgeConfig,
    ForgeError,
    collect,
    prepare,
    run,
)


class FakeTask:
    def __init__(self, tmp: Path, shapes=None, dtype="bf16"):
        self.dir = tmp
        self.name = "silu__2048x2048"
        self.op = "aten::silu"
        self.arch = "sm89"
        self.dtype_name = dtype
        self.shapes = shapes or [[2048, 2048]]
        self.target_ratio = 1.3
        self.allow_complex = True

    def make_inputs(self, device="cuda", seed=0):
        return []


@pytest.fixture()
def task_dir(tmp_path):
    d = tmp_path / "task"
    d.mkdir()
    (d / "reference.py").write_text(
        "def make_inputs(device='cpu', dtype=None, seed=0):\n    return []\n"
        "def run(x):\n    return x\n",
        encoding="utf-8",
    )
    return d


def test_prepare_writes_forge_layout(tmp_path, task_dir):
    task = FakeTask(task_dir)
    prepared = prepare(task, tmp_path / "ws", ForgeConfig())

    assert prepared.task_dir.exists()
    data = tomllib.loads((prepared.task_dir / "task.toml").read_text())
    assert data["task"]["language"] == "cuda"
    assert data["task"]["arch"] == "sm89"
    assert data["reference"]["run"] == "reference/run.py:run"
    assert data["target"]["ratio"] == pytest.approx(1.3)
    assert data["budget"]["hours"] == pytest.approx(0.2)

    shapes = json.loads((prepared.task_dir / "shapes.json").read_text())
    assert shapes[0]["axes"] == {"d0": 2048, "d1": 2048}

    # reference/run.py 必须真的能委托到原实现
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fk_ref", prepared.task_dir / "reference" / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run("x") == "x"


def test_prepare_rejects_multi_input(tmp_path, task_dir):
    task = FakeTask(task_dir, shapes=[[2048, 2048], [2048]])
    with pytest.raises(ForgeError):
        prepare(task, tmp_path / "ws", ForgeConfig())


def _make_artifact(root: Path, ratios, kernel="// kernel\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps({"id": "run1", "status": "stopped", "target_met": True}), encoding="utf-8"
    )
    op_dir = root / "operators" / "silu"
    op_dir.mkdir(parents=True)
    (op_dir / "kernel.cu").write_text(kernel, encoding="utf-8")
    lines = [json.dumps({"median_ratio": r, "sha": f"s{i}"}) for i, r in enumerate(ratios)]
    (op_dir / "paired_verdict.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_collect_picks_best_ratio(tmp_path):
    art = _make_artifact(tmp_path / "art", [1.01, 1.05, 0.99])
    cand = collect(art)
    assert cand.language == "cuda"
    assert cand.meta["forge_ratio"] == pytest.approx(1.05)
    assert cand.meta["target_met"] is True
    assert "// kernel" in cand.source


def test_collect_falls_back_to_ws_kernel(tmp_path):
    art = tmp_path / "art2"
    (art / "ws").mkdir(parents=True)
    (art / "ws" / "kernel.cu").write_text("// ws kernel\n", encoding="utf-8")
    cand = collect(art)
    assert "ws kernel" in cand.source
    assert cand.meta["forge_ratio"] is None


def test_collect_skips_scaffold_and_prefers_ws(tmp_path):
    art = _make_artifact(tmp_path / "art3", [1.02])
    # 把 operators 里的 kernel 换成脚手架（ForgeKernel 的初始状态）
    (art / "operators" / "silu" / "kernel.cu").write_text(
        'torch::Tensor run(torch::Tensor x) { TORCH_CHECK(false, "write the kernel"); }',
        encoding="utf-8",
    )
    (art / "ws").mkdir()
    (art / "ws" / "kernel.cu").write_text("// real optimized kernel\n", encoding="utf-8")

    cand = collect(art)
    assert "real optimized kernel" in cand.source
    assert cand.meta["forge_ratio"] == pytest.approx(1.02)


def test_collect_errors_without_kernel(tmp_path):
    art = tmp_path / "empty"
    art.mkdir()
    with pytest.raises(ForgeError):
        collect(art)


def test_run_uses_subprocess_and_returns_rc(tmp_path, task_dir):
    task = FakeTask(task_dir)
    prepared = prepare(task, tmp_path / "ws", ForgeConfig(budget_hours=0.001))
    result = run(prepared, ForgeConfig(budget_hours=0.001, create_run=False),
                 stub_cmd=["/bin/sh", "-c", "exit 2"])
    assert result.returncode == 2
    assert result.ok is True   # 2 = 未达目标但 run 正常结束


def test_run_timeout_is_isolated(tmp_path, task_dir):
    task = FakeTask(task_dir)
    prepared = prepare(task, tmp_path / "ws", ForgeConfig())
    config = ForgeConfig(budget_hours=0.0002, timeout_margin_s=0, create_run=False)
    result = run(prepared, config, stub_cmd=["/bin/sh", "-c", "sleep 5"])
    assert result.ok is False
    assert result.returncode == -9
    assert "timed out" in result.log_tail


def test_cuda_source_guard():
    good = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def("run", &run);\n}'
    assert validate_cuda_source(good) is None
    assert "PYBIND11_MODULE" in (validate_cuda_source("int main(){}") or "")
    assert "m.def" in (validate_cuda_source('PYBIND11_MODULE(X, m){ m.def("launch", &f); }') or "")


def test_parse_declared_flags():
    src = "// fk-cuda-cflags: -gencode=arch=compute_89,code=sm_89\nint x;"
    assert parse_declared_flags(src) == ["-gencode=arch=compute_89,code=sm_89"]
    assert parse_declared_flags("int x;") == []
