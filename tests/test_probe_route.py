"""MoE 路由探针：纯逻辑单测（不需要 sglang/GPU）。"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delivery"))

from optframe_delivery import probe_route as probe


def setup_function(_):
    probe.reset_state()


def test_sample_counts_experts(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_LOG", tmp_path / "route.json")
    ids = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int32)
    probe._sample(ids)
    st = probe._state
    assert st["sampled"] == 1
    assert st["tokens"] == 6
    assert st["expert_hist"][0] == 2
    assert st["expert_hist"][1] == 2
    assert st["expert_hist"][2] == 2
    assert st["active_per_sample"] == [3]
    data = json.loads((tmp_path / "route.json").read_text())
    assert data["sampled"] == 1
    assert data["samples"] == [[2, 3, 2, 6]]


def test_sample_ignores_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_LOG", tmp_path / "route.json")
    ids = torch.tensor([[0, 999]], dtype=torch.int32)
    probe._sample(ids)
    assert probe._state["expert_hist"][0] == 1
    assert probe._state["tokens"] == 2


def test_install_without_sglang_is_safe():
    assert probe.install() in (True, False)
