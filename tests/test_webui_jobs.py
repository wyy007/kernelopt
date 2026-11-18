"""控制台 e2e 动作的参数校验与 argv 构建（不需要 GPU）。"""

from __future__ import annotations

import pytest

from optframe.webui import jobs


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(jobs, "_python", lambda: "python3")
    monkeypatch.setattr(jobs, "_model_path", lambda value: str(value))


def test_e2e_abba_passes_order_and_min_gain():
    argv = jobs._build_e2e({
        "confirm": True, "order": "abba", "blocks": 8,
        "min_gain": 1.02, "seconds": 30,
    })
    assert "--order" in argv and argv[argv.index("--order") + 1] == "abba"
    assert "--min-gain" in argv and argv[argv.index("--min-gain") + 1] == "1.02"
    assert argv[argv.index("--blocks") + 1] == "8"


def test_e2e_default_is_alternate_without_min_gain_flag():
    argv = jobs._build_e2e({"confirm": True})
    assert argv[argv.index("--order") + 1] == "alternate"
    assert argv[argv.index("--min-gain") + 1] == "1.015"


def test_e2e_requires_confirm():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"blocks": 6})


def test_e2e_rejects_bad_order():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "order": "random"})


def test_e2e_abba_requires_multiple_of_four_blocks():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "order": "abba", "blocks": 6})


def test_e2e_rejects_min_gain_at_or_below_one():
    with pytest.raises(jobs.ParamError):
        jobs._build_e2e({"confirm": True, "min_gain": 1.0})
