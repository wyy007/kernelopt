"""控制台配置（驱动模型）读写。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optframe.webui import settings


def test_defaults_when_file_missing(tmp_path):
    data = settings.load_settings(tmp_path / "none.json")
    assert data["llm_enabled"] is True
    assert data["llm_model"] == "ds-v4"


def test_save_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    saved = settings.save_settings(
        {"llm_enabled": False, "llm_model": "qwen3.5-9b", "llm_repeat": 5}, p
    )
    assert saved["llm_enabled"] is False
    assert saved["llm_model"] == "qwen3.5-9b"
    assert saved["llm_repeat"] == 5
    assert settings.load_settings(p)["llm_model"] == "qwen3.5-9b"


def test_save_clamps_numbers(tmp_path):
    p = tmp_path / "s.json"
    saved = settings.save_settings({"llm_n": 0, "llm_rounds": -3, "llm_repeat": 0}, p)
    assert saved["llm_n"] == 1
    assert saved["llm_rounds"] == 1
    assert saved["llm_repeat"] == 1
