"""preflight 自检工具的单元测试（不依赖 GPU：全部用假环境事实）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optframe import preflight as pf


def fake_info(arch="sm120", available=True):
    return {
        "available": available,
        "name": "NVIDIA RTX PRO 5000 72GB Blackwell",
        "capability": [12, 0],
        "arch": arch,
        "memory_gib": 71.7,
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "triton": "3.6.0",
        "triton_arch": arch,
    }


# ---------------------------------------------------------------- 基础

def test_parse_version():
    assert pf.parse_version("12.8") == (12, 8)
    assert pf.parse_version("release 13.0, V13.0.88") == (13, 0)
    assert pf.parse_version(None) is None
    assert pf.parse_version("n/a") is None


def test_report_worst():
    r = pf.Report()
    r.add("a", pf.PASS, "ok")
    assert r.worst == pf.PASS
    r.add("b", pf.WARN, "warn")
    assert r.worst == pf.WARN
    r.add("c", pf.FAIL, "fail")
    assert r.worst == pf.FAIL


# ---------------------------------------------------------------- 设备/工具链

def test_check_device_pass():
    r = pf.Report()
    pf.check_device(fake_info(), r)
    assert r.findings[-1].status == pf.PASS
    assert "sm120" in r.findings[-1].detail


def test_check_device_fail_without_cuda():
    r = pf.Report()
    pf.check_device({"available": False, "error": "CUDA unavailable"}, r)
    assert r.findings[-1].status == pf.FAIL


def test_toolkit_same_major_passes(monkeypatch):
    monkeypatch.setattr(pf, "nvcc_version", lambda: (12, 8))
    r = pf.Report()
    pf.check_toolkit(fake_info(), r)
    assert r.findings[-1].status == pf.PASS


def test_toolkit_cross_major_without_cudart_fails(monkeypatch):
    monkeypatch.setattr(pf, "nvcc_version", lambda: (13, 0))
    monkeypatch.setattr(pf, "find_torch_cudart", lambda: None)
    r = pf.Report()
    pf.check_toolkit(fake_info(), r)
    assert r.findings[-1].status == pf.FAIL
    assert "no kernel image" in r.findings[-1].hint


def test_toolkit_cross_major_with_cudart_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "nvcc_version", lambda: (13, 0))
    lib = tmp_path / "libcudart.so.12"
    lib.write_text("stub")
    monkeypatch.setattr(pf, "find_torch_cudart", lambda: lib)
    r = pf.Report()
    pf.check_toolkit(fake_info(), r)
    assert r.findings[-1].status == pf.PASS


def test_triton_mismatch_warns():
    info = fake_info()
    info["triton_arch"] = "sm89"
    r = pf.Report()
    pf.check_triton(info, r)
    assert r.findings[-1].status == pf.WARN


# ---------------------------------------------------------------- 基线

def test_arch_baseline_missing_fails(tmp_path):
    r = pf.Report()
    pf.check_arch_baseline(fake_info(), tmp_path, r)
    assert r.findings[-1].status == pf.FAIL


def test_arch_baseline_match_and_mismatch(tmp_path):
    (tmp_path / "L20.json").write_text(json.dumps({"name": "L20"}))
    r = pf.Report()
    pf.check_arch_baseline(fake_info(), tmp_path, r)
    assert r.findings[-1].status == pf.WARN
    assert "L20" in r.findings[-1].detail

    (tmp_path / "RTX.json").write_text(json.dumps({"name": "RTX PRO 5000"}))
    r2 = pf.Report()
    pf.check_arch_baseline(fake_info(), tmp_path, r2)
    assert r2.findings[-1].status == pf.PASS


# ---------------------------------------------------------------- 任务

def _task(path: Path, name: str, arch: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "task.toml").write_text(
        f'[task]\nname = "{name}"\nop = "aten::silu"\narch = "{arch}"\n'
        'dtype = "bf16"\nshapes = [[8, 8]]\n',
        encoding="utf-8",
    )


def test_tasks_all_match(tmp_path):
    _task(tmp_path / "t1", "t1", "sm120")
    _task(tmp_path / "t2", "t2", "sm120")
    r = pf.Report()
    pf.check_tasks(fake_info(), [tmp_path], r)
    assert r.findings[-1].status == pf.PASS


def test_tasks_drift_warns(tmp_path):
    _task(tmp_path / "t1", "t1", "sm89")
    _task(tmp_path / "t2", "t2", "sm120")
    r = pf.Report()
    pf.check_tasks(fake_info(), [tmp_path], r)
    finding = r.findings[-1]
    assert finding.status == pf.WARN
    assert "1/2" in finding.detail


def test_tasks_missing_skips(tmp_path):
    r = pf.Report()
    pf.check_tasks(fake_info(), [tmp_path / "nope"], r)
    assert r.findings[-1].status == pf.SKIP


# ---------------------------------------------------------------- 硬编码

def test_scan_hardcoded_gencode(tmp_path):
    cu = tmp_path / "k.cu"
    cu.write_text("// fk-cuda-cflags: -gencode=arch=compute_89,code=sm_89\n")
    hits = pf.scan_hardcoded_gencode([tmp_path])
    codes = {code for _, code in hits}
    assert "89" in codes


def test_hardcoded_warns_on_mismatch(tmp_path):
    (tmp_path / "k.cu").write_text("// -gencode=arch=compute_89,code=sm_89\n")
    r = pf.Report()
    pf.check_hardcoded(fake_info(), [tmp_path], r)
    assert r.findings[-1].status == pf.WARN


def test_hardcoded_pass_when_matching(tmp_path):
    (tmp_path / "k.cu").write_text("// -gencode=arch=compute_120,code=sm_120\n")
    r = pf.Report()
    pf.check_hardcoded(fake_info(), [tmp_path], r)
    assert r.findings[-1].status == pf.PASS


# ---------------------------------------------------------------- hw_kb

def test_hw_kb_missing_arch_warns(tmp_path):
    (tmp_path / "hw_kb" / "cuda" / "sm89").mkdir(parents=True)
    r = pf.Report()
    pf.check_hw_kb(fake_info(), tmp_path, r)
    assert r.findings[-1].status == pf.WARN
    assert "sm120" in r.findings[-1].detail


def test_hw_kb_present_passes(tmp_path):
    (tmp_path / "hw_kb" / "cuda" / "sm120").mkdir(parents=True)
    r = pf.Report()
    pf.check_hw_kb(fake_info(), tmp_path, r)
    assert r.findings[-1].status == pf.PASS


# ---------------------------------------------------------------- 汇总

def test_run_checks_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "probe_device", lambda: {"available": False, "error": "no gpu"})
    report = pf.run_checks(arch_dir=tmp_path, task_dirs=[tmp_path], forge_root=None)
    assert report.worst == pf.FAIL
    assert any(f.check == "device" and f.status == pf.FAIL for f in report.findings)


def test_render_contains_symbols():
    r = pf.Report()
    r.add("device", pf.PASS, "ok")
    text = pf.render(r)
    assert "✅" in text and "结论: PASS" in text
