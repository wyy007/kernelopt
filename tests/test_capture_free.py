"""free_gpu：只杀 sglang、等显存释放（不需要 GPU）。"""

from pathlib import Path

from optframe.webui import capture as cap


def test_free_gpu_already_free(monkeypatch):
    monkeypatch.setattr(cap, "gpu_used_gb", lambda: 1.0)
    ok, why = cap.free_gpu(5.0)
    assert ok and "空闲" in why


def test_free_gpu_without_force_refuses(monkeypatch):
    monkeypatch.setattr(cap, "gpu_used_gb", lambda: 50.0)
    ok, why = cap.free_gpu(5.0)
    assert not ok and "不空闲" in why


def test_free_gpu_force_kills_sglang_then_waits(monkeypatch):
    seq = iter([50.0, 50.0, 1.0])
    monkeypatch.setattr(cap, "gpu_used_gb", lambda: next(seq))
    calls = []
    monkeypatch.setattr(cap, "sh", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(cap.time, "sleep", lambda s: None)
    ok, why = cap.free_gpu(5.0, force=True)
    assert ok and "已释放" in why
    assert any("pkill -9 -f sglang.launch_server" in " ".join(a) for a in calls)


def test_trace_stats_retry_recovers(monkeypatch):
    seen = {"n": 0}

    def fake(path, window_est=True):
        seen["n"] += 1
        if seen["n"] < 3:
            raise EOFError("truncated")
        return {"ok": 1}

    monkeypatch.setattr(cap, "trace_stats", fake)
    monkeypatch.setattr(cap.time, "sleep", lambda s: None)
    assert cap.trace_stats_retry(Path("x")) == {"ok": 1}
    assert seen["n"] == 3
