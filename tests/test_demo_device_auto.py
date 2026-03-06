from __future__ import annotations

import torch

from kws.demo import realtime


def test_auto_device_picks_fastest(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_bench(checkpoint, device_name: str, iters: int) -> float:  # noqa: ARG001
        return 8.0 if device_name == "cpu" else 20.0

    monkeypatch.setattr(realtime, "_benchmark_runtime_device", fake_bench)

    device, label, timings = realtime._resolve_runtime_device(
        preferred="auto",
        checkpoint={"config": {}},
        benchmark_iters=5,
    )

    assert device.type == "cpu"
    assert label == "auto->cpu"
    assert timings["cpu"] == 8.0
    assert timings["mps"] == 20.0


def test_auto_device_falls_back_when_no_finite_latency(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(realtime, "_benchmark_runtime_device", lambda checkpoint, device_name, iters: float("inf"))

    device, label, timings = realtime._resolve_runtime_device(
        preferred="auto",
        checkpoint={"config": {}},
        benchmark_iters=3,
    )

    assert device.type in {"cpu", "mps", "cuda"}
    assert label.startswith("auto->")
    assert all(v == float("inf") for v in timings.values())


def test_explicit_device_bypasses_auto_benchmark(monkeypatch) -> None:
    called = {"value": False}

    def fake_bench(checkpoint, device_name: str, iters: int) -> float:  # noqa: ARG001
        called["value"] = True
        return 1.0

    monkeypatch.setattr(realtime, "_benchmark_runtime_device", fake_bench)

    device, label, timings = realtime._resolve_runtime_device(
        preferred="cpu",
        checkpoint={"config": {}},
        benchmark_iters=5,
    )

    assert device.type == "cpu"
    assert label == "cpu"
    assert timings == {}
    assert called["value"] is False
