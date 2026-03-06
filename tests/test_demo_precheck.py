from __future__ import annotations

from types import SimpleNamespace

from kws.demo import realtime


class _FakeSD:
    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


def test_resolve_stream_device_no_input(monkeypatch) -> None:
    fake = _FakeSD([
        {"name": "spk", "max_input_channels": 0, "max_output_channels": 2},
    ])
    monkeypatch.setattr(realtime, "sd", fake)

    out = realtime._resolve_stream_device("")
    assert isinstance(out, realtime.MicPrecheckResult)
    assert out.state == realtime.MIC_NO_DEVICE


def test_resolve_stream_device_by_id(monkeypatch) -> None:
    fake = _FakeSD([
        {"name": "mic0", "max_input_channels": 1, "max_output_channels": 0},
    ])
    monkeypatch.setattr(realtime, "sd", fake)

    out = realtime._resolve_stream_device("0")
    assert out == 0


def test_resolve_stream_device_by_name(monkeypatch) -> None:
    fake = _FakeSD([
        {"name": "My USB Mic", "max_input_channels": 1, "max_output_channels": 0},
    ])
    monkeypatch.setattr(realtime, "sd", fake)

    out = realtime._resolve_stream_device("usb")
    assert out == 0


def test_classify_stream_error_permission() -> None:
    state = realtime._classify_stream_error(RuntimeError("microphone permission denied"))
    assert state == realtime.MIC_PERMISSION_DENIED


def test_classify_precheck_signal_accepts_quiet_real_mic() -> None:
    state, threshold = realtime.classify_precheck_signal(median_rms=0.00167, peak95=0.0135, min_rms=0.001)
    assert state == realtime.MIC_RUNNING
    assert 0.001 <= threshold <= 0.003


def test_classify_precheck_signal_detects_permission_denied() -> None:
    state, threshold = realtime.classify_precheck_signal(median_rms=0.0, peak95=0.0, min_rms=0.001)
    assert state == realtime.MIC_PERMISSION_DENIED
    assert threshold >= 0.001


def test_classify_precheck_signal_detects_no_signal() -> None:
    state, _threshold = realtime.classify_precheck_signal(median_rms=0.0002, peak95=0.0010, min_rms=0.001)
    assert state == realtime.MIC_NO_SIGNAL
