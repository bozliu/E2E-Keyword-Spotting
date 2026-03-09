from __future__ import annotations

import numpy as np
import torch
import pytest

from kws.constants import CLIP_SAMPLES, COMMAND31_LABELS
from kws.demo import realtime
from kws.models.common import DualTaskOutput


class _FakeFrontend:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.zeros(80, 126, dtype=torch.float32)


class _FakeModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> DualTaskOutput:  # noqa: ARG002
        command_logits = torch.zeros(1, len(COMMAND31_LABELS), dtype=torch.float32)
        command_logits[0, COMMAND31_LABELS.index("yes")] = 10.0
        wake_logits = torch.tensor([10.0], dtype=torch.float32)
        embedding = torch.zeros(1, 8, dtype=torch.float32)
        return DualTaskOutput(command_logits=command_logits, wake_logits=wake_logits, embedding=embedding)


def _make_engine(*, backend: str) -> realtime.RealtimeEngine:
    gate = realtime.GateStateMachine(
        mode="fixed",
        open_threshold=0.5,
        close_threshold=0.4,
        cmd_conf_threshold=0.3,
        hold_seconds=0.2,
        adaptive=realtime.AdaptiveGateConfig(calibration_seconds=0.0),
    )
    return realtime.RealtimeEngine(
        model=_FakeModel(),
        frontend=_FakeFrontend(),
        device=torch.device("cpu"),
        command31_labels=COMMAND31_LABELS,
        wheel="kws12",
        gate=gate,
        hop_seconds=0.0,
        ema_alpha=0.3,
        hold_ms=200.0,
        selected_device_label="cpu",
        input_device_name="Test Mic",
        stream_sample_rate=16000.0,
        model_sample_rate=16000,
        audio_seconds=1.0,
        mic_precheck_seconds=0.0,
        mic_min_rms=0.0,
        auto_gain=True,
        target_rms=0.05,
        max_gain_db=18.0,
        display_conf_thr=0.0,
        display_wake_thr=0.0,
        vote_window=1,
        vote_min_count=1,
        passive_profile=None,
        keyword_calibration=None,
        verifier=None,
        runtime_label_backend=backend,
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="mps",
    )


def test_parse_args_defaults_to_accuracy_first(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["realtime"])
    args = realtime.parse_args()
    assert args.demo_profile == "accuracy-first"
    assert args.device == "auto"
    assert args.runtime_label_backend == ""
    assert args.external_kws_model == realtime.ENSEMBLE_AST_SUPERB_MODEL_ID


def test_resolve_realtime_profile_defaults_to_external_ensemble(monkeypatch) -> None:
    monkeypatch.setattr(realtime.torch.backends.mps, "is_available", lambda: True)
    profile = realtime.resolve_realtime_profile(
        demo_profile="accuracy-first",
        detector_device_preference="auto",
        runtime_label_backend="",
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="auto",
        wheel="kws12",
    )
    assert profile.detector_device_preference == "mps"
    assert profile.runtime_label_backend == "external-ensemble"
    assert profile.external_kws_device == "mps"


def test_resolve_realtime_profile_requires_mps_for_accuracy_first(monkeypatch) -> None:
    monkeypatch.setattr(realtime.torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires Apple MPS"):
        realtime.resolve_realtime_profile(
            demo_profile="accuracy-first",
            detector_device_preference="auto",
            runtime_label_backend="",
            external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
            external_kws_device="auto",
            wheel="kws12",
        )


def test_realtime_engine_detector_backend_matches_yes() -> None:
    engine = _make_engine(backend="detector")
    snap = engine.process_chunk(
        np.full(CLIP_SAMPLES, 0.01, dtype=np.float32),
        now=1.0,
        now_wall=1.0,
        queue_fill_ratio=0.0,
    )
    assert snap is not None
    assert snap.active_label == "yes"
    assert snap.command_label == "yes"
    assert snap.runtime_label_backend == "detector"


def test_realtime_engine_external_backend_falls_back_to_detector(monkeypatch) -> None:
    def _boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("hf boom")

    monkeypatch.setattr(realtime, "predict_kws12_from_waveforms", _boom)
    engine = _make_engine(backend="external-ensemble")
    snap = engine.process_chunk(
        np.full(CLIP_SAMPLES, 0.01, dtype=np.float32),
        now=1.0,
        now_wall=1.0,
        queue_fill_ratio=0.0,
    )
    assert snap is not None
    assert snap.active_label == "yes"
    assert snap.runtime_label_backend == "detector-fallback"
    assert "hf boom" in snap.backend_note
