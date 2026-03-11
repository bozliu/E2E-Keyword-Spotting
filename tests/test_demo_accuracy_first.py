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
        external_ensemble_calibration=None,
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


def test_resolve_realtime_profile_realtime_variant_defaults_to_external_ensemble(monkeypatch) -> None:
    monkeypatch.setattr(realtime.torch.backends.mps, "is_available", lambda: True)
    profile = realtime.resolve_realtime_profile(
        demo_profile="accuracy-first-realtime",
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

    monkeypatch.setattr(realtime, "predict_ensemble_ast_superb_from_waveforms", _boom)
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


def test_resolve_keyword_calibration_requires_realtime_file(tmp_path) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first-realtime",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="mps",
    )
    with pytest.raises(FileNotFoundError, match="requires a realtime calibration file"):
        realtime._resolve_keyword_calibration(
            checkpoint_path=checkpoint_path,
            checkpoint_payload={},
            profile=profile,
            keyword_calibration_path="",
        )


def test_resolve_external_ensemble_calibration_requires_realtime_file(tmp_path) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first-realtime",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="mps",
    )
    with pytest.raises(FileNotFoundError, match="requires an external ensemble realtime calibration file"):
        realtime._resolve_external_ensemble_calibration(
            checkpoint_path=checkpoint_path,
            profile=profile,
            external_ensemble_calibration_path="",
        )


def test_resolve_segment_decoder_requires_marker_or_artifact(tmp_path) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first-realtime",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="mps",
    )
    with pytest.raises(FileNotFoundError, match="requires segment_decoder_realtime.pt"):
        realtime._resolve_segment_decoder(
            checkpoint_path=checkpoint_path,
            profile=profile,
            keyword_calibration={},
            runtime_device=torch.device("cpu"),
        )

    decoder, path, disabled = realtime._resolve_segment_decoder(
        checkpoint_path=checkpoint_path,
        profile=profile,
        keyword_calibration={"segment_decoder_disabled": True},
        runtime_device=torch.device("cpu"),
    )
    assert decoder is None
    assert path is None
    assert disabled is True


def test_resolve_realtime_specialist_requires_artifact_and_calibration(tmp_path) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first-realtime",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model=realtime.ENSEMBLE_AST_SUPERB_MODEL_ID,
        external_kws_device="mps",
    )
    with pytest.raises(FileNotFoundError, match="requires realtime_specialist_calibration.json"):
        realtime._resolve_realtime_specialist_calibration(
            checkpoint_path=checkpoint_path,
            profile=profile,
        )
    cal_path = checkpoint_path.parent / "realtime_specialist_calibration.json"
    cal_path.write_text('{"enabled": true, "default": {"accept_prob": 0.6, "min_margin": 0.04, "trigger_prob": 0.2}}', encoding="utf-8")
    calibration, resolved_cal_path = realtime._resolve_realtime_specialist_calibration(
        checkpoint_path=checkpoint_path,
        profile=profile,
    )
    assert resolved_cal_path == cal_path.resolve()
    assert calibration["enabled"] is True
    with pytest.raises(FileNotFoundError, match="requires realtime_specialist.pt"):
        realtime._resolve_realtime_specialist(
            checkpoint_path=checkpoint_path,
            profile=profile,
            runtime_device=torch.device("cpu"),
        )


def test_hard_word_specialist_rescues_unknown_on(monkeypatch) -> None:
    engine = _make_engine(backend="external-ensemble")
    engine.realtime_specialist = object()
    engine.realtime_specialist_calibration = {
        "enabled": True,
        "default": {"accept_prob": 0.6, "min_margin": 0.04, "trigger_prob": 0.2, "role": "rescue"},
        "per_label": {
            "on": {"accept_prob": 0.5, "min_margin": 0.02, "trigger_prob": 0.16, "role": "rescue"},
            "up": {"accept_prob": 0.72, "min_margin": 0.10, "trigger_prob": 0.22, "role": "guard"},
        },
        "unknown_trigger_prob": 0.18,
    }
    active_segment = realtime.ActiveSegmentState(
        stats=realtime.SegmentFeatureStats(thresholds=np.full((len(realtime.SEGMENT_TARGET_LABELS),), 0.18, dtype=np.float32)),
        started_at=0.0,
        last_signal_at=0.2,
        hard_peak_label="on",
        hard_peak_prob=0.32,
        hard_peak_time=0.1,
        hard_peak_waveform=np.full((CLIP_SAMPLES,), 0.04, dtype=np.float32),
    )
    monkeypatch.setattr(
        realtime,
        "predict_realtime_specialist",
        lambda specialist, waveform: np.asarray([0.82, 0.04, 0.04, 0.03, 0.07], dtype=np.float32),
    )
    label, prob, margin, probs = engine._apply_hard_word_specialist(
        active_segment=active_segment,
        accepted_label=None,
        top_label="unknown",
        top_prob=0.44,
        margin=0.05,
    )
    assert label == "on"
    assert prob > 0.81
    assert probs is not None


def test_hard_word_specialist_guards_up_to_unknown(monkeypatch) -> None:
    engine = _make_engine(backend="external-ensemble")
    engine.realtime_specialist = object()
    engine.realtime_specialist_calibration = {
        "enabled": True,
        "default": {"accept_prob": 0.6, "min_margin": 0.04, "trigger_prob": 0.2, "role": "rescue"},
        "per_label": {
            "up": {"accept_prob": 0.72, "min_margin": 0.10, "trigger_prob": 0.22, "role": "guard"},
        },
        "unknown_trigger_prob": 0.18,
    }
    active_segment = realtime.ActiveSegmentState(
        stats=realtime.SegmentFeatureStats(thresholds=np.full((len(realtime.SEGMENT_TARGET_LABELS),), 0.18, dtype=np.float32)),
        started_at=0.0,
        last_signal_at=0.2,
        hard_peak_label="up",
        hard_peak_prob=0.51,
        hard_peak_time=0.1,
        hard_peak_waveform=np.full((CLIP_SAMPLES,), 0.04, dtype=np.float32),
    )
    monkeypatch.setattr(
        realtime,
        "predict_realtime_specialist",
        lambda specialist, waveform: np.asarray([0.05, 0.06, 0.05, 0.12, 0.72], dtype=np.float32),
    )
    label, _prob, _margin, _probs = engine._apply_hard_word_specialist(
        active_segment=active_segment,
        accepted_label="up",
        top_label="up",
        top_prob=0.79,
        margin=0.11,
    )
    assert label is None
