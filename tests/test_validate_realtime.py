from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kws.constants import KWS12_TO_INDEX
from kws.demo import validate_realtime


def test_summarize_clip_frames_uses_majority_then_earliest() -> None:
    frames = [
        (0.50, SimpleNamespace(prompt_status="MATCH", active_label="yes", gate_open=True, command_label="yes")),
        (0.55, SimpleNamespace(prompt_status="MATCH", active_label="no", gate_open=True, command_label="no")),
        (0.60, SimpleNamespace(prompt_status="MATCH", active_label="yes", gate_open=True, command_label="yes")),
        (0.65, SimpleNamespace(prompt_status="LISTENING", active_label=None, gate_open=True, command_label="yes")),
    ]
    pred, no_match, latency_ms = validate_realtime._summarize_clip_frames(frames, window_start=0.40, window_end=0.90)
    assert pred == KWS12_TO_INDEX["yes"]
    assert no_match is False
    assert latency_ms == pytest.approx(100.0)


def test_summarize_clip_frames_falls_back_to_silence() -> None:
    frames = [
        (0.50, SimpleNamespace(prompt_status="LISTENING", active_label=None, gate_open=False, command_label="silence")),
        (0.60, SimpleNamespace(prompt_status="LISTENING", active_label=None, gate_open=False, command_label="silence")),
    ]
    pred, no_match, latency_ms = validate_realtime._summarize_clip_frames(frames, window_start=0.40, window_end=0.90)
    assert pred == KWS12_TO_INDEX["silence"]
    assert no_match is True
    assert latency_ms is None


def test_validate_realtime_main_writes_expected_payload(tmp_path: Path, monkeypatch) -> None:
    fake_bundle = SimpleNamespace(
        resolved_profile=SimpleNamespace(
            demo_profile="accuracy-first",
            runtime_label_backend="external-ensemble",
            external_kws_model="ensemble/ast-superb-kws12",
            external_kws_device="mps",
        ),
    )
    fake_records = [
        SimpleNamespace(command_label=2),
        SimpleNamespace(command_label=0),
    ]

    monkeypatch.setattr(validate_realtime, "load_realtime_demo", lambda **kwargs: fake_bundle)
    monkeypatch.setattr(validate_realtime, "_manifest_records", lambda manifest_path, limit_per_class=0: fake_records)
    outputs = iter(
        [
            (KWS12_TO_INDEX["yes"], False, 120.0),
            (KWS12_TO_INDEX["silence"], True, None),
        ]
    )
    monkeypatch.setattr(validate_realtime, "_predict_clip", lambda **kwargs: next(outputs))
    output_path = tmp_path / "realtime.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_realtime",
            "--split",
            "valid",
            "--output",
            str(output_path),
        ],
    )
    validate_realtime.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["runtime_label_backend"] == "external-ensemble"
    assert payload["external_kws_model_id"] == "ensemble/ast-superb-kws12"
    assert "per_class_kws12" in payload
    assert payload["avg_match_latency_ms"] == 120.0


def test_run_validation_passes_keyword_calibration_override(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_load_realtime_demo(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            resolved_profile=SimpleNamespace(
                demo_profile="accuracy-first-realtime",
                runtime_label_backend="external-ensemble",
                external_kws_model="ensemble/ast-superb-kws12",
                external_kws_device="mps",
            ),
        )

    monkeypatch.setattr(validate_realtime, "load_realtime_demo", _fake_load_realtime_demo)
    monkeypatch.setattr(validate_realtime, "_manifest_records", lambda manifest_path, limit_per_class=0: [])
    args = SimpleNamespace(
        checkpoint="auto",
        demo_profile="accuracy-first-realtime",
        device="mps",
        selection_profile="stable",
        keyword_calibration_path="/tmp/realtime.json",
        external_ensemble_calibration_path="/tmp/external.json",
        wheel="kws12",
        runtime_label_backend="",
        external_kws_model="ensemble/ast-superb-kws12",
        external_kws_device="mps",
        sensitivity_profile="strict",
        split="valid",
        limit_per_class=0,
    )
    payload = validate_realtime.run_validation(args)
    assert seen["keyword_calibration_path"] == "/tmp/realtime.json"
    assert seen["external_ensemble_calibration_path"] == "/tmp/external.json"
    assert payload["runtime_label_backend"] == "external-ensemble"


def test_predict_clip_counts_final_flushed_match_inside_window(monkeypatch, tmp_path: Path) -> None:
    class _FakeGate:
        open_threshold = 0.6
        close_threshold = 0.5

    class _FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            self.gate = _FakeGate()

        def bypass_precheck(self) -> None:
            return None

        def process_chunk(self, chunk, *, now, now_wall, queue_fill_ratio=0.0):
            return None

        def flush_pending_segment(self, *, now, now_wall):
            return SimpleNamespace(
                prompt_status="MATCH",
                active_label="yes",
                gate_open=True,
                command_label="yes",
            )

    monkeypatch.setattr(validate_realtime, "RealtimeEngine", _FakeEngine)
    monkeypatch.setattr(validate_realtime, "load_audio", lambda path, sample_rate: SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((16000,), dtype=np.float32)))))
    monkeypatch.setattr(validate_realtime, "_estimate_utterance_bounds", lambda waveform, sample_rate: (0, int(0.2 * sample_rate)))

    bundle = SimpleNamespace(
        model=object(),
        frontend=object(),
        runtime_device="cpu",
        command31_labels=["silence"],
        wheel="kws12",
        selected_device_label="cpu",
        sample_rate=16000,
        audio_seconds=1.0,
        keyword_calibration={},
        external_ensemble_calibration={},
        segment_decoder=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_calibration={},
        verifier=None,
        resolved_profile=SimpleNamespace(
            demo_profile="cpu-baseline",
            runtime_label_backend="detector",
            external_kws_model="ensemble/ast-superb-kws12",
            external_kws_device="mps",
        ),
    )
    record = SimpleNamespace(path=str(tmp_path / "clip.wav"))
    args = SimpleNamespace(
        split="valid",
        gate_mode="adaptive",
        threshold=None,
        wake_open_thr=0.6,
        wake_close_thr=0.5,
        calibration_seconds=2.0,
        cmd_conf_thr=None,
        hold_ms=300.0,
        display_conf_thr=None,
        display_wake_thr=None,
        vote_window=None,
        vote_min_count=None,
        hop_seconds=0.10,
        ema_alpha=0.35,
        pre_silence_seconds=0.4,
        post_silence_seconds=0.4,
        match_tail_seconds=0.3,
    )
    tuning = SimpleNamespace(
        cmd_conf_thr=0.35,
        open_offset=0.12,
        close_offset=0.05,
        open_floor=0.25,
        close_floor=0.15,
        display_conf_thr=0.40,
        display_wake_thr=0.40,
        vote_window=4,
        vote_min_count=2,
    )

    pred, no_match, _latency_ms = validate_realtime._predict_clip(
        bundle=bundle,
        record=record,
        args=args,
        tuning=tuning,
    )
    assert pred == KWS12_TO_INDEX["yes"]
    assert no_match is False


def test_predict_clip_uses_trace_replay_for_accuracy_first_realtime(monkeypatch, tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        resolved_profile=SimpleNamespace(
            demo_profile="accuracy-first-realtime",
            runtime_label_backend="external-ensemble",
            external_kws_model="ensemble/ast-superb-kws12",
            external_kws_device="mps",
        ),
    )
    record = SimpleNamespace(path=str(tmp_path / "clip.wav"))
    args = SimpleNamespace()
    tuning = SimpleNamespace()
    seen: dict[str, object] = {}

    monkeypatch.setattr(validate_realtime, "collect_clip_trace", lambda **kwargs: seen.setdefault("trace", {"target_kws12": 2}) or seen["trace"])
    monkeypatch.setattr(validate_realtime, "replay_clip_trace", lambda **kwargs: (KWS12_TO_INDEX["yes"], False, 120.0))

    pred, no_match, latency_ms = validate_realtime._predict_clip(
        bundle=bundle,
        record=record,
        args=args,
        tuning=tuning,
    )
    assert seen["trace"]["target_kws12"] == 2
    assert pred == KWS12_TO_INDEX["yes"]
    assert no_match is False
    assert latency_ms == 120.0
