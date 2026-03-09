from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
