from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kws.constants import KWS12_TO_INDEX
from kws.demo import cache_realtime_traces, replay_realtime


def test_cache_realtime_traces_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    fake_bundle = SimpleNamespace(
        resolved_profile=SimpleNamespace(
            demo_profile="accuracy-first-realtime",
            runtime_label_backend="external-ensemble",
            external_kws_model="ensemble/ast-superb-kws12",
            external_kws_device="mps",
        ),
        keyword_calibration_path=tmp_path / "keyword_calibration_realtime.json",
        external_ensemble_calibration_path=tmp_path / "external_ensemble_realtime_calibration.json",
    )
    fake_records = [SimpleNamespace(path=tmp_path / "a.wav", command_label=2)]
    monkeypatch.setattr(cache_realtime_traces, "load_realtime_demo", lambda **kwargs: fake_bundle)
    monkeypatch.setattr(cache_realtime_traces, "_records_for_split", lambda split, limit_per_class=0: fake_records)
    monkeypatch.setattr(
        cache_realtime_traces,
        "collect_clip_trace",
        lambda **kwargs: {
            "record_path": str((tmp_path / "a.wav").resolve()),
            "target_kws12": int(KWS12_TO_INDEX["yes"]),
            "window_start": 0.4,
            "window_end": 0.9,
            "timestamps": [0.5],
            "detector_command_probs": [[0.0]],
            "detector_wake_probs": [0.9],
            "ast_probs": [[0.0] * 12],
            "superb_probs": [[0.0] * 12],
        },
    )
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cache_realtime_traces",
            "--cache-root",
            str(cache_root),
        ],
    )
    cache_realtime_traces.main()
    manifest_path = cache_root / "valid" / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["demo_profile"] == "accuracy-first-realtime"
    assert payload["num_eval_samples"] == 1
    assert len(payload["entries"]) == 1


def test_replay_realtime_main_writes_expected_payload(tmp_path: Path, monkeypatch) -> None:
    fake_bundle = SimpleNamespace(
        resolved_profile=SimpleNamespace(
            runtime_label_backend="external-ensemble",
            external_kws_model="ensemble/ast-superb-kws12",
            external_kws_device="mps",
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"trace_path": str(tmp_path / "0.npz")},
                    {"trace_path": str(tmp_path / "1.npz")},
                ]
            }
        ),
        encoding="utf-8",
    )
    traces = iter(
        [
            {"target_kws12": int(KWS12_TO_INDEX["yes"])},
            {"target_kws12": int(KWS12_TO_INDEX["silence"])},
        ]
    )
    outputs = iter(
        [
            (int(KWS12_TO_INDEX["yes"]), False, 120.0),
            (int(KWS12_TO_INDEX["silence"]), True, None),
        ]
    )
    monkeypatch.setattr(replay_realtime, "load_realtime_demo", lambda **kwargs: fake_bundle)
    monkeypatch.setattr(replay_realtime, "load_trace", lambda path: next(traces))
    monkeypatch.setattr(replay_realtime, "replay_clip_trace", lambda **kwargs: next(outputs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay_realtime",
            "--trace-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    replay_realtime.main()
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["runtime_label_backend"] == "external-ensemble"
    assert payload["external_kws_model_id"] == "ensemble/ast-superb-kws12"
    assert payload["avg_match_latency_ms"] == 120.0
