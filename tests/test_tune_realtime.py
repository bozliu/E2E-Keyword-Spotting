from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from kws.constants import COMMAND31_TO_INDEX
from kws.demo import realtime, tune_realtime


def _base_seed_report() -> dict[str, object]:
    per_class = {
        "silence": {"precision": 0.99, "recall": 1.0, "top_confusions": []},
        "unknown": {"precision": 0.96, "recall": 0.98, "top_confusions": [{"label": "up", "count": 20}]},
    }
    for label in ("yes", "no", "down", "left", "right", "stop"):
        per_class[label] = {"precision": 0.98, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 5}]}
    per_class["on"] = {"precision": 0.99, "recall": 0.78, "top_confusions": [{"label": "unknown", "count": 20}]}
    per_class["off"] = {"precision": 0.98, "recall": 0.82, "top_confusions": [{"label": "unknown", "count": 18}]}
    per_class["go"] = {"precision": 0.99, "recall": 0.89, "top_confusions": [{"label": "unknown", "count": 12}]}
    per_class["up"] = {"precision": 0.82, "recall": 0.94, "top_confusions": [{"label": "unknown", "count": 10}]}
    return {
        "split": "valid",
        "per_class_kws12": per_class,
        "min_kws12_precision": 0.82,
        "min_kws12_recall": 0.78,
        "unknown_to_target_rate": 0.01,
        "passed": False,
    }


def test_tune_realtime_writes_calibration_and_report(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    base_calibration = {
        "defaults": {
            "command_conf_threshold": 0.35,
            "vote_window": 4,
            "vote_min_count": 2,
            "prototype_bonus_max": 0.04,
            "min_margin": 0.0,
            "highlight_hold_ms": 220,
            "external_force_open_conf_threshold": 0.80,
        },
        "keywords": {
            "on": {"command_conf_threshold": 0.20, "vote_window": 6, "vote_min_count": 4, "prototype_bonus_max": 0.08, "min_margin": 0.16, "highlight_hold_ms": 320, "external_force_open_conf_threshold": 0.80, "support": 10},
            "off": {"command_conf_threshold": 0.40, "vote_window": 5, "vote_min_count": 3, "prototype_bonus_max": 0.08, "min_margin": 0.12, "highlight_hold_ms": 280, "external_force_open_conf_threshold": 0.80, "support": 10},
            "go": {"command_conf_threshold": 0.75, "vote_window": 5, "vote_min_count": 3, "prototype_bonus_max": 0.08, "min_margin": 0.12, "highlight_hold_ms": 280, "external_force_open_conf_threshold": 0.80, "support": 10},
            "up": {"command_conf_threshold": 0.45, "vote_window": 5, "vote_min_count": 2, "prototype_bonus_max": 0.04, "min_margin": 0.08, "highlight_hold_ms": 220, "external_force_open_conf_threshold": 0.80, "support": 10},
        },
    }
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model="ensemble/ast-superb-kws12",
        external_kws_device="mps",
    )
    bundle = realtime.LoadedRealtimeDemo(
        checkpoint_path=checkpoint_path,
        checkpoint_payload={},
        runtime_device=torch.device("cpu"),
        selected_device_label="cpu",
        model=torch.nn.Identity(),
        frontend=SimpleNamespace(),
        command31_labels=[],
        wheel="kws12",
        keyword_calibration=base_calibration,
        keyword_calibration_path=tmp_path / "keyword_calibration.json",
        external_ensemble_calibration={"version": 1, "defaults": {"unknown_superb_weight": 1.2}, "per_label_bias": {}},
        external_ensemble_calibration_path=tmp_path / "external_ensemble_calibration.json",
        segment_decoder=None,
        segment_decoder_path=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_path=None,
        realtime_specialist_calibration={},
        realtime_specialist_calibration_path=None,
        sample_rate=16_000,
        clip_samples=16_000,
        audio_seconds=1.0,
        verifier=None,
        resolved_profile=profile,
    )
    monkeypatch.setattr(tune_realtime, "load_realtime_demo", lambda **kwargs: bundle)
    monkeypatch.setattr(tune_realtime, "_load_seed_report", lambda args: _base_seed_report())
    monkeypatch.setattr(
        tune_realtime,
        "_manifest_records",
        lambda manifest_path, limit_per_class=0: [
            SimpleNamespace(command_label=COMMAND31_TO_INDEX["on"]),
            SimpleNamespace(command_label=COMMAND31_TO_INDEX["off"]),
            SimpleNamespace(command_label=COMMAND31_TO_INDEX["go"]),
            SimpleNamespace(command_label=COMMAND31_TO_INDEX["up"]),
        ],
    )

    def _fake_evaluate_records(*, bundle, records, args, tuning):  # noqa: ARG001
        keywords = bundle.keyword_calibration.get("keywords", {})
        on_thr = float(keywords.get("on", {}).get("external_force_open_conf_threshold", 0.80))
        off_thr = float(keywords.get("off", {}).get("external_force_open_conf_threshold", 0.80))
        go_thr = float(keywords.get("go", {}).get("external_force_open_conf_threshold", 0.80))
        up_conf = float(keywords.get("up", {}).get("command_conf_threshold", 0.45))
        passed = on_thr < 0.8 and off_thr < 0.8 and go_thr < 0.8 and up_conf > 0.45
        per_class = _base_seed_report()["per_class_kws12"]
        if passed:
            per_class = {
                **per_class,
                "on": {"precision": 0.97, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 8}]},
                "off": {"precision": 0.97, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 7}]},
                "go": {"precision": 0.98, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 5}]},
                "up": {"precision": 0.96, "recall": 0.95, "top_confusions": [{"label": "unknown", "count": 4}]},
            }
        return {
            "split": args.split,
            "per_class_kws12": per_class,
            "min_kws12_precision": 0.96 if passed else 0.82,
            "min_kws12_recall": 0.95 if passed else 0.78,
            "unknown_to_target_rate": 0.01,
            "passed": passed,
        }

    monkeypatch.setattr(tune_realtime, "evaluate_records", _fake_evaluate_records)
    output_calibration = tmp_path / "keyword_calibration_realtime.json"
    output_external_calibration = tmp_path / "external_ensemble_realtime_calibration.json"
    output_report = tmp_path / "realtime_tuning_valid.json"
    failure_report = tmp_path / "realtime_tuning_failure.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_realtime",
            "--output-calibration",
            str(output_calibration),
            "--output-external-calibration",
            str(output_external_calibration),
            "--output-report",
            str(output_report),
            "--failure-report",
            str(failure_report),
        ],
    )

    tune_realtime.main()

    assert output_calibration.exists()
    assert output_external_calibration.exists()
    assert output_report.exists()
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["tuned_profile"] == "accuracy-first-realtime"
    assert payload["selected_candidate"] != "baseline"
    assert payload["passed"] is True
    calibration = json.loads(output_calibration.read_text(encoding="utf-8"))
    assert calibration["keywords"]["on"]["external_force_open_conf_threshold"] < 0.8
    assert calibration["keywords"]["up"]["command_conf_threshold"] > 0.45
    external_calibration = json.loads(output_external_calibration.read_text(encoding="utf-8"))
    assert "per_label_bias" in external_calibration
    assert not failure_report.exists()


def test_tune_realtime_trains_segment_decoder_when_rules_still_fail(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model="ensemble/ast-superb-kws12",
        external_kws_device="mps",
    )
    bundle = realtime.LoadedRealtimeDemo(
        checkpoint_path=checkpoint_path,
        checkpoint_payload={},
        runtime_device=torch.device("cpu"),
        selected_device_label="cpu",
        model=torch.nn.Identity(),
        frontend=SimpleNamespace(),
        command31_labels=[],
        wheel="kws12",
        keyword_calibration={"defaults": {"command_conf_threshold": 0.35}, "keywords": {}},
        keyword_calibration_path=tmp_path / "keyword_calibration.json",
        external_ensemble_calibration={"version": 1, "defaults": {"unknown_superb_weight": 1.2}, "per_label_bias": {}},
        external_ensemble_calibration_path=tmp_path / "external.json",
        segment_decoder=None,
        segment_decoder_path=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_path=None,
        realtime_specialist_calibration={},
        realtime_specialist_calibration_path=None,
        sample_rate=16_000,
        clip_samples=16_000,
        audio_seconds=1.0,
        verifier=None,
        resolved_profile=profile,
    )
    monkeypatch.setattr(tune_realtime, "load_realtime_demo", lambda **kwargs: bundle)
    monkeypatch.setattr(tune_realtime, "_load_seed_report", lambda args: _base_seed_report())
    monkeypatch.setattr(tune_realtime, "_manifest_records", lambda manifest_path, limit_per_class=0: [])

    valid_manifest_path = tmp_path / "valid_manifest.json"
    train_manifest_path = tmp_path / "train_manifest.json"
    test_manifest_path = tmp_path / "test_manifest.json"
    for path in (valid_manifest_path, train_manifest_path, test_manifest_path):
        path.write_text(json.dumps({"entries": [{"trace_path": str(tmp_path / "dummy.npz"), "target_kws12": 2}]}), encoding="utf-8")

    def _fake_trace_manifest(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(tune_realtime, "_load_trace_manifest", _fake_trace_manifest)
    monkeypatch.setattr(
        tune_realtime,
        "_collect_segment_samples",
        lambda manifest, bundle, args, tuning, keyword_calibration, external_calibration: (
            np.stack(
                [
                    np.full((tune_realtime.SEGMENT_FEATURE_DIM,), 0.2, dtype=np.float32),
                    np.full((tune_realtime.SEGMENT_FEATURE_DIM,), 0.8, dtype=np.float32),
                ],
                axis=0,
            ),
            np.asarray(
                [
                    tune_realtime.SEGMENT_DECODER_LABEL_TO_INDEX["unknown"],
                    tune_realtime.SEGMENT_DECODER_LABEL_TO_INDEX["on"],
                ],
                dtype=np.int64,
            ),
        ),
    )

    def _fake_evaluate_trace_manifest(*, bundle, args, manifest, entries_override=None):  # noqa: ARG001
        passed = bundle.segment_decoder is not None
        per_class = _base_seed_report()["per_class_kws12"]
        if passed:
            per_class = {
                **per_class,
                "on": {"precision": 0.97, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 8}]},
                "off": {"precision": 0.97, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 7}]},
                "go": {"precision": 0.98, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 5}]},
                "up": {"precision": 0.96, "recall": 0.95, "top_confusions": [{"label": "unknown", "count": 4}]},
            }
        return {
            "split": args.split,
            "per_class_kws12": per_class,
            "min_kws12_precision": 0.96 if passed else 0.82,
            "min_kws12_recall": 0.95 if passed else 0.78,
            "unknown_to_target_rate": 0.01,
            "passed": passed,
        }

    monkeypatch.setattr(tune_realtime, "evaluate_trace_manifest", _fake_evaluate_trace_manifest)

    output_report = tmp_path / "out_valid.json"
    output_test = tmp_path / "out_test.json"
    output_cal = tmp_path / "keyword_calibration_realtime.json"
    output_ext = tmp_path / "external_ensemble_realtime_calibration.json"
    output_seg = tmp_path / "segment_decoder_realtime.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_realtime",
            "--trace-manifest",
            str(valid_manifest_path),
            "--train-trace-manifest",
            str(train_manifest_path),
            "--test-trace-manifest",
            str(test_manifest_path),
            "--output-report",
            str(output_report),
            "--output-test-report",
            str(output_test),
            "--output-calibration",
            str(output_cal),
            "--output-external-calibration",
            str(output_ext),
            "--output-segment-decoder",
            str(output_seg),
        ],
    )
    tune_realtime.main()
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["segment_decoder_disabled"] is False
    assert output_seg.exists()


def test_tune_realtime_writes_specialist_reports_and_flags(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "best_kws12.pt"
    checkpoint_path.write_text("stub", encoding="utf-8")
    profile = realtime.ResolvedRealtimeProfile(
        demo_profile="accuracy-first",
        detector_device_preference="mps",
        runtime_label_backend="external-ensemble",
        external_kws_model="ensemble/ast-superb-kws12",
        external_kws_device="mps",
    )
    bundle = realtime.LoadedRealtimeDemo(
        checkpoint_path=checkpoint_path,
        checkpoint_payload={},
        runtime_device=torch.device("cpu"),
        selected_device_label="cpu",
        model=torch.nn.Identity(),
        frontend=SimpleNamespace(),
        command31_labels=[],
        wheel="kws12",
        keyword_calibration={"defaults": {"command_conf_threshold": 0.35}, "keywords": {}},
        keyword_calibration_path=tmp_path / "keyword_calibration.json",
        external_ensemble_calibration={"version": 1, "defaults": {"unknown_superb_weight": 1.2}, "per_label_bias": {}},
        external_ensemble_calibration_path=tmp_path / "external.json",
        segment_decoder=None,
        segment_decoder_path=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_path=None,
        realtime_specialist_calibration={},
        realtime_specialist_calibration_path=None,
        sample_rate=16_000,
        clip_samples=16_000,
        audio_seconds=1.0,
        verifier=None,
        resolved_profile=profile,
    )
    monkeypatch.setattr(tune_realtime, "load_realtime_demo", lambda **kwargs: bundle)
    monkeypatch.setattr(tune_realtime, "_load_seed_report", lambda args: _base_seed_report())
    monkeypatch.setattr(tune_realtime, "_manifest_records", lambda manifest_path, limit_per_class=0: [])

    valid_manifest_path = tmp_path / "valid_manifest.json"
    train_manifest_path = tmp_path / "train_manifest.json"
    test_manifest_path = tmp_path / "test_manifest.json"
    dummy_entry = {"entries": [{"trace_path": str(tmp_path / "dummy.npz"), "target_kws12": 8}]}
    for path in (valid_manifest_path, train_manifest_path, test_manifest_path):
        path.write_text(json.dumps(dummy_entry), encoding="utf-8")

    monkeypatch.setattr(tune_realtime, "_load_trace_manifest", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(
        tune_realtime,
        "_collect_segment_samples",
        lambda manifest, bundle, args, tuning, keyword_calibration, external_calibration: (
            np.zeros((0, tune_realtime.SEGMENT_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        tune_realtime,
        "_collect_manifest_specialist_samples",
        lambda split, clip_samples, other_target_quota, unknown_quota, silence_quota: (
            np.stack(
                [
                    np.full((16000,), 0.10, dtype=np.float32),
                    np.full((16000,), 0.20, dtype=np.float32),
                ],
                axis=0,
            ),
            np.asarray([0, 4], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        tune_realtime,
        "_collect_specialist_samples",
        lambda manifest, bundle, args: (
            np.stack(
                [
                    np.full((16000,), 0.15, dtype=np.float32),
                    np.full((16000,), 0.25, dtype=np.float32),
                ],
                axis=0,
            ),
            np.asarray([0, 4], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        tune_realtime,
        "train_realtime_specialist",
        lambda **kwargs: (torch.nn.Identity(), np.zeros((64,), dtype=np.float32), np.ones((64,), dtype=np.float32)),
    )
    monkeypatch.setattr(
        tune_realtime,
        "save_realtime_specialist_artifact",
        lambda path, **kwargs: Path(path).write_text("stub", encoding="utf-8"),
    )
    monkeypatch.setattr(
        tune_realtime,
        "load_realtime_specialist_artifact",
        lambda path, device=None: SimpleNamespace(path=Path(path), device=device),
    )
    monkeypatch.setattr(
        tune_realtime,
        "_predict_probs",
        lambda *args, **kwargs: np.asarray(
            [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.05, 0.05, 0.05, 0.05, 0.80],
                [0.88, 0.03, 0.03, 0.03, 0.03],
                [0.04, 0.04, 0.04, 0.04, 0.84],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        tune_realtime,
        "_calibrate_from_valid_logits",
        lambda probs, labels: {
            "enabled": True,
            "default": {"accept_prob": 0.55, "min_margin": 0.02, "trigger_prob": 0.16, "role": "rescue"},
            "per_label": {
                "on": {"accept_prob": 0.52, "min_margin": 0.01, "trigger_prob": 0.14, "role": "rescue"},
                "off": {"accept_prob": 0.56, "min_margin": 0.02, "trigger_prob": 0.16, "role": "rescue"},
                "go": {"accept_prob": 0.56, "min_margin": 0.02, "trigger_prob": 0.16, "role": "rescue"},
                "up": {"accept_prob": 0.72, "min_margin": 0.10, "trigger_prob": 0.22, "role": "guard"},
            },
        },
    )

    def _fake_evaluate_trace_manifest(*, bundle, args, manifest, entries_override=None):  # noqa: ARG001
        passed = bundle.realtime_specialist is not None and bool(bundle.realtime_specialist_calibration.get("enabled", False))
        per_class = _base_seed_report()["per_class_kws12"]
        if passed:
            per_class = {
                **per_class,
                "on": {"precision": 0.97, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 6}]},
                "off": {"precision": 0.98, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 5}]},
                "go": {"precision": 0.98, "recall": 0.96, "top_confusions": [{"label": "unknown", "count": 4}]},
                "up": {"precision": 0.96, "recall": 0.95, "top_confusions": [{"label": "unknown", "count": 4}]},
            }
        return {
            "split": args.split,
            "per_class_kws12": per_class,
            "min_kws12_precision": 0.96 if passed else 0.82,
            "min_kws12_recall": 0.95 if passed else 0.78,
            "unknown_to_target_rate": 0.01,
            "no_match_rate": 0.4,
            "passed": passed,
        }

    monkeypatch.setattr(tune_realtime, "evaluate_trace_manifest", _fake_evaluate_trace_manifest)

    output_report = tmp_path / "out_valid.json"
    output_test = tmp_path / "out_test.json"
    output_valid_specialist = tmp_path / "specialist_valid.json"
    output_test_specialist = tmp_path / "specialist_test.json"
    output_specialist = tmp_path / "realtime_specialist.pt"
    output_specialist_cal = tmp_path / "realtime_specialist_calibration.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_realtime",
            "--trace-manifest",
            str(valid_manifest_path),
            "--train-trace-manifest",
            str(train_manifest_path),
            "--test-trace-manifest",
            str(test_manifest_path),
            "--output-report",
            str(output_report),
            "--output-test-report",
            str(output_test),
            "--output-realtime-specialist",
            str(output_specialist),
            "--output-realtime-specialist-calibration",
            str(output_specialist_cal),
            "--output-realtime-specialist-valid-report",
            str(output_valid_specialist),
            "--output-realtime-specialist-test-report",
            str(output_test_specialist),
        ],
    )

    tune_realtime.main()

    assert output_specialist.exists()
    assert output_specialist_cal.exists()
    assert output_valid_specialist.exists()
    assert output_test_specialist.exists()
    valid_payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert valid_payload["selected_candidate_specialist_enabled"] is True
    assert any(item["specialist_enabled"] for item in valid_payload["focused_candidate_summaries"])
    specialist_valid_payload = json.loads(output_valid_specialist.read_text(encoding="utf-8"))
    specialist_test_payload = json.loads(output_test_specialist.read_text(encoding="utf-8"))
    assert "hard_word_macro_f1" in specialist_valid_payload
    assert "hard_word_macro_f1" in specialist_test_payload
