from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from kws.demo import realtime, train_realtime_specialist


def test_train_realtime_specialist_cli_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
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
        keyword_calibration={},
        keyword_calibration_path=tmp_path / "keyword_calibration.json",
        external_ensemble_calibration={},
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
    monkeypatch.setattr(train_realtime_specialist, "load_realtime_demo", lambda **kwargs: bundle)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": [{"trace_path": str(tmp_path / "dummy.npz"), "target_kws12": 8}]}), encoding="utf-8")
    monkeypatch.setattr(train_realtime_specialist, "_load_manifest", lambda path: json.loads(manifest_path.read_text(encoding="utf-8")))
    monkeypatch.setattr(
        train_realtime_specialist,
        "_collect_specialist_samples",
        lambda manifest, bundle, args: (
            np.stack(
                [
                    np.full((16000,), 0.05, dtype=np.float32),
                    np.full((16000,), 0.15, dtype=np.float32),
                ],
                axis=0,
            ),
            np.asarray([0, 4], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        train_realtime_specialist,
        "_collect_manifest_specialist_samples",
        lambda split, clip_samples, other_target_quota, unknown_quota, silence_quota: (
            np.stack(
                [
                    np.full((16000,), 0.04, dtype=np.float32),
                    np.full((16000,), 0.18, dtype=np.float32),
                ],
                axis=0,
            ),
            np.asarray([0, 4], dtype=np.int64),
        ),
    )
    specialist_path = tmp_path / "realtime_specialist.pt"
    specialist_cal_path = tmp_path / "realtime_specialist_calibration.json"
    valid_report = tmp_path / "valid.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_realtime_specialist",
            "--train-trace-manifest",
            str(manifest_path),
            "--valid-trace-manifest",
            str(manifest_path),
            "--output-specialist",
            str(specialist_path),
            "--output-calibration",
            str(specialist_cal_path),
            "--output-valid-report",
            str(valid_report),
            "--epochs",
            "2",
            "--batch-size",
            "2",
        ],
    )
    train_realtime_specialist.main()
    assert specialist_path.exists()
    assert specialist_cal_path.exists()
    assert valid_report.exists()
    payload = json.loads(valid_report.read_text(encoding="utf-8"))
    assert "hard_word_macro_f1" in payload
