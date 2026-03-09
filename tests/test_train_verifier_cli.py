from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from kws.data.dataset import Batch
from kws.train_verifier import build_batch_reject_mask
from kws import train_verifier as train_verifier_main


class _DummyTeacherCache:
    def __init__(self, **kwargs) -> None:  # noqa: D401, ARG002
        self.feature_dim = 4

    def load_features(self, paths, *, device):
        return torch.ones(len(paths), self.feature_dim, dtype=torch.float32, device=device)


def test_build_batch_reject_mask_marks_configured_hard_negatives() -> None:
    batch = Batch(
        features=torch.randn(3, 80, 24),
        command_labels=torch.tensor([0, 1, 2], dtype=torch.long),
        wake_labels=torch.zeros(3, dtype=torch.float32),
        paths=[
            "data/synthetic/hard_negatives/a.wav",
            "data/local/speech_commands_split/train/yes/example.wav",
            "data/custom/example.wav",
        ],
        sources=["synthetic_tts", "local_sc", "curated"],
        difficulty_buckets=["hard_negative", None, "hard_negative"],
    )

    mask = build_batch_reject_mask(
        batch,
        verifier_cfg={
            "reject_sources": ["synthetic_tts"],
            "reject_difficulty_buckets": ["hard_negative"],
            "reject_path_substrings": ["hard_negatives"],
        },
    )

    assert mask.tolist() == [True, False, True]


def test_train_verifier_main_writes_artifacts_next_to_detector_run(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / "configs"
    detector_dir = workspace / "outputs" / "demo_mhatt_small_focus"
    config_dir.mkdir(parents=True)
    detector_dir.mkdir(parents=True)

    detector_ckpt = detector_dir / "best_kws12.pt"
    torch.save({"model_state": {}, "config": {"model": {"name": "mhatt_crnn"}}}, detector_ckpt)

    config_path = config_dir / "verifier_test.yaml"
    cfg = {
        "seed": 7,
        "run_name": "verifier_test",
        "model": {
            "name": "kws12_verifier",
            "conv_channels": 8,
            "num_blocks": 1,
            "attn_dim": 16,
            "num_heads": 2,
            "dropout": 0.0,
        },
        "features": {
            "sample_rate": 16000,
            "audio_seconds": 1.0,
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 80,
            "f_min": 20.0,
            "f_max": 7600.0,
        },
        "data": {
            "manifests_dir": "data/processed/manifests",
            "local": {
                "speech_commands_split": "data/local/speech_commands_split",
                "silence_ratio": 0.1,
                "xiaomi_mirror": "data/legacy/KWS/XiaoMi/speech_commands",
            },
            "external": {"hi_mia": {"enabled": False}},
        },
        "training": {
            "device": "cpu",
            "output_dir": "outputs",
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "teacher": {
                "enabled": True,
                "model_id": "dummy/wavlm",
                "cache_dir": "cache/test_teacher",
                "dropout": 0.0,
            },
            "verifier": {
                "labels": ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "reject"],
                "reject_label": "reject",
                "margin": 0.1,
                "reject_weight": 1.5,
                "label_smoothing": 0.0,
                "min_accept_prob": 0.6,
                "min_margin": 0.05,
                "source_checkpoint": "outputs/demo_mhatt_small_focus/best_kws12.pt",
                "reject_difficulty_buckets": ["hard_negative"],
                "reject_path_substrings": ["hard_negatives"],
            },
            "loss_weights": {
                "ce": 1.0,
                "margin": 0.1,
                "distill_logits": 0.1,
                "distill_embed": 0.05,
            },
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def _make_batch() -> Batch:
        return Batch(
            features=torch.randn(4, 80, 24),
            command_labels=torch.tensor([0, 2, 3, 2], dtype=torch.long),
            wake_labels=torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            paths=[
                str(workspace / "data" / "local" / "silence.wav"),
                str(workspace / "data" / "local" / "yes.wav"),
                str(workspace / "data" / "synthetic" / "hard_negatives" / "confuser.wav"),
                str(workspace / "data" / "local" / "yes2.wav"),
            ],
            sources=["local_sc", "local_sc", "synthetic_tts", "local_sc"],
            difficulty_buckets=[None, None, "hard_negative", None],
        )

    dataloaders = SimpleNamespace(
        train=[_make_batch()],
        valid=[_make_batch()],
        test=[_make_batch()],
        stats={"train_samples": 4, "valid_samples": 4, "test_samples": 4},
    )

    monkeypatch.setattr(train_verifier_main, "parse_args", lambda: Namespace(config=str(config_path), seed=3))
    monkeypatch.setattr(train_verifier_main, "prepare_data", lambda cfg, project_root: {"prepared": True})
    monkeypatch.setattr(train_verifier_main, "create_dataloaders", lambda cfg, project_root: dataloaders)
    monkeypatch.setattr(train_verifier_main, "pick_device", lambda preferred: torch.device("cpu"))
    monkeypatch.setattr(train_verifier_main, "WavLMFeatureCache", _DummyTeacherCache)
    monkeypatch.setattr(
        train_verifier_main,
        "run_repo_preflight",
        lambda *args, **kwargs: {
            "active_kws_file": str(workspace / "src" / "kws" / "__init__.py"),
            "mps_available": False,
            "teacher_model_id": "dummy/wavlm",
            "is_clean": True,
            "manifest_audit": {
                "is_clean": True,
                "manifests_dir": str(workspace / "data" / "processed" / "manifests"),
                "manifests": {},
                "aggregate": {"record_count": 0, "missing_file_count": 0, "source_counts": {}, "split_counts": {}},
                "missing_manifest_files": [],
            },
        },
    )

    train_verifier_main.main()

    best_path = detector_dir / "best_kws12_verifier.pt"
    last_path = detector_dir / "last_verifier.pt"
    calibration_path = detector_dir / "verifier_calibration.json"
    test_metrics_path = detector_dir / "verifier_test_metrics.json"
    history_path = detector_dir / "verifier_metrics_history.jsonl"
    resolved_config_path = detector_dir / "resolved_verifier_config.json"
    dataset_stats_path = detector_dir / "verifier_dataset_stats.json"
    audit_path = detector_dir / "verifier_data_audit.json"

    assert best_path.exists()
    assert last_path.exists()
    assert calibration_path.exists()
    assert test_metrics_path.exists()
    assert history_path.exists()
    assert resolved_config_path.exists()
    assert dataset_stats_path.exists()
    assert audit_path.exists()

    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    history_rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert payload["source_detector_checkpoint"] == str(detector_ckpt.resolve())
    assert payload["verifier_labels"][-1] == "reject"
    assert "verifier_calibration" in payload
    assert calibration["default"]["min_accept_prob"] >= 0.5
    assert calibration["default"]["min_margin"] >= 0.0
    assert "yes" in calibration["per_label"]
    assert history_rows[0]["valid_metrics"]["verifier_macro_f1"] >= 0.0
    assert test_metrics["metrics"]["verifier_macro_f1"] >= 0.0
