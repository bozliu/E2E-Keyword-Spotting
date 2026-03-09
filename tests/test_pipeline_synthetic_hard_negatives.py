from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio

from kws.constants import COMMAND31_TO_INDEX
from kws.data.manifest import ManifestRecord, write_manifest
from kws.data.pipeline import create_dataloaders, prepare_data


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def test_pipeline_includes_synthetic_hard_negative_manifests(tmp_path: Path) -> None:
    project_root = tmp_path
    split_root = project_root / "data" / "local" / "speech_commands_split"
    manifests_root = project_root / "data" / "processed" / "manifests"

    for split, label in (("train", "yes"), ("valid", "yes"), ("test", "no")):
        _write_wav(split_root / split / label / f"{label}_{split}.wav")
    _write_wav(split_root / "train" / "_background_noise_" / "bg.wav", seconds=3.0)

    synthetic_train = project_root / "data" / "synthetic" / "hard_negatives" / "train" / "yes" / "confuser.wav"
    synthetic_valid = project_root / "data" / "synthetic" / "hard_negatives" / "valid" / "yes" / "confuser.wav"
    _write_wav(synthetic_train)
    _write_wav(synthetic_valid)
    write_manifest(
        [
            ManifestRecord(
                path=str(synthetic_train.resolve()),
                source="synthetic_hard_negative",
                split="train",
                command_label=COMMAND31_TO_INDEX["bird"],
                wake_label=0,
                sr=16000,
                transcript="bird",
                is_synthetic=True,
                difficulty_bucket="hard_negative",
            )
        ],
        manifests_root / "synthetic_hard_negative_train.jsonl",
    )
    write_manifest(
        [
            ManifestRecord(
                path=str(synthetic_valid.resolve()),
                source="synthetic_hard_negative",
                split="valid",
                command_label=COMMAND31_TO_INDEX["bird"],
                wake_label=0,
                sr=16000,
                transcript="bird",
                is_synthetic=True,
                difficulty_bucket="hard_negative",
            )
        ],
        manifests_root / "synthetic_hard_negative_valid.jsonl",
    )

    cfg = {
        "data": {
            "manifests_dir": "data/processed/manifests",
            "local": {
                "speech_commands_split": "data/local/speech_commands_split",
                "silence_ratio": 0.1,
                "xiaomi_mirror": "",
            },
            "external": {"hi_mia": {"enabled": False}},
            "synthetic": {
                "hard_negatives": {
                    "enabled": True,
                }
            },
        },
        "features": {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 40,
            "f_min": 20.0,
            "f_max": 7600.0,
        },
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "augment": False,
            "keyword_focus": {},
        },
    }

    stats = prepare_data(cfg, project_root)
    dataloaders = create_dataloaders(cfg, project_root)

    assert stats["synthetic_hard_negative"] == {"train": 1, "valid": 1, "test": 0}
    assert dataloaders.stats["with_synthetic_hard_negatives"] is True
    assert len(dataloaders.train.dataset.records) == 3
    assert len(dataloaders.valid.dataset.records) == 3
