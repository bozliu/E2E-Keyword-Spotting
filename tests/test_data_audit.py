from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torchaudio

from kws.constants import COMMAND31_TO_INDEX, SILENCE_LABEL
from kws.data.audit import audit_manifests
from kws.data.local_speech_commands import build_local_manifests
from kws.data.manifest import ManifestRecord, write_manifest


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def test_audit_manifests_reports_counts_for_generated_local_manifests(tmp_path: Path) -> None:
    root = tmp_path / "speech_commands_split"
    for split in ("train", "valid", "test"):
        for cls in ("yes", "no"):
            _write_wav(root / split / cls / f"{cls}_{split}.wav")
    _write_wav(root / "train" / "_background_noise_" / "bg.wav", seconds=3.0)

    build_local_manifests(root=root, output_dir=tmp_path / "manifests", silence_ratio=0.2)
    report = audit_manifests(tmp_path / "manifests")

    assert report["is_clean"] is True
    assert report["aggregate"]["record_count"] == 9
    assert report["aggregate"]["missing_file_count"] == 0
    assert report["aggregate"]["source_counts"]["local_speech_commands"] == 6
    assert report["aggregate"]["source_counts"]["local_silence"] == 3


def test_audit_manifests_flags_missing_audio_and_missing_manifest(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    existing_audio = tmp_path / "audio" / "ok.wav"
    _write_wav(existing_audio)
    write_manifest(
        [
            ManifestRecord(
                path=str(existing_audio.resolve()),
                source="local_speech_commands",
                split="train",
                command_label=COMMAND31_TO_INDEX["yes"],
                wake_label=1,
                sr=16000,
            ),
            ManifestRecord(
                path=str((tmp_path / "audio" / "missing.wav").resolve()),
                source="local_silence",
                split="train",
                command_label=COMMAND31_TO_INDEX[SILENCE_LABEL],
                wake_label=0,
                sr=16000,
            ),
        ],
        manifests_dir / "sample.jsonl",
    )

    report = audit_manifests(
        manifests_dir,
        manifest_names=["sample.jsonl", "missing.jsonl"],
        sample_missing_limit=5,
    )

    assert report["is_clean"] is False
    assert len(report["missing_manifest_files"]) == 1
    assert report["aggregate"]["record_count"] == 2
    assert report["aggregate"]["missing_file_count"] == 1
    assert report["manifests"]["sample.jsonl"]["missing_file_count"] == 1
    assert report["manifests"]["sample.jsonl"]["source_counts"] == {
        "local_silence": 1,
        "local_speech_commands": 1,
    }
