from __future__ import annotations

import math
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from kws.data.manifest import read_manifest
from kws.data.speech_commands import restore_speech_commands_dataset


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def _build_speech_commands_archive(tmp_path: Path) -> Path:
    raw_root = tmp_path / "speech_commands_raw"
    _write_wav(raw_root / "yes" / "yes_train.wav")
    _write_wav(raw_root / "yes" / "yes_valid.wav")
    _write_wav(raw_root / "no" / "no_test.wav")
    _write_wav(raw_root / "cat" / "cat_train.wav")
    _write_wav(raw_root / "_background_noise_" / "bg.wav", seconds=3.0)
    (raw_root / "validation_list.txt").write_text("yes/yes_valid.wav\n", encoding="utf-8")
    (raw_root / "testing_list.txt").write_text("no/no_test.wav\n", encoding="utf-8")

    archive_path = tmp_path / "speech_commands_fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as handle:
        for path in sorted(raw_root.rglob("*")):
            handle.add(path, arcname=path.relative_to(raw_root))
    return archive_path


def test_restore_speech_commands_dataset_rebuilds_official_split_and_manifests(tmp_path: Path) -> None:
    archive_path = _build_speech_commands_archive(tmp_path)
    split_root = tmp_path / "speech_commands_split"
    manifests_dir = tmp_path / "manifests"

    restored = restore_speech_commands_dataset(
        split_root=split_root,
        manifests_dir=manifests_dir,
        archive_path=archive_path,
        version="v1",
        silence_ratio=0.2,
        force=True,
    )

    assert (split_root / "train" / "yes" / "yes_train.wav").exists()
    assert (split_root / "train" / "cat" / "cat_train.wav").exists()
    assert (split_root / "valid" / "yes" / "yes_valid.wav").exists()
    assert (split_root / "test" / "no" / "no_test.wav").exists()
    assert (split_root / "train" / "_background_noise_" / "bg.wav").exists()
    assert not (split_root / "valid" / "_background_noise_").exists()
    assert not (split_root / "test" / "_background_noise_").exists()

    stats = restored["stats"]
    assert stats["audio_files"] == {"train": 2, "valid": 1, "test": 1}
    assert stats["background_noise_files"] == 1
    assert stats["manifest_records"] == {"train": 3, "valid": 2, "test": 2}
    assert (manifests_dir / "speech_commands_stats.json").exists()
    assert (manifests_dir / "local_train.jsonl").exists()
    assert (manifests_dir / "local_valid.jsonl").exists()
    assert (manifests_dir / "local_test.jsonl").exists()

    train_records = read_manifest(manifests_dir / "local_train.jsonl")
    train_sources = {record.source for record in train_records}
    assert "local_speech_commands" in train_sources
    assert "local_silence" in train_sources
