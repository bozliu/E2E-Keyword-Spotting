from __future__ import annotations

import math
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from kws.data import download_external


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def _build_archive(tmp_path: Path) -> Path:
    raw_root = tmp_path / "speech_commands_raw"
    _write_wav(raw_root / "yes" / "yes_train.wav")
    _write_wav(raw_root / "yes" / "yes_valid.wav")
    _write_wav(raw_root / "no" / "no_test.wav")
    _write_wav(raw_root / "_background_noise_" / "bg.wav", seconds=3.0)
    (raw_root / "validation_list.txt").write_text("yes/yes_valid.wav\n", encoding="utf-8")
    (raw_root / "testing_list.txt").write_text("no/no_test.wav\n", encoding="utf-8")

    archive_path = tmp_path / "speech_commands_cli_fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as handle:
        for path in sorted(raw_root.rglob("*")):
            handle.add(path, arcname=path.relative_to(raw_root))
    return archive_path


def test_download_external_cli_restores_speech_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    archive_path = _build_archive(tmp_path)
    split_root = tmp_path / "speech_commands_split"
    manifests_dir = tmp_path / "manifests"
    download_dir = tmp_path / "downloads"

    monkeypatch.setattr(download_external, "ensure_repo_import", lambda project_root: Path(project_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_external",
            "--dataset",
            "speech_commands",
            "--root",
            str(split_root),
            "--manifests-dir",
            str(manifests_dir),
            "--download-dir",
            str(download_dir),
            "--archive",
            str(archive_path),
            "--force",
        ],
    )

    download_external.main()
    output = capsys.readouterr().out

    assert "Restored Speech Commands v1" in output
    assert (split_root / "train" / "_background_noise_" / "bg.wav").exists()
    assert (manifests_dir / "local_train.jsonl").exists()
    assert (manifests_dir / "speech_commands_stats.json").exists()
