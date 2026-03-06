from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torchaudio

from kws.constants import COMMAND31_TO_INDEX, SILENCE_LABEL
from kws.data.local_speech_commands import build_local_manifests, detect_optional_test_mirror


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def test_local_manifest_and_mirror_detection(tmp_path: Path) -> None:
    root = tmp_path / "speech_commands_split"
    for split in ["train", "valid", "test"]:
        for cls in ["yes", "no", "cat"]:
            _write_wav(root / split / cls / f"{cls}_{split}.wav")
    _write_wav(root / "train" / "_background_noise_" / "bg.wav", seconds=3.0)

    manifests = build_local_manifests(root=root, output_dir=tmp_path / "manifests", silence_ratio=0.2)

    assert len(manifests["train"]) > 0
    assert any(r.command_label == COMMAND31_TO_INDEX[SILENCE_LABEL] for r in manifests["train"])

    mirror = tmp_path / "mirror"
    _write_wav(mirror / "yes" / "yes_test.wav")
    # copy a matching key from test split
    src = root / "test" / "yes" / "yes_test.wav"
    _write_wav(src)

    stats = detect_optional_test_mirror(manifests["test"], mirror)
    assert stats["mirror_files"] >= 1
