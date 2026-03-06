from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from kws.data.hi_mia import build_himia_manifests


def _write_wav(path: Path, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.randn(1, sr) * 0.01
    torchaudio.save(path, wav, sr)


def test_himia_manifest_builder(tmp_path: Path) -> None:
    root = tmp_path / "hi_mia"
    _write_wav(root / "train" / "SPEECHDATA" / "wav" / "A" / "a.wav")
    _write_wav(root / "dev" / "SPEECHDATA" / "wav" / "B" / "b.wav")
    _write_wav(root / "test" / "SPEECHDATA" / "wav" / "C" / "c.wav")

    manifests = build_himia_manifests(root, tmp_path / "manifests")
    assert len(manifests["train"]) == 1
    assert len(manifests["valid"]) == 1
    assert len(manifests["test"]) == 1
    assert manifests["train"][0].wake_label == 1
    assert manifests["train"][0].command_label is None
