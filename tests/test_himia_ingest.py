from __future__ import annotations

import os
from pathlib import Path

import json
import torch
import torchaudio

from kws.data.hi_mia import (
    HI_MIA_DOWNLOAD_MAX_WORKERS,
    build_himia_manifests,
    build_himia_manifests_with_status,
    download_hi_mia,
)


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


def test_himia_manifest_builder_writes_status(tmp_path: Path) -> None:
    root = tmp_path / "hi_mia"
    _write_wav(root / "train" / "SPEECHDATA" / "wav" / "A" / "a.wav")
    _write_wav(root / "dev" / "SPEECHDATA" / "wav" / "B" / "b.wav")
    _write_wav(root / "test" / "SPEECHDATA" / "wav" / "C" / "c.wav")

    manifests, status = build_himia_manifests_with_status(root, tmp_path / "manifests")

    assert len(manifests["train"]) == 1
    assert status["reduced_data_mode"] is False
    assert status["source_file_counts"]["train"] == 1
    status_path = tmp_path / "manifests" / "hi_mia_status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["present_source_splits"] == ["train", "dev", "test"]


def test_download_hi_mia_uses_repo_local_xet_cache(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    def _fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / "cache")

    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr("kws.data.hi_mia.snapshot_download", _fake_snapshot_download)

    target = download_hi_mia(tmp_path / "hi_mia")

    expected_cache = (target / ".cache" / "hf_xet").resolve()
    assert os.environ["HF_XET_CACHE"] == str(expected_cache)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert calls["local_dir"] == str(target)
    assert calls["max_workers"] == HI_MIA_DOWNLOAD_MAX_WORKERS
