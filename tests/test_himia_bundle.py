from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

from backports import zstd
import numpy as np
import pytest
import torch
import torchaudio

from kws.data.hi_mia import build_himia_manifests_with_status
from kws.data.himia_bundle import (
    HI_MIA_MANIFEST_NAMES,
    import_himia_bundle,
    prepare_himia_bundle,
)


def _write_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = 0.1 * np.sin(2 * math.pi * 220 * t)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


def _make_full_himia_tree(project_root: Path) -> dict:
    data_root = project_root / "data" / "external" / "hi_mia"
    manifests_root = project_root / "data" / "processed" / "manifests"
    reports_root = project_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    _write_wav(data_root / "train" / "SPEECHDATA" / "wav" / "A" / "a.wav")
    _write_wav(data_root / "dev" / "SPEECHDATA" / "wav" / "B" / "b.wav")
    _write_wav(data_root / "test" / "SPEECHDATA" / "wav" / "C" / "c.wav")
    manifests, status = build_himia_manifests_with_status(data_root, manifests_root)
    stats = {
        "manifest_counts": {split: len(records) for split, records in manifests.items()},
        "reduced_data_mode": bool(status.get("reduced_data_mode", False)),
        "present_source_splits": status.get("present_source_splits", []),
        "source_file_counts": status.get("source_file_counts", {}),
        "build_mode": status.get("build_mode", "official"),
    }
    (manifests_root / "hi_mia_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return status


def _make_partial_himia_tree(project_root: Path) -> dict:
    data_root = project_root / "data" / "external" / "hi_mia"
    manifests_root = project_root / "data" / "processed" / "manifests"
    _write_wav(data_root / "dev" / "SPEECHDATA" / "wav" / "B" / "b.wav")
    manifests, status = build_himia_manifests_with_status(data_root, manifests_root)
    stats = {
        "manifest_counts": {split: len(records) for split, records in manifests.items()},
        "reduced_data_mode": bool(status.get("reduced_data_mode", False)),
        "present_source_splits": status.get("present_source_splits", []),
        "source_file_counts": status.get("source_file_counts", {}),
        "build_mode": status.get("build_mode", "fallback_repartition"),
    }
    (manifests_root / "hi_mia_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return status


def test_prepare_himia_bundle_packages_full_restore(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    status = _make_full_himia_tree(project_root)

    manifest = prepare_himia_bundle(
        project_root,
        bundle_path="artifacts/hi_mia_full_bundle.tar.zst",
        manifest_path="artifacts/hi_mia_full_bundle_manifest.json",
        audit_output_path="reports/hi_mia_remote_audit.json",
    )

    assert manifest["audit_is_clean"] is True
    assert manifest["status"]["reduced_data_mode"] is False
    bundle_path = Path(manifest["bundle_path"])
    assert bundle_path.exists()
    assert manifest["bundle_sha256"]

    with zstd.open(bundle_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as tar:
            names = [member.name for member in tar]
    assert any(name.startswith("data/external/hi_mia/train/") for name in names)
    assert any(name.startswith("data/external/hi_mia/dev/") for name in names)
    assert any(name.startswith("data/external/hi_mia/test/") for name in names)
    for name in HI_MIA_MANIFEST_NAMES:
        assert f"data/processed/manifests/{name}" in names
    assert manifest["status"]["present_source_splits"] == status["present_source_splits"]


def test_import_himia_bundle_replaces_partial_restore(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    _make_full_himia_tree(remote_root)
    _make_partial_himia_tree(local_root)

    manifest = prepare_himia_bundle(
        remote_root,
        bundle_path="artifacts/hi_mia_full_bundle.tar.zst",
        manifest_path="artifacts/hi_mia_full_bundle_manifest.json",
        audit_output_path="reports/hi_mia_remote_audit.json",
    )
    result = import_himia_bundle(
        local_root,
        bundle_path=manifest["bundle_path"],
        audit_output_path="reports/hi_mia_import_audit.json",
    )

    imported_status = result["status"]
    assert imported_status["reduced_data_mode"] is False
    assert set(imported_status["present_source_splits"]) >= {"train", "dev", "test"}
    assert (local_root / "data" / "external" / "hi_mia" / "train").exists()
    assert (local_root / "data" / "external" / "hi_mia" / "test").exists()
    assert Path(result["backup_root"]).exists()
    assert (Path(result["backup_root"]) / "data" / "external" / "hi_mia" / "dev").exists()
    audit = json.loads((local_root / "reports" / "hi_mia_import_audit.json").read_text(encoding="utf-8"))
    assert audit["is_clean"] is True


def test_import_himia_bundle_rejects_reduced_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    bundle_path = project_root / "artifacts" / "reduced.tar.zst"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    reduced_status = {
        "root": "/tmp/hi_mia",
        "reduced_data_mode": True,
        "present_source_splits": ["dev"],
        "source_file_counts": {"train": 0, "dev": 1, "test": 0, "test_v2": 0},
        "build_mode": "fallback_repartition",
        "manifest_counts": {"train": 1, "valid": 1, "test": 1},
    }

    with zstd.open(bundle_path, "wb") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for rel_path, payload in {
                "data/processed/manifests/hi_mia_status.json": json.dumps(reduced_status).encode("utf-8"),
                "data/processed/manifests/hi_mia_train.jsonl": b"",
                "data/processed/manifests/hi_mia_valid.jsonl": b"",
                "data/processed/manifests/hi_mia_test.jsonl": b"",
                "data/processed/manifests/hi_mia_stats.json": json.dumps({"reduced_data_mode": True}).encode("utf-8"),
            }.items():
                info = tarfile.TarInfo(name=rel_path)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="not full"):
        import_himia_bundle(project_root, bundle_path=bundle_path)
