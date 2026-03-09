"""HI-MIA dataset integration."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download

from kws.constants import SAMPLE_RATE
from kws.data.manifest import ManifestRecord, write_manifest

HI_MIA_REPO_ID = "AISHELL/HI-MIA"
HI_MIA_REQUIRED_SOURCE_SPLITS: Tuple[str, ...] = ("train", "dev", "test")
HI_MIA_OPTIONAL_SOURCE_SPLITS: Tuple[str, ...] = ("test_v2",)
HI_MIA_ALL_SOURCE_SPLITS: Tuple[str, ...] = HI_MIA_REQUIRED_SOURCE_SPLITS + HI_MIA_OPTIONAL_SOURCE_SPLITS
HI_MIA_DOWNLOAD_MAX_WORKERS = 16


def _configure_hf_xet_cache(target: Path) -> Path:
    cache_dir = (target / ".cache" / "hf_xet").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(cache_dir)
    hf_constants.HF_XET_CACHE = str(cache_dir)
    return cache_dir


def _disable_hf_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hf_constants.HF_HUB_DISABLE_XET = True


def download_hi_mia(output_dir: str | Path) -> Path:
    """Download HI-MIA from Hugging Face dataset repo."""
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    _configure_hf_xet_cache(target)
    _disable_hf_xet()

    snapshot_download(
        repo_id=HI_MIA_REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        max_workers=HI_MIA_DOWNLOAD_MAX_WORKERS,
    )
    return target


def _iter_wavs(root: Path, split_folder: str) -> Iterable[Path]:
    split_root = root / split_folder
    if not split_root.exists():
        return []
    return sorted(split_root.rglob("*.wav"))


def collect_himia_source_status(root: str | Path) -> Dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    source_file_counts = {
        split: len(list(_iter_wavs(root_path, split)))
        for split in HI_MIA_ALL_SOURCE_SPLITS
    }
    present_source_splits = [
        split for split in HI_MIA_ALL_SOURCE_SPLITS
        if int(source_file_counts.get(split, 0)) > 0
    ]
    reduced_data_mode = any(int(source_file_counts.get(split, 0)) <= 0 for split in HI_MIA_REQUIRED_SOURCE_SPLITS)
    return {
        "root": str(root_path),
        "reduced_data_mode": bool(reduced_data_mode),
        "present_source_splits": present_source_splits,
        "source_file_counts": source_file_counts,
    }


def build_himia_manifests_with_status(
    root: str | Path,
    output_dir: str | Path,
    limit_per_split: int | None = None,
) -> tuple[Dict[str, List[ManifestRecord]], Dict[str, object]]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "train",
        "dev": "valid",
        "test": "test",
        "test_v2": "test",
    }
    source_status = collect_himia_source_status(root_path)
    manifests: Dict[str, List[ManifestRecord]] = defaultdict(list)
    build_mode = "official"
    existing_split_files: Dict[str, List[Path]] = {}
    for src_split, mapped_split in split_mapping.items():
        wavs = _iter_wavs(root_path, src_split)
        if not wavs:
            continue
        existing_split_files[src_split] = list(wavs)
        for idx, wav_path in enumerate(wavs):
            if limit_per_split is not None and idx >= limit_per_split:
                break
            manifests[mapped_split].append(
                ManifestRecord(
                    path=str(wav_path.resolve()),
                    source="hi_mia",
                    split=mapped_split,
                    command_label=None,
                    wake_label=1,
                    sr=SAMPLE_RATE,
                )
            )

    # Fallback: if train/test are missing but one split exists (often partial download),
    # repartition available wavs to keep the training pipeline runnable.
    if not manifests["train"] or not manifests["test"]:
        all_wavs: List[Path] = []
        for wavs in existing_split_files.values():
            all_wavs.extend(wavs)
        all_wavs = sorted(all_wavs)
        if all_wavs:
            manifests = defaultdict(list)
            n = len(all_wavs)
            train_end = int(0.8 * n)
            valid_end = int(0.9 * n)
            repartition = {
                "train": all_wavs[:train_end],
                "valid": all_wavs[train_end:valid_end],
                "test": all_wavs[valid_end:],
            }
            for split_name, wavs in repartition.items():
                for idx, wav_path in enumerate(wavs):
                    if limit_per_split is not None and idx >= limit_per_split:
                        break
                    manifests[split_name].append(
                        ManifestRecord(
                            path=str(wav_path.resolve()),
                            source="hi_mia",
                            split=split_name,
                            command_label=None,
                            wake_label=1,
                            sr=SAMPLE_RATE,
                        )
                    )
            build_mode = "fallback_repartition"
            source_status["reduced_data_mode"] = True

    for split, recs in manifests.items():
        write_manifest(recs, output_path / f"hi_mia_{split}.jsonl")

    status = {
        **source_status,
        "build_mode": build_mode,
        "limit_per_split": int(limit_per_split) if limit_per_split is not None else None,
        "manifest_counts": {split: len(recs) for split, recs in manifests.items()},
    }
    (output_path / "hi_mia_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifests, status


def build_himia_manifests(
    root: str | Path,
    output_dir: str | Path,
    limit_per_split: int | None = None,
) -> Dict[str, List[ManifestRecord]]:
    manifests, _status = build_himia_manifests_with_status(
        root=root,
        output_dir=output_dir,
        limit_per_split=limit_per_split,
    )
    return manifests
