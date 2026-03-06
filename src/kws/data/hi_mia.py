"""HI-MIA dataset integration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import snapshot_download

from kws.constants import SAMPLE_RATE
from kws.data.manifest import ManifestRecord, write_manifest

HI_MIA_REPO_ID = "AISHELL/HI-MIA"


def download_hi_mia(output_dir: str | Path) -> Path:
    """Download HI-MIA from Hugging Face dataset repo."""
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HI_MIA_REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        max_workers=4,
    )
    return target


def _iter_wavs(root: Path, split_folder: str) -> Iterable[Path]:
    split_root = root / split_folder
    if not split_root.exists():
        return []
    return sorted(split_root.rglob("*.wav"))


def build_himia_manifests(
    root: str | Path,
    output_dir: str | Path,
    limit_per_split: int | None = None,
) -> Dict[str, List[ManifestRecord]]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "train",
        "dev": "valid",
        "test": "test",
        "test_v2": "test",
    }

    manifests: Dict[str, List[ManifestRecord]] = defaultdict(list)

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
            # Rebuild all splits from one pool to avoid split leakage.
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

    for split, recs in manifests.items():
        write_manifest(recs, output_path / f"hi_mia_{split}.jsonl")

    return manifests
