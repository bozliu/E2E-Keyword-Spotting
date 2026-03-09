"""Speech Commands download and official split restoration helpers."""

from __future__ import annotations

import json
import subprocess
import shutil
import tarfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

from kws.constants import SPEECH_COMMANDS_30
from kws.data.local_speech_commands import build_local_manifests

SPEECH_COMMANDS_VERSION_INFO: Mapping[str, Mapping[str, str]] = {
    "v1": {
        "archive_name": "speech_commands_v0.01.tar.gz",
        "url": "https://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
    },
    "v2": {
        "archive_name": "speech_commands_v0.02.tar.gz",
        "url": "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    },
}


def _strip_archive_suffix(name: str) -> str:
    if name.endswith(".tar.gz"):
        return name[:-7]
    if name.endswith(".tgz"):
        return name[:-4]
    return Path(name).stem


def _safe_extract(archive: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in archive.getmembers():
        member_path = (target_root / member.name).resolve()
        if not member_path.is_relative_to(target_root):
            raise ValueError(f"Refusing to extract outside target dir: {member.name}")
    archive.extractall(target_root, filter="data")


def _resolve_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    split_files = (root / "validation_list.txt", root / "testing_list.txt")
    if all(p.exists() for p in split_files):
        return root

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        nested_split_files = (child / "validation_list.txt", child / "testing_list.txt")
        if all(p.exists() for p in nested_split_files):
            return child

    raise FileNotFoundError(f"Could not locate Speech Commands root under {root}")


def _download_with_curl(url: str, archive: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for the Speech Commands download fallback")
    archive.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [curl, "-L", "--fail", "--retry", "3", "-o", str(archive), url],
        [curl, "-L", "--fail", "--retry", "3", "--insecure", "-o", str(archive), url],
    ]
    last_error: subprocess.CalledProcessError | None = None
    for command in commands:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            archive.unlink(missing_ok=True)
    if last_error is None:
        raise RuntimeError(f"Speech Commands download fallback failed for {url}")
    raise RuntimeError(
        f"Speech Commands download fallback failed for {url}: {last_error.stderr.strip()}"
    ) from last_error


def _read_split_list(path: str | Path) -> set[str]:
    items: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rel_path = line.strip().replace("\\", "/")
            if rel_path:
                items.add(rel_path)
    return items


def download_speech_commands_archive(
    output_dir: str | Path,
    *,
    version: str = "v1",
    archive_path: str | Path | None = None,
) -> Path:
    """Download Speech Commands archive unless a local archive is provided."""
    if version not in SPEECH_COMMANDS_VERSION_INFO:
        raise ValueError(f"Unsupported Speech Commands version: {version}")

    if archive_path is not None:
        archive = Path(archive_path).expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Speech Commands archive not found: {archive}")
        return archive

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    info = SPEECH_COMMANDS_VERSION_INFO[version]
    archive = target_dir / info["archive_name"]
    if not archive.exists():
        try:
            urllib.request.urlretrieve(info["url"], archive)
        except urllib.error.URLError:
            archive.unlink(missing_ok=True)
            _download_with_curl(info["url"], archive)
    return archive


def extract_speech_commands_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Extract Speech Commands archive and return the dataset root."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Speech Commands archive not found: {archive}")

    target_dir = Path(output_dir).expanduser().resolve() / _strip_archive_suffix(archive.name)
    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:*") as handle:
            _safe_extract(handle, target_dir)

    return _resolve_dataset_root(target_dir)


def materialize_speech_commands_split(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    force: bool = True,
    labels: Iterable[str] = SPEECH_COMMANDS_30,
) -> Dict[str, object]:
    """Copy Speech Commands files into train/valid/test using the official split lists."""
    dataset_root = _resolve_dataset_root(source_root)
    target_root = Path(output_dir).expanduser().resolve()
    label_set = set(labels)

    if force:
        for split in ("train", "valid", "test"):
            shutil.rmtree(target_root / split, ignore_errors=True)

    valid_entries = _read_split_list(dataset_root / "validation_list.txt")
    test_entries = _read_split_list(dataset_root / "testing_list.txt")
    overlap = valid_entries & test_entries
    if overlap:
        raise ValueError(f"Split files overlap for {len(overlap)} entries")

    split_counts: Counter[str] = Counter()
    class_counts: Dict[str, Counter[str]] = {split: Counter() for split in ("train", "valid", "test")}
    skipped_labels: Counter[str] = Counter()
    background_noise_files = 0

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))

        if class_name == "_background_noise_":
            for wav_path in wav_files:
                dest = target_root / "train" / class_name / wav_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav_path, dest)
                background_noise_files += 1
            continue

        if class_name not in label_set:
            if wav_files:
                skipped_labels[class_name] += len(wav_files)
            continue

        for wav_path in wav_files:
            rel_path = wav_path.relative_to(dataset_root).as_posix()
            split = "train"
            if rel_path in valid_entries:
                split = "valid"
            elif rel_path in test_entries:
                split = "test"

            dest = target_root / split / class_name / wav_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, dest)
            split_counts[split] += 1
            class_counts[split][class_name] += 1

    if not any(split_counts.values()):
        raise FileNotFoundError(f"No Speech Commands wavs found under {dataset_root}")
    if background_noise_files == 0:
        raise FileNotFoundError(f"Missing _background_noise_ wavs under {dataset_root}")

    return {
        "dataset_root": str(dataset_root),
        "split_root": str(target_root),
        "audio_files": {split: int(split_counts.get(split, 0)) for split in ("train", "valid", "test")},
        "background_noise_files": background_noise_files,
        "class_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in class_counts.items()
        },
        "skipped_labels": dict(sorted(skipped_labels.items())),
    }


def restore_speech_commands_dataset(
    split_root: str | Path,
    manifests_dir: str | Path,
    *,
    version: str = "v1",
    download_dir: str | Path | None = None,
    archive_path: str | Path | None = None,
    silence_ratio: float = 0.10,
    limit_per_class: int | None = None,
    force: bool = True,
) -> Dict[str, object]:
    """Restore Speech Commands into local split folders and regenerate manifests."""
    split_root_path = Path(split_root).expanduser().resolve()
    manifests_root = Path(manifests_dir).expanduser().resolve()
    manifests_root.mkdir(parents=True, exist_ok=True)

    effective_download_dir = (
        Path(download_dir).expanduser().resolve()
        if download_dir is not None
        else (split_root_path.parent / "_downloads").resolve()
    )
    effective_download_dir.mkdir(parents=True, exist_ok=True)

    archive = download_speech_commands_archive(
        effective_download_dir,
        version=version,
        archive_path=archive_path,
    )
    extracted_root = extract_speech_commands_archive(
        archive,
        effective_download_dir / "extracted",
        force=force,
    )
    split_stats = materialize_speech_commands_split(
        extracted_root,
        split_root_path,
        force=force,
    )
    manifests = build_local_manifests(
        root=split_root_path,
        output_dir=manifests_root,
        silence_ratio=silence_ratio,
        limit_per_class=limit_per_class,
    )
    manifest_stats = {split: len(records) for split, records in manifests.items()}

    stats = {
        "version": version,
        "archive_path": str(archive),
        "extracted_root": str(extracted_root),
        "split_root": str(split_root_path),
        "audio_files": split_stats["audio_files"],
        "background_noise_files": split_stats["background_noise_files"],
        "class_counts": split_stats["class_counts"],
        "manifest_records": manifest_stats,
        "manifests_dir": str(manifests_root),
    }
    with (manifests_root / "speech_commands_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    return {
        "stats": stats,
        "manifests": manifests,
    }
