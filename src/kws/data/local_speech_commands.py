"""Local Google Speech Commands split adapter."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from kws.constants import (
    COMMAND31_TO_INDEX,
    SAMPLE_RATE,
    SILENCE_LABEL,
    SPEECH_COMMANDS_30,
    TARGET_KEYWORDS_10,
)
from kws.data.manifest import ManifestRecord, write_manifest


def _iter_wavs(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*.wav")):
        if path.is_file():
            yield path


def build_local_manifests(
    root: str | Path,
    output_dir: str | Path,
    silence_ratio: float = 0.10,
    limit_per_class: int | None = None,
) -> Dict[str, List[ManifestRecord]]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    background_dir = root_path / "train" / "_background_noise_"
    if not background_dir.exists():
        raise FileNotFoundError(f"Missing background noise directory: {background_dir}")
    background_noise_files = sorted(str(p) for p in background_dir.glob("*.wav"))
    if not background_noise_files:
        raise FileNotFoundError(f"No .wav files in {background_dir}")

    manifests: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        split_dir = root_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        class_counts: Dict[str, int] = defaultdict(int)
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name.startswith("_"):
                continue
            if class_name not in SPEECH_COMMANDS_30:
                continue

            for wav_path in _iter_wavs(class_dir):
                if limit_per_class is not None and class_counts[class_name] >= limit_per_class:
                    break
                command_label = COMMAND31_TO_INDEX[class_name]
                wake_label = 1 if class_name in TARGET_KEYWORDS_10 else 0
                manifests[split].append(
                    ManifestRecord(
                        path=str(wav_path.resolve()),
                        source="local_speech_commands",
                        split=split,
                        command_label=command_label,
                        wake_label=wake_label,
                        sr=SAMPLE_RATE,
                    )
                )
                class_counts[class_name] += 1

        silence_count = max(1, int(len(manifests[split]) * silence_ratio))
        for idx in range(silence_count):
            noise_file = background_noise_files[idx % len(background_noise_files)]
            manifests[split].append(
                ManifestRecord(
                    path=noise_file,
                    source="local_silence",
                    split=split,
                    command_label=COMMAND31_TO_INDEX[SILENCE_LABEL],
                    wake_label=0,
                    sr=SAMPLE_RATE,
                )
            )

        write_manifest(manifests[split], output_path / f"local_{split}.jsonl")

    return manifests


def detect_optional_test_mirror(local_test_manifest: Sequence[ManifestRecord], mirror_root: str | Path) -> Dict[str, int]:
    """Check duplication against an optional secondary mirror folder if present."""
    mirror_path = Path(mirror_root).expanduser().resolve()
    if not mirror_path.exists():
        return {"mirror_files": 0, "overlap": 0}

    local_keys = {
        (Path(rec.path).parent.name, Path(rec.path).name)
        for rec in local_test_manifest
        if rec.source == "local_speech_commands"
    }

    mirror_files = 0
    overlap = 0
    for class_dir in mirror_path.iterdir():
        if not class_dir.is_dir():
            continue
        for wav_path in class_dir.glob("*.wav"):
            mirror_files += 1
            key = (class_dir.name, wav_path.name)
            if key in local_keys:
                overlap += 1

    return {"mirror_files": mirror_files, "overlap": overlap}
