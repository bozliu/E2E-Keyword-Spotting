"""Prepare a Speech Commands split directory from the official dataset lists."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Set

from kws.constants import SPEECH_COMMANDS_30


SplitMode = Literal["symlink", "copy"]


@dataclass
class SplitStats:
    train: int = 0
    valid: int = 0
    test: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def _read_list(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _iter_command_wavs(dataset_root: Path) -> Iterable[tuple[str, Path, str]]:
    for class_name in SPEECH_COMMANDS_30:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            relative = wav_path.relative_to(dataset_root).as_posix()
            yield class_name, wav_path, relative


def _materialize(source: Path, target: Path, mode: SplitMode) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return
    os.symlink(source, target)


def prepare_speech_commands_split(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    mode: SplitMode = "symlink",
    overwrite: bool = False,
) -> Dict[str, int]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root {output_path} already exists and is not empty. "
            "Pass overwrite=True or remove it first."
        )
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_set = _read_list(dataset_path / "validation_list.txt")
    test_set = _read_list(dataset_path / "testing_list.txt")

    stats = SplitStats()
    for class_name, wav_path, relative in _iter_command_wavs(dataset_path):
        if relative in test_set:
            split = "test"
            stats.test += 1
        elif relative in valid_set:
            split = "valid"
            stats.valid += 1
        else:
            split = "train"
            stats.train += 1
        _materialize(wav_path, output_path / split / class_name / wav_path.name, mode=mode)

    background_dir = dataset_path / "_background_noise_"
    if background_dir.exists():
        for wav_path in sorted(background_dir.glob("*.wav")):
            _materialize(wav_path, output_path / "train" / "_background_noise_" / wav_path.name, mode=mode)

    return stats.as_dict()
