"""Manifest record helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import json


@dataclass
class ManifestRecord:
    path: str
    source: str
    split: str
    command_label: int | None
    wake_label: int | None
    sr: int
    speaker_id: str | None = None
    transcript: str | None = None
    accent_group: str | None = None
    l1_group: str | None = None
    is_synthetic: bool = False
    difficulty_bucket: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_manifest(records: Iterable[ManifestRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(rec.to_json() + "\n")


def read_manifest(path: str | Path) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(ManifestRecord(**payload))
    return records
