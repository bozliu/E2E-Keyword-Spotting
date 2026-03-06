"""Data schema used across data loaders and training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sample:
    audio: torch.Tensor
    sr: int
    command_label: int | None
    wake_label: int | None
    source: str
    path: str
    speaker_id: str | None = None
    transcript: str | None = None
    accent_group: str | None = None
    l1_group: str | None = None
    is_synthetic: bool = False
    difficulty_bucket: str | None = None
