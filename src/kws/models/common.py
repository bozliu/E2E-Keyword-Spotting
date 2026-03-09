"""Common model interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DualTaskOutput:
    command_logits: torch.Tensor
    wake_logits: torch.Tensor
    embedding: torch.Tensor


@dataclass
class VerifierOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
