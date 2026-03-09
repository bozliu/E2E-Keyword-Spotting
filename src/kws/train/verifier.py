"""Verifier-specific labels, teacher heads, and loss helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from kws.constants import IGNORE_INDEX, INDEX_TO_COMMAND31, KWS12_LABELS, command31_to_kws12

VERIFIER_REJECT_LABEL = "reject"
VERIFIER_LABELS = [*KWS12_LABELS, VERIFIER_REJECT_LABEL]
VERIFIER_TO_INDEX = {label: idx for idx, label in enumerate(VERIFIER_LABELS)}


@dataclass(frozen=True)
class VerifierTargets:
    labels: torch.Tensor
    reject_mask: torch.Tensor


@dataclass
class VerifierTeacherTargets:
    verifier_logits: torch.Tensor
    projected_embedding: torch.Tensor


class VerifierTeacherHeads(nn.Module):
    """Trainable heads on top of frozen SSL teacher features for verifier distillation."""

    def __init__(
        self,
        *,
        feature_dim: int,
        student_dim: int,
        num_labels: int = len(VERIFIER_LABELS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(128, student_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.verifier_head = nn.Linear(hidden_dim, num_labels)
        self.embedding_head = nn.Linear(hidden_dim, student_dim)

    def forward(self, pooled_features: torch.Tensor) -> VerifierTeacherTargets:
        x = self.backbone(self.norm(pooled_features))
        return VerifierTeacherTargets(
            verifier_logits=self.verifier_head(x),
            projected_embedding=self.embedding_head(x),
        )


def build_verifier_targets(
    command_targets: torch.Tensor,
    *,
    reject_mask: torch.Tensor | None = None,
) -> VerifierTargets:
    labels = torch.full_like(command_targets, fill_value=IGNORE_INDEX)
    valid = command_targets != IGNORE_INDEX
    if valid.any():
        labels_valid = command_targets[valid].detach().cpu().tolist()
        out = []
        for idx in labels_valid:
            out.append(command31_to_kws12(INDEX_TO_COMMAND31[int(idx)]))
        labels[valid] = torch.tensor(out, dtype=labels.dtype, device=labels.device)

    effective_reject = (
        reject_mask.to(device=command_targets.device, dtype=torch.bool)
        if reject_mask is not None
        else torch.zeros_like(command_targets, dtype=torch.bool)
    )
    labels[effective_reject & valid] = VERIFIER_TO_INDEX[VERIFIER_REJECT_LABEL]
    return VerifierTargets(labels=labels, reject_mask=effective_reject & valid)


def verifier_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    return F.cross_entropy(
        logits[valid],
        targets[valid],
        label_smoothing=float(max(0.0, label_smoothing)),
    )


def verifier_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reject_weight: float = 1.5,
    margin: float = 0.15,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    logits_v = logits[valid]
    targets_v = targets[valid]
    true_scores = logits_v.gather(1, targets_v.unsqueeze(1)).squeeze(1)
    masked = logits_v.masked_fill(
        F.one_hot(targets_v, num_classes=logits_v.size(1)).to(dtype=torch.bool),
        float("-inf"),
    )
    rival_scores = masked.max(dim=1).values
    sample_weight = torch.ones_like(true_scores)
    sample_weight = torch.where(
        targets_v == VERIFIER_TO_INDEX[VERIFIER_REJECT_LABEL],
        sample_weight.new_full(sample_weight.shape, float(max(1.0, reject_weight))),
        sample_weight,
    )
    return torch.relu(margin - (true_scores - rival_scores)).mul(sample_weight).mean()


def verifier_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    temp = float(max(1e-3, temperature))
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def load_verifier_checkpoint(path: str | bytes | "os.PathLike[str]") -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Verifier checkpoint payload must be a mapping.")
    return payload
