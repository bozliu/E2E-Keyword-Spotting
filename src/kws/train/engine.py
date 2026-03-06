"""Training engine for dual-task KWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from kws.constants import COMMAND31_TO_INDEX
from kws.constants import COMMAND31_LABELS, IGNORE_INDEX, KWS12_LABELS, command31_to_kws12
from kws.losses.confusion import confusion_aware_embedding_loss
from kws.losses.prototype import prototype_cosine_loss
from kws.train.metrics import (
    compute_command_metrics,
    compute_keyword_breakdown,
    compute_kws12_accuracy,
    compute_kws12_breakdown,
    compute_wake_metrics,
)
from kws.train.teacher import TeacherHeads, WavLMFeatureCache


_COMMAND31_TO_KWS12 = torch.tensor([command31_to_kws12(label) for label in COMMAND31_LABELS], dtype=torch.long)
_KWS12_GROUPS = tuple(
    torch.tensor([idx for idx, label in enumerate(COMMAND31_LABELS) if command31_to_kws12(label) == kws_idx], dtype=torch.long)
    for kws_idx in range(len(KWS12_LABELS))
)


def _safe_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    valid = targets != ignore_index
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def _build_keyword_weight_vector(
    *,
    keyword_ce_weights: Mapping[str, float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if not keyword_ce_weights:
        return None
    weights = torch.ones(len(COMMAND31_LABELS), dtype=torch.float32, device=device)
    for keyword, value in keyword_ce_weights.items():
        idx = COMMAND31_TO_INDEX.get(str(keyword))
        if idx is None:
            continue
        weights[idx] = float(max(0.1, value))
    return weights


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    valid = targets != ignore_index
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    if class_weights is None:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return F.cross_entropy(logits, targets, weight=class_weights, ignore_index=ignore_index)


def _distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    temp = float(max(1e-3, temperature))
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float | object]


def pick_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    lambda_command: float,
    lambda_kws12: float,
    lambda_wake: float,
    lambda_aux: float,
    lambda_confusion: float,
    aux_margin: float,
    audio_seconds: float,
    confusion_margin: float = 0.20,
    teacher_cache: Optional[WavLMFeatureCache] = None,
    teacher_heads: Optional[TeacherHeads] = None,
    lambda_distill_logits: float = 0.0,
    lambda_distill_embed: float = 0.0,
    keyword_ce_weights: Mapping[str, float] | None = None,
    confusion_groups: Mapping[str, Sequence[str]] | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(mode=training)
    if teacher_heads is not None:
        teacher_heads.train(mode=training)

    total_loss = 0.0
    total_items = 0

    all_command_preds = []
    all_command_targets = []
    all_wake_scores = []
    all_wake_targets = []

    iterator = tqdm(loader, leave=False)
    class_weights = _build_keyword_weight_vector(keyword_ce_weights=keyword_ce_weights, device=device)

    def _map_command_targets_to_kws12(command_targets: torch.Tensor) -> torch.Tensor:
        mapped = command_targets.clone()
        valid = mapped != IGNORE_INDEX
        if valid.any():
            lookup = _COMMAND31_TO_KWS12.to(command_targets.device)
            mapped[valid] = lookup[command_targets[valid]]
        return mapped

    def _aggregate_command_logits_to_kws12(command_logits: torch.Tensor) -> torch.Tensor:
        grouped = []
        for group in _KWS12_GROUPS:
            idx = group.to(command_logits.device)
            grouped.append(torch.logsumexp(command_logits.index_select(1, idx), dim=1))
        return torch.stack(grouped, dim=1)

    for batch in iterator:
        features = batch.features.to(device)
        command_targets = batch.command_labels.to(device)
        wake_targets = batch.wake_labels.to(device)
        kws12_targets = _map_command_targets_to_kws12(command_targets)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(features)
            command_loss = _weighted_cross_entropy(
                output.command_logits,
                command_targets,
                class_weights=class_weights,
                ignore_index=IGNORE_INDEX,
            )
            kws12_logits = _aggregate_command_logits_to_kws12(output.command_logits)
            kws12_loss = _safe_cross_entropy(kws12_logits, kws12_targets, ignore_index=IGNORE_INDEX)
            wake_loss = F.binary_cross_entropy_with_logits(output.wake_logits, wake_targets)
            aux_loss = prototype_cosine_loss(output.embedding, command_targets, margin=aux_margin)
            confusion_loss = confusion_aware_embedding_loss(
                output.embedding,
                command_targets,
                confusion_groups=confusion_groups or {},
                margin=confusion_margin,
            )
            distill_logits_loss = output.embedding.new_tensor(0.0)
            distill_embed_loss = output.embedding.new_tensor(0.0)
            teacher_supervision_loss = output.embedding.new_tensor(0.0)

            if training and teacher_cache is not None and teacher_heads is not None:
                teacher_features = teacher_cache.load_features(batch.paths, device=device)
                teacher_targets = teacher_heads(teacher_features)
                teacher_supervision_loss = (
                    _safe_cross_entropy(teacher_targets.command_logits, command_targets, ignore_index=IGNORE_INDEX)
                    + _safe_cross_entropy(teacher_targets.kws12_logits, kws12_targets, ignore_index=IGNORE_INDEX)
                )
                distill_logits_loss = 0.5 * (
                    _distill_kl(output.command_logits, teacher_targets.command_logits)
                    + _distill_kl(kws12_logits, teacher_targets.kws12_logits)
                )
                student_embed = F.normalize(output.embedding, dim=-1)
                teacher_embed = F.normalize(teacher_targets.projected_embedding.detach(), dim=-1)
                distill_embed_loss = F.mse_loss(student_embed, teacher_embed)

            loss = (
                lambda_command * command_loss
                + lambda_kws12 * kws12_loss
                + lambda_wake * wake_loss
                + lambda_aux * aux_loss
                + lambda_confusion * confusion_loss
                + lambda_distill_logits * distill_logits_loss
                + lambda_distill_embed * distill_embed_loss
                + 0.5 * teacher_supervision_loss
            )

            if training:
                loss.backward()
                optimizer.step()

        bs = features.size(0)
        total_items += bs
        total_loss += float(loss.detach().cpu().item()) * bs

        command_preds = output.command_logits.detach().argmax(dim=1).cpu().numpy()
        command_targs = command_targets.detach().cpu().numpy()
        wake_scores = torch.sigmoid(output.wake_logits.detach()).cpu().numpy()
        wake_targs = wake_targets.detach().cpu().numpy()

        all_command_preds.append(command_preds)
        all_command_targets.append(command_targs)
        all_wake_scores.append(wake_scores)
        all_wake_targets.append(wake_targs)

        iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    if total_items == 0:
        return EpochResult(loss=0.0, metrics={})

    command_preds = np.concatenate(all_command_preds)
    command_targets = np.concatenate(all_command_targets)
    wake_scores = np.concatenate(all_wake_scores)
    wake_targets = np.concatenate(all_wake_targets)

    command_metrics = compute_command_metrics(command_preds, command_targets)
    kws12_acc = compute_kws12_accuracy(command_preds, command_targets)
    kws12_breakdown = compute_kws12_breakdown(command_preds, command_targets)
    keyword_breakdown = compute_keyword_breakdown(command_preds, command_targets)
    wake_metrics = compute_wake_metrics(wake_scores, wake_targets, audio_seconds=audio_seconds)

    metrics: Dict[str, float | object] = {
        "kws12_acc": kws12_acc,
        **command_metrics,
        **kws12_breakdown,
        **keyword_breakdown,
        **wake_metrics,
    }
    return EpochResult(loss=total_loss / total_items, metrics=metrics)
