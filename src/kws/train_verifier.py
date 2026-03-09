"""CLI entrypoint for training the standalone KWS12 verifier."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from tqdm import tqdm

from kws.config import load_yaml
from kws.constants import IGNORE_INDEX
from kws.data.audit import audit_manifests
from kws.data.dataset import Batch
from kws.data.hard_negatives import DEFAULT_SAY_RATES, DEFAULT_SAY_VOICES, generate_hard_negative_dataset
from kws.data.pipeline import create_dataloaders, prepare_data
from kws.env import ensure_repo_import, run_repo_preflight
from kws.models import create_model
from kws.train.engine import pick_device
from kws.train.teacher import WavLMFeatureCache
from kws.train.verifier import (
    VERIFIER_LABELS,
    VERIFIER_REJECT_LABEL,
    VERIFIER_TO_INDEX,
    VerifierTeacherHeads,
    build_verifier_targets,
    verifier_cross_entropy,
    verifier_distillation_loss,
    verifier_margin_loss,
)


@dataclass
class VerifierEpochResult:
    loss: float
    metrics: Dict[str, float | object]
    preds: np.ndarray
    targets: np.ndarray
    top_probs: np.ndarray
    margins: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KWS12 verifier")
    parser.add_argument("--config", type=str, required=True, help="Path to verifier YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_string_set(values: object, *, default: Sequence[str] = ()) -> set[str]:
    if values is None:
        return {str(item).strip().lower() for item in default if str(item).strip()}
    if isinstance(values, (str, bytes)):
        items = [values]
    elif isinstance(values, Sequence):
        items = list(values)
    else:
        items = list(default)
    return {str(item).strip().lower() for item in items if str(item).strip()}


def resolve_source_checkpoint(cfg: Mapping[str, object], project_root: str | Path) -> Path | None:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        return None
    verifier_cfg = training_cfg.get("verifier", {})
    if not isinstance(verifier_cfg, Mapping):
        return None
    raw_path = str(verifier_cfg.get("source_checkpoint", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path(project_root).resolve() / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_relative_path(project_root: str | Path, raw_path: str, default: str = "") -> Path:
    path = Path(raw_path or default).expanduser()
    if not path.is_absolute():
        path = (Path(project_root).resolve() / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_output_dir(cfg: Mapping[str, object], project_root: str | Path) -> Path:
    source_checkpoint = resolve_source_checkpoint(cfg, project_root)
    if source_checkpoint is not None:
        if not source_checkpoint.exists():
            raise FileNotFoundError(f"Verifier source checkpoint not found: {source_checkpoint}")
        output_dir = source_checkpoint.parent
    else:
        training_cfg = cfg.get("training", {})
        if not isinstance(training_cfg, Mapping):
            training_cfg = {}
        run_name = str(cfg.get("run_name") or f"verifier_{time.strftime('%Y%m%d_%H%M%S')}")
        output_dir = (Path(project_root).resolve() / str(training_cfg.get("output_dir", "outputs")) / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def maybe_generate_hard_negatives(cfg: Mapping[str, object], project_root: str | Path) -> Dict[str, object] | None:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        return None
    synthetic_cfg = data_cfg.get("synthetic", {})
    if not isinstance(synthetic_cfg, Mapping):
        return None
    hard_negative_cfg = synthetic_cfg.get("hard_negatives", {})
    if not isinstance(hard_negative_cfg, Mapping) or not bool(hard_negative_cfg.get("enabled", False)):
        return None

    source_checkpoint = resolve_source_checkpoint(cfg, project_root)
    if source_checkpoint is None:
        raw_source = str(hard_negative_cfg.get("source_checkpoint", "")).strip()
        if raw_source:
            source_checkpoint = _resolve_relative_path(project_root, raw_source)
    if source_checkpoint is None:
        raise ValueError("Hard-negative generation requires training.verifier.source_checkpoint or data.synthetic.hard_negatives.source_checkpoint")

    output_dir = _resolve_relative_path(
        project_root,
        str(hard_negative_cfg.get("output_dir", "")),
        "data/synthetic/hard_negatives",
    )
    manifests_dir = _resolve_relative_path(
        project_root,
        str(data_cfg.get("manifests_dir", "")),
        "data/processed/manifests",
    )
    raw_selection_report = str(hard_negative_cfg.get("selection_report", "")).strip()
    selection_report = _resolve_relative_path(project_root, raw_selection_report) if raw_selection_report else None

    raw_voices = hard_negative_cfg.get("voices", list(DEFAULT_SAY_VOICES))
    voices = tuple(str(item) for item in raw_voices) if isinstance(raw_voices, Sequence) and not isinstance(raw_voices, (str, bytes)) else DEFAULT_SAY_VOICES
    raw_rates = hard_negative_cfg.get("rates", list(DEFAULT_SAY_RATES))
    rates = tuple(int(item) for item in raw_rates) if isinstance(raw_rates, Sequence) and not isinstance(raw_rates, (str, bytes)) else DEFAULT_SAY_RATES

    manifests = generate_hard_negative_dataset(
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        source_checkpoint=source_checkpoint,
        selection_report=selection_report,
        voices=voices,
        rates=rates,
        overwrite=bool(hard_negative_cfg.get("overwrite", False)),
        train_ratio=float(hard_negative_cfg.get("train_ratio", 0.8)),
    )
    return {
        "output_dir": str(output_dir),
        "manifests_dir": str(manifests_dir),
        "counts": {split: len(records) for split, records in manifests.items()},
        "source_checkpoint": str(source_checkpoint),
        "selection_report": str(selection_report) if selection_report is not None else "",
    }


def build_audit_manifest_names(cfg: Mapping[str, object]) -> list[str]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    external_cfg = data_cfg.get("external", {})
    if not isinstance(external_cfg, Mapping):
        external_cfg = {}
    manifests = ["local_train.jsonl", "local_valid.jsonl", "local_test.jsonl"]
    hi_cfg = external_cfg.get("hi_mia", {})
    if isinstance(hi_cfg, Mapping) and bool(hi_cfg.get("enabled", False)):
        manifests.extend(["hi_mia_train.jsonl", "hi_mia_valid.jsonl", "hi_mia_test.jsonl"])
    mswc_cfg = external_cfg.get("mswc", {})
    if isinstance(mswc_cfg, Mapping) and bool(mswc_cfg.get("enabled", False)):
        manifests.extend(["mswc_train.jsonl", "mswc_valid.jsonl", "mswc_test.jsonl"])
    l2_cfg = external_cfg.get("l2_arctic_eval", {})
    if isinstance(l2_cfg, Mapping) and bool(l2_cfg.get("enabled", False)):
        manifests.append("l2_arctic_eval_test.jsonl")
    synthetic_cfg = data_cfg.get("synthetic", {})
    if isinstance(synthetic_cfg, Mapping):
        hard_negative_cfg = synthetic_cfg.get("hard_negatives", {})
        if isinstance(hard_negative_cfg, Mapping) and bool(hard_negative_cfg.get("enabled", False)):
            manifests.extend(["synthetic_hard_negative_train.jsonl", "synthetic_hard_negative_valid.jsonl"])
    return manifests


def resolve_verifier_labels(cfg: Mapping[str, object]) -> tuple[str, ...]:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    verifier_cfg = training_cfg.get("verifier", {})
    if not isinstance(verifier_cfg, Mapping):
        verifier_cfg = {}
    labels = verifier_cfg.get("labels", VERIFIER_LABELS)
    parsed = tuple(str(label) for label in labels) if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)) else tuple(VERIFIER_LABELS)
    if parsed != tuple(VERIFIER_LABELS):
        raise ValueError(
            "Verifier training currently requires the canonical KWS12+reject label order: "
            f"{tuple(VERIFIER_LABELS)}"
        )
    reject_label = str(verifier_cfg.get("reject_label", VERIFIER_REJECT_LABEL))
    if reject_label != VERIFIER_REJECT_LABEL:
        raise ValueError(f"Verifier reject label must be '{VERIFIER_REJECT_LABEL}', got '{reject_label}'")
    return parsed


def build_batch_reject_mask(batch: Batch, verifier_cfg: Mapping[str, object] | None = None) -> torch.Tensor:
    verifier_cfg = verifier_cfg if isinstance(verifier_cfg, Mapping) else {}
    reject_sources = _normalize_string_set(verifier_cfg.get("reject_sources"))
    reject_difficulty_buckets = _normalize_string_set(
        verifier_cfg.get("reject_difficulty_buckets"),
        default=("hard_negative",),
    )
    reject_path_substrings = _normalize_string_set(
        verifier_cfg.get("reject_path_substrings"),
        default=("hard_negative", "hard_negatives"),
    )

    mask: list[bool] = []
    difficulties = list(batch.difficulty_buckets or [])
    for idx, path in enumerate(batch.paths):
        source = batch.sources[idx] if idx < len(batch.sources) else ""
        difficulty = difficulties[idx] if idx < len(difficulties) else None
        path_lower = str(path).lower()
        is_reject = False
        if reject_sources and str(source).strip().lower() in reject_sources:
            is_reject = True
        if not is_reject and difficulty is not None and str(difficulty).strip().lower() in reject_difficulty_buckets:
            is_reject = True
        if not is_reject and reject_path_substrings:
            is_reject = any(token in path_lower for token in reject_path_substrings)
        mask.append(bool(is_reject))
    return torch.tensor(mask, dtype=torch.bool)


def compute_verifier_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float | object]:
    valid = targets != IGNORE_INDEX
    empty = {
        "verifier_acc": 0.0,
        "verifier_macro_f1": 0.0,
        "verifier_non_reject_precision": 0.0,
        "verifier_non_reject_recall": 0.0,
        "verifier_reject_precision": 0.0,
        "verifier_reject_recall": 0.0,
        "per_class_verifier": {
            label: {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "predicted": 0,
                "top_confusions": [],
            }
            for label in VERIFIER_LABELS
        },
        "min_verifier_precision": 0.0,
        "min_verifier_recall": 0.0,
        "reject_samples": 0,
    }
    if valid.sum() == 0:
        return empty

    preds_v = preds[valid]
    targets_v = targets[valid]
    cm = confusion_matrix(targets_v, preds_v, labels=np.arange(len(VERIFIER_LABELS)))
    per_class: Dict[str, Dict[str, object]] = {}
    for idx, label in enumerate(VERIFIER_LABELS):
        row = cm[idx]
        col = cm[:, idx]
        support = int(row.sum())
        predicted = int(col.sum())
        tp = int(cm[idx, idx])
        precision = float(tp / max(predicted, 1))
        recall = float(tp / max(support, 1))
        denom = precision + recall
        f1 = float((2.0 * precision * recall) / denom) if denom > 0.0 else 0.0
        confusions = [
            {"label": VERIFIER_LABELS[j], "count": int(count)}
            for j, count in enumerate(row.tolist())
            if j != idx and int(count) > 0
        ]
        confusions.sort(key=lambda item: (-item["count"], str(item["label"])))
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "top_confusions": confusions[:3],
        }

    reject_idx = VERIFIER_TO_INDEX[VERIFIER_REJECT_LABEL]
    pred_non_reject = preds_v != reject_idx
    true_non_reject = targets_v != reject_idx
    exact_accept = pred_non_reject & true_non_reject & (preds_v == targets_v)
    min_precision = min(float(stats["precision"]) for stats in per_class.values()) if per_class else 0.0
    min_recall = min(float(stats["recall"]) for stats in per_class.values()) if per_class else 0.0

    return {
        "verifier_acc": float((preds_v == targets_v).mean()),
        "verifier_macro_f1": float(f1_score(targets_v, preds_v, average="macro")),
        "verifier_non_reject_precision": float(exact_accept.sum() / max(int(pred_non_reject.sum()), 1)),
        "verifier_non_reject_recall": float(exact_accept.sum() / max(int(true_non_reject.sum()), 1)),
        "verifier_reject_precision": float(per_class[VERIFIER_REJECT_LABEL]["precision"]),
        "verifier_reject_recall": float(per_class[VERIFIER_REJECT_LABEL]["recall"]),
        "per_class_verifier": per_class,
        "min_verifier_precision": float(min_precision),
        "min_verifier_recall": float(min_recall),
        "reject_samples": int((targets_v == reject_idx).sum()),
    }


def fit_verifier_calibration(
    *,
    top_probs: np.ndarray,
    margins: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    verifier_cfg: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    verifier_cfg = verifier_cfg if isinstance(verifier_cfg, Mapping) else {}
    default_prob = float(verifier_cfg.get("min_accept_prob", 0.65))
    default_margin = float(verifier_cfg.get("min_margin", 0.08))
    precision_target = float(verifier_cfg.get("calibration_target_precision", 0.95))
    fit_split = str(verifier_cfg.get("calibration_fit_split", "valid"))

    def _candidate_grid() -> tuple[list[float], list[float]]:
        prob_candidates = sorted({round(float(x), 2) for x in np.linspace(0.50, 0.95, 10)} | {round(default_prob, 2)})
        margin_candidates = sorted({round(float(x), 2) for x in np.linspace(0.00, 0.30, 16)} | {round(default_margin, 2)})
        return prob_candidates, margin_candidates

    def _fit_grid(
        *,
        accepted_base: np.ndarray,
        correct_mask: np.ndarray,
        positive_mask: np.ndarray,
        reject_mask: np.ndarray,
    ) -> Dict[str, float | int]:
        prob_candidates, margin_candidates = _candidate_grid()
        best: tuple[bool, float, float, float, float] | None = None
        best_payload: Dict[str, float | int] | None = None
        for prob_threshold in prob_candidates:
            for margin_threshold in margin_candidates:
                accepted = accepted_base & (probs_v >= prob_threshold) & (margins_v >= margin_threshold)
                correct_accept = accepted & correct_mask
                accept_precision = float(correct_accept.sum() / max(int(accepted.sum()), 1))
                accept_recall = float(correct_accept.sum() / max(int(positive_mask.sum()), 1))
                reject_escape_rate = float((accepted & reject_mask).sum() / max(int(reject_mask.sum()), 1))
                meets_target = accept_precision >= precision_target
                key = (
                    meets_target,
                    accept_recall,
                    accept_precision,
                    -float(prob_threshold),
                    -float(margin_threshold),
                )
                if best is None or key > best:
                    best = key
                    best_payload = {
                        "min_accept_prob": float(prob_threshold),
                        "min_margin": float(margin_threshold),
                        "accept_precision": accept_precision,
                        "accept_recall": accept_recall,
                        "reject_escape_rate": reject_escape_rate,
                        "num_accepted": int(accepted.sum()),
                        "num_positive_samples": int(positive_mask.sum()),
                    }
        if best_payload is None:
            return {
                "min_accept_prob": float(default_prob),
                "min_margin": float(default_margin),
                "accept_precision": 0.0,
                "accept_recall": 0.0,
                "reject_escape_rate": 0.0,
                "num_accepted": 0,
                "num_positive_samples": int(positive_mask.sum()),
            }
        return best_payload

    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return {
            "default": {
                "min_accept_prob": default_prob,
                "min_margin": default_margin,
                "accept_precision": 0.0,
                "accept_recall": 0.0,
                "reject_escape_rate": 0.0,
                "num_accepted": 0,
                "num_positive_samples": 0,
            },
            "per_label": {},
            "target_precision": precision_target,
            "fit_split": fit_split,
            "num_eval_samples": 0,
            "min_accept_prob": default_prob,
            "min_margin": default_margin,
            "accept_precision": 0.0,
            "accept_recall": 0.0,
            "reject_escape_rate": 0.0,
            "num_accepted": 0,
        }

    probs_v = top_probs[valid]
    margins_v = margins[valid]
    preds_v = preds[valid]
    targets_v = targets[valid]
    reject_idx = VERIFIER_TO_INDEX[VERIFIER_REJECT_LABEL]
    true_non_reject = targets_v != reject_idx
    true_reject = targets_v == reject_idx
    default_payload = _fit_grid(
        accepted_base=(preds_v != reject_idx),
        correct_mask=(preds_v == targets_v) & true_non_reject,
        positive_mask=true_non_reject,
        reject_mask=true_reject,
    )
    per_label: Dict[str, Dict[str, float | int]] = {}
    for label in VERIFIER_LABELS:
        if label == VERIFIER_REJECT_LABEL:
            continue
        label_idx = VERIFIER_TO_INDEX[label]
        per_label[label] = _fit_grid(
            accepted_base=(preds_v == label_idx),
            correct_mask=(targets_v == label_idx),
            positive_mask=(targets_v == label_idx),
            reject_mask=true_reject,
        )

    return {
        "default": default_payload,
        "per_label": per_label,
        "target_precision": precision_target,
        "fit_split": fit_split,
        "num_eval_samples": int(valid.sum()),
        "min_accept_prob": float(default_payload["min_accept_prob"]),
        "min_margin": float(default_payload["min_margin"]),
        "accept_precision": float(default_payload["accept_precision"]),
        "accept_recall": float(default_payload["accept_recall"]),
        "reject_escape_rate": float(default_payload["reject_escape_rate"]),
        "num_accepted": int(default_payload["num_accepted"]),
    }


def _build_teacher_stack(
    *,
    cfg: Mapping[str, object],
    project_root: Path,
    device: torch.device,
    student_dim: int,
    num_labels: int,
) -> tuple[WavLMFeatureCache | None, VerifierTeacherHeads | None]:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    teacher_cfg = training_cfg.get("teacher", {})
    if not isinstance(teacher_cfg, Mapping) or not bool(teacher_cfg.get("enabled", False)):
        return None, None

    audio_seconds = float(cfg.get("features", {}).get("audio_seconds", 1.0))
    sample_rate = int(cfg.get("features", {}).get("sample_rate", 16_000))
    teacher_cache = WavLMFeatureCache(
        model_id=str(teacher_cfg.get("model_id", "microsoft/wavlm-base-plus")),
        cache_dir=(project_root / str(teacher_cfg.get("cache_dir", "cache/wavlm_verifier_teacher"))).resolve(),
        device=device,
        clip_samples=int(round(audio_seconds * sample_rate)),
        sample_rate=sample_rate,
    )
    teacher_heads = VerifierTeacherHeads(
        feature_dim=teacher_cache.feature_dim,
        student_dim=student_dim,
        num_labels=num_labels,
        dropout=float(teacher_cfg.get("dropout", 0.1)),
    ).to(device)
    return teacher_cache, teacher_heads


def _optimizer_from_config(
    *,
    cfg: Mapping[str, object],
    trainable_params: Sequence[nn.Parameter],
) -> torch.optim.Optimizer:
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("lr", 8e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    params = list(trainable_params)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_verifier_epoch(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    verifier_cfg: Mapping[str, object] | None = None,
    loss_weights: Mapping[str, object] | None = None,
    teacher_cache: WavLMFeatureCache | None = None,
    teacher_heads: VerifierTeacherHeads | None = None,
) -> VerifierEpochResult:
    verifier_cfg = verifier_cfg if isinstance(verifier_cfg, Mapping) else {}
    loss_weights = loss_weights if isinstance(loss_weights, Mapping) else {}
    training = optimizer is not None
    model.train(mode=training)
    if teacher_heads is not None:
        teacher_heads.train(mode=training)

    label_smoothing = float(verifier_cfg.get("label_smoothing", 0.0))
    margin = float(verifier_cfg.get("margin", 0.15))
    reject_weight = float(verifier_cfg.get("reject_weight", 1.5))
    lambda_ce = float(loss_weights.get("ce", 1.0))
    lambda_margin = float(loss_weights.get("margin", 0.25))
    lambda_distill_logits = float(loss_weights.get("distill_logits", 0.2 if teacher_heads is not None else 0.0))
    lambda_distill_embed = float(loss_weights.get("distill_embed", 0.05 if teacher_heads is not None else 0.0))

    total_loss = 0.0
    total_items = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_top_probs: list[np.ndarray] = []
    all_margins: list[np.ndarray] = []

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        features = batch.features.to(device)
        command_targets = batch.command_labels.to(device)
        reject_mask = build_batch_reject_mask(batch, verifier_cfg=verifier_cfg).to(device)
        verifier_targets = build_verifier_targets(command_targets, reject_mask=reject_mask).labels

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(features)
            ce_loss = verifier_cross_entropy(output.logits, verifier_targets, label_smoothing=label_smoothing)
            margin_loss = verifier_margin_loss(
                output.logits,
                verifier_targets,
                reject_weight=reject_weight,
                margin=margin,
            )
            distill_logits_loss = output.embedding.new_tensor(0.0)
            distill_embed_loss = output.embedding.new_tensor(0.0)
            teacher_supervision_loss = output.embedding.new_tensor(0.0)

            if training and teacher_cache is not None and teacher_heads is not None:
                teacher_features = teacher_cache.load_features(batch.paths, device=device)
                teacher_targets = teacher_heads(teacher_features)
                teacher_supervision_loss = verifier_cross_entropy(
                    teacher_targets.verifier_logits,
                    verifier_targets,
                    label_smoothing=label_smoothing,
                )
                distill_logits_loss = verifier_distillation_loss(output.logits, teacher_targets.verifier_logits)
                student_embed = F.normalize(output.embedding, dim=-1)
                teacher_embed = F.normalize(teacher_targets.projected_embedding.detach(), dim=-1)
                distill_embed_loss = F.mse_loss(student_embed, teacher_embed)

            loss = (
                lambda_ce * ce_loss
                + lambda_margin * margin_loss
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

        probs = torch.softmax(output.logits.detach(), dim=-1)
        top_values, top_indices = torch.topk(probs, k=min(2, probs.size(1)), dim=-1)
        margins = (
            (top_values[:, 0] - top_values[:, 1])
            if top_values.size(1) > 1
            else top_values[:, 0]
        )

        all_preds.append(top_indices[:, 0].cpu().numpy())
        all_targets.append(verifier_targets.detach().cpu().numpy())
        all_top_probs.append(top_values[:, 0].cpu().numpy())
        all_margins.append(margins.cpu().numpy())

        iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    if total_items == 0:
        return VerifierEpochResult(
            loss=0.0,
            metrics=compute_verifier_metrics(np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)),
            preds=np.zeros((0,), dtype=np.int64),
            targets=np.zeros((0,), dtype=np.int64),
            top_probs=np.zeros((0,), dtype=np.float32),
            margins=np.zeros((0,), dtype=np.float32),
        )

    preds = np.concatenate(all_preds) if all_preds else np.zeros((0,), dtype=np.int64)
    targets = np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=np.int64)
    top_probs = np.concatenate(all_top_probs) if all_top_probs else np.zeros((0,), dtype=np.float32)
    margins = np.concatenate(all_margins) if all_margins else np.zeros((0,), dtype=np.float32)
    return VerifierEpochResult(
        loss=total_loss / total_items,
        metrics=compute_verifier_metrics(preds, targets),
        preds=preds,
        targets=targets,
        top_probs=top_probs,
        margins=margins,
    )


def _best_metric_tuple(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics.get("verifier_non_reject_precision", 0.0)),
        float(metrics.get("verifier_non_reject_recall", 0.0)),
        float(metrics.get("verifier_macro_f1", 0.0)),
    )


def main() -> None:
    args = parse_args()
    bundle = load_yaml(args.config)
    cfg = bundle.raw
    project_root = bundle.path.parent.parent.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    ensure_repo_import(repo_root)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))
    set_seed(seed)
    verifier_labels = resolve_verifier_labels(cfg)

    device = pick_device(str(cfg.get("training", {}).get("device", "auto")))
    output_dir = resolve_output_dir(cfg, project_root)
    hard_negative_stats = maybe_generate_hard_negatives(cfg, project_root)
    stats = prepare_data(cfg, project_root)
    manifests_root = _resolve_relative_path(
        project_root,
        str(cfg.get("data", {}).get("manifests_dir", "")) if isinstance(cfg.get("data", {}), Mapping) else "",
        "data/processed/manifests",
    )
    teacher_cfg = cfg.get("training", {}).get("teacher", {})
    preflight = run_repo_preflight(
        project_root,
        manifests_dir=manifests_root,
        manifest_names=build_audit_manifest_names(cfg),
        teacher_model_id=str(teacher_cfg.get("model_id", "")).strip() if isinstance(teacher_cfg, Mapping) and bool(teacher_cfg.get("enabled", False)) else None,
        teacher_cache_dir=(project_root / str(teacher_cfg.get("cache_dir", ""))).resolve() if isinstance(teacher_cfg, Mapping) and bool(teacher_cfg.get("enabled", False)) and str(teacher_cfg.get("cache_dir", "")).strip() else None,
        require_mps=str(cfg.get("training", {}).get("device", "auto")).strip().lower() == "mps",
    )
    audit_report = preflight["manifest_audit"]
    (output_dir / "verifier_data_audit.json").write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not audit_report.get("is_clean", False):
        raise RuntimeError(
            "Manifest audit failed before verifier training. "
            f"See {output_dir / 'verifier_data_audit.json'} for details."
        )
    dataloaders = create_dataloaders(cfg, project_root)

    model = create_model(
        cfg["model"],
        n_mels=int(cfg["features"]["n_mels"]),
        num_commands=len(verifier_labels),
    ).to(device)
    student_dim = int(getattr(getattr(model, "logits", None), "in_features", 0))
    if student_dim <= 0:
        raise AttributeError("Could not infer verifier student dim from model.logits.in_features")

    teacher_cache, teacher_heads = _build_teacher_stack(
        cfg=cfg,
        project_root=project_root,
        device=device,
        student_dim=student_dim,
        num_labels=len(verifier_labels),
    )
    trainable_params = list(model.parameters()) + (list(teacher_heads.parameters()) if teacher_heads is not None else [])
    optimizer = _optimizer_from_config(cfg=cfg, trainable_params=trainable_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("training", {}).get("lr_gamma", 0.5)),
        patience=int(cfg.get("training", {}).get("lr_patience", 2)),
    )

    verifier_cfg = cfg.get("training", {}).get("verifier", {})
    if not isinstance(verifier_cfg, Mapping):
        verifier_cfg = {}
    loss_weights = cfg.get("training", {}).get("loss_weights", {})
    if not isinstance(loss_weights, Mapping):
        loss_weights = {}

    with (output_dir / "resolved_verifier_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)
    with (output_dir / "verifier_dataset_stats.json").open("w", encoding="utf-8") as handle:
        payload = dict(stats)
        if hard_negative_stats is not None:
            payload["generated_hard_negatives"] = hard_negative_stats
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    epochs = int(cfg.get("training", {}).get("epochs", 10))
    history_path = output_dir / "verifier_metrics_history.jsonl"
    source_checkpoint = resolve_source_checkpoint(cfg, project_root)
    best_tuple = (-1.0, -1.0, -1.0)

    for epoch in range(1, epochs + 1):
        train_result = run_verifier_epoch(
            model=model,
            loader=dataloaders.train,
            device=device,
            optimizer=optimizer,
            verifier_cfg=verifier_cfg,
            loss_weights=loss_weights,
            teacher_cache=teacher_cache,
            teacher_heads=teacher_heads,
        )
        valid_result = run_verifier_epoch(
            model=model,
            loader=dataloaders.valid,
            device=device,
            optimizer=None,
            verifier_cfg=verifier_cfg,
            loss_weights=loss_weights,
        )

        scheduler.step(float(valid_result.metrics.get("verifier_macro_f1", 0.0)))
        calibration = fit_verifier_calibration(
            top_probs=valid_result.top_probs,
            margins=valid_result.margins,
            preds=valid_result.preds,
            targets=valid_result.targets,
            verifier_cfg=verifier_cfg,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "valid_loss": valid_result.loss,
            "train_metrics": train_result.metrics,
            "valid_metrics": valid_result.metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "verifier_calibration": calibration,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "metrics": valid_result.metrics,
            "device": str(device),
            "verifier_labels": list(verifier_labels),
            "verifier_calibration": calibration,
            "teacher_state": teacher_heads.state_dict() if teacher_heads is not None else None,
            "source_detector_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
        }
        torch.save(checkpoint, output_dir / "last_verifier.pt")

        current_tuple = _best_metric_tuple(valid_result.metrics)
        if current_tuple > best_tuple:
            best_tuple = current_tuple
            torch.save(checkpoint, output_dir / "best_kws12_verifier.pt")
            (output_dir / "verifier_calibration.json").write_text(
                json.dumps(calibration, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(
            f"[epoch {epoch:02d}] train_loss={train_result.loss:.4f} "
            f"valid_loss={valid_result.loss:.4f} "
            f"non_reject_precision={float(valid_result.metrics.get('verifier_non_reject_precision', 0.0)):.4f} "
            f"non_reject_recall={float(valid_result.metrics.get('verifier_non_reject_recall', 0.0)):.4f} "
            f"macro_f1={float(valid_result.metrics.get('verifier_macro_f1', 0.0)):.4f}"
        )

    best_checkpoint_path = output_dir / "best_kws12_verifier.pt"
    if best_checkpoint_path.exists():
        best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        test_result = run_verifier_epoch(
            model=model,
            loader=dataloaders.test,
            device=device,
            optimizer=None,
            verifier_cfg=verifier_cfg,
            loss_weights=loss_weights,
        )
        test_payload = {
            "loss": test_result.loss,
            "metrics": test_result.metrics,
            "verifier_calibration": best_payload.get("verifier_calibration", {}),
        }
        (output_dir / "verifier_test_metrics.json").write_text(
            json.dumps(test_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Verifier training complete. Outputs at: {output_dir}")


if __name__ == "__main__":
    main()
