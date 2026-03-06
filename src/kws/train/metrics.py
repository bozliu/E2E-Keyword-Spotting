"""Training and evaluation metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from kws.constants import COMMAND31_LABELS, IGNORE_INDEX, INDEX_TO_COMMAND31, command31_to_kws12
from kws.utils.keyword_focus import compute_keyword_balance, compute_per_keyword_metrics


def compute_command_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return {
            "command_acc": 0.0,
            "command_macro_f1": 0.0,
            "command_confusion": [],
        }

    preds_v = preds[valid]
    targets_v = targets[valid]
    acc = float((preds_v == targets_v).mean())
    macro_f1 = float(f1_score(targets_v, preds_v, average="macro"))
    cm = confusion_matrix(targets_v, preds_v, labels=np.arange(len(COMMAND31_LABELS))).tolist()
    return {
        "command_acc": acc,
        "command_macro_f1": macro_f1,
        "command_confusion": cm,
    }


def _to_kws12_indices(indices: np.ndarray) -> np.ndarray:
    mapped: List[int] = []
    for idx in indices:
        label_name = INDEX_TO_COMMAND31[int(idx)]
        mapped.append(command31_to_kws12(label_name))
    return np.array(mapped, dtype=np.int64)


def compute_kws12_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return 0.0
    preds_kws12 = _to_kws12_indices(preds[valid])
    targets_kws12 = _to_kws12_indices(targets[valid])
    return float((preds_kws12 == targets_kws12).mean())


def compute_kws12_breakdown(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return {
            "kws12_target_precision": 0.0,
            "kws12_target_recall": 0.0,
            "kws12_unknown_to_target_errors": 0.0,
            "kws12_unknown_to_target_rate": 0.0,
        }

    preds_kws12 = _to_kws12_indices(preds[valid])
    targets_kws12 = _to_kws12_indices(targets[valid])

    pred_target = preds_kws12 > 1
    true_target = targets_kws12 > 1
    exact_target_match = pred_target & true_target & (preds_kws12 == targets_kws12)

    precision = float(exact_target_match.sum() / max(int(pred_target.sum()), 1))
    recall = float(exact_target_match.sum() / max(int(true_target.sum()), 1))

    unknown_mask = targets_kws12 == 1
    unknown_to_target = pred_target & unknown_mask
    unknown_count = int(unknown_mask.sum())

    return {
        "kws12_target_precision": precision,
        "kws12_target_recall": recall,
        "kws12_unknown_to_target_errors": float(unknown_to_target.sum()),
        "kws12_unknown_to_target_rate": float(unknown_to_target.sum() / max(unknown_count, 1)),
    }


def compute_keyword_breakdown(preds: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    per_keyword = compute_per_keyword_metrics(preds, targets)
    return {
        "per_keyword": per_keyword,
        **compute_keyword_balance(per_keyword),
    }


def compute_wake_metrics(scores: np.ndarray, targets: np.ndarray, audio_seconds: float = 1.0) -> Dict[str, float]:
    if len(scores) == 0:
        return {
            "wake_roc_auc": 0.0,
            "wake_frr_at_1fa_per_hour": 1.0,
            "wake_far_at_1fa_per_hour": 0.0,
        }

    targets = targets.astype(np.int64)
    if len(np.unique(targets)) < 2:
        return {
            "wake_roc_auc": 0.0,
            "wake_frr_at_1fa_per_hour": 1.0,
            "wake_far_at_1fa_per_hour": 0.0,
        }

    fpr, tpr, _ = roc_curve(targets, scores)
    far_per_hour = fpr * (3600.0 / audio_seconds)
    frr = 1.0 - tpr

    # choose the operating point closest to <= 1 false alarm / hour.
    mask = far_per_hour <= 1.0
    if mask.any():
        idx = np.where(mask)[0][-1]
    else:
        idx = np.argmin(np.abs(far_per_hour - 1.0))

    return {
        "wake_roc_auc": float(roc_auc_score(targets, scores)),
        "wake_frr_at_1fa_per_hour": float(frr[idx]),
        "wake_far_at_1fa_per_hour": float(far_per_hour[idx]),
    }
