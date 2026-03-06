"""Keyword-level analysis, focus mining, calibration, and weighting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np

from kws.constants import (
    COMMAND31_LABELS,
    COMMAND31_TO_INDEX,
    IGNORE_INDEX,
    INDEX_TO_COMMAND31,
    KWS12_LABELS,
    TARGET_KEYWORDS_10,
    command31_to_kws12,
)
from kws.data.manifest import ManifestRecord


DEFAULT_WEAK_KEYWORDS = ("off", "no", "go", "down", "on")
DEFAULT_KEYWORD_FOCUS_TOP_K = 5
DEFAULT_KEYWORD_CE_WEIGHT = 1.5
DEFAULT_KEYWORD_OVERSAMPLE = 2.5
DEFAULT_CONFUSION_OVERSAMPLE = 2.0
DEFAULT_PROTOTYPE_BONUS_MAX = 0.04
DEFAULT_WEAK_PROTOTYPE_BONUS_MAX = 0.08
DEFAULT_HIGHLIGHT_HOLD_MS = 220
DEFAULT_WEAK_HIGHLIGHT_HOLD_MS = 280
DEFAULT_MIN_MARGIN = 0.0
DEFAULT_TARGET_CONFUSION_MARGIN = 0.08
DEFAULT_WEAK_CONFUSION_MARGIN = 0.12

DEFAULT_CONFUSION_GROUPS: Dict[str, tuple[str, ...]] = {
    "off": ("up", "on"),
    "no": ("go", "down"),
    "go": ("no", "down", "up"),
    "down": ("go", "no"),
    "on": ("off", "up"),
    "left": ("right",),
    "right": ("left",),
}

DEFAULT_RUNTIME_CONFUSION_GROUPS: Dict[str, tuple[str, ...]] = {
    "off": ("on", "up"),
    "on": ("off", "five", "one"),
    "no": ("go", "down", "zero"),
    "go": ("no", "down"),
    "down": ("go", "no"),
    "left": ("right", "bed", "yes"),
    "right": ("left",),
}

DEFAULT_FOCUS_KEYWORDS = ("left", "on", "down")
DEFAULT_FOCUS_PAIRS: Dict[str, tuple[str, ...]] = {
    "left": ("right", "bed", "yes"),
    "on": ("off", "five", "one"),
    "down": ("go", "no"),
}
DEFAULT_FOCUS_RUNTIME_OVERRIDES: Dict[str, Dict[str, float | int]] = {
    "on": {"min_margin": 0.16, "vote_window": 6, "vote_min_count": 4, "highlight_hold_ms": 320},
    "down": {"min_margin": 0.14, "vote_window": 5, "vote_min_count": 3, "highlight_hold_ms": 280},
    "left": {"min_margin": 0.12, "vote_window": 5, "vote_min_count": 3, "highlight_hold_ms": 280},
}


def _normalize_focus_keywords(focus: Mapping[str, object] | None = None) -> list[str]:
    raw = (focus or {}).get("focus_keywords", DEFAULT_FOCUS_KEYWORDS)
    if not isinstance(raw, (list, tuple)):
        raw = DEFAULT_FOCUS_KEYWORDS
    out = []
    for keyword in raw:
        label = str(keyword).strip()
        if label and label in TARGET_KEYWORDS_10 and label not in out:
            out.append(label)
    return out or list(DEFAULT_FOCUS_KEYWORDS)


def _normalize_focus_pairs(focus: Mapping[str, object] | None = None) -> Dict[str, list[str]]:
    merged: Dict[str, list[str]] = {key: list(values) for key, values in DEFAULT_FOCUS_PAIRS.items()}
    raw = (focus or {}).get("focus_pairs", {})
    if isinstance(raw, Mapping):
        for keyword, values in raw.items():
            label = str(keyword).strip()
            if not label or label not in TARGET_KEYWORDS_10:
                continue
            merged.setdefault(label, [])
            if isinstance(values, (list, tuple)):
                for value in values:
                    rival = str(value).strip()
                    if rival and rival in COMMAND31_TO_INDEX and rival not in merged[label]:
                        merged[label].append(rival)
    return merged


def compute_focus_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    focus_keywords: Sequence[str] | None = None,
    focus_pairs: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, object]:
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = targets != IGNORE_INDEX
    preds_v = preds[valid]
    targets_v = targets[valid]

    chosen_keywords = [str(keyword) for keyword in (focus_keywords or DEFAULT_FOCUS_KEYWORDS) if str(keyword) in TARGET_KEYWORDS_10]
    pair_map = _normalize_focus_pairs({"focus_pairs": dict(focus_pairs or {})})

    recalls: list[float] = []
    total_support = 0
    total_confusions = 0
    out: Dict[str, Dict[str, object]] = {}
    for keyword in chosen_keywords:
        idx = COMMAND31_TO_INDEX[keyword]
        true_mask = targets_v == idx
        support = int(true_mask.sum())
        predicted = preds_v[true_mask]
        recall = float((predicted == idx).sum() / max(support, 1))
        recalls.append(recall)
        competitors = []
        confusion_count = 0
        for rival in pair_map.get(keyword, []):
            rival_idx = COMMAND31_TO_INDEX.get(str(rival))
            count = int((predicted == rival_idx).sum()) if rival_idx is not None else 0
            confusion_count += count
            competitors.append({"label": str(rival), "count": count})
        total_support += support
        total_confusions += confusion_count
        out[keyword] = {
            "support": support,
            "recall": recall,
            "count": int(confusion_count),
            "rate": float(confusion_count / max(support, 1)),
            "pairs": competitors,
        }

    return {
        "focus_keywords": chosen_keywords,
        "focus_pairs": {key: list(pair_map.get(key, [])) for key in chosen_keywords},
        "focus_keyword_recall_mean": float(sum(recalls) / max(len(recalls), 1)),
        "focus_pair_confusions": out,
        "focus_pair_confusion_rate": float(total_confusions / max(total_support, 1)),
    }


def load_keyword_focus(path: str | Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def save_keyword_focus(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def save_keyword_calibration(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_keyword_calibration(path: str | Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def build_command_class_weights(focus: Mapping[str, object] | None) -> np.ndarray:
    weights = np.ones(len(COMMAND31_LABELS), dtype=np.float32)
    focus = focus or {}
    keyword_weights = focus.get("keyword_ce_weights", {}) if isinstance(focus, Mapping) else {}
    if not isinstance(keyword_weights, Mapping):
        keyword_weights = {}
    for keyword in TARGET_KEYWORDS_10:
        idx = COMMAND31_TO_INDEX[keyword]
        weights[idx] = float(keyword_weights.get(keyword, 1.0))
    return weights


def build_kws12_class_weights(focus: Mapping[str, object] | None) -> np.ndarray:
    weights = np.ones(len(KWS12_LABELS), dtype=np.float32)
    focus = focus or {}
    keyword_weights = focus.get("keyword_ce_weights", {}) if isinstance(focus, Mapping) else {}
    if not isinstance(keyword_weights, Mapping):
        keyword_weights = {}
    for idx, keyword in enumerate(TARGET_KEYWORDS_10, start=2):
        weights[idx] = float(keyword_weights.get(keyword, 1.0))
    return weights


def _to_kws12_indices(indices: np.ndarray) -> np.ndarray:
    mapped = [command31_to_kws12(INDEX_TO_COMMAND31[int(idx)]) for idx in indices]
    return np.asarray(mapped, dtype=np.int64)


def aggregate_command_probs_to_kws12(
    command_probs: np.ndarray,
    command31_labels: Sequence[str] = COMMAND31_LABELS,
) -> np.ndarray:
    probs = np.asarray(command_probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[None, :]
    kws12_probs = np.zeros((probs.shape[0], len(KWS12_LABELS)), dtype=np.float32)
    for idx, label in enumerate(command31_labels):
        kws12_probs[:, command31_to_kws12(label)] += probs[:, idx]
    return kws12_probs if command_probs.ndim > 1 else kws12_probs[0]


def compute_per_keyword_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    top_confusions: int = 3,
) -> Dict[str, Dict[str, object]]:
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = targets != IGNORE_INDEX
    preds_v = preds[valid]
    targets_v = targets[valid]

    out: Dict[str, Dict[str, object]] = {}
    for keyword in TARGET_KEYWORDS_10:
        idx = COMMAND31_TO_INDEX[keyword]
        true_mask = targets_v == idx
        pred_mask = preds_v == idx
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        support = int(true_mask.sum())
        predicted = int(pred_mask.sum())
        precision = float(tp / max(predicted, 1))
        recall = float(tp / max(support, 1))
        denom = precision + recall
        f1 = float((2.0 * precision * recall) / denom) if denom > 0.0 else 0.0

        confusion_counts: MutableMapping[str, int] = {}
        if support > 0:
            wrong_preds = preds_v[np.logical_and(true_mask, ~pred_mask)]
            for wrong_idx in wrong_preds.tolist():
                label = INDEX_TO_COMMAND31[int(wrong_idx)]
                confusion_counts[label] = confusion_counts.get(label, 0) + 1
        sorted_confusions = sorted(
            confusion_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[: max(1, int(top_confusions))]
        out[keyword] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "top_confusions": [{"label": label, "count": int(count)} for label, count in sorted_confusions],
        }
    return out


def compute_keyword_balance(per_keyword: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
    recalls = [float(stats.get("recall", 0.0)) for stats in per_keyword.values()]
    if not recalls:
        return {"bottom3_keyword_recall": 0.0, "keyword_balance_gap": 0.0}

    recalls_sorted = sorted(recalls)
    bottom_k = recalls_sorted[: min(3, len(recalls_sorted))]
    return {
        "bottom3_keyword_recall": float(sum(bottom_k) / max(len(bottom_k), 1)),
        "keyword_balance_gap": float(max(recalls) - min(recalls)),
    }


def compute_per_keyword_from_confusion(confusion: Sequence[Sequence[int]]) -> Dict[str, Dict[str, object]]:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] < len(COMMAND31_LABELS) or cm.shape[1] < len(COMMAND31_LABELS):
        return {kw: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "predicted": 0, "top_confusions": []} for kw in TARGET_KEYWORDS_10}

    out: Dict[str, Dict[str, object]] = {}
    for keyword in TARGET_KEYWORDS_10:
        idx = COMMAND31_TO_INDEX[keyword]
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
            {"label": COMMAND31_LABELS[j], "count": int(count)}
            for j, count in enumerate(row.tolist())
            if j != idx and int(count) > 0
        ]
        confusions.sort(key=lambda item: (-item["count"], item["label"]))
        out[keyword] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "top_confusions": confusions[:3],
        }
    return out


def mine_keyword_focus(
    per_keyword: Mapping[str, Mapping[str, object]],
    *,
    top_k: int = DEFAULT_KEYWORD_FOCUS_TOP_K,
) -> Dict[str, object]:
    scored = []
    for keyword in TARGET_KEYWORDS_10:
        stats = per_keyword.get(keyword, {})
        recall = float(stats.get("recall", 0.0))
        support = int(stats.get("support", 0))
        if support <= 0:
            continue
        scored.append((recall, support, keyword))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))

    weak_keywords = [kw for _recall, _support, kw in scored[: max(1, min(int(top_k), len(scored)))]]
    if not weak_keywords:
        weak_keywords = list(DEFAULT_WEAK_KEYWORDS)

    confusable_groups: Dict[str, list[str]] = {}
    for keyword in TARGET_KEYWORDS_10:
        stats = per_keyword.get(keyword, {})
        mined = []
        for item in stats.get("top_confusions", []):
            label = str(item.get("label", "")).strip()
            if label and label != keyword and label not in mined:
                mined.append(label)
        for label in DEFAULT_CONFUSION_GROUPS.get(keyword, ()):
            if label not in mined:
                mined.append(label)
        confusable_groups[keyword] = mined[:3]

    keyword_weights = {
        keyword: float(DEFAULT_KEYWORD_CE_WEIGHT if keyword in weak_keywords else 1.0)
        for keyword in TARGET_KEYWORDS_10
    }
    oversample_weights = {
        keyword: float(DEFAULT_KEYWORD_OVERSAMPLE if keyword in weak_keywords else 1.0)
        for keyword in TARGET_KEYWORDS_10
    }

    return {
        "weak_keywords": weak_keywords,
        "confusable_groups": confusable_groups,
        "keyword_ce_weights": keyword_weights,
        "keyword_oversample_weights": oversample_weights,
        "confusion_negative_oversample": float(DEFAULT_CONFUSION_OVERSAMPLE),
        **compute_keyword_balance(per_keyword),
    }


def build_keyword_focus_report(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    top_k: int = DEFAULT_KEYWORD_FOCUS_TOP_K,
    focus_keywords: Sequence[str] | None = None,
    focus_pairs: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, object]:
    per_keyword = compute_per_keyword_metrics(preds, targets)
    summary = mine_keyword_focus(per_keyword, top_k=top_k)
    focus_metrics = compute_focus_metrics(
        preds,
        targets,
        focus_keywords=focus_keywords,
        focus_pairs=focus_pairs,
    )
    return {"per_keyword": per_keyword, **summary, **focus_metrics}


def fit_keyword_calibration(
    command_probs: np.ndarray,
    wake_probs: np.ndarray,
    targets: np.ndarray,
    *,
    focus: Mapping[str, object] | None = None,
    beta_weak: float = 1.35,
    beta_default: float = 1.0,
) -> Dict[str, object]:
    command_probs = np.asarray(command_probs, dtype=np.float32)
    wake_probs = np.asarray(wake_probs, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64)
    valid = targets != IGNORE_INDEX
    command_probs = command_probs[valid]
    wake_probs = wake_probs[valid]
    targets = targets[valid]

    kws12_targets = _to_kws12_indices(targets)
    kws12_probs = aggregate_command_probs_to_kws12(command_probs)
    weak_keywords = set(str(x) for x in (focus or {}).get("weak_keywords", DEFAULT_WEAK_KEYWORDS))
    focus_keywords = set(_normalize_focus_keywords(focus))
    focus_pairs = _normalize_focus_pairs(focus)
    confusable_groups = dict(DEFAULT_RUNTIME_CONFUSION_GROUPS)
    raw_confusable = (focus or {}).get("confusable_groups", {})
    if isinstance(raw_confusable, Mapping):
        for key, values in raw_confusable.items():
            label = str(key).strip()
            if not label or label not in TARGET_KEYWORDS_10 or not isinstance(values, (list, tuple)):
                continue
            confusable_groups[label] = tuple(
                rival
                for value in values
                for rival in [str(value).strip()]
                if rival in COMMAND31_TO_INDEX
            )
    for key, values in focus_pairs.items():
        merged = []
        for label in list(values) + list(confusable_groups.get(key, ())):
            rival = str(label).strip()
            if rival and rival not in merged:
                merged.append(rival)
        confusable_groups[key] = tuple(merged)

    defaults = {
        "command_conf_threshold": 0.35,
        "vote_window": 4,
        "vote_min_count": 2,
        "prototype_bonus_max": DEFAULT_PROTOTYPE_BONUS_MAX,
        "min_margin": DEFAULT_MIN_MARGIN,
        "highlight_hold_ms": DEFAULT_HIGHLIGHT_HOLD_MS,
    }
    keywords: Dict[str, Dict[str, float | int]] = {}

    for kws_idx, keyword in enumerate(TARGET_KEYWORDS_10, start=2):
        scores = kws12_probs[:, kws_idx] if len(kws12_probs) else np.asarray([], dtype=np.float32)
        positives = kws12_targets == kws_idx
        beta = beta_weak if keyword in weak_keywords else beta_default
        best_thr = float(defaults["command_conf_threshold"])
        best_score = -1.0

        for thr in np.linspace(0.15, 0.95, num=17):
            pred_pos = scores >= thr
            tp = float(np.logical_and(pred_pos, positives).sum())
            fp = float(np.logical_and(pred_pos, ~positives).sum())
            fn = float(np.logical_and(~pred_pos, positives).sum())
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            beta_sq = beta * beta
            f_beta = (1.0 + beta_sq) * precision * recall / max((beta_sq * precision) + recall, 1e-8)
            fp_rate = fp / max(float((~positives).sum()), 1.0)
            objective = float(f_beta - (0.25 * fp_rate))
            if objective > best_score:
                best_score = objective
                best_thr = float(thr)

        is_weak = keyword in weak_keywords
        has_target_confusion = any(label in TARGET_KEYWORDS_10 for label in confusable_groups.get(keyword, ()))
        vote_window = 5 if (is_weak or has_target_confusion) else 4
        vote_min_count = 3 if (is_weak or best_thr < 0.30) else 2
        prototype_bonus = DEFAULT_WEAK_PROTOTYPE_BONUS_MAX if is_weak else DEFAULT_PROTOTYPE_BONUS_MAX
        if is_weak:
            min_margin = DEFAULT_WEAK_CONFUSION_MARGIN
            highlight_hold_ms = DEFAULT_WEAK_HIGHLIGHT_HOLD_MS
        elif has_target_confusion:
            min_margin = DEFAULT_TARGET_CONFUSION_MARGIN
            highlight_hold_ms = DEFAULT_HIGHLIGHT_HOLD_MS
        else:
            min_margin = DEFAULT_MIN_MARGIN
            highlight_hold_ms = DEFAULT_HIGHLIGHT_HOLD_MS
        if keyword in focus_keywords:
            overrides = DEFAULT_FOCUS_RUNTIME_OVERRIDES.get(keyword, {})
            vote_window = max(vote_window, int(overrides.get("vote_window", vote_window)))
            vote_min_count = max(vote_min_count, int(overrides.get("vote_min_count", vote_min_count)))
            min_margin = max(min_margin, float(overrides.get("min_margin", min_margin)))
            highlight_hold_ms = max(highlight_hold_ms, int(overrides.get("highlight_hold_ms", highlight_hold_ms)))
        keywords[keyword] = {
            "command_conf_threshold": float(best_thr),
            "vote_window": int(vote_window),
            "vote_min_count": int(vote_min_count),
            "prototype_bonus_max": float(prototype_bonus),
            "min_margin": float(min_margin),
            "highlight_hold_ms": int(highlight_hold_ms),
            "support": int(positives.sum()),
        }

    return {
        "version": 1,
        "weak_keywords": sorted(weak_keywords),
        "focus_keywords": sorted(focus_keywords),
        "focus_pairs": {key: list(values) for key, values in focus_pairs.items()},
        "defaults": defaults,
        "confusable_groups": {key: list(values) for key, values in confusable_groups.items()},
        "keywords": keywords,
    }


def compute_accent_slices(
    records: Sequence[ManifestRecord],
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(records) != len(preds) or len(records) != len(targets):
        return {}

    grouped: Dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        accent = (rec.accent_group or "").strip()
        if not accent:
            continue
        grouped.setdefault(accent, []).append(idx)

    out: Dict[str, Dict[str, object]] = {}
    for accent, indices in grouped.items():
        part_preds = preds[indices]
        part_targets = targets[indices]
        per_keyword = compute_per_keyword_metrics(part_preds, part_targets)
        out[accent] = {
            "num_samples": len(indices),
            "per_keyword": per_keyword,
            **compute_keyword_balance(per_keyword),
        }
    return out
