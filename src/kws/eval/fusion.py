"""Detector + verifier fusion helpers shared by eval, analysis, and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from kws.constants import KWS12_LABELS, KWS12_TO_INDEX, TARGET_KEYWORDS_10, UNKNOWN_LABEL, command31_to_kws12
from kws.demo.verifier_runtime import resolve_calibration_thresholds
from kws.train.metrics import compute_kws12_breakdown_from_indices
from kws.train.verifier import VERIFIER_REJECT_LABEL


@dataclass(frozen=True)
class FusionOutputs:
    detector_preds_kws12: np.ndarray
    fused_preds_kws12: np.ndarray
    detector_probs_kws12: np.ndarray
    detector_margins: np.ndarray
    verify_candidate_labels: tuple[str | None, ...]
    should_verify_mask: np.ndarray
    verifier_accept_mask: np.ndarray
    verifier_top_labels: tuple[str | None, ...]
    verifier_top_probs: np.ndarray
    verifier_margins: np.ndarray


def aggregate_command_probs_to_kws12_matrix(
    command_probs: np.ndarray,
    command31_labels: Sequence[str],
) -> np.ndarray:
    probs = np.asarray(command_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"command_probs must be rank-2, got shape={probs.shape}")
    kws12 = np.zeros((probs.shape[0], len(KWS12_LABELS)), dtype=np.float32)
    for idx, label in enumerate(command31_labels):
        kws12[:, command31_to_kws12(str(label))] += probs[:, idx].astype(np.float32, copy=False)
    return kws12


def _top_margin(values: np.ndarray) -> tuple[int, float]:
    top_idx = int(np.argmax(values))
    if values.size <= 1:
        return top_idx, float(values[top_idx])
    top_values = np.partition(values, -2)[-2:]
    margin = float(top_values[-1] - top_values[-2])
    return top_idx, margin


def select_verifier_candidate(
    *,
    detector_label: str,
    detector_margin: float,
    detector_probs_kws12: np.ndarray,
    decision_profile: str,
    margin_trigger: float,
) -> str | None:
    profile = str(decision_profile).strip().lower()
    if profile == "fast":
        return None
    if detector_label in TARGET_KEYWORDS_10:
        return detector_label
    if profile != "stable" or float(detector_margin) >= float(margin_trigger):
        return None
    target_scores = np.asarray(detector_probs_kws12[2:], dtype=np.float32)
    if target_scores.size == 0:
        return None
    target_idx = int(np.argmax(target_scores))
    return TARGET_KEYWORDS_10[target_idx]


def fuse_detector_and_verifier(
    *,
    command_probs: np.ndarray,
    command31_labels: Sequence[str],
    verifier_probs: np.ndarray | None = None,
    verifier_labels: Sequence[str] | None = None,
    verifier_calibration: Mapping[str, object] | None = None,
    decision_profile: str = "stable",
    margin_trigger: float = 0.20,
) -> FusionOutputs:
    detector_probs_kws12 = aggregate_command_probs_to_kws12_matrix(command_probs, command31_labels)
    num_items = int(detector_probs_kws12.shape[0])
    detector_preds = np.zeros((num_items,), dtype=np.int64)
    fused_preds = np.zeros((num_items,), dtype=np.int64)
    detector_margins = np.zeros((num_items,), dtype=np.float32)
    should_verify = np.zeros((num_items,), dtype=bool)
    verifier_accept = np.zeros((num_items,), dtype=bool)
    verifier_top_probs = np.zeros((num_items,), dtype=np.float32)
    verifier_margins = np.zeros((num_items,), dtype=np.float32)
    verify_candidates: list[str | None] = []
    top_labels: list[str | None] = []

    verifier_probs_arr = None if verifier_probs is None else np.asarray(verifier_probs, dtype=np.float32)
    verifier_label_list = tuple(str(label) for label in verifier_labels) if verifier_labels is not None else ()

    for idx in range(num_items):
        kws12_row = detector_probs_kws12[idx]
        detector_idx, detector_margin = _top_margin(kws12_row)
        detector_label = KWS12_LABELS[detector_idx]
        detector_preds[idx] = detector_idx
        fused_preds[idx] = detector_idx
        detector_margins[idx] = float(detector_margin)

        candidate_label = select_verifier_candidate(
            detector_label=detector_label,
            detector_margin=float(detector_margin),
            detector_probs_kws12=kws12_row,
            decision_profile=decision_profile,
            margin_trigger=margin_trigger,
        )
        verify_candidates.append(candidate_label)
        if verifier_probs_arr is None or not verifier_label_list or candidate_label is None:
            top_labels.append(None)
            continue

        should_verify[idx] = True
        verifier_row = verifier_probs_arr[idx]
        verifier_idx, verifier_margin = _top_margin(verifier_row)
        verifier_label = verifier_label_list[verifier_idx]
        verifier_prob = float(verifier_row[verifier_idx])
        verifier_top_probs[idx] = verifier_prob
        verifier_margins[idx] = float(verifier_margin)
        top_labels.append(verifier_label)
        min_accept_prob, min_margin = resolve_calibration_thresholds(verifier_calibration, candidate_label)
        accepted = bool(
            verifier_label == candidate_label
            and verifier_label != VERIFIER_REJECT_LABEL
            and verifier_prob >= float(min_accept_prob)
            and float(verifier_margin) >= float(min_margin)
        )
        verifier_accept[idx] = accepted
        fused_preds[idx] = KWS12_TO_INDEX[candidate_label if accepted else UNKNOWN_LABEL]

    return FusionOutputs(
        detector_preds_kws12=detector_preds,
        fused_preds_kws12=fused_preds,
        detector_probs_kws12=detector_probs_kws12,
        detector_margins=detector_margins,
        verify_candidate_labels=tuple(verify_candidates),
        should_verify_mask=should_verify,
        verifier_accept_mask=verifier_accept,
        verifier_top_labels=tuple(top_labels),
        verifier_top_probs=verifier_top_probs,
        verifier_margins=verifier_margins,
    )


def compute_fused_payload(
    *,
    command_probs: np.ndarray,
    command31_labels: Sequence[str],
    targets_kws12: np.ndarray,
    verifier_probs: np.ndarray | None = None,
    verifier_labels: Sequence[str] | None = None,
    verifier_calibration: Mapping[str, object] | None = None,
    decision_profile: str = "stable",
    margin_trigger: float = 0.20,
) -> dict:
    outputs = fuse_detector_and_verifier(
        command_probs=command_probs,
        command31_labels=command31_labels,
        verifier_probs=verifier_probs,
        verifier_labels=verifier_labels,
        verifier_calibration=verifier_calibration,
        decision_profile=decision_profile,
        margin_trigger=margin_trigger,
    )
    detector_metrics = compute_kws12_breakdown_from_indices(outputs.detector_preds_kws12, targets_kws12)
    fused_metrics = compute_kws12_breakdown_from_indices(outputs.fused_preds_kws12, targets_kws12)
    return {
        "decision_profile": str(decision_profile),
        "verifier_margin_trigger": float(margin_trigger),
        "detector_preds_kws12": outputs.detector_preds_kws12,
        "fused_preds_kws12": outputs.fused_preds_kws12,
        "detector_metrics": detector_metrics,
        "fused_metrics": fused_metrics,
        "detector_margins": outputs.detector_margins,
        "verify_candidate_labels": list(outputs.verify_candidate_labels),
        "verify_rate": float(outputs.should_verify_mask.mean()) if outputs.should_verify_mask.size else 0.0,
        "verifier_accept_rate": float(outputs.verifier_accept_mask.mean()) if outputs.verifier_accept_mask.size else 0.0,
        "verifier_top_labels": list(outputs.verifier_top_labels),
        "verifier_top_probs": outputs.verifier_top_probs,
        "verifier_margins": outputs.verifier_margins,
    }


def compute_hybrid_fused_payload(
    *,
    command_probs: np.ndarray,
    command31_labels: Sequence[str],
    targets_kws12: np.ndarray,
    internal_verifier_probs: np.ndarray | None = None,
    internal_verifier_labels: Sequence[str] | None = None,
    internal_verifier_calibration: Mapping[str, object] | None = None,
    external_verifier_probs: np.ndarray | None = None,
    external_verifier_labels: Sequence[str] | None = None,
    external_verifier_calibration: Mapping[str, object] | None = None,
    decision_profile: str = "stable",
    margin_trigger: float = 0.20,
) -> dict:
    base = fuse_detector_and_verifier(
        command_probs=command_probs,
        command31_labels=command31_labels,
        verifier_probs=None,
        verifier_labels=None,
        verifier_calibration=None,
        decision_profile=decision_profile,
        margin_trigger=margin_trigger,
    )
    internal = (
        fuse_detector_and_verifier(
            command_probs=command_probs,
            command31_labels=command31_labels,
            verifier_probs=internal_verifier_probs,
            verifier_labels=internal_verifier_labels,
            verifier_calibration=internal_verifier_calibration,
            decision_profile=decision_profile,
            margin_trigger=margin_trigger,
        )
        if internal_verifier_probs is not None and internal_verifier_labels is not None
        else None
    )
    external = (
        fuse_detector_and_verifier(
            command_probs=command_probs,
            command31_labels=command31_labels,
            verifier_probs=external_verifier_probs,
            verifier_labels=external_verifier_labels,
            verifier_calibration=external_verifier_calibration,
            decision_profile=decision_profile,
            margin_trigger=margin_trigger,
        )
        if external_verifier_probs is not None and external_verifier_labels is not None
        else None
    )

    fused_preds = np.array(base.detector_preds_kws12, copy=True)
    combined_accept = np.zeros_like(base.should_verify_mask, dtype=bool)
    for idx, candidate_label in enumerate(base.verify_candidate_labels):
        if candidate_label is None:
            continue
        accepted = False
        if internal is not None and bool(internal.verifier_accept_mask[idx]):
            accepted = True
        if external is not None and bool(external.verifier_accept_mask[idx]):
            accepted = True
        combined_accept[idx] = accepted
        fused_preds[idx] = KWS12_TO_INDEX[candidate_label] if accepted else KWS12_TO_INDEX[UNKNOWN_LABEL]

    detector_metrics = compute_kws12_breakdown_from_indices(base.detector_preds_kws12, targets_kws12)
    fused_metrics = compute_kws12_breakdown_from_indices(fused_preds, targets_kws12)
    return {
        "decision_profile": str(decision_profile),
        "verifier_margin_trigger": float(margin_trigger),
        "detector_preds_kws12": base.detector_preds_kws12,
        "fused_preds_kws12": fused_preds,
        "detector_metrics": detector_metrics,
        "fused_metrics": fused_metrics,
        "detector_margins": base.detector_margins,
        "verify_candidate_labels": list(base.verify_candidate_labels),
        "verify_rate": float(base.should_verify_mask.mean()) if base.should_verify_mask.size else 0.0,
        "verifier_accept_rate": float(combined_accept.mean()) if combined_accept.size else 0.0,
        "internal_verifier_accept_rate": float(internal.verifier_accept_mask.mean()) if internal is not None and internal.verifier_accept_mask.size else 0.0,
        "external_verifier_accept_rate": float(external.verifier_accept_mask.mean()) if external is not None and external.verifier_accept_mask.size else 0.0,
    }
