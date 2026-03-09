"""Runtime helpers for optional KWS12 verifier checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from kws.models import create_model
from kws.train.verifier import VERIFIER_LABELS, VERIFIER_REJECT_LABEL, load_verifier_checkpoint


@dataclass(frozen=True)
class LoadedVerifier:
    checkpoint_path: Path
    runtime_device: torch.device
    model: torch.nn.Module
    labels: tuple[str, ...]
    calibration: Mapping[str, object]
    min_accept_prob: float
    min_margin: float


@dataclass(frozen=True)
class VerifierDecision:
    accepted: bool
    top_label: str
    confidence: float
    margin: float


def resolve_calibration_thresholds(
    calibration: Mapping[str, object] | None,
    candidate_label: str | None,
) -> tuple[float, float]:
    calibration = calibration if isinstance(calibration, Mapping) else {}
    default_payload = calibration.get("default", {})
    if not isinstance(default_payload, Mapping):
        default_payload = {}
    per_label = calibration.get("per_label", {})
    if not isinstance(per_label, Mapping):
        per_label = {}
    entry = per_label.get(str(candidate_label), {}) if candidate_label else {}
    if not isinstance(entry, Mapping):
        entry = {}
    default_prob = float(default_payload.get("min_accept_prob", calibration.get("min_accept_prob", 0.65)))
    default_margin = float(default_payload.get("min_margin", calibration.get("min_margin", 0.08)))
    return (
        float(entry.get("min_accept_prob", default_prob)),
        float(entry.get("min_margin", default_margin)),
    )


def _load_verifier_labels(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw_labels = payload.get("verifier_labels")
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
        labels = tuple(str(item) for item in raw_labels)
        if labels:
            return labels
    cfg = payload.get("config", {})
    if isinstance(cfg, Mapping):
        training = cfg.get("training", {})
        if isinstance(training, Mapping):
            verifier_cfg = training.get("verifier", {})
            if isinstance(verifier_cfg, Mapping):
                raw_labels = verifier_cfg.get("labels")
                if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
                    labels = tuple(str(item) for item in raw_labels)
                    if labels:
                        return labels
    return tuple(VERIFIER_LABELS)


def load_runtime_verifier(
    detector_checkpoint_path: str | Path,
    *,
    device: torch.device,
    verifier_path: str | Path | None = None,
) -> LoadedVerifier | None:
    detector_checkpoint_path = Path(detector_checkpoint_path).expanduser().resolve()
    path = (
        Path(verifier_path).expanduser().resolve()
        if verifier_path is not None and str(verifier_path).strip()
        else detector_checkpoint_path.parent / "best_kws12_verifier.pt"
    )
    if not path.exists():
        return None

    payload = load_verifier_checkpoint(path)
    cfg = payload.get("config", {})
    if not isinstance(cfg, Mapping):
        raise TypeError(f"Verifier checkpoint at {path} is missing a config mapping.")
    features = cfg.get("features", {})
    if not isinstance(features, Mapping):
        raise TypeError(f"Verifier checkpoint at {path} is missing features config.")
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise TypeError(f"Verifier checkpoint at {path} is missing model config.")
    labels = _load_verifier_labels(payload)
    model = create_model(
        dict(model_cfg),
        n_mels=int(features.get("n_mels", 80)),
        num_commands=len(labels),
    )
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise TypeError(f"Verifier checkpoint at {path} is missing model_state.")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    calibration = payload.get("verifier_calibration", {})
    if not isinstance(calibration, Mapping):
        calibration = {}
    if not calibration:
        calibration_path = path.parent / "verifier_calibration.json"
        if calibration_path.exists():
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except Exception:
                calibration = {}
        if not isinstance(calibration, Mapping):
            calibration = {}
    min_accept_prob, min_margin = resolve_calibration_thresholds(calibration, None)

    return LoadedVerifier(
        checkpoint_path=path,
        runtime_device=device,
        model=model,
        labels=labels,
        calibration=calibration,
        min_accept_prob=min_accept_prob,
        min_margin=min_margin,
    )


def verify_keyword(
    verifier: LoadedVerifier | None,
    features: torch.Tensor,
    *,
    candidate_label: str | None,
) -> VerifierDecision | None:
    if verifier is None or not candidate_label:
        return None
    with torch.no_grad():
        out = verifier.model(features.to(verifier.runtime_device))
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
    top_prob, top_idx = torch.max(probs, dim=0)
    top_label = verifier.labels[int(top_idx)]
    if probs.numel() > 1:
        top_values = torch.topk(probs, k=min(2, probs.numel())).values
        margin = float((top_values[0] - top_values[-1]).item()) if top_values.numel() == 2 else float(top_values[0].item())
    else:
        margin = float(top_prob.item())
    min_accept_prob, min_margin = resolve_calibration_thresholds(verifier.calibration, candidate_label)
    accepted = bool(
        top_label == str(candidate_label)
        and top_label != VERIFIER_REJECT_LABEL
        and float(top_prob.item()) >= float(min_accept_prob)
        and margin >= float(min_margin)
    )
    return VerifierDecision(
        accepted=accepted,
        top_label=str(top_label),
        confidence=float(top_prob.item()),
        margin=margin,
    )
