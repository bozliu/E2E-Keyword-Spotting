"""Transformers-backed external KWS inference and calibration helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from kws.constants import CLIP_SAMPLES, KWS12_LABELS, KWS12_TO_INDEX, TARGET_KEYWORDS_10, UNKNOWN_LABEL
from kws.data.audio import load_audio, pad_or_trim

DEFAULT_EXTERNAL_VERIFIER_MODEL_ID = "superb/wav2vec2-base-superb-ks"
DEFAULT_EXTERNAL_AUX_MODEL_ID = "MIT/ast-finetuned-speech-commands-v2"
ENSEMBLE_AST_SUPERB_MODEL_ID = "ensemble/ast-superb-kws12"
SUPPORTED_EXTERNAL_MODEL_IDS = (
    DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
    DEFAULT_EXTERNAL_AUX_MODEL_ID,
    ENSEMBLE_AST_SUPERB_MODEL_ID,
)

_UNKNOWN_ALIASES = {"unknown", "_unknown_", "<unk>"}
_SILENCE_ALIASES = {"silence", "_silence_", "_background_noise_"}
DEFAULT_EXTERNAL_ENSEMBLE_DEFAULTS: dict[str, float] = {
    "silence_weight": 1.0,
    "unknown_superb_weight": 1.2,
    "unknown_ast_weight": 1.2,
    "target_ast_weight": 1.0,
    "target_superb_residual_weight": 0.2,
    "target_global_scale": 1.0,
    "unknown_bias": 0.0,
}


@dataclass(frozen=True)
class ExternalKWSBatchResult:
    model_id: str
    runtime_device: str
    probs: np.ndarray
    top_indices: np.ndarray
    top_labels: tuple[str, ...]
    margins: np.ndarray


@dataclass(frozen=True)
class ExternalEnsembleBatchResult:
    model_id: str
    runtime_device: str
    ast_probs: np.ndarray
    superb_probs: np.ndarray
    probs: np.ndarray
    top_indices: np.ndarray
    top_labels: tuple[str, ...]
    margins: np.ndarray


@dataclass(frozen=True)
class CachedExternalTeacherTargets:
    probs: torch.Tensor
    sample_weights: torch.Tensor
    agreement_mask: torch.Tensor
    disagreement_mask: torch.Tensor


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(model_id).strip().lower()).strip("_")


def _pick_device(preferred: str = "auto") -> torch.device:
    raw = str(preferred).strip().lower()
    if raw == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if raw == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _label_to_kws12_index(label: str) -> int:
    normalized = _normalize_label(label)
    if normalized in TARGET_KEYWORDS_10:
        return KWS12_TO_INDEX[normalized]
    if normalized in _SILENCE_ALIASES:
        return KWS12_TO_INDEX["silence"]
    if normalized in _UNKNOWN_ALIASES:
        return KWS12_TO_INDEX[UNKNOWN_LABEL]
    return KWS12_TO_INDEX[UNKNOWN_LABEL]


@lru_cache(maxsize=8)
def _load_hf_components(model_id: str, device_type: str):
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except Exception as exc:  # pragma: no cover - dependency error
        raise ImportError("transformers is required for external HF KWS integration") from exc

    model_id = str(model_id).strip()
    if model_id not in SUPPORTED_EXTERNAL_MODEL_IDS:
        raise ValueError(
            f"Unsupported external HF model: {model_id}. "
            f"Supported ids: {', '.join(SUPPORTED_EXTERNAL_MODEL_IDS)}"
        )
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    device = _pick_device(device_type)
    model = model.to(device)
    model.eval()
    return extractor, model, device


def _to_numpy_waveform(waveform: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = np.asarray(waveform, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _aggregate_raw_probs_to_kws12(raw_probs: np.ndarray, id2label: Mapping[int, str]) -> np.ndarray:
    kws12 = np.zeros((len(KWS12_LABELS),), dtype=np.float32)
    for idx, prob in enumerate(np.asarray(raw_probs, dtype=np.float32).reshape(-1).tolist()):
        label = str(id2label.get(int(idx), idx))
        kws12[_label_to_kws12_index(label)] += float(prob)
    total = float(kws12.sum())
    if total <= 0.0:
        kws12[KWS12_TO_INDEX[UNKNOWN_LABEL]] = 1.0
    elif abs(total - 1.0) > 1e-5:
        kws12 = kws12 / total
    return kws12


def default_external_ensemble_calibration(
    *,
    model_id: str = ENSEMBLE_AST_SUPERB_MODEL_ID,
    mode: str = "offline",
) -> dict[str, object]:
    return {
        "version": 1,
        "model_id": str(model_id),
        "mode": str(mode),
        "defaults": dict(DEFAULT_EXTERNAL_ENSEMBLE_DEFAULTS),
        "per_label_bias": {label: 0.0 for label in TARGET_KEYWORDS_10},
    }


def save_external_ensemble_calibration(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_external_ensemble_calibration(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _external_ensemble_defaults(calibration: Mapping[str, object] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_EXTERNAL_ENSEMBLE_DEFAULTS)
    if not isinstance(calibration, Mapping):
        return resolved
    defaults = calibration.get("defaults", {})
    if not isinstance(defaults, Mapping):
        return resolved
    for key, fallback in DEFAULT_EXTERNAL_ENSEMBLE_DEFAULTS.items():
        try:
            resolved[key] = float(defaults.get(key, fallback))
        except (TypeError, ValueError):
            resolved[key] = float(fallback)
    return resolved


def _external_ensemble_biases(calibration: Mapping[str, object] | None) -> dict[str, float]:
    biases = {label: 0.0 for label in TARGET_KEYWORDS_10}
    if not isinstance(calibration, Mapping):
        return biases
    per_label = calibration.get("per_label_bias", {})
    if not isinstance(per_label, Mapping):
        return biases
    for label in TARGET_KEYWORDS_10:
        try:
            biases[label] = float(per_label.get(label, 0.0))
        except (TypeError, ValueError):
            biases[label] = 0.0
    return biases


def _blend_ast_superb_probs(
    ast_probs: np.ndarray,
    superb_probs: np.ndarray,
    *,
    calibration: Mapping[str, object] | None = None,
    silence_weight: float = 1.0,
    unknown_superb_weight: float = 1.2,
    unknown_ast_weight: float = 1.2,
    target_ast_weight: float = 1.0,
    target_superb_residual_weight: float = 0.2,
    target_global_scale: float = 1.0,
    unknown_bias: float = 0.0,
) -> np.ndarray:
    ast = np.asarray(ast_probs, dtype=np.float32)
    superb = np.asarray(superb_probs, dtype=np.float32)
    if ast.shape != superb.shape:
        raise ValueError(f"AST/SUPERB probs must share the same shape, got {ast.shape} vs {superb.shape}")
    if calibration:
        defaults = _external_ensemble_defaults(calibration)
        silence_weight = defaults["silence_weight"]
        unknown_superb_weight = defaults["unknown_superb_weight"]
        unknown_ast_weight = defaults["unknown_ast_weight"]
        target_ast_weight = defaults["target_ast_weight"]
        target_superb_residual_weight = defaults["target_superb_residual_weight"]
        target_global_scale = defaults["target_global_scale"]
        unknown_bias = defaults["unknown_bias"]
    per_label_bias = _external_ensemble_biases(calibration)
    blended = np.zeros_like(ast)
    blended[:, 0] = float(silence_weight) * superb[:, 0]
    blended[:, 1] = (
        float(unknown_superb_weight) * superb[:, 1]
        + float(unknown_ast_weight) * ast[:, 1]
        + float(unknown_bias)
    )
    blended[:, 2:] = (
        (float(target_ast_weight) * ast[:, 2:])
        + (float(target_superb_residual_weight) * superb[:, 2:])
    ) * float(target_global_scale)
    for offset, label in enumerate(TARGET_KEYWORDS_10):
        blended[:, offset + 2] += float(per_label_bias.get(label, 0.0))
    np.maximum(blended, 0.0, out=blended)
    denom = blended.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return blended / denom


def blend_ast_superb_probs(
    ast_probs: np.ndarray,
    superb_probs: np.ndarray,
    *,
    calibration: Mapping[str, object] | None = None,
) -> np.ndarray:
    return _blend_ast_superb_probs(ast_probs, superb_probs, calibration=calibration)


def _top_indices_labels_and_margins(probs: np.ndarray) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected probs shape [N, C], got {arr.shape}")
    top_indices = arr.argmax(axis=1).astype(np.int64, copy=False)
    labels = tuple(KWS12_LABELS[int(idx)] for idx in top_indices.tolist())
    margins = np.zeros((arr.shape[0],), dtype=np.float32)
    if arr.shape[1] <= 1:
        margins[:] = arr[:, 0]
        return top_indices, labels, margins
    for row_idx, row in enumerate(arr):
        top2 = np.partition(row, -2)[-2:]
        margins[row_idx] = float(top2[-1] - top2[-2])
    return top_indices, labels, margins


def predict_kws12_from_waveforms(
    waveforms: Sequence[np.ndarray | torch.Tensor | Sequence[float]],
    *,
    model_id: str = DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
    device: str = "auto",
    sample_rate: int = 16_000,
) -> ExternalKWSBatchResult:
    if str(model_id).strip() == ENSEMBLE_AST_SUPERB_MODEL_ID:
        fused = predict_ensemble_ast_superb_from_waveforms(
            waveforms,
            device=device,
            sample_rate=sample_rate,
        )
        return ExternalKWSBatchResult(
            model_id=str(model_id),
            runtime_device=str(fused.runtime_device),
            probs=fused.probs,
            top_indices=fused.top_indices,
            top_labels=fused.top_labels,
            margins=fused.margins,
        )

    normalized = [_to_numpy_waveform(waveform) for waveform in waveforms]
    extractor, model, runtime_device = _load_hf_components(str(model_id), str(device))
    inputs = extractor(normalized, sampling_rate=int(sample_rate), return_tensors="pt", padding=True)
    inputs = {name: tensor.to(runtime_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        raw_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    id2label = getattr(model.config, "id2label", {})
    probs = np.stack([_aggregate_raw_probs_to_kws12(row, id2label) for row in raw_probs], axis=0) if raw_probs.size else np.zeros((0, len(KWS12_LABELS)), dtype=np.float32)
    top_indices, top_labels, margins = _top_indices_labels_and_margins(probs)
    return ExternalKWSBatchResult(
        model_id=str(model_id),
        runtime_device=str(runtime_device),
        probs=probs,
        top_indices=top_indices,
        top_labels=top_labels,
        margins=margins,
    )


def predict_ensemble_ast_superb_from_waveforms(
    waveforms: Sequence[np.ndarray | torch.Tensor | Sequence[float]],
    *,
    device: str = "auto",
    sample_rate: int = 16_000,
    calibration: Mapping[str, object] | None = None,
) -> ExternalEnsembleBatchResult:
    ast = predict_kws12_from_waveforms(
        waveforms,
        model_id=DEFAULT_EXTERNAL_AUX_MODEL_ID,
        device=device,
        sample_rate=sample_rate,
    )
    superb = predict_kws12_from_waveforms(
        waveforms,
        model_id=DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
        device=device,
        sample_rate=sample_rate,
    )
    probs = _blend_ast_superb_probs(ast.probs, superb.probs, calibration=calibration)
    top_indices, top_labels, margins = _top_indices_labels_and_margins(probs)
    return ExternalEnsembleBatchResult(
        model_id=ENSEMBLE_AST_SUPERB_MODEL_ID,
        runtime_device=str(ast.runtime_device),
        ast_probs=ast.probs,
        superb_probs=superb.probs,
        probs=probs,
        top_indices=top_indices,
        top_labels=top_labels,
        margins=margins,
    )


def predict_kws12_from_paths(
    paths: Sequence[str | Path],
    *,
    model_id: str = DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
    device: str = "auto",
    sample_rate: int = 16_000,
    clip_samples: int = CLIP_SAMPLES,
) -> ExternalKWSBatchResult:
    waveforms = []
    for path in paths:
        waveform = load_audio(path, sample_rate=int(sample_rate))
        waveforms.append(pad_or_trim(waveform, target_samples=int(clip_samples)))
    return predict_kws12_from_waveforms(
        waveforms,
        model_id=model_id,
        device=device,
        sample_rate=sample_rate,
    )


def collect_external_probs_from_loader(
    loader,
    *,
    model_id: str,
    device: str = "auto",
    sample_rate: int = 16_000,
    clip_samples: int = CLIP_SAMPLES,
) -> dict[str, np.ndarray]:
    all_probs = []
    all_targets = []
    for batch in loader:
        result = predict_kws12_from_paths(
            batch.paths,
            model_id=model_id,
            device=device,
            sample_rate=sample_rate,
            clip_samples=clip_samples,
        )
        all_probs.append(result.probs)
        all_targets.append(batch.command_labels.detach().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(KWS12_LABELS)), dtype=np.float32)
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64)
    return {
        "probs": probs,
        "targets": targets,
    }


def benchmark_external_latency_ms(
    *,
    model_id: str,
    device: str = "auto",
    sample_rate: int = 16_000,
    runs: int = 10,
) -> float:
    zero = np.zeros((int(sample_rate),), dtype=np.float32)
    predict_kws12_from_waveforms([zero], model_id=model_id, device=device, sample_rate=sample_rate)
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() and _pick_device(device).type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if start is not None else None
    if start is not None and end is not None:
        start.record()
        for _ in range(max(1, int(runs))):
            predict_kws12_from_waveforms([zero], model_id=model_id, device=device, sample_rate=sample_rate)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / max(1, int(runs)))

    import time

    begin = time.perf_counter()
    for _ in range(max(1, int(runs))):
        predict_kws12_from_waveforms([zero], model_id=model_id, device=device, sample_rate=sample_rate)
    return float((time.perf_counter() - begin) * 1000.0 / max(1, int(runs)))


def fit_external_verifier_calibration(
    *,
    probs: np.ndarray,
    targets_kws12: np.ndarray,
    model_id: str,
    backend: str = "external",
    precision_target: float = 0.95,
    fit_split: str = "valid",
) -> dict[str, object]:
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets_kws12, dtype=np.int64)
    if probs.size == 0 or targets.size == 0:
        return {
            "default": {"min_accept_prob": 0.65, "min_margin": 0.08, "accept_precision": 0.0, "accept_recall": 0.0, "num_accepted": 0},
            "per_label": {},
            "backend": str(backend),
            "model_id": str(model_id),
            "fit_split": str(fit_split),
            "target_precision": float(precision_target),
            "num_eval_samples": 0,
        }

    preds = probs.argmax(axis=1).astype(np.int64, copy=False)
    top_probs = probs[np.arange(probs.shape[0]), preds]
    margins = np.zeros((probs.shape[0],), dtype=np.float32)
    if probs.shape[1] > 1:
        for idx, row in enumerate(probs):
            top2 = np.partition(row, -2)[-2:]
            margins[idx] = float(top2[-1] - top2[-2])
    else:
        margins[:] = top_probs

    prob_candidates = sorted({round(float(v), 2) for v in np.linspace(0.50, 0.99, 13)} | {0.65})
    margin_candidates = sorted({round(float(v), 2) for v in np.linspace(0.00, 0.40, 21)} | {0.08})

    def _fit(candidate_idx: int | None) -> dict[str, float | int]:
        if candidate_idx is None:
            accepted_base = preds >= KWS12_TO_INDEX[TARGET_KEYWORDS_10[0]]
            correct_mask = (preds == targets) & (targets >= KWS12_TO_INDEX[TARGET_KEYWORDS_10[0]])
            positive_mask = targets >= KWS12_TO_INDEX[TARGET_KEYWORDS_10[0]]
            negative_mask = ~positive_mask
        else:
            accepted_base = preds == candidate_idx
            correct_mask = targets == candidate_idx
            positive_mask = targets == candidate_idx
            negative_mask = targets != candidate_idx

        best_key: tuple[bool, float, float, float, float] | None = None
        best_payload: dict[str, float | int] | None = None
        for prob_thr in prob_candidates:
            for margin_thr in margin_candidates:
                accepted = accepted_base & (top_probs >= prob_thr) & (margins >= margin_thr)
                precision = float((accepted & correct_mask).sum() / max(int(accepted.sum()), 1))
                recall = float((accepted & correct_mask).sum() / max(int(positive_mask.sum()), 1))
                key = (
                    precision >= float(precision_target),
                    recall,
                    precision,
                    -float(prob_thr),
                    -float(margin_thr),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "min_accept_prob": float(prob_thr),
                        "min_margin": float(margin_thr),
                        "accept_precision": precision,
                        "accept_recall": recall,
                        "num_accepted": int(accepted.sum()),
                        "false_accepts": int((accepted & negative_mask).sum()),
                    }
        if best_payload is None:
            return {
                "min_accept_prob": 0.65,
                "min_margin": 0.08,
                "accept_precision": 0.0,
                "accept_recall": 0.0,
                "num_accepted": 0,
                "false_accepts": 0,
            }
        return best_payload

    default_payload = _fit(None)
    per_label = {
        label: _fit(KWS12_TO_INDEX[label])
        for label in TARGET_KEYWORDS_10
    }
    return {
        "default": default_payload,
        "per_label": per_label,
        "backend": str(backend),
        "model_id": str(model_id),
        "fit_split": str(fit_split),
        "target_precision": float(precision_target),
        "num_eval_samples": int(targets.size),
        "min_accept_prob": float(default_payload["min_accept_prob"]),
        "min_margin": float(default_payload["min_margin"]),
    }


class ExternalKWSLogitCache:
    """On-disk cache for imported external teacher KWS12 probabilities."""

    def __init__(
        self,
        *,
        primary_model_id: str,
        aux_model_id: str | None,
        cache_dir: str | Path,
        device: str,
        clip_samples: int,
        sample_rate: int = 16_000,
        agreement_weight: float = 0.25,
    ) -> None:
        self.primary_model_id = str(primary_model_id)
        self.aux_model_id = str(aux_model_id).strip() if aux_model_id else ""
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device)
        self.clip_samples = int(clip_samples)
        self.sample_rate = int(sample_rate)
        self.agreement_weight = float(max(0.0, agreement_weight))
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_path(self, audio_path: str) -> Path:
        resolved = str(Path(audio_path).expanduser().resolve())
        digest = hashlib.sha1(
            f"{self.primary_model_id}|{self.aux_model_id}|{self.sample_rate}|{self.clip_samples}|{resolved}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def load_targets(self, paths: Iterable[str], *, device: torch.device) -> CachedExternalTeacherTargets:
        paths_list = [str(path) for path in paths]
        cached: list[dict[str, object] | None] = [None] * len(paths_list)
        missing_indices: list[int] = []
        missing_paths: list[str] = []

        for idx, path in enumerate(paths_list):
            cache_path = self._cache_path(path)
            if cache_path.exists():
                cached[idx] = json.loads(cache_path.read_text(encoding="utf-8"))
                self.cache_hits += 1
            else:
                missing_indices.append(idx)
                missing_paths.append(path)

        if missing_paths:
            primary = predict_kws12_from_paths(
                missing_paths,
                model_id=self.primary_model_id,
                device=self.device,
                sample_rate=self.sample_rate,
                clip_samples=self.clip_samples,
            )
            aux = (
                predict_kws12_from_paths(
                    missing_paths,
                    model_id=self.aux_model_id,
                    device=self.device,
                    sample_rate=self.sample_rate,
                    clip_samples=self.clip_samples,
                )
                if self.aux_model_id
                else None
            )
            for local_idx, global_idx in enumerate(missing_indices):
                primary_probs = primary.probs[local_idx].astype(np.float32, copy=False)
                aux_probs = aux.probs[local_idx].astype(np.float32, copy=False) if aux is not None else None
                agreement = bool(aux is None or int(primary.top_indices[local_idx]) == int(aux.top_indices[local_idx]))
                disagreement = bool(aux is not None and not agreement)
                blended = (
                    0.5 * (primary_probs + aux_probs)
                    if aux_probs is not None and agreement
                    else primary_probs
                )
                payload = {
                    "probs": blended.tolist(),
                    "agreement": agreement,
                    "disagreement": disagreement,
                }
                cache_path = self._cache_path(paths_list[global_idx])
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                cached[global_idx] = payload
                self.cache_misses += 1

        probs = torch.tensor([entry["probs"] for entry in cached if entry is not None], dtype=torch.float32, device=device)
        agreement_mask = torch.tensor([bool(entry["agreement"]) for entry in cached if entry is not None], dtype=torch.bool, device=device)
        disagreement_mask = torch.tensor([bool(entry["disagreement"]) for entry in cached if entry is not None], dtype=torch.bool, device=device)
        sample_weights = torch.ones((len(paths_list),), dtype=torch.float32, device=device)
        sample_weights = torch.where(
            agreement_mask,
            sample_weights * float(1.0 + self.agreement_weight),
            sample_weights,
        )
        return CachedExternalTeacherTargets(
            probs=probs,
            sample_weights=sample_weights,
            agreement_mask=agreement_mask,
            disagreement_mask=disagreement_mask,
        )
