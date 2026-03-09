"""Rank checkpoints for demo selection."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from kws.constants import COMMAND31_LABELS, KWS12_LABELS, command31_to_kws12
from kws.data.audio import MelFrontend
from kws.models import create_model
from kws.train.engine import pick_device
from kws.utils.keyword_focus import compute_keyword_balance, compute_per_keyword_from_confusion


DEFAULT_FAST_REPORT_PATH = Path("reports/demo_model_ranking.json")
DEFAULT_BALANCED_REPORT_PATH = Path("reports/demo_model_ranking_balanced.json")
DEFAULT_STABLE_REPORT_PATH = Path("reports/demo_model_ranking_stable.json")
DEFAULT_REPORT_PATH = DEFAULT_FAST_REPORT_PATH
DEFAULT_LATENCY_TARGET_MS = 100.0
DEFAULT_LATENCY_FALLBACK_MS = 130.0
DEFAULT_SELECTION_PROFILE = "stable"
SELECTION_PROFILES = ("stable", "balanced", "fast")


@dataclass(frozen=True)
class Candidate:
    run: str
    checkpoint: Path
    runtime_device: str
    kws12_acc: float
    min_kws12_precision: float
    min_kws12_recall: float
    kws12_target_recall: float
    kws12_unknown_to_target_rate: float
    focus_keyword_recall_mean: float
    focus_pair_confusion_rate: float
    bottom3_keyword_recall: float
    keyword_balance_gap: float
    wake_frr_at_1fa_per_hour: float
    latency_ms: float
    latency_bucket: int
    latency_target_met: bool
    selection_passed_fp_guardrail: bool
    score: float


def _iter_candidate_checkpoints(root: Path) -> Iterable[Tuple[str, Path]]:
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        best_kws = run_dir / "best_kws12.pt"
        best_wake = run_dir / "best_wake_frr.pt"
        if best_kws.exists():
            yield run_dir.name, best_kws
        elif best_wake.exists():
            yield run_dir.name, best_wake


def _best_valid_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {
            "kws12_acc": -1.0,
            "min_kws12_precision": 0.0,
            "min_kws12_recall": 0.0,
            "wake_frr_at_1fa_per_hour": 1.0,
            "kws12_target_recall": 0.0,
            "kws12_unknown_to_target_rate": 1.0,
            "focus_keyword_recall_mean": 0.0,
            "focus_pair_confusion_rate": 1.0,
            "bottom3_keyword_recall": 0.0,
            "keyword_balance_gap": 1.0,
        }
    best = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            valid = row.get("valid_metrics", {})
            candidate = {
                "kws12_acc": float(valid.get("kws12_acc", -1.0)),
                "min_kws12_precision": float(valid.get("min_kws12_precision", valid.get("kws12_target_precision", valid.get("kws12_acc", 0.0)))),
                "min_kws12_recall": float(valid.get("min_kws12_recall", valid.get("kws12_target_recall", 0.0))),
                "wake_frr_at_1fa_per_hour": float(valid.get("wake_frr_at_1fa_per_hour", 1.0)),
                "kws12_target_recall": float(valid.get("kws12_target_recall", 0.0)),
                "kws12_unknown_to_target_rate": float(valid.get("kws12_unknown_to_target_rate", 1.0)),
                "focus_keyword_recall_mean": float(valid.get("focus_keyword_recall_mean", 0.0)),
                "focus_pair_confusion_rate": float(valid.get("focus_pair_confusion_rate", 1.0)),
                "bottom3_keyword_recall": float(valid.get("bottom3_keyword_recall", 0.0)),
                "keyword_balance_gap": float(valid.get("keyword_balance_gap", 1.0)),
            }
            if best is None or candidate["kws12_acc"] > best["kws12_acc"]:
                best = candidate
    if best is None:
        best = {
            "kws12_acc": -1.0,
            "min_kws12_precision": 0.0,
            "min_kws12_recall": 0.0,
            "wake_frr_at_1fa_per_hour": 1.0,
            "kws12_target_recall": 0.0,
            "kws12_unknown_to_target_rate": 1.0,
            "focus_keyword_recall_mean": 0.0,
            "focus_pair_confusion_rate": 1.0,
            "bottom3_keyword_recall": 0.0,
            "keyword_balance_gap": 1.0,
        }
    return best


def _report_path_for_profile(selection_profile: str) -> Path:
    selection_profile = str(selection_profile).lower().strip()
    if selection_profile == "fast":
        return DEFAULT_FAST_REPORT_PATH
    if selection_profile == "balanced":
        return DEFAULT_BALANCED_REPORT_PATH
    return DEFAULT_STABLE_REPORT_PATH


def _normalize_selection_profile(selection_profile: str) -> str:
    profile = str(selection_profile).lower().strip()
    if profile not in SELECTION_PROFILES:
        raise ValueError(f"Unknown selection profile: {selection_profile}")
    return profile


def _per_class_kws12_from_command_confusion(confusion: object) -> Dict[str, Dict[str, float]]:
    cm31 = np.asarray(confusion, dtype=np.int64)
    if cm31.ndim != 2 or cm31.shape[0] < len(COMMAND31_LABELS) or cm31.shape[1] < len(COMMAND31_LABELS):
        return {}
    size = len(KWS12_LABELS)
    cm12 = np.zeros((size, size), dtype=np.int64)
    for src_idx, src_label in enumerate(COMMAND31_LABELS):
        src_kws = int(command31_to_kws12(src_label))
        for dst_idx, dst_label in enumerate(COMMAND31_LABELS):
            dst_kws = int(command31_to_kws12(dst_label))
            cm12[src_kws, dst_kws] += int(cm31[src_idx, dst_idx])

    out: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(KWS12_LABELS):
        row = cm12[idx]
        col = cm12[:, idx]
        support = int(row.sum())
        predicted = int(col.sum())
        tp = int(cm12[idx, idx])
        precision = float(tp / max(predicted, 1))
        recall = float(tp / max(support, 1))
        denom = precision + recall
        f1 = float((2.0 * precision * recall) / denom) if denom > 0.0 else 0.0
        confusions = [
            {"label": KWS12_LABELS[j], "count": int(count)}
            for j, count in enumerate(row.tolist())
            if j != idx and int(count) > 0
        ]
        confusions.sort(key=lambda item: (-item["count"], item["label"]))
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "top_confusions": confusions[:3],
        }
    return out


def _extract_per_class_kws12(payload: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    per_class = payload.get("per_class_kws12") or metrics.get("per_class_kws12")
    if isinstance(per_class, dict) and per_class:
        return {
            str(label): {
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1": float(stats.get("f1", 0.0)),
                "support": int(stats.get("support", 0)),
                "predicted": int(stats.get("predicted", 0)),
                "top_confusions": list(stats.get("top_confusions", [])),
            }
            for label, stats in per_class.items()
            if isinstance(stats, dict)
        }
    return _per_class_kws12_from_command_confusion(metrics.get("command_confusion", []))


def _min_per_class_metric(per_class: Dict[str, Dict[str, float]], key: str) -> float:
    values = [float(stats.get(key, 0.0)) for label, stats in per_class.items() if label in KWS12_LABELS]
    return float(min(values)) if values else 0.0


def _report_metrics_for_run(run: str, ckpt_path: Path) -> Dict[str, float]:
    project_root = ckpt_path.parent.parent.parent.resolve()
    reports_root = project_root / "reports"
    candidates = [ckpt_path.parent / "demo_analysis.json"]
    candidates.extend(sorted(reports_root.glob(f"{run}*analysis*.json")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            per_class = _extract_per_class_kws12(payload, metrics)
            keyword_focus = payload.get("keyword_focus", {}) if isinstance(payload.get("keyword_focus", {}), dict) else {}
            per_keyword = payload.get("per_keyword") or metrics.get("per_keyword") or {}
            if not per_keyword:
                per_keyword = compute_per_keyword_from_confusion(metrics.get("command_confusion", []))
            keyword_balance = compute_keyword_balance(per_keyword)
            return {
                "kws12_acc": float(metrics.get("kws12_acc", -1.0)),
                "min_kws12_precision": float(
                    payload.get(
                        "min_kws12_precision",
                        metrics.get(
                            "min_kws12_precision",
                            metrics.get("kws12_target_precision", _min_per_class_metric(per_class, "precision")),
                        ),
                    )
                ),
                "min_kws12_recall": float(
                    payload.get(
                        "min_kws12_recall",
                        metrics.get(
                            "min_kws12_recall",
                            metrics.get("kws12_target_recall", _min_per_class_metric(per_class, "recall")),
                        ),
                    )
                ),
                "wake_frr_at_1fa_per_hour": float(metrics.get("wake_frr_at_1fa_per_hour", 1.0)),
                "kws12_target_recall": float(metrics.get("kws12_target_recall", 0.0)),
                "kws12_unknown_to_target_rate": float(metrics.get("kws12_unknown_to_target_rate", 1.0)),
                "focus_keyword_recall_mean": float(payload.get("focus_keyword_recall_mean", keyword_focus.get("focus_keyword_recall_mean", 0.0))),
                "focus_pair_confusion_rate": float(payload.get("focus_pair_confusion_rate", keyword_focus.get("focus_pair_confusion_rate", 1.0))),
                "bottom3_keyword_recall": float(payload.get("bottom3_keyword_recall", keyword_balance.get("bottom3_keyword_recall", 0.0))),
                "keyword_balance_gap": float(payload.get("keyword_balance_gap", keyword_balance.get("keyword_balance_gap", 1.0))),
            }
        except Exception:
            continue
    return {}


def _feature_frames(cfg: Dict[str, object]) -> Tuple[int, int]:
    features = cfg.get("features", {})
    sr = int(features.get("sample_rate", 16000))
    audio_seconds = float(features.get("audio_seconds", 1.0))
    hop = int(features.get("hop_length", 128))
    n_mels = int(features.get("n_mels", 80))
    n_samples = int(round(sr * audio_seconds))
    frames = int(n_samples // hop) + 1
    return n_mels, frames


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _aggregate_command_probs_to_kws12(command_probs: np.ndarray, command31_labels: List[str]) -> np.ndarray:
    kws12_probs = np.zeros(len(KWS12_LABELS), dtype=np.float32)
    for idx, label in enumerate(command31_labels):
        kws12_probs[command31_to_kws12(label)] += float(command_probs[idx])
    return kws12_probs


def benchmark_latency_ms(checkpoint: Dict[str, object], device: torch.device, iters: int = 30) -> float:
    cfg = checkpoint["config"]
    features = cfg["features"]
    sample_rate = int(features.get("sample_rate", 16000))
    audio_seconds = float(features.get("audio_seconds", 1.0))
    n_samples = int(round(sample_rate * audio_seconds))
    num_commands = len(checkpoint["label_set"])

    model = create_model(cfg["model"], n_mels=int(features.get("n_mels", 80)), num_commands=num_commands)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    frontend = MelFrontend(
        sample_rate=sample_rate,
        n_fft=int(features.get("n_fft", 1024)),
        hop_length=int(features.get("hop_length", 128)),
        n_mels=int(features.get("n_mels", 80)),
        f_min=float(features.get("f_min", 20.0)),
        f_max=float(features.get("f_max", 7600.0)),
    )
    command31_labels = list(checkpoint["label_set"])
    waveform = torch.randn(n_samples, dtype=torch.float32) * 0.01

    def _run_once() -> None:
        feature = frontend(waveform)
        mean = feature.mean()
        std = feature.std().clamp(min=1e-5)
        x = ((feature - mean) / std).unsqueeze(0).to(device)
        out = model(x)
        probs = torch.softmax(out.command_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        wake_prob = float(torch.sigmoid(out.wake_logits).squeeze(0).item())
        kws12_probs = _aggregate_command_probs_to_kws12(probs, command31_labels)
        _ = KWS12_LABELS[int(np.argmax(kws12_probs))]
        _ = wake_prob >= 0.5

    with torch.no_grad():
        for _ in range(10):
            _run_once()
        if device.type in ("cuda", "mps"):
            _synchronize(device)

        timings: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _run_once()
            if device.type in ("cuda", "mps"):
                _synchronize(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return float(np.median(np.asarray(timings, dtype=np.float64)))


def _normalize(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        return [0.5 for _ in values]
    return [float((v - vmin) / (vmax - vmin)) for v in values]


def rank_checkpoints(
    *,
    outputs_root: Path,
    device: torch.device | str,
    metric_balance: Tuple[float, float] = (0.7, 0.3),
    benchmark_iters: int = 30,
    latency_target_ms: float = DEFAULT_LATENCY_TARGET_MS,
    latency_fallback_ms: float = DEFAULT_LATENCY_FALLBACK_MS,
    selection_profile: str = DEFAULT_SELECTION_PROFILE,
) -> List[Candidate]:
    selection_profile = _normalize_selection_profile(selection_profile)
    outputs_root = outputs_root.expanduser().resolve()
    if isinstance(device, str):
        requested = str(device).lower().strip()
        if requested == "auto":
            runtime_devices = [torch.device("cpu")]
            if torch.backends.mps.is_available():
                runtime_devices.append(torch.device("mps"))
            if torch.cuda.is_available():
                runtime_devices.append(torch.device("cuda"))
        else:
            runtime_devices = [pick_device(requested)]
    else:
        runtime_devices = [device]

    candidates: List[Tuple[str, Path, Dict[str, float]]] = []
    for run, ckpt_path in _iter_candidate_checkpoints(outputs_root):
        metrics_path = ckpt_path.parent / "metrics_history.jsonl"
        valid = _best_valid_metrics(metrics_path)
        fallback = _report_metrics_for_run(run, ckpt_path)
        needs_fallback = (
            float(valid.get("kws12_target_recall", 0.0)) <= 0.0
            or float(valid.get("focus_keyword_recall_mean", 0.0)) <= 0.0
            or float(valid.get("focus_pair_confusion_rate", 1.0)) >= 0.999
            or float(valid.get("bottom3_keyword_recall", 0.0)) <= 0.0
            or float(valid.get("keyword_balance_gap", 1.0)) >= 0.999
        )
        if needs_fallback and fallback:
            valid = {
                "kws12_acc": float(valid.get("kws12_acc", fallback.get("kws12_acc", -1.0))),
                "min_kws12_precision": float(
                    valid.get(
                        "min_kws12_precision",
                        fallback.get("min_kws12_precision", valid.get("kws12_target_precision", valid.get("kws12_acc", 0.0))),
                    )
                ),
                "min_kws12_recall": float(
                    valid.get(
                        "min_kws12_recall",
                        fallback.get("min_kws12_recall", valid.get("kws12_target_recall", 0.0)),
                    )
                ),
                "wake_frr_at_1fa_per_hour": float(
                    valid.get("wake_frr_at_1fa_per_hour", fallback.get("wake_frr_at_1fa_per_hour", 1.0))
                ),
                "kws12_target_recall": float(fallback.get("kws12_target_recall", valid.get("kws12_target_recall", 0.0))),
                "kws12_unknown_to_target_rate": float(
                    fallback.get("kws12_unknown_to_target_rate", valid.get("kws12_unknown_to_target_rate", 1.0))
                ),
                "focus_keyword_recall_mean": float(
                    fallback.get("focus_keyword_recall_mean", valid.get("focus_keyword_recall_mean", 0.0))
                ),
                "focus_pair_confusion_rate": float(
                    fallback.get("focus_pair_confusion_rate", valid.get("focus_pair_confusion_rate", 1.0))
                ),
                "bottom3_keyword_recall": float(
                    fallback.get("bottom3_keyword_recall", valid.get("bottom3_keyword_recall", 0.0))
                ),
                "keyword_balance_gap": float(
                    fallback.get("keyword_balance_gap", valid.get("keyword_balance_gap", 1.0))
                ),
            }
        candidates.append((run, ckpt_path, valid))

    if not candidates:
        return []

    baseline_unknown_rate = None
    for run, _ckpt_path, valid in candidates:
        if run == "quick_mhatt":
            baseline_unknown_rate = float(valid.get("kws12_unknown_to_target_rate", 1.0))
            break
    if baseline_unknown_rate is None:
        baseline_unknown_rate = min(float(valid.get("kws12_unknown_to_target_rate", 1.0)) for _, _, valid in candidates)
    fp_guardrail = float(baseline_unknown_rate + 0.02)

    ranked: List[Candidate] = []
    for run, ckpt_path, valid in candidates:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for runtime_device in runtime_devices:
            latency_ms = benchmark_latency_ms(ckpt, device=runtime_device, iters=benchmark_iters)
            if latency_ms <= latency_target_ms:
                latency_bucket = 0
            elif latency_ms <= latency_fallback_ms:
                latency_bucket = 1
            else:
                latency_bucket = 2
            target_recall = float(valid.get("kws12_target_recall", 0.0))
            min_precision = float(valid.get("min_kws12_precision", 0.0))
            min_recall = float(valid.get("min_kws12_recall", 0.0))
            unknown_to_target_rate = float(valid.get("kws12_unknown_to_target_rate", 1.0))
            focus_recall = float(valid.get("focus_keyword_recall_mean", 0.0))
            focus_conf_rate = float(valid.get("focus_pair_confusion_rate", 1.0))
            bottom3_recall = float(valid.get("bottom3_keyword_recall", 0.0))
            balance_gap = float(valid.get("keyword_balance_gap", 1.0))
            frr = float(valid.get("wake_frr_at_1fa_per_hour", 1.0))
            acc = float(valid.get("kws12_acc", -1.0))
            passed_fp_guardrail = bool(unknown_to_target_rate <= fp_guardrail)
            score = (
                (2.4 * focus_recall)
                - (1.4 * focus_conf_rate)
                + (1.8 * bottom3_recall)
                + (1.4 * target_recall)
                + acc
                - (0.30 * unknown_to_target_rate)
                - (0.25 * balance_gap)
                - (0.05 * float(latency_bucket))
                - (0.02 * frr)
                - (float(latency_ms) / 10000.0)
            )
            ranked.append(
                Candidate(
                    run=run,
                    checkpoint=ckpt_path.resolve(),
                    runtime_device=runtime_device.type,
                    kws12_acc=acc,
                    min_kws12_precision=min_precision,
                    min_kws12_recall=min_recall,
                    kws12_target_recall=target_recall,
                    kws12_unknown_to_target_rate=unknown_to_target_rate,
                    focus_keyword_recall_mean=focus_recall,
                    focus_pair_confusion_rate=focus_conf_rate,
                    bottom3_keyword_recall=bottom3_recall,
                    keyword_balance_gap=balance_gap,
                    wake_frr_at_1fa_per_hour=frr,
                    latency_ms=float(latency_ms),
                    latency_bucket=int(latency_bucket),
                    latency_target_met=bool(latency_ms <= latency_target_ms),
                    selection_passed_fp_guardrail=passed_fp_guardrail,
                    score=score,
                )
            )

    if selection_profile == "fast":
        ranked.sort(
            key=lambda c: (
                c.latency_bucket,
                int(not c.selection_passed_fp_guardrail),
                -c.focus_keyword_recall_mean,
                c.focus_pair_confusion_rate,
                -c.kws12_target_recall,
                -c.kws12_acc,
                c.latency_ms,
                c.wake_frr_at_1fa_per_hour,
                c.run,
            )
        )
    elif selection_profile == "balanced":
        ranked.sort(
            key=lambda c: (
                c.latency_bucket,
                int(not c.selection_passed_fp_guardrail),
                -c.min_kws12_precision,
                -c.focus_keyword_recall_mean,
                c.focus_pair_confusion_rate,
                -c.min_kws12_recall,
                c.kws12_unknown_to_target_rate,
                -c.kws12_acc,
                c.latency_ms,
                c.run,
            )
        )
    else:
        ranked.sort(
            key=lambda c: (
                c.latency_bucket,
                int(not c.selection_passed_fp_guardrail),
                -c.min_kws12_precision,
                -c.min_kws12_recall,
                c.kws12_unknown_to_target_rate,
                -c.kws12_acc,
                c.latency_ms,
                c.run,
            )
        )
    return ranked


def select_best_checkpoint(
    *,
    outputs_root: str | Path = "outputs",
    device: str = "auto",
    metric_balance: Tuple[float, float] = (0.7, 0.3),
    benchmark_iters: int = 30,
    report_path: str | Path | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
    selection_profile: str = DEFAULT_SELECTION_PROFILE,
) -> Tuple[Path, str]:
    selection_profile = _normalize_selection_profile(selection_profile)
    if report_path is None or str(report_path).strip() == "":
        report_path = _report_path_for_profile(selection_profile)
    report_path = Path(report_path).expanduser().resolve()
    if use_cache and not rebuild and report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            if str(payload.get("selection_profile", selection_profile)).lower().strip() != selection_profile:
                raise ValueError("selection profile mismatch")
            chosen = Path(payload.get("chosen", {}).get("checkpoint", "")).expanduser()
            runtime_device = str(payload.get("chosen", {}).get("runtime_device", "")).strip()
            if chosen.exists() and runtime_device:
                return chosen.resolve(), runtime_device
        except Exception:
            pass

    ranked = rank_checkpoints(
        outputs_root=Path(outputs_root),
        device=device,
        metric_balance=metric_balance,
        benchmark_iters=int(benchmark_iters),
        selection_profile=selection_profile,
    )
    if not ranked:
        raise FileNotFoundError(f"No checkpoints found under {Path(outputs_root).resolve()}")

    guardrail_candidates = [c for c in ranked if c.latency_target_met and c.selection_passed_fp_guardrail]
    chosen = guardrail_candidates[0] if guardrail_candidates else next(
        (c for c in ranked if c.run == "quick_mhatt" and c.runtime_device == "cpu"),
        ranked[0],
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_profile": selection_profile,
        "metric_balance": {"accuracy": float(metric_balance[0]), "latency": float(metric_balance[1])},
        "requested_device": str(device),
        "latency_target_ms": DEFAULT_LATENCY_TARGET_MS,
        "latency_fallback_ms": DEFAULT_LATENCY_FALLBACK_MS,
        "fp_guardrail_margin": 0.02,
        "candidates": [
            {
                "run": c.run,
                "checkpoint": str(c.checkpoint),
                "runtime_device": c.runtime_device,
                "kws12_acc": c.kws12_acc,
                "min_kws12_precision": c.min_kws12_precision,
                "min_kws12_recall": c.min_kws12_recall,
                "kws12_target_recall": c.kws12_target_recall,
                "kws12_unknown_to_target_rate": c.kws12_unknown_to_target_rate,
                "focus_keyword_recall_mean": c.focus_keyword_recall_mean,
                "focus_pair_confusion_rate": c.focus_pair_confusion_rate,
                "bottom3_keyword_recall": c.bottom3_keyword_recall,
                "keyword_balance_gap": c.keyword_balance_gap,
                "wake_frr_at_1fa_per_hour": c.wake_frr_at_1fa_per_hour,
                "latency_ms": c.latency_ms,
                "latency_bucket": c.latency_bucket,
                "latency_target_met": c.latency_target_met,
                "selection_passed_fp_guardrail": c.selection_passed_fp_guardrail,
                "score": c.score,
            }
            for c in ranked
        ],
        "chosen": {
            "run": chosen.run,
            "checkpoint": str(chosen.checkpoint),
            "runtime_device": chosen.runtime_device,
            "kws12_acc": chosen.kws12_acc,
            "min_kws12_precision": chosen.min_kws12_precision,
            "min_kws12_recall": chosen.min_kws12_recall,
            "kws12_target_recall": chosen.kws12_target_recall,
            "kws12_unknown_to_target_rate": chosen.kws12_unknown_to_target_rate,
            "focus_keyword_recall_mean": chosen.focus_keyword_recall_mean,
            "focus_pair_confusion_rate": chosen.focus_pair_confusion_rate,
            "bottom3_keyword_recall": chosen.bottom3_keyword_recall,
            "keyword_balance_gap": chosen.keyword_balance_gap,
            "wake_frr_at_1fa_per_hour": chosen.wake_frr_at_1fa_per_hour,
            "latency_ms": chosen.latency_ms,
            "latency_bucket": chosen.latency_bucket,
            "latency_target_met": chosen.latency_target_met,
            "selection_passed_fp_guardrail": chosen.selection_passed_fp_guardrail,
            "score": chosen.score,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return chosen.checkpoint, chosen.runtime_device


def _parse_metric_balance(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("metric balance must be 'acc,lat' e.g. 0.7,0.3")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank checkpoints for demo selection")
    parser.add_argument("--root", type=str, default="outputs", help="Outputs root directory")
    parser.add_argument("--device", type=str, default="auto", help="auto|mps|cpu|cuda")
    parser.add_argument("--selection-profile", type=str, default=DEFAULT_SELECTION_PROFILE, choices=list(SELECTION_PROFILES))
    parser.add_argument("--metric-balance", type=str, default="0.7,0.3", help="accuracy,latency weights")
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_balance = _parse_metric_balance(args.metric_balance)
    chosen, runtime_device = select_best_checkpoint(
        outputs_root=args.root,
        device=args.device,
        metric_balance=metric_balance,
        benchmark_iters=int(args.benchmark_iters),
        report_path=args.output or None,
        use_cache=not args.no_cache,
        rebuild=bool(args.rebuild),
        selection_profile=args.selection_profile,
    )
    report_path = Path(args.output).expanduser().resolve() if args.output else _report_path_for_profile(args.selection_profile).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    print(f"Chosen checkpoint: {chosen} on {runtime_device}")


if __name__ == "__main__":
    main()
