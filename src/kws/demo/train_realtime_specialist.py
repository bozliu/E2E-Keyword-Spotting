"""Train a hard-word realtime specialist from cached stream traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kws.constants import INDEX_TO_COMMAND31, KWS12_LABELS, TARGET_KEYWORDS_10, UNKNOWN_LABEL, command31_to_kws12
from kws.data.audio import load_audio
from kws.data.manifest import read_manifest
from kws.demo.realtime import (
    DEFAULT_DEMO_PROFILE,
    REALTIME_TUNED_DEMO_PROFILE,
    ResolvedRealtimeProfile,
    get_sensitivity_tuning,
    load_realtime_demo,
)
from kws.demo.realtime_specialist import (
    HARD_WORD_SPECIALIST_LABEL_TO_INDEX,
    HARD_WORD_SPECIALIST_TARGETS,
    default_realtime_specialist_calibration,
    save_realtime_specialist_artifact,
    save_realtime_specialist_calibration,
    summarize_realtime_specialist_predictions,
    train_realtime_specialist,
)
from kws.demo.realtime_trace import estimate_utterance_bounds, load_trace, reconstruct_trace_frame_waveforms, replay_trace_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the realtime hard-word specialist from cached traces.")
    parser.add_argument("--demo-profile", type=str, default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--selection-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--wheel", type=str, default="kws12", choices=["kws12", "target10"])
    parser.add_argument("--runtime-label-backend", type=str, default="")
    parser.add_argument("--external-kws-model", type=str, default="ensemble/ast-superb-kws12")
    parser.add_argument("--external-kws-device", type=str, default="mps")
    parser.add_argument("--keyword-calibration-path", type=str, default="")
    parser.add_argument("--external-ensemble-calibration-path", type=str, default="")
    parser.add_argument("--train-trace-manifest", type=str, default="cache/realtime_traces/train/manifest.json")
    parser.add_argument("--valid-trace-manifest", type=str, default="cache/realtime_traces/valid/manifest.json")
    parser.add_argument("--test-trace-manifest", type=str, default="")
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--valid-split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--other-target-quota", type=int, default=180)
    parser.add_argument("--unknown-quota", type=int, default=600)
    parser.add_argument("--silence-quota", type=int, default=180)
    parser.add_argument("--gate-mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--sensitivity-profile", type=str, default="strict", choices=["high", "balanced", "strict"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--wake-open-thr", type=float, default=0.6)
    parser.add_argument("--wake-close-thr", type=float, default=0.5)
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--cmd-conf-thr", type=float, default=None)
    parser.add_argument("--display-conf-thr", type=float, default=None)
    parser.add_argument("--display-wake-thr", type=float, default=None)
    parser.add_argument("--vote-window", type=int, default=None)
    parser.add_argument("--vote-min-count", type=int, default=None)
    parser.add_argument("--hop-seconds", type=float, default=0.10)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--hold-ms", type=float, default=300.0)
    parser.add_argument("--pre-silence-seconds", type=float, default=0.4)
    parser.add_argument("--post-silence-seconds", type=float, default=0.4)
    parser.add_argument("--match-tail-seconds", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--output-specialist", type=str, default="")
    parser.add_argument("--output-calibration", type=str, default="")
    parser.add_argument("--output-valid-report", type=str, default="reports/realtime_specialist_valid.json")
    parser.add_argument("--output-test-report", type=str, default="reports/realtime_specialist_test.json")
    return parser.parse_args()


def _load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Trace manifest not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _bundle_with_realtime_profile(bundle):
    profile = ResolvedRealtimeProfile(
        demo_profile=REALTIME_TUNED_DEMO_PROFILE,
        detector_device_preference=bundle.resolved_profile.detector_device_preference,
        runtime_label_backend=bundle.resolved_profile.runtime_label_backend,
        external_kws_model=bundle.resolved_profile.external_kws_model,
        external_kws_device=bundle.resolved_profile.external_kws_device,
    )
    return replace(bundle, resolved_profile=profile)


def _segment_waveform(
    *,
    timestamps: np.ndarray,
    frame_waveforms: list[np.ndarray],
    segment: dict[str, Any],
) -> np.ndarray:
    if not frame_waveforms:
        raise ValueError("frame_waveforms must be non-empty")
    peak_time = float(segment.get("hard_peak_time", 0.0))
    if peak_time <= 0.0:
        peak_time = 0.5 * (float(segment.get("start_time", 0.0)) + float(segment.get("end_time", 0.0)))
    idx = int(np.argmin(np.abs(timestamps - peak_time)))
    idx = max(0, min(idx, len(frame_waveforms) - 1))
    return np.asarray(frame_waveforms[idx], dtype=np.float32)


def _collect_specialist_samples(manifest: dict[str, Any], *, bundle, args) -> tuple[np.ndarray, np.ndarray]:
    tuning = get_sensitivity_tuning(args.sensitivity_profile)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for entry in manifest.get("entries", []):
        trace = load_trace(entry["trace_path"])
        segments = replay_trace_segments(bundle=bundle, trace=trace, args=args, tuning=tuning)
        if not segments:
            continue
        frame_waveforms = reconstruct_trace_frame_waveforms(trace=trace, bundle=bundle, args=args)
        timestamps = np.asarray(trace["timestamps"], dtype=np.float32)
        target_idx = int(trace.get("target_kws12", -1))
        target_label = KWS12_LABELS[target_idx] if 0 <= target_idx < len(KWS12_LABELS) else UNKNOWN_LABEL
        window_start = float(trace.get("window_start", 0.0))
        window_end = float(trace.get("window_end", window_start))
        for segment in segments:
            start_time = float(segment.get("start_time", 0.0))
            end_time = float(segment.get("end_time", start_time))
            overlaps = min(end_time, window_end) > max(start_time, window_start)
            hard_peak_prob = float(segment.get("hard_peak_prob", 0.0))
            accepted_label = str(segment.get("accepted_label") or "")
            hard_peak_label = str(segment.get("hard_peak_label") or "")
            should_keep = (
                target_label in HARD_WORD_SPECIALIST_TARGETS
                or accepted_label in HARD_WORD_SPECIALIST_TARGETS
                or hard_peak_label in HARD_WORD_SPECIALIST_TARGETS
                or hard_peak_prob >= 0.18
            )
            if not should_keep:
                continue
            label = target_label if overlaps and target_label in HARD_WORD_SPECIALIST_TARGETS else "other"
            waveforms.append(_segment_waveform(timestamps=timestamps, frame_waveforms=frame_waveforms, segment=segment))
            labels.append(int(HARD_WORD_SPECIALIST_LABEL_TO_INDEX[label]))
    if not waveforms:
        return np.zeros((0, bundle.clip_samples), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(waveforms, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _collect_manifest_specialist_samples(
    split: str,
    *,
    clip_samples: int,
    other_target_quota: int,
    unknown_quota: int,
    silence_quota: int,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = Path("data/processed/manifests") / f"local_{split}.jsonl"
    records = read_manifest(manifest_path)
    target_counts = {label: 0 for label in TARGET_KEYWORDS_10 if label not in HARD_WORD_SPECIALIST_TARGETS}
    unknown_seen = 0
    silence_seen = 0
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        if record.command_label is None:
            continue
        kws12_label = KWS12_LABELS[command31_to_kws12(INDEX_TO_COMMAND31[int(record.command_label)])]
        if kws12_label in HARD_WORD_SPECIALIST_TARGETS:
            label = kws12_label
        elif kws12_label == "unknown":
            if unknown_seen >= int(unknown_quota):
                continue
            unknown_seen += 1
            label = "other"
        elif kws12_label == "silence":
            if silence_seen >= int(silence_quota):
                continue
            silence_seen += 1
            label = "other"
        elif kws12_label in target_counts:
            if target_counts[kws12_label] >= int(other_target_quota):
                continue
            target_counts[kws12_label] += 1
            label = "other"
        else:
            continue
        waveform = load_audio(record.path, sample_rate=16_000).detach().cpu().numpy().astype(np.float32, copy=False)
        start_idx, end_idx = estimate_utterance_bounds(waveform, 16_000)
        center = 0.5 * (float(start_idx) + float(end_idx))
        half = clip_samples / 2.0
        crop_start = max(0, int(round(center - half)))
        crop_end = min(waveform.shape[0], crop_start + clip_samples)
        crop_start = max(0, crop_end - clip_samples)
        waveform = waveform[crop_start:crop_end]
        if waveform.shape[0] < clip_samples:
            padded = np.zeros((clip_samples,), dtype=np.float32)
            offset = max(0, (clip_samples - waveform.shape[0]) // 2)
            padded[offset : offset + waveform.shape[0]] = waveform
            waveform = padded
        waveforms.append(waveform.astype(np.float32, copy=False))
        labels.append(int(HARD_WORD_SPECIALIST_LABEL_TO_INDEX[label]))
    if not waveforms:
        return np.zeros((0, clip_samples), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(waveforms, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _concat_samples(*parts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    waveforms = [wave for wave, labels in parts if wave.size and labels.size]
    labels = [labels for wave, labels in parts if wave.size and labels.size]
    if not waveforms:
        return np.zeros((0, 16000), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(waveforms, axis=0), np.concatenate(labels, axis=0)


def _calibrate_from_valid_logits(valid_probs: np.ndarray, valid_labels: np.ndarray) -> dict[str, object]:
    calibration = default_realtime_specialist_calibration()
    per_label = calibration["per_label"]
    for label in HARD_WORD_SPECIALIST_TARGETS:
        label_idx = HARD_WORD_SPECIALIST_LABEL_TO_INDEX[label]
        role = "guard" if label == "up" else "rescue"
        accept_grid = (0.60, 0.64, 0.68, 0.72, 0.76, 0.80) if role == "guard" else (0.46, 0.50, 0.54, 0.58, 0.62, 0.66)
        margin_grid = (0.06, 0.08, 0.10, 0.12, 0.14) if role == "guard" else (0.00, 0.02, 0.04, 0.06, 0.08)
        best = None
        for accept_prob in accept_grid:
            for min_margin in margin_grid:
                preds = []
                for probs in valid_probs:
                    top_idx = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])
                    top_label = list(HARD_WORD_SPECIALIST_LABEL_TO_INDEX.keys())[top_idx]
                    if probs.size > 1:
                        top2 = np.partition(probs, -2)[-2:]
                        margin = float(top2[-1] - top2[-2])
                    else:
                        margin = top_prob
                    if top_label == label and top_prob >= accept_prob and margin >= min_margin:
                        preds.append(label_idx)
                    else:
                        preds.append(HARD_WORD_SPECIALIST_LABEL_TO_INDEX["other"])
                preds_arr = np.asarray(preds, dtype=np.int64)
                tp = float(np.sum((preds_arr == label_idx) & (valid_labels == label_idx)))
                fp = float(np.sum((preds_arr == label_idx) & (valid_labels != label_idx)))
                fn = float(np.sum((preds_arr != label_idx) & (valid_labels == label_idx)))
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                if role == "guard":
                    objective = (precision < 0.95, -precision, -recall, fp)
                else:
                    objective = (precision < 0.95, -(recall if precision >= 0.95 else 0.0), -(precision if precision >= 0.95 else precision), fp)
                if best is None or objective < best[0]:
                    best = (objective, accept_prob, min_margin, precision, recall)
        if best is not None:
            _objective, accept_prob, min_margin, _precision, _recall = best
            entry = per_label.setdefault(label, {})
            entry["accept_prob"] = float(accept_prob)
            entry["min_margin"] = float(min_margin)
            entry["trigger_prob"] = float(max(0.14, accept_prob - (0.22 if role == "guard" else 0.28)))
            entry["role"] = role
    return calibration


def _predict_probs(model, waveforms: np.ndarray, *, device, sample_rate: int, target_samples: int, n_mels: int, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    from kws.demo.realtime_specialist import _stack_features  # local import to keep module surface small

    features = _stack_features(
        np.asarray(waveforms, dtype=np.float32),
        sample_rate=sample_rate,
        target_samples=target_samples,
        n_mels=n_mels,
    )
    x = ((features - feature_mean) / np.maximum(feature_std, 1e-5)).astype(np.float32, copy=False)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
    return probs


def _resolve_output_specialist(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_specialist).strip():
        return Path(str(args.output_specialist).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "realtime_specialist.pt").resolve()


def _resolve_output_calibration(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_calibration).strip():
        return Path(str(args.output_calibration).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "realtime_specialist_calibration.json").resolve()


def main() -> None:
    args = parse_args()
    base_bundle = load_realtime_demo(
        checkpoint=args.checkpoint,
        demo_profile=args.demo_profile,
        detector_device_preference=args.device,
        selection_profile=args.selection_profile,
        keyword_calibration_path=args.keyword_calibration_path,
        external_ensemble_calibration_path=args.external_ensemble_calibration_path,
        wheel=args.wheel,
        runtime_label_backend=args.runtime_label_backend,
        external_kws_model=args.external_kws_model,
        external_kws_device=args.external_kws_device,
        ranking_iters=8,
        no_cache_ranking=False,
        rebuild_ranking=False,
        device_auto_bench_iters=6,
    )
    bundle = _bundle_with_realtime_profile(base_bundle)
    train_manifest = _load_manifest(args.train_trace_manifest)
    valid_manifest = _load_manifest(args.valid_trace_manifest)
    test_manifest = _load_manifest(args.test_trace_manifest) if str(args.test_trace_manifest).strip() else None

    train_manifest_waveforms, train_manifest_labels = _collect_manifest_specialist_samples(
        str(args.train_split),
        clip_samples=int(bundle.clip_samples),
        other_target_quota=int(args.other_target_quota),
        unknown_quota=int(args.unknown_quota),
        silence_quota=int(args.silence_quota),
    )
    valid_manifest_waveforms, valid_manifest_labels = _collect_manifest_specialist_samples(
        str(args.valid_split),
        clip_samples=int(bundle.clip_samples),
        other_target_quota=max(40, int(args.other_target_quota) // 2),
        unknown_quota=max(120, int(args.unknown_quota) // 2),
        silence_quota=max(40, int(args.silence_quota) // 2),
    )
    train_trace_waveforms, train_trace_labels = _collect_specialist_samples(train_manifest, bundle=bundle, args=args)
    valid_trace_waveforms, valid_trace_labels = _collect_specialist_samples(valid_manifest, bundle=bundle, args=args)
    train_waveforms, train_labels = _concat_samples(
        (train_manifest_waveforms, train_manifest_labels),
        (train_trace_waveforms, train_trace_labels),
    )
    valid_waveforms, valid_labels = _concat_samples(
        (valid_manifest_waveforms, valid_manifest_labels),
        (valid_trace_waveforms, valid_trace_labels),
    )
    model, feature_mean, feature_std = train_realtime_specialist(
        train_waveforms=train_waveforms,
        train_labels=train_labels,
        valid_waveforms=valid_waveforms,
        valid_labels=valid_labels,
        device=bundle.runtime_device,
        sample_rate=int(bundle.sample_rate),
        target_samples=int(bundle.clip_samples),
        n_mels=int(args.n_mels),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )

    specialist_path = _resolve_output_specialist(bundle, args)
    calibration_path = _resolve_output_calibration(bundle, args)
    save_realtime_specialist_artifact(
        specialist_path,
        model=model,
        sample_rate=int(bundle.sample_rate),
        target_samples=int(bundle.clip_samples),
        n_mels=int(args.n_mels),
        hidden_dim=int(args.hidden_dim),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    valid_probs = _predict_probs(
        model,
        valid_waveforms,
        device=bundle.runtime_device,
        sample_rate=int(bundle.sample_rate),
        target_samples=int(bundle.clip_samples),
        n_mels=int(args.n_mels),
        feature_mean=feature_mean,
        feature_std=feature_std,
    ) if valid_waveforms.size else np.zeros((0, len(HARD_WORD_SPECIALIST_LABEL_TO_INDEX)), dtype=np.float32)
    calibration = _calibrate_from_valid_logits(valid_probs, valid_labels) if valid_probs.size else default_realtime_specialist_calibration()
    save_realtime_specialist_calibration(calibration_path, calibration)

    valid_metrics = summarize_realtime_specialist_predictions(valid_probs, valid_labels) if valid_probs.size else {
        "per_label": {},
        "macro_f1": 0.0,
        "hard_word_macro_f1": 0.0,
        "hard_word_min_precision": 0.0,
        "hard_word_min_recall": 0.0,
    }

    valid_report = {
        "num_train_samples": int(train_labels.shape[0]),
        "num_valid_samples": int(valid_labels.shape[0]),
        "num_train_manifest_samples": int(train_manifest_labels.shape[0]),
        "num_train_trace_samples": int(train_trace_labels.shape[0]),
        "num_valid_manifest_samples": int(valid_manifest_labels.shape[0]),
        "num_valid_trace_samples": int(valid_trace_labels.shape[0]),
        "specialist_path": str(specialist_path),
        "calibration_path": str(calibration_path),
        "labels": list(HARD_WORD_SPECIALIST_LABEL_TO_INDEX.keys()),
        "n_mels": int(args.n_mels),
        "hidden_dim": int(args.hidden_dim),
        **valid_metrics,
    }
    output_valid = Path(args.output_valid_report).expanduser().resolve()
    output_valid.parent.mkdir(parents=True, exist_ok=True)
    output_valid.write_text(json.dumps(valid_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if test_manifest is not None:
        test_manifest_waveforms, test_manifest_labels = _collect_manifest_specialist_samples(
            "test",
            clip_samples=int(bundle.clip_samples),
            other_target_quota=max(40, int(args.other_target_quota) // 2),
            unknown_quota=max(120, int(args.unknown_quota) // 2),
            silence_quota=max(40, int(args.silence_quota) // 2),
        )
        test_trace_waveforms, test_trace_labels = _collect_specialist_samples(test_manifest, bundle=bundle, args=args)
        test_waveforms, test_labels = _concat_samples(
            (test_manifest_waveforms, test_manifest_labels),
            (test_trace_waveforms, test_trace_labels),
        )
        test_report = {
            "num_test_segments": int(test_labels.shape[0]),
            "specialist_path": str(specialist_path),
            "calibration_path": str(calibration_path),
            "n_mels": int(args.n_mels),
            "hidden_dim": int(args.hidden_dim),
        }
        if test_waveforms.size and test_labels.size:
            test_probs = _predict_probs(
                model,
                test_waveforms,
                device=bundle.runtime_device,
                sample_rate=int(bundle.sample_rate),
                target_samples=int(bundle.clip_samples),
                n_mels=int(args.n_mels),
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            test_report.update(summarize_realtime_specialist_predictions(test_probs, test_labels))
        output_test = Path(args.output_test_report).expanduser().resolve()
        output_test.parent.mkdir(parents=True, exist_ok=True)
        output_test.write_text(json.dumps(test_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(valid_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
