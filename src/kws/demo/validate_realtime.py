"""Automated realtime-style validation for the desktop KWS demo."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kws.constants import IGNORE_INDEX, INDEX_TO_COMMAND31, KWS12_TO_INDEX, UNKNOWN_LABEL, command31_to_kws12
from kws.data.audio import load_audio
from kws.data.manifest import ManifestRecord, read_manifest
from kws.demo.realtime import (
    DEFAULT_DEMO_PROFILE,
    DEFAULT_RUNTIME_LABEL_BACKEND,
    AdaptiveGateConfig,
    GateStateMachine,
    RealtimeEngine,
    get_sensitivity_tuning,
    load_realtime_demo,
)
from kws.demo.realtime_trace import collect_clip_trace, replay_clip_trace
from kws.train.metrics import compute_kws12_breakdown_from_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the realtime KWS desktop demo on local manifests.")
    parser.add_argument("--demo-profile", type=str, default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--selection-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--keyword-calibration-path", type=str, default="")
    parser.add_argument("--external-ensemble-calibration-path", type=str, default="")
    parser.add_argument("--wheel", type=str, default="kws12", choices=["kws12", "target10"])
    parser.add_argument("--runtime-label-backend", type=str, default="")
    parser.add_argument("--external-kws-model", type=str, default="ensemble/ast-superb-kws12")
    parser.add_argument("--external-kws-device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
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
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _resolve_gate_args(args: argparse.Namespace) -> tuple[str, float, float]:
    gate_mode = str(args.gate_mode).strip().lower()
    if args.threshold is not None:
        wake_open_thr = float(args.threshold)
        wake_close_thr = max(0.0, wake_open_thr - 0.1)
        gate_mode = "fixed"
    else:
        wake_open_thr = float(args.wake_open_thr)
        wake_close_thr = float(args.wake_close_thr)
    return gate_mode, wake_open_thr, wake_close_thr


def _manifest_records(manifest_path: Path, limit_per_class: int = 0) -> list[ManifestRecord]:
    records = read_manifest(manifest_path)
    if int(limit_per_class) <= 0:
        return records
    per_class: dict[int, int] = {}
    out: list[ManifestRecord] = []
    for rec in records:
        if rec.command_label is None or int(rec.command_label) == IGNORE_INDEX:
            continue
        key = int(rec.command_label)
        seen = per_class.get(key, 0)
        if seen >= int(limit_per_class):
            continue
        per_class[key] = seen + 1
        out.append(rec)
    return out


def _estimate_utterance_bounds(waveform: np.ndarray, sample_rate: int) -> tuple[int, int]:
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0, 0
    frame = max(1, int(round(sample_rate * 0.02)))
    hop = max(1, int(round(sample_rate * 0.01)))
    rms_values: list[float] = []
    peaks: list[float] = []
    starts: list[int] = []
    for start in range(0, max(1, x.size - frame + 1), hop):
        chunk = x[start : start + frame]
        rms_values.append(float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)) + 1e-12)))
        peaks.append(float(np.max(np.abs(chunk))) if chunk.size else 0.0)
        starts.append(start)
    if not rms_values:
        return 0, x.size
    rms_arr = np.asarray(rms_values, dtype=np.float64)
    peak_arr = np.asarray(peaks, dtype=np.float64)
    thr = max(0.002, float(np.percentile(rms_arr, 85.0)) * 0.35, float(np.percentile(peak_arr, 90.0)) * 0.05)
    active = np.where((rms_arr >= thr) | (peak_arr >= max(0.01, thr * 4.0)))[0]
    if active.size == 0:
        return 0, x.size
    first = int(starts[int(active[0])])
    last = min(x.size, int(starts[int(active[-1])] + frame))
    return first, max(first + 1, last)


def _build_stream_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    calibration_seconds: float,
    pre_silence_seconds: float,
    post_silence_seconds: float,
) -> tuple[np.ndarray, int, int]:
    calibration = np.zeros((int(round(float(calibration_seconds) * sample_rate)),), dtype=np.float32)
    pre = np.zeros((int(round(float(pre_silence_seconds) * sample_rate)),), dtype=np.float32)
    post = np.zeros((int(round(float(post_silence_seconds) * sample_rate)),), dtype=np.float32)
    utt_start, utt_end = _estimate_utterance_bounds(waveform, sample_rate)
    stream = np.concatenate([calibration, pre, waveform.astype(np.float32, copy=False), post], axis=0)
    offset = int(calibration.size + pre.size)
    return stream, int(offset + utt_start), int(offset + utt_end)


def _iter_hop_chunks(stream: np.ndarray, hop_samples: int) -> list[np.ndarray]:
    padded = np.asarray(stream, dtype=np.float32)
    remainder = int(padded.size % hop_samples)
    if remainder:
        padded = np.pad(padded, (0, hop_samples - remainder))
    return [padded[start : start + hop_samples] for start in range(0, padded.size, hop_samples)]


def _summarize_clip_frames(
    frames: list[tuple[float, object]],
    *,
    window_start: float,
    window_end: float,
) -> tuple[int, bool, float | None]:
    match_frames = [
        (ts, snap)
        for ts, snap in frames
        if window_start <= ts <= window_end and getattr(snap, "prompt_status", "") == "MATCH" and getattr(snap, "active_label", None) is not None
    ]
    no_match = len(match_frames) == 0
    if match_frames:
        first_seen: dict[str, float] = {}
        counts: Counter[str] = Counter()
        for ts, snap in match_frames:
            label = str(getattr(snap, "active_label"))
            counts[label] += 1
            first_seen.setdefault(label, ts)
        pred_label = sorted(counts, key=lambda label: (-counts[label], first_seen[label]))[0]
        latency_ms = max(0.0, (first_seen[pred_label] - window_start) * 1000.0)
        return KWS12_TO_INDEX[pred_label], no_match, latency_ms

    silence_frames = [
        snap
        for ts, snap in frames
        if window_start <= ts <= window_end and str(getattr(snap, "command_label", "")).strip().lower() == "silence"
    ]
    if silence_frames and all(not bool(getattr(snap, "gate_open", False)) for snap in silence_frames):
        return KWS12_TO_INDEX["silence"], True, None
    return KWS12_TO_INDEX[UNKNOWN_LABEL], True, None


def _predict_clip(
    *,
    bundle,
    record: ManifestRecord,
    args: argparse.Namespace,
    tuning,
) -> tuple[int, bool, float | None]:
    if (
        bundle.resolved_profile.demo_profile == "accuracy-first-realtime"
        and bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND
    ):
        trace = collect_clip_trace(bundle=bundle, record=record, args=args)
        return replay_clip_trace(bundle=bundle, trace=trace, args=args, tuning=tuning)

    gate_mode, wake_open_thr, wake_close_thr = _resolve_gate_args(args)
    adaptive_cfg = AdaptiveGateConfig(
        calibration_seconds=float(args.calibration_seconds),
        open_offset=tuning.open_offset,
        close_offset=tuning.close_offset,
        open_floor=tuning.open_floor,
        close_floor=tuning.close_floor,
    )
    gate = GateStateMachine(
        mode=gate_mode,
        open_threshold=wake_open_thr,
        close_threshold=wake_close_thr,
        cmd_conf_threshold=float(args.cmd_conf_thr) if args.cmd_conf_thr is not None else tuning.cmd_conf_thr,
        hold_seconds=max(0.0, float(args.hold_ms) / 1000.0),
        adaptive=adaptive_cfg,
    )
    engine = RealtimeEngine(
        model=bundle.model,
        frontend=bundle.frontend,
        device=bundle.runtime_device,
        command31_labels=bundle.command31_labels,
        wheel=bundle.wheel,
        gate=gate,
        hop_seconds=float(args.hop_seconds),
        ema_alpha=float(args.ema_alpha),
        hold_ms=float(args.hold_ms),
        selected_device_label=bundle.selected_device_label,
        input_device_name=f"manifest:{args.split}",
        stream_sample_rate=float(bundle.sample_rate),
        model_sample_rate=int(bundle.sample_rate),
        audio_seconds=float(bundle.audio_seconds),
        mic_precheck_seconds=0.0,
        mic_min_rms=0.0,
        auto_gain=True,
        target_rms=0.05,
        max_gain_db=18.0,
        display_conf_thr=float(args.display_conf_thr) if args.display_conf_thr is not None else tuning.display_conf_thr,
        display_wake_thr=float(args.display_wake_thr) if args.display_wake_thr is not None else tuning.display_wake_thr,
        vote_window=int(args.vote_window) if args.vote_window is not None else tuning.vote_window,
        vote_min_count=int(args.vote_min_count) if args.vote_min_count is not None else tuning.vote_min_count,
        passive_profile=None,
        keyword_calibration=bundle.keyword_calibration,
        external_ensemble_calibration=bundle.external_ensemble_calibration,
        segment_decoder=bundle.segment_decoder,
        segment_decoder_disabled=bundle.segment_decoder_disabled,
        realtime_specialist=bundle.realtime_specialist,
        realtime_specialist_calibration=bundle.realtime_specialist_calibration,
        segment_runtime_enabled=bundle.resolved_profile.demo_profile == "accuracy-first-realtime",
        verifier=bundle.verifier,
        runtime_label_backend=bundle.resolved_profile.runtime_label_backend,
        external_kws_model=bundle.resolved_profile.external_kws_model,
        external_kws_device=bundle.resolved_profile.external_kws_device,
    )
    engine.bypass_precheck()

    waveform = load_audio(record.path, sample_rate=int(bundle.sample_rate)).detach().cpu().numpy().astype(np.float32, copy=False)
    stream, utt_start, utt_end = _build_stream_waveform(
        waveform,
        sample_rate=int(bundle.sample_rate),
        calibration_seconds=float(args.calibration_seconds) if str(args.gate_mode).strip().lower() == "adaptive" else 0.0,
        pre_silence_seconds=float(args.pre_silence_seconds),
        post_silence_seconds=float(args.post_silence_seconds),
    )
    hop_samples = max(1, int(round(float(args.hop_seconds) * bundle.sample_rate)))
    window_start = float(utt_start) / float(bundle.sample_rate)
    window_end = float(min(stream.size, utt_end + int(round(float(args.match_tail_seconds) * bundle.sample_rate)))) / float(bundle.sample_rate)

    frames: list[tuple[float, object]] = []
    sim_now = 0.0
    for chunk in _iter_hop_chunks(stream, hop_samples):
        sim_now += float(chunk.size) / float(bundle.sample_rate)
        snapshot = engine.process_chunk(chunk, now=sim_now, now_wall=sim_now, queue_fill_ratio=0.0)
        if snapshot is not None:
            frames.append((sim_now, snapshot))
    flushed = engine.flush_pending_segment(now=sim_now, now_wall=sim_now)
    if flushed is not None:
        # Keep the final flushed match inside the same evaluation window that replay uses.
        frames.append((window_end, flushed))

    return _summarize_clip_frames(frames, window_start=window_start, window_end=window_end)


def evaluate_records(
    *,
    bundle,
    records: list[ManifestRecord],
    args: argparse.Namespace,
    tuning,
) -> dict[str, object]:
    preds: list[int] = []
    targets: list[int] = []
    no_match_count = 0
    latencies_ms: list[float] = []
    for rec in records:
        if rec.command_label is None or int(rec.command_label) == IGNORE_INDEX:
            continue
        pred, no_match, latency_ms = _predict_clip(
            bundle=bundle,
            record=rec,
            args=args,
            tuning=tuning,
        )
        preds.append(int(pred))
        targets.append(command31_to_kws12(INDEX_TO_COMMAND31[int(rec.command_label)]))
        no_match_count += int(no_match)
        if latency_ms is not None:
            latencies_ms.append(float(latency_ms))

    preds_arr = np.asarray(preds, dtype=np.int64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    metrics = compute_kws12_breakdown_from_indices(preds_arr, targets_arr)
    return {
        "split": args.split,
        "num_eval_samples": int(targets_arr.size),
        "runtime_label_backend": bundle.resolved_profile.runtime_label_backend,
        "external_kws_model_id": (
            bundle.resolved_profile.external_kws_model
            if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND
            else ""
        ),
        "external_kws_device": (
            bundle.resolved_profile.external_kws_device
            if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND
            else ""
        ),
        "per_class_kws12": metrics.get("per_class_kws12", {}),
        "min_kws12_precision": float(metrics.get("min_kws12_precision", 0.0)),
        "min_kws12_recall": float(metrics.get("min_kws12_recall", 0.0)),
        "unknown_to_target_rate": float(metrics.get("kws12_unknown_to_target_rate", 0.0)),
        "no_match_rate": float(no_match_count / max(int(targets_arr.size), 1)),
        "avg_match_latency_ms": float(np.mean(np.asarray(latencies_ms, dtype=np.float64))) if latencies_ms else None,
        "p95_match_latency_ms": float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 95.0)) if latencies_ms else None,
        "goal": {
            "min_kws12_precision": 0.95,
            "min_kws12_recall": 0.95,
            "unknown_to_target_rate_max": 0.02,
        },
        "passed": bool(
            float(metrics.get("min_kws12_precision", 0.0)) >= 0.95
            and float(metrics.get("min_kws12_recall", 0.0)) >= 0.95
            and float(metrics.get("kws12_unknown_to_target_rate", 1.0)) <= 0.02
        ),
    }


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    bundle = load_realtime_demo(
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
    tuning = get_sensitivity_tuning(args.sensitivity_profile)
    manifest_path = Path("data/processed/manifests") / f"local_{args.split}.jsonl"
    records = _manifest_records(manifest_path, limit_per_class=int(args.limit_per_class))
    return evaluate_records(bundle=bundle, records=records, args=args, tuning=tuning)


def main() -> None:
    args = parse_args()
    payload = run_validation(args)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (Path.cwd() / "reports" / f"realtime_{args.demo_profile}_{args.split}.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
