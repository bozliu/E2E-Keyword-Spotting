"""Trace caching and replay helpers for full-stream realtime validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchaudio.transforms as ta_transforms

from kws.constants import CLIP_SECONDS, IGNORE_INDEX, INDEX_TO_COMMAND31, KWS12_LABELS, KWS12_TO_INDEX, UNKNOWN_LABEL, command31_to_kws12
from kws.data.audio import load_audio, pad_or_trim
from kws.data.manifest import ManifestRecord
from kws.demo.realtime import (
    DEFAULT_RUNTIME_LABEL_BACKEND,
    AdaptiveGateConfig,
    GateStateMachine,
    RealtimeEngine,
    _command31_probs_from_kws12_probs,
)
from kws.external import blend_ast_superb_probs, predict_ensemble_ast_superb_from_waveforms


def resolve_gate_args(args) -> tuple[str, float, float]:
    gate_mode = str(args.gate_mode).strip().lower()
    if args.threshold is not None:
        wake_open_thr = float(args.threshold)
        wake_close_thr = max(0.0, wake_open_thr - 0.1)
        gate_mode = "fixed"
    else:
        wake_open_thr = float(args.wake_open_thr)
        wake_close_thr = float(args.wake_close_thr)
    return gate_mode, wake_open_thr, wake_close_thr


def estimate_utterance_bounds(waveform: np.ndarray, sample_rate: int) -> tuple[int, int]:
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


def build_stream_waveform(
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
    utt_start, utt_end = estimate_utterance_bounds(waveform, sample_rate)
    stream = np.concatenate([calibration, pre, waveform.astype(np.float32, copy=False), post], axis=0)
    offset = int(calibration.size + pre.size)
    return stream, int(offset + utt_start), int(offset + utt_end)


def iter_hop_chunks(stream: np.ndarray, hop_samples: int) -> list[np.ndarray]:
    padded = np.asarray(stream, dtype=np.float32)
    remainder = int(padded.size % hop_samples)
    if remainder:
        padded = np.pad(padded, (0, hop_samples - remainder))
    return [padded[start : start + hop_samples] for start in range(0, padded.size, hop_samples)]


def summarize_clip_frames(
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
    if match_frames:
        first_seen: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ts, snap in match_frames:
            label = str(getattr(snap, "active_label"))
            counts[label] = counts.get(label, 0) + 1
            first_seen.setdefault(label, ts)
        pred_label = sorted(counts, key=lambda label: (-counts[label], first_seen[label]))[0]
        latency_ms = max(0.0, (first_seen[pred_label] - window_start) * 1000.0)
        return KWS12_TO_INDEX[pred_label], False, latency_ms

    silence_frames = [
        snap
        for ts, snap in frames
        if window_start <= ts <= window_end and str(getattr(snap, "command_label", "")).strip().lower() == "silence"
    ]
    if silence_frames and all(not bool(getattr(snap, "gate_open", False)) for snap in silence_frames):
        return KWS12_TO_INDEX["silence"], True, None
    return KWS12_TO_INDEX[UNKNOWN_LABEL], True, None


def _trace_target_index(record: ManifestRecord) -> int:
    if record.command_label is None or int(record.command_label) == IGNORE_INDEX:
        return KWS12_TO_INDEX[UNKNOWN_LABEL]
    return int(command31_to_kws12(INDEX_TO_COMMAND31[int(record.command_label)]))


def collect_clip_trace(*, bundle, record: ManifestRecord, args) -> dict[str, Any]:
    waveform = load_audio(record.path, sample_rate=int(bundle.sample_rate)).detach().cpu().numpy().astype(np.float32, copy=False)
    stream, utt_start, utt_end = build_stream_waveform(
        waveform,
        sample_rate=int(bundle.sample_rate),
        calibration_seconds=float(args.calibration_seconds) if str(args.gate_mode).strip().lower() == "adaptive" else 0.0,
        pre_silence_seconds=float(args.pre_silence_seconds),
        post_silence_seconds=float(args.post_silence_seconds),
    )
    window_start = float(utt_start) / float(bundle.sample_rate)
    window_end = float(min(stream.size, utt_end + int(round(float(args.match_tail_seconds) * bundle.sample_rate)))) / float(bundle.sample_rate)
    timestamps, model_waveforms = collect_trace_frame_waveforms_from_stream(
        stream=stream,
        sample_rate=int(bundle.sample_rate),
        audio_seconds=float(bundle.audio_seconds),
        hop_seconds=float(args.hop_seconds),
    )

    detector_command_probs: list[np.ndarray] = []
    detector_wake_probs: list[float] = []
    detector_top_kws12_index: list[int] = []
    detector_top_margin: list[float] = []
    ast_probs: list[np.ndarray] = []
    superb_probs: list[np.ndarray] = []
    batch_size = 24
    for start_idx in range(0, len(model_waveforms), batch_size):
        batch_waveforms = model_waveforms[start_idx : start_idx + batch_size]
        if not batch_waveforms:
            continue
        feature_batch: list[torch.Tensor] = []
        for waveform_np in batch_waveforms:
            feature = bundle.frontend(torch.from_numpy(waveform_np))
            mean = feature.mean()
            std = feature.std().clamp(min=1e-5)
            feature_batch.append((feature - mean) / std)
        x_input = torch.stack(feature_batch, dim=0).to(bundle.runtime_device)
        with torch.no_grad():
            out = bundle.model(x_input)
            batch_detector = torch.softmax(out.command_logits, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
            batch_wake = torch.sigmoid(out.wake_logits).reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND:
            pair = predict_ensemble_ast_superb_from_waveforms(
                batch_waveforms,
                device=bundle.resolved_profile.external_kws_device,
                sample_rate=int(bundle.sample_rate),
                calibration=None,
            )
            batch_ast = pair.ast_probs.astype(np.float32, copy=False)
            batch_superb = pair.superb_probs.astype(np.float32, copy=False)
        else:
            batch_ast = np.zeros((len(batch_waveforms), len(KWS12_LABELS)), dtype=np.float32)
            batch_superb = np.zeros((len(batch_waveforms), len(KWS12_LABELS)), dtype=np.float32)

        for idx, detector_probs in enumerate(batch_detector):
            wake_prob = float(batch_wake[idx])
            detector_kws12 = np.zeros((len(KWS12_LABELS),), dtype=np.float32)
            for command_idx, label in enumerate(bundle.command31_labels):
                detector_kws12[command31_to_kws12(label)] += float(detector_probs[command_idx])
            det_top_idx = int(np.argmax(detector_kws12))
            if detector_kws12.size > 1:
                top2 = np.partition(detector_kws12, -2)[-2:]
                det_margin = float(top2[-1] - top2[-2])
            else:
                det_margin = float(detector_kws12[det_top_idx])
            detector_command_probs.append(detector_probs.copy())
            detector_wake_probs.append(wake_prob)
            detector_top_kws12_index.append(det_top_idx)
            detector_top_margin.append(det_margin)
            ast_probs.append(batch_ast[idx].copy())
            superb_probs.append(batch_superb[idx].copy())

    return {
        "record_path": str(Path(record.path).resolve()),
        "target_kws12": int(_trace_target_index(record)),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "timestamps": np.asarray(timestamps, dtype=np.float32),
        "detector_command_probs": np.asarray(detector_command_probs, dtype=np.float32),
        "detector_wake_probs": np.asarray(detector_wake_probs, dtype=np.float32),
        "detector_top_kws12_index": np.asarray(detector_top_kws12_index, dtype=np.int64),
        "detector_top_margin": np.asarray(detector_top_margin, dtype=np.float32),
        "ast_probs": np.asarray(ast_probs, dtype=np.float32),
        "superb_probs": np.asarray(superb_probs, dtype=np.float32),
    }


def collect_trace_frame_waveforms_from_stream(
    *,
    stream: np.ndarray,
    sample_rate: int,
    audio_seconds: float,
    hop_seconds: float,
) -> tuple[list[float], list[np.ndarray]]:
    hop_samples = max(1, int(round(float(hop_seconds) * sample_rate)))
    stream_clip_samples = int(round(float(audio_seconds) * float(sample_rate)))
    model_clip_samples = int(round(float(audio_seconds) * float(sample_rate)))
    ring = np.zeros((stream_clip_samples,), dtype=np.float32)
    ring_linear = np.zeros((stream_clip_samples,), dtype=np.float32)
    write_idx = 0
    size = 0
    last_infer_at = 0.0

    timestamps: list[float] = []
    model_waveforms: list[np.ndarray] = []

    sim_now = 0.0
    for chunk in iter_hop_chunks(stream, hop_samples):
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n >= stream_clip_samples:
            ring[:] = x[-stream_clip_samples:]
            write_idx = 0
            size = stream_clip_samples
        else:
            tail = stream_clip_samples - write_idx
            if n <= tail:
                ring[write_idx : write_idx + n] = x
            else:
                ring[write_idx:] = x[:tail]
                ring[: n - tail] = x[tail:]
            write_idx = (write_idx + n) % stream_clip_samples
            size = min(stream_clip_samples, size + n)
        sim_now += float(n) / float(sample_rate)
        if size < stream_clip_samples:
            continue
        if (sim_now - last_infer_at) < float(hop_seconds):
            continue
        last_infer_at = sim_now

        if write_idx == 0:
            raw_waveform_np = ring.copy()
        else:
            tail = stream_clip_samples - write_idx
            ring_linear[:tail] = ring[write_idx:]
            ring_linear[tail:] = ring[:write_idx]
            raw_waveform_np = ring_linear.copy()

        rms = float(np.sqrt(np.mean(np.square(raw_waveform_np, dtype=np.float64)) + 1e-12))
        target_rms = 0.05
        max_gain_db = 18.0
        max_gain_lin = 10.0 ** (max_gain_db / 20.0)
        desired_gain = target_rms / max(rms, 1e-8)
        gain_lin = min(max_gain_lin, max(1.0, desired_gain))
        waveform_np = raw_waveform_np.copy()
        waveform_np *= float(gain_lin)
        np.clip(waveform_np, -1.0, 1.0, out=waveform_np)

        waveform_tensor = torch.from_numpy(waveform_np)
        waveform_tensor = pad_or_trim(waveform_tensor, target_samples=model_clip_samples)
        waveform_model_np = waveform_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        timestamps.append(float(sim_now))
        model_waveforms.append(waveform_model_np.copy())
    return timestamps, model_waveforms


def reconstruct_trace_frame_waveforms(*, trace: Mapping[str, Any], bundle, args) -> list[np.ndarray]:
    waveform = load_audio(str(trace["record_path"]), sample_rate=int(bundle.sample_rate)).detach().cpu().numpy().astype(np.float32, copy=False)
    stream, _utt_start, _utt_end = build_stream_waveform(
        waveform,
        sample_rate=int(bundle.sample_rate),
        calibration_seconds=float(args.calibration_seconds) if str(args.gate_mode).strip().lower() == "adaptive" else 0.0,
        pre_silence_seconds=float(args.pre_silence_seconds),
        post_silence_seconds=float(args.post_silence_seconds),
    )
    timestamps, model_waveforms = collect_trace_frame_waveforms_from_stream(
        stream=stream,
        sample_rate=int(bundle.sample_rate),
        audio_seconds=float(bundle.audio_seconds),
        hop_seconds=float(args.hop_seconds),
    )
    expected = int(np.asarray(trace["timestamps"], dtype=np.float32).shape[0])
    if len(model_waveforms) > expected:
        return model_waveforms[:expected]
    if len(model_waveforms) < expected and model_waveforms:
        pad_waveform = model_waveforms[-1]
        model_waveforms = model_waveforms + [pad_waveform.copy() for _ in range(expected - len(model_waveforms))]
    return model_waveforms


def save_trace(path: str | Path, trace: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "record_path": np.asarray([str(trace["record_path"])], dtype=object),
        "target_kws12": np.asarray([int(trace["target_kws12"])], dtype=np.int64),
        "window_start": np.asarray([float(trace["window_start"])], dtype=np.float32),
        "window_end": np.asarray([float(trace["window_end"])], dtype=np.float32),
        "timestamps": np.asarray(trace["timestamps"], dtype=np.float32),
        "detector_command_probs": np.asarray(trace["detector_command_probs"], dtype=np.float32),
        "detector_wake_probs": np.asarray(trace["detector_wake_probs"], dtype=np.float32),
        "detector_top_kws12_index": np.asarray(
            trace.get("detector_top_kws12_index", np.zeros((len(trace["timestamps"]),), dtype=np.int64)),
            dtype=np.int64,
        ),
        "detector_top_margin": np.asarray(
            trace.get("detector_top_margin", np.zeros((len(trace["timestamps"]),), dtype=np.float32)),
            dtype=np.float32,
        ),
        "ast_probs": np.asarray(trace["ast_probs"], dtype=np.float32),
        "superb_probs": np.asarray(trace["superb_probs"], dtype=np.float32),
    }
    np.savez_compressed(target, **payload)
    return target


def load_trace(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=True) as payload:
        return {
            "record_path": str(payload["record_path"][0]),
            "target_kws12": int(payload["target_kws12"][0]),
            "window_start": float(payload["window_start"][0]),
            "window_end": float(payload["window_end"][0]),
            "timestamps": payload["timestamps"].astype(np.float32, copy=False),
            "detector_command_probs": payload["detector_command_probs"].astype(np.float32, copy=False),
            "detector_wake_probs": payload["detector_wake_probs"].astype(np.float32, copy=False),
            "detector_top_kws12_index": payload["detector_top_kws12_index"].astype(np.int64, copy=False) if "detector_top_kws12_index" in payload else np.zeros((payload["timestamps"].shape[0],), dtype=np.int64),
            "detector_top_margin": payload["detector_top_margin"].astype(np.float32, copy=False) if "detector_top_margin" in payload else np.zeros((payload["timestamps"].shape[0],), dtype=np.float32),
            "ast_probs": payload["ast_probs"].astype(np.float32, copy=False),
            "superb_probs": payload["superb_probs"].astype(np.float32, copy=False),
        }


def _build_replay_engine(*, bundle, args, tuning) -> RealtimeEngine:
    gate_mode, wake_open_thr, wake_close_thr = resolve_gate_args(args)
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
        input_device_name="trace-replay",
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
    return engine


def replay_trace_segments(*, bundle, trace: Mapping[str, Any], args, tuning) -> list[dict[str, Any]]:
    engine = _build_replay_engine(bundle=bundle, args=args, tuning=tuning)

    timestamps = np.asarray(trace["timestamps"], dtype=np.float32)
    detector_probs = np.asarray(trace["detector_command_probs"], dtype=np.float32)
    wake_probs = np.asarray(trace["detector_wake_probs"], dtype=np.float32)
    ast_probs = np.asarray(trace["ast_probs"], dtype=np.float32)
    superb_probs = np.asarray(trace["superb_probs"], dtype=np.float32)
    frame_waveforms = reconstruct_trace_frame_waveforms(trace=trace, bundle=bundle, args=args)
    for idx, now in enumerate(timestamps.tolist()):
        det_probs = detector_probs[idx]
        if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND and ast_probs.size and superb_probs.size:
            fused_probs = blend_ast_superb_probs(
                ast_probs[idx : idx + 1],
                superb_probs[idx : idx + 1],
                calibration=bundle.external_ensemble_calibration,
            )[0]
            decision_command_probs = _command31_probs_from_kws12_probs(fused_probs, bundle.command31_labels)
            backend_name = f"{DEFAULT_RUNTIME_LABEL_BACKEND}@trace"
        else:
            fused_probs = np.zeros((len(KWS12_LABELS),), dtype=np.float32)
            decision_command_probs = det_probs
            for command_idx, label in enumerate(bundle.command31_labels):
                fused_probs[command31_to_kws12(label)] += float(det_probs[command_idx])
            backend_name = "detector"
        engine._process_runtime_step(
            detector_probs=det_probs,
            wake_prob=float(wake_probs[idx]),
            decision_command_probs=decision_command_probs,
            runtime_kws12_probs=fused_probs,
            model_waveform_np=frame_waveforms[idx] if idx < len(frame_waveforms) else None,
            backend_name=backend_name,
            backend_note="trace-replay",
            now=float(now),
            now_wall=float(now),
            queue_fill_ratio=0.0,
            latency_ms=0.0,
        )
    engine.flush_pending_segment(
        now=float(timestamps[-1]) if timestamps.size else float(trace["window_end"]),
        now_wall=float(timestamps[-1]) if timestamps.size else float(trace["window_end"]),
    )
    return engine.drain_finalized_segments()


def replay_clip_trace(*, bundle, trace: Mapping[str, Any], args, tuning) -> tuple[int, bool, float | None]:
    engine = _build_replay_engine(bundle=bundle, args=args, tuning=tuning)

    frames: list[tuple[float, object]] = []
    timestamps = np.asarray(trace["timestamps"], dtype=np.float32)
    detector_probs = np.asarray(trace["detector_command_probs"], dtype=np.float32)
    wake_probs = np.asarray(trace["detector_wake_probs"], dtype=np.float32)
    ast_probs = np.asarray(trace["ast_probs"], dtype=np.float32)
    superb_probs = np.asarray(trace["superb_probs"], dtype=np.float32)
    frame_waveforms = reconstruct_trace_frame_waveforms(trace=trace, bundle=bundle, args=args)
    for idx, now in enumerate(timestamps.tolist()):
        det_probs = detector_probs[idx]
        if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND and ast_probs.size and superb_probs.size:
            fused_probs = blend_ast_superb_probs(
                ast_probs[idx : idx + 1],
                superb_probs[idx : idx + 1],
                calibration=bundle.external_ensemble_calibration,
            )[0]
            decision_command_probs = _command31_probs_from_kws12_probs(fused_probs, bundle.command31_labels)
            backend_name = f"{DEFAULT_RUNTIME_LABEL_BACKEND}@trace"
        else:
            fused_probs = np.zeros((len(KWS12_LABELS),), dtype=np.float32)
            decision_command_probs = det_probs
            for command_idx, label in enumerate(bundle.command31_labels):
                fused_probs[command31_to_kws12(label)] += float(det_probs[command_idx])
            backend_name = "detector"
        snapshot = engine._process_runtime_step(
            detector_probs=det_probs,
            wake_prob=float(wake_probs[idx]),
            decision_command_probs=decision_command_probs,
            runtime_kws12_probs=fused_probs,
            model_waveform_np=frame_waveforms[idx] if idx < len(frame_waveforms) else None,
            backend_name=backend_name,
            backend_note="trace-replay",
            now=float(now),
            now_wall=float(now),
            queue_fill_ratio=0.0,
            latency_ms=0.0,
        )
        frames.append((float(now), snapshot))
    flushed = engine.flush_pending_segment(
        now=float(timestamps[-1]) if timestamps.size else float(trace["window_end"]),
        now_wall=float(timestamps[-1]) if timestamps.size else float(trace["window_end"]),
    )
    if flushed is not None:
        frames.append((float(trace["window_end"]), flushed))

    return summarize_clip_frames(
        frames,
        window_start=float(trace["window_start"]),
        window_end=float(trace["window_end"]),
    )


def trace_manifest_payload(*, split: str, bundle, args, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "split": str(split),
        "demo_profile": str(bundle.resolved_profile.demo_profile),
        "runtime_label_backend": str(bundle.resolved_profile.runtime_label_backend),
        "external_kws_model": str(bundle.resolved_profile.external_kws_model),
        "external_kws_device": str(bundle.resolved_profile.external_kws_device),
        "keyword_calibration_path": str(bundle.keyword_calibration_path) if bundle.keyword_calibration_path else "",
        "external_ensemble_calibration_path": (
            str(bundle.external_ensemble_calibration_path) if bundle.external_ensemble_calibration_path else ""
        ),
        "realtime_specialist_path": str(bundle.realtime_specialist_path) if getattr(bundle, "realtime_specialist_path", None) else "",
        "realtime_specialist_calibration_path": (
            str(bundle.realtime_specialist_calibration_path)
            if getattr(bundle, "realtime_specialist_calibration_path", None)
            else ""
        ),
        "hop_seconds": float(args.hop_seconds),
        "ema_alpha": float(args.ema_alpha),
        "hold_ms": float(args.hold_ms),
        "pre_silence_seconds": float(args.pre_silence_seconds),
        "post_silence_seconds": float(args.post_silence_seconds),
        "match_tail_seconds": float(args.match_tail_seconds),
        "calibration_seconds": float(args.calibration_seconds),
        "num_eval_samples": int(len(entries)),
        "entries": list(entries),
    }


def write_trace_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
