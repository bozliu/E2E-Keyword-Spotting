#!/usr/bin/env python3
"""Render a small deterministic GIF from the realtime accuracy-first demo path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from kws.constants import IGNORE_INDEX, INDEX_TO_COMMAND31, KWS12_LABELS, command31_to_kws12
from kws.data.audio import load_audio
from kws.data.manifest import read_manifest
from kws.demo.realtime import (
    AdaptiveGateConfig,
    GateStateMachine,
    RealtimeEngine,
    get_sensitivity_tuning,
    load_realtime_demo,
)
from kws.demo.validate_realtime import _build_stream_waveform, _estimate_utterance_bounds, _iter_hop_chunks
from kws.external import predict_kws12_from_paths


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GIF for the public release README.")
    parser.add_argument("--label", type=str, default="go", choices=KWS12_LABELS[2:])
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--external-kws-device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="docs/assets/realtime_accuracy_first_demo.gif")
    parser.add_argument("--metadata-output", type=str, default="docs/assets/data/realtime_demo_clip.json")
    parser.add_argument("--max-search", type=int, default=32)
    return parser.parse_args()


def _pick_record(label: str, split: str, max_search: int, device: str) -> dict[str, Any]:
    manifest_path = ROOT / "data" / "processed" / "manifests" / f"local_{split}.jsonl"
    records = read_manifest(manifest_path)
    candidates = []
    for rec in records:
        if rec.command_label is None or int(rec.command_label) == IGNORE_INDEX:
            continue
        kws12 = command31_to_kws12(INDEX_TO_COMMAND31[int(rec.command_label)])
        if kws12 != label:
            continue
        candidates.append(rec)
        if len(candidates) >= max_search:
            break
    if not candidates:
        raise RuntimeError(f"No manifest records found for label={label} split={split}")

    result = predict_kws12_from_paths([str(rec.path) for rec in candidates], model_id="ensemble/ast-superb-kws12", device=device)
    preds = result.probs.argmax(axis=1)
    for rec, pred in zip(candidates, preds):
        if KWS12_LABELS[int(pred)] == label:
            return {"record": rec, "model_id": result.model_id, "runtime_device": result.runtime_device}
    return {"record": candidates[0], "model_id": result.model_id, "runtime_device": result.runtime_device}


def _collect_frames(args: argparse.Namespace, label: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    selected = _pick_record(label, args.split, int(args.max_search), args.external_kws_device)
    record = selected["record"]
    bundle = load_realtime_demo(
        checkpoint=args.checkpoint,
        demo_profile="accuracy-first",
        detector_device_preference=args.device,
        selection_profile="stable",
        wheel="kws12",
        runtime_label_backend="external-ensemble",
        external_kws_model="ensemble/ast-superb-kws12",
        external_kws_device=args.external_kws_device,
        ranking_iters=8,
        no_cache_ranking=False,
        rebuild_ranking=False,
        device_auto_bench_iters=6,
    )
    tuning = get_sensitivity_tuning("strict")
    gate = GateStateMachine(
        mode="adaptive",
        open_threshold=0.6,
        close_threshold=0.5,
        cmd_conf_threshold=tuning.cmd_conf_thr,
        hold_seconds=0.3,
        adaptive=AdaptiveGateConfig(
            calibration_seconds=2.0,
            open_offset=tuning.open_offset,
            close_offset=tuning.close_offset,
            open_floor=tuning.open_floor,
            close_floor=tuning.close_floor,
        ),
    )
    engine = RealtimeEngine(
        model=bundle.model,
        frontend=bundle.frontend,
        device=bundle.runtime_device,
        command31_labels=bundle.command31_labels,
        wheel="kws12",
        gate=gate,
        hop_seconds=0.10,
        ema_alpha=0.35,
        hold_ms=300.0,
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
        display_conf_thr=tuning.display_conf_thr,
        display_wake_thr=tuning.display_wake_thr,
        vote_window=tuning.vote_window,
        vote_min_count=tuning.vote_min_count,
        passive_profile=None,
        keyword_calibration=bundle.keyword_calibration,
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
        calibration_seconds=2.0,
        pre_silence_seconds=0.4,
        post_silence_seconds=0.4,
    )
    hop_samples = max(1, int(round(0.10 * bundle.sample_rate)))
    frames: list[dict[str, Any]] = []
    sim_now = 0.0
    for chunk in _iter_hop_chunks(stream, hop_samples):
        sim_now += float(chunk.size) / float(bundle.sample_rate)
        snapshot = engine.process_chunk(chunk, now=sim_now, now_wall=sim_now, queue_fill_ratio=0.0)
        if snapshot is None:
            continue
        frames.append(
            {
                "time": sim_now,
                "prompt_status": snapshot.prompt_status,
                "active_label": snapshot.active_label,
                "display_label": snapshot.display_label,
                "command_label": snapshot.command_label,
                "gate_open": bool(snapshot.gate_open),
                "command_conf": float(snapshot.command_conf),
                "wake_prob": float(snapshot.wake_prob),
                "latency_ms": float(snapshot.latency_ms),
                "backend": snapshot.runtime_label_backend,
            }
        )

    meta = {
        "label": label,
        "split": args.split,
        "sample_rate": int(bundle.sample_rate),
        "utterance_bounds_seconds": [
            float(utt_start) / float(bundle.sample_rate),
            float(utt_end) / float(bundle.sample_rate),
        ],
        "model_id": selected["model_id"],
        "runtime_device": selected["runtime_device"],
        "num_frames": len(frames),
    }
    return stream, frames, meta


def _render_gif(stream: np.ndarray, frames: list[dict[str, Any]], meta: dict[str, Any], output_path: Path) -> None:
    times = np.linspace(0.0, float(stream.size) / float(meta["sample_rate"]), num=stream.size, endpoint=False)
    frame_ids = np.linspace(0, max(len(frames) - 1, 0), num=min(18, max(1, len(frames))), dtype=int)
    sampled = [frames[idx] for idx in frame_ids]
    start_sec, end_sec = meta["utterance_bounds_seconds"]

    fig, (ax_wave, ax_info) = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    fig.patch.set_facecolor("#f7f4ed")
    ax_wave.set_facecolor("#fffaf0")
    ax_info.set_facecolor("#fffaf0")

    ax_wave.plot(times, stream, color="#1768ac", linewidth=1.0)
    ax_wave.axvspan(start_sec, end_sec, color="#5ec576", alpha=0.18, label="Utterance window")
    cursor = ax_wave.axvline(0.0, color="#b7410e", linewidth=2.0)
    title = ax_wave.set_title("", fontsize=14, fontweight="bold")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.legend(loc="upper right")

    info_text = ax_info.text(
        0.02,
        0.94,
        "",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        transform=ax_info.transAxes,
    )
    ax_info.axis("off")

    writer = PillowWriter(fps=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output_path), dpi=120):
        for frame in sampled:
            cursor.set_xdata([frame["time"], frame["time"]])
            title.set_text(f"Realtime accuracy-first demo example: '{meta['label']}'")
            info_text.set_text(
                "\n".join(
                    [
                        f"True label      : {meta['label']}",
                        f"Current time    : {frame['time']:.2f}s",
                        f"Prompt status   : {frame['prompt_status']}",
                        f"Display label   : {frame['display_label']}",
                        f"Active label    : {frame['active_label'] or '-'}",
                        f"Gate open       : {frame['gate_open']}",
                        f"Command conf    : {frame['command_conf']:.3f}",
                        f"Wake prob       : {frame['wake_prob']:.3f}",
                        f"Backend         : {frame['backend']}",
                        f"Latency         : {frame['latency_ms']:.2f} ms",
                    ]
                )
            )
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.external_kws_device != "mps":
        raise RuntimeError("The public release GIF should be rendered with --external-kws-device mps.")
    stream, frames, meta = _collect_frames(args, args.label)
    output_path = (ROOT / args.output).resolve()
    metadata_path = (ROOT / args.metadata_output).resolve()
    _render_gif(stream, frames, meta, output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metadata": meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
