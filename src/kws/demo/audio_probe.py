"""Lightweight microphone probe for debugging input capture without loading a model."""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np

from kws.demo.realtime import MicPrecheckResult, _compute_rms, _resolve_audio_input_spec, sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe microphone capture and print chunk RMS/peak.")
    parser.add_argument("--audio-device", type=str, default="", help="Optional sounddevice device id/name")
    parser.add_argument("--duration", type=float, default=3.0, help="Probe duration in seconds")
    parser.add_argument("--hop-seconds", type=float, default=0.10, help="Chunk duration in seconds")
    parser.add_argument("--sample-rate", type=float, default=0.0, help="Optional stream sample rate override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sd is None:
        raise RuntimeError("sounddevice is not installed.")

    resolved = _resolve_audio_input_spec(args.audio_device)
    if isinstance(resolved, MicPrecheckResult):
        raise RuntimeError(resolved.message)

    stream_sample_rate = float(args.sample_rate) if float(args.sample_rate) > 0.0 else float(resolved.default_samplerate)
    blocksize = max(1, int(round(stream_sample_rate * float(args.hop_seconds))))
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):  # type: ignore[override]
        if status:
            print(f"Audio status: {status}")
        q.put(np.asarray(indata[:, 0], dtype=np.float32).copy())

    print(
        f"input={resolved.name} index={resolved.index} sr={stream_sample_rate:.0f} "
        f"blocksize={blocksize} duration={float(args.duration):.1f}s"
    )
    started = time.monotonic()
    chunk_idx = 0
    with sd.InputStream(
        samplerate=stream_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=int(resolved.index),
        callback=callback,
    ):
        while (time.monotonic() - started) < float(args.duration):
            chunk = q.get(timeout=max(float(args.hop_seconds) * 2.0, 0.5))
            rms = _compute_rms(chunk)
            peak = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0
            print(f"chunk={chunk_idx:03d} rms={rms:.6f} peak={peak:.6f}")
            chunk_idx += 1


if __name__ == "__main__":
    main()
