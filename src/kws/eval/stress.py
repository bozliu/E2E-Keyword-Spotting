"""Stress evaluation helpers for offline KWS analysis."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
import torchaudio.functional as AF

from kws.constants import CLIP_SAMPLES, IGNORE_INDEX, SAMPLE_RATE
from kws.data.audio import MelFrontend, load_audio, pad_or_trim
from kws.data.manifest import ManifestRecord
from kws.train.metrics import (
    compute_command_metrics,
    compute_keyword_breakdown,
    compute_kws12_accuracy,
    compute_kws12_breakdown,
    compute_wake_metrics,
)


def _simple_reverb(waveform: torch.Tensor) -> torch.Tensor:
    n_taps = 48
    t = torch.linspace(0.0, 1.0, steps=n_taps, dtype=waveform.dtype)
    rir = torch.exp(-15.0 * t)
    rir[0] = 1.0
    rir[10] += 0.12
    rir[24] += 0.08
    rir = rir / rir.abs().sum().clamp(min=1e-6)
    out = torch.nn.functional.conv1d(
        waveform.view(1, 1, -1),
        rir.view(1, 1, -1),
        padding=n_taps - 1,
    ).view(-1)
    return out[: waveform.numel()]


def _speed_perturb(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    orig_freq = max(1, int(round(SAMPLE_RATE * float(factor))))
    return AF.resample(waveform.unsqueeze(0), orig_freq=orig_freq, new_freq=SAMPLE_RATE).squeeze(0)


def _stress_scenarios() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "noise": lambda wav: (wav + 0.01 * torch.randn_like(wav)).clamp(-1.0, 1.0),
        "reverb": lambda wav: _simple_reverb(wav).clamp(-1.0, 1.0),
        "speed_0.9": lambda wav: pad_or_trim(_speed_perturb(wav, 0.9), target_samples=CLIP_SAMPLES),
    }


def run_stress_eval(
    *,
    records: Iterable[ManifestRecord],
    model: torch.nn.Module,
    frontend: MelFrontend,
    device: torch.device,
    audio_seconds: float,
    max_records: int = 256,
) -> Dict[str, Dict[str, float | object]]:
    selected = list(records)[: max(1, int(max_records))]
    if not selected:
        return {}

    model.eval()
    results: Dict[str, Dict[str, float | object]] = {}
    scenarios = _stress_scenarios()

    with torch.no_grad():
        for name, transform in scenarios.items():
            command_preds: List[int] = []
            command_targets: List[int] = []
            wake_scores: List[float] = []
            wake_targets: List[float] = []

            for rec in selected:
                waveform = load_audio(rec.path, sample_rate=SAMPLE_RATE)
                waveform = pad_or_trim(waveform, target_samples=CLIP_SAMPLES)
                waveform = transform(waveform)
                waveform = pad_or_trim(waveform, target_samples=CLIP_SAMPLES)

                feature = frontend(waveform)
                mean = feature.mean()
                std = feature.std().clamp(min=1e-5)
                x = ((feature - mean) / std).unsqueeze(0).to(device)
                out = model(x)

                command_preds.append(int(out.command_logits.argmax(dim=-1).item()))
                command_targets.append(int(rec.command_label if rec.command_label is not None else IGNORE_INDEX))
                wake_scores.append(float(torch.sigmoid(out.wake_logits).item()))
                wake_targets.append(float(rec.wake_label if rec.wake_label is not None else 0.0))

            preds = np.asarray(command_preds, dtype=np.int64)
            targets = np.asarray(command_targets, dtype=np.int64)
            scores = np.asarray(wake_scores, dtype=np.float32)
            wakes = np.asarray(wake_targets, dtype=np.float32)

            results[name] = {
                "kws12_acc": compute_kws12_accuracy(preds, targets),
                **compute_kws12_breakdown(preds, targets),
                **compute_keyword_breakdown(preds, targets),
                **compute_command_metrics(preds, targets),
                **compute_wake_metrics(scores, wakes, audio_seconds=audio_seconds),
            }

    return results
