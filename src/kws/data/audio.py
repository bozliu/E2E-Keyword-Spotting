"""Audio helpers for KWS."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torchaudio

from kws.constants import CLIP_SAMPLES, SAMPLE_RATE


EPS = 1e-6


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def pad_or_trim(waveform: torch.Tensor, target_samples: int = CLIP_SAMPLES) -> torch.Tensor:
    if waveform.numel() > target_samples:
        return waveform[:target_samples]
    if waveform.numel() < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform


def random_crop_or_pad(waveform: torch.Tensor, target_samples: int = CLIP_SAMPLES) -> torch.Tensor:
    if waveform.numel() == target_samples:
        return waveform
    if waveform.numel() < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))
    max_start = waveform.numel() - target_samples
    start = random.randint(0, max_start)
    return waveform[start : start + target_samples]


def waveform_to_log_mel(
    waveform: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    mel = mel_transform(waveform.unsqueeze(0))
    mel = torch.log(mel + EPS)
    return mel.squeeze(0)


@dataclass
class MelFrontend:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = 1024
    hop_length: int = 128
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 7600.0

    def __post_init__(self) -> None:
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=True,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform_to_log_mel(waveform, self.transform)
