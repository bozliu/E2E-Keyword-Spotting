"""Torch dataset and dataloader helpers for dual-task KWS."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from kws.constants import CLIP_SAMPLES, IGNORE_INDEX, SAMPLE_RATE
from kws.data.audio import MelFrontend, load_audio, pad_or_trim, random_crop_or_pad
from kws.data.manifest import ManifestRecord, read_manifest


@dataclass
class Batch:
    features: torch.Tensor
    command_labels: torch.Tensor
    wake_labels: torch.Tensor
    paths: List[str]
    sources: List[str]
    transcripts: List[str | None] = field(default_factory=list)
    accent_groups: List[str | None] = field(default_factory=list)
    l1_groups: List[str | None] = field(default_factory=list)
    difficulty_buckets: List[str | None] = field(default_factory=list)


@dataclass(frozen=True)
class AugmentConfig:
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    loudness_min: float = 0.65
    loudness_max: float = 1.35
    max_shift_frac: float = 0.20
    gaussian_noise_scale: float = 0.005
    background_mix_prob: float = 0.35
    background_snr_min_db: float = 6.0
    background_snr_max_db: float = 20.0
    reverb_prob: float = 0.20
    reverb_decay: float = 18.0
    specaugment_time_masks: int = 2
    specaugment_time_width: int = 10
    specaugment_freq_masks: int = 2
    specaugment_freq_width: int = 8


def parse_augment_config(raw: object, *, enabled_default: bool = True) -> tuple[bool, AugmentConfig]:
    if isinstance(raw, bool):
        return bool(raw), AugmentConfig()
    if not isinstance(raw, Mapping):
        return bool(enabled_default), AugmentConfig()

    enabled = bool(raw.get("enabled", enabled_default))
    speed = tuple(float(v) for v in raw.get("speed_factors", (0.9, 1.0, 1.1)))
    return (
        enabled,
        AugmentConfig(
            speed_factors=speed or (1.0,),
            loudness_min=float(raw.get("loudness_min", 0.65)),
            loudness_max=float(raw.get("loudness_max", 1.35)),
            max_shift_frac=float(raw.get("max_shift_frac", 0.20)),
            gaussian_noise_scale=float(raw.get("gaussian_noise_scale", 0.005)),
            background_mix_prob=float(raw.get("background_mix_prob", 0.35)),
            background_snr_min_db=float(raw.get("background_snr_min_db", 6.0)),
            background_snr_max_db=float(raw.get("background_snr_max_db", 20.0)),
            reverb_prob=float(raw.get("reverb_prob", 0.20)),
            reverb_decay=float(raw.get("reverb_decay", 18.0)),
            specaugment_time_masks=int(raw.get("specaugment_time_masks", 2)),
            specaugment_time_width=int(raw.get("specaugment_time_width", 10)),
            specaugment_freq_masks=int(raw.get("specaugment_freq_masks", 2)),
            specaugment_freq_width=int(raw.get("specaugment_freq_width", 8)),
        ),
    )


class KWSSampleDataset(Dataset):
    """Manifest-driven waveform dataset."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        split: str,
        augment: bool = False,
        augment_cfg: AugmentConfig | None = None,
    ) -> None:
        self.records = list(records)
        self.split = split
        self.augment = augment
        self.augment_cfg = augment_cfg or AugmentConfig()
        self._background_noise_paths = [
            str(Path(rec.path).expanduser().resolve())
            for rec in self.records
            if rec.source == "local_silence"
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _apply_speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        factor = random.choice(self.augment_cfg.speed_factors)
        if abs(float(factor) - 1.0) < 1e-4:
            return waveform
        orig_freq = max(1, int(round(SAMPLE_RATE * float(factor))))
        return AF.resample(waveform.unsqueeze(0), orig_freq=orig_freq, new_freq=SAMPLE_RATE).squeeze(0)

    def _mix_background(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self._background_noise_paths or random.random() > self.augment_cfg.background_mix_prob:
            return waveform

        noise_path = random.choice(self._background_noise_paths)
        noise = load_audio(noise_path, sample_rate=SAMPLE_RATE)
        noise = random_crop_or_pad(noise, target_samples=CLIP_SAMPLES)
        signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-4)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-4)
        snr_db = random.uniform(self.augment_cfg.background_snr_min_db, self.augment_cfg.background_snr_max_db)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)
        return waveform + noise

    def _apply_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.augment_cfg.reverb_prob:
            return waveform
        n_taps = random.randint(16, 64)
        decay = random.uniform(self.augment_cfg.reverb_decay * 0.7, self.augment_cfg.reverb_decay * 1.3)
        t = torch.linspace(0.0, 1.0, steps=n_taps, dtype=waveform.dtype)
        rir = torch.exp(-decay * t)
        rir[0] = 1.0
        if n_taps > 4:
            rir[random.randint(1, n_taps - 1)] += random.uniform(0.05, 0.20)
        rir = rir / rir.abs().sum().clamp(min=1e-6)
        reverbed = torch.nn.functional.conv1d(
            waveform.view(1, 1, -1),
            rir.view(1, 1, -1),
            padding=n_taps - 1,
        ).view(-1)
        return reverbed[: waveform.numel()]

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._apply_speed_perturb(waveform)
        waveform = random_crop_or_pad(waveform, target_samples=CLIP_SAMPLES)
        if random.random() < 0.7:
            gain = random.uniform(self.augment_cfg.loudness_min, self.augment_cfg.loudness_max)
            waveform = waveform * gain
        if random.random() < 0.6:
            shift = random.randint(
                -int(self.augment_cfg.max_shift_frac * CLIP_SAMPLES),
                int(self.augment_cfg.max_shift_frac * CLIP_SAMPLES),
            )
            if shift > 0:
                waveform = torch.nn.functional.pad(waveform, (shift, 0))[:CLIP_SAMPLES]
            elif shift < 0:
                waveform = torch.nn.functional.pad(waveform, (0, -shift))[-shift : -shift + CLIP_SAMPLES]
        waveform = self._mix_background(waveform)
        waveform = self._apply_reverb(waveform)
        noise_scale = random.uniform(0.0, self.augment_cfg.gaussian_noise_scale)
        waveform = waveform + noise_scale * torch.randn_like(waveform)
        return waveform.clamp(min=-1.0, max=1.0)

    def __getitem__(self, index: int) -> Dict[str, object]:
        rec = self.records[index]
        waveform = load_audio(rec.path, sample_rate=SAMPLE_RATE)

        if rec.source == "local_silence" or self.split == "train":
            waveform = random_crop_or_pad(waveform, target_samples=CLIP_SAMPLES)
        else:
            waveform = pad_or_trim(waveform, target_samples=CLIP_SAMPLES)

        if self.augment and self.split == "train":
            waveform = self._augment_waveform(waveform)

        command_label = rec.command_label if rec.command_label is not None else IGNORE_INDEX
        wake_label = rec.wake_label if rec.wake_label is not None else 0

        return {
            "audio": waveform,
            "sr": rec.sr,
            "command_label": int(command_label),
            "wake_label": int(wake_label),
            "source": rec.source,
            "path": rec.path,
            "speaker_id": rec.speaker_id,
            "transcript": rec.transcript,
            "accent_group": rec.accent_group,
            "l1_group": rec.l1_group,
            "is_synthetic": bool(rec.is_synthetic),
            "difficulty_bucket": rec.difficulty_bucket,
        }


class CollateWithFrontend:
    """Pickle-safe collate callable for multiprocessing dataloaders."""

    def __init__(self, frontend: MelFrontend, *, specaugment_cfg: AugmentConfig | None = None) -> None:
        self.frontend = frontend
        self.specaugment_cfg = specaugment_cfg

    def _apply_specaugment(self, feat: torch.Tensor) -> torch.Tensor:
        cfg = self.specaugment_cfg
        if cfg is None:
            return feat

        out = feat.clone()
        batch, n_mels, frames = out.shape
        for b in range(batch):
            for _ in range(max(0, cfg.specaugment_time_masks)):
                width = min(frames, random.randint(0, max(0, cfg.specaugment_time_width)))
                if width <= 0 or frames <= width:
                    continue
                start = random.randint(0, frames - width)
                out[b, :, start : start + width] = 0.0
            for _ in range(max(0, cfg.specaugment_freq_masks)):
                width = min(n_mels, random.randint(0, max(0, cfg.specaugment_freq_width)))
                if width <= 0 or n_mels <= width:
                    continue
                start = random.randint(0, n_mels - width)
                out[b, start : start + width, :] = 0.0
        return out

    def __call__(self, samples: List[Dict[str, object]]) -> Batch:
        audios = [sample["audio"] for sample in samples]
        features = [self.frontend(audio) for audio in audios]
        feat = torch.stack(features, dim=0)

        # Per-sample CMVN for stability.
        mean = feat.mean(dim=(1, 2), keepdim=True)
        std = feat.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        feat = (feat - mean) / std
        feat = self._apply_specaugment(feat)

        command_labels = torch.tensor([sample["command_label"] for sample in samples], dtype=torch.long)
        wake_labels = torch.tensor([sample["wake_label"] for sample in samples], dtype=torch.float32)
        paths = [str(sample["path"]) for sample in samples]
        sources = [str(sample["source"]) for sample in samples]
        transcripts = [sample.get("transcript") for sample in samples]
        accent_groups = [sample.get("accent_group") for sample in samples]
        l1_groups = [sample.get("l1_group") for sample in samples]
        difficulty_buckets = [sample.get("difficulty_bucket") for sample in samples]
        return Batch(
            features=feat,
            command_labels=command_labels,
            wake_labels=wake_labels,
            paths=paths,
            sources=sources,
            transcripts=transcripts,
            accent_groups=accent_groups,
            l1_groups=l1_groups,
            difficulty_buckets=difficulty_buckets,
        )


def build_collate_fn(frontend: MelFrontend, *, specaugment_cfg: AugmentConfig | None = None) -> CollateWithFrontend:
    return CollateWithFrontend(frontend, specaugment_cfg=specaugment_cfg)


def load_manifests(paths: Iterable[str | Path]) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        records.extend(read_manifest(p))
    return records
