"""Lightweight hard-word specialist for realtime-only stream recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from kws.constants import CLIP_SAMPLES, SAMPLE_RATE
from kws.data.audio import MelFrontend, pad_or_trim


HARD_WORD_SPECIALIST_LABELS = ("on", "off", "go", "up", "other")
HARD_WORD_SPECIALIST_TARGETS = HARD_WORD_SPECIALIST_LABELS[:-1]
HARD_WORD_SPECIALIST_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(HARD_WORD_SPECIALIST_LABELS)}
HARD_WORD_SPECIALIST_INDEX_TO_LABEL = {idx: label for label, idx in HARD_WORD_SPECIALIST_LABEL_TO_INDEX.items()}


def default_realtime_specialist_calibration() -> dict[str, object]:
    return {
        "version": 2,
        "mode": "realtime-specialist",
        "enabled": True,
        "default": {
            "accept_prob": 0.66,
            "min_margin": 0.08,
            "trigger_prob": 0.24,
            "role": "rescue",
        },
        "per_label": {
            "on": {"accept_prob": 0.56, "min_margin": 0.02, "trigger_prob": 0.16, "role": "rescue"},
            "off": {"accept_prob": 0.58, "min_margin": 0.03, "trigger_prob": 0.18, "role": "rescue"},
            "go": {"accept_prob": 0.58, "min_margin": 0.03, "trigger_prob": 0.18, "role": "rescue"},
            "up": {"accept_prob": 0.72, "min_margin": 0.10, "trigger_prob": 0.22, "role": "guard"},
        },
        "unknown_trigger_prob": 0.18,
    }


def load_realtime_specialist_calibration(path: str | Path) -> dict[str, object]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def save_realtime_specialist_calibration(path: str | Path, calibration: dict[str, object]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


@dataclass(frozen=True)
class LoadedRealtimeSpecialist:
    path: Path
    model: "RealtimeHardWordSpecialist"
    device: torch.device
    sample_rate: int
    target_samples: int
    n_mels: int
    hidden_dim: int
    labels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray


class _AttentivePool1d(nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        self.score = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(self.proj(x))), dim=-1)
        return torch.sum(x * weights, dim=-1)


class RealtimeHardWordSpecialist(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = 64,
        hidden_dim: int = 96,
        output_dim: int = len(HARD_WORD_SPECIALIST_LABELS),
    ) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.hidden_dim = int(hidden_dim)
        self.mel_stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.attentive_pool = _AttentivePool1d(hidden_dim, max(hidden_dim // 2, 32))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.18),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mel_stem(x)
        y = y.mean(dim=2)
        y = self.temporal(y)
        y = self.attentive_pool(y)
        return self.head(y)


def _make_frontend(*, sample_rate: int = SAMPLE_RATE, n_mels: int = 64) -> MelFrontend:
    return MelFrontend(
        sample_rate=int(sample_rate),
        n_fft=512,
        hop_length=128,
        n_mels=int(n_mels),
        f_min=20.0,
        f_max=7600.0,
    )


def waveform_to_specialist_feature(
    waveform: np.ndarray | torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    target_samples: int = CLIP_SAMPLES,
    n_mels: int = 64,
) -> np.ndarray:
    if isinstance(waveform, np.ndarray):
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32).reshape(-1))
    else:
        tensor = waveform.detach().cpu().float().reshape(-1)
    tensor = pad_or_trim(tensor, target_samples=int(target_samples))
    frontend = _make_frontend(sample_rate=int(sample_rate), n_mels=int(n_mels))
    mel = frontend(tensor).float()
    mean = mel.mean()
    std = mel.std().clamp(min=1e-5)
    mel = (mel - mean) / std
    return mel.unsqueeze(0).numpy().astype(np.float32, copy=False)


def _stack_features(
    waveforms: np.ndarray,
    *,
    sample_rate: int,
    target_samples: int,
    n_mels: int,
) -> np.ndarray:
    items = [
        waveform_to_specialist_feature(
            waveform,
            sample_rate=sample_rate,
            target_samples=target_samples,
            n_mels=n_mels,
        )
        for waveform in waveforms
    ]
    return np.stack(items, axis=0).astype(np.float32, copy=False)


def _augment_waveforms(
    waveforms: np.ndarray,
    labels: np.ndarray,
    *,
    target_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    augmented_waveforms: list[np.ndarray] = [np.asarray(waveforms, dtype=np.float32)]
    augmented_labels: list[np.ndarray] = [np.asarray(labels, dtype=np.int64)]
    hard_set = {
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["on"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["off"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["go"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["up"],
    }
    max_shift = max(1, int(target_samples * 0.06))
    for waveform, label in zip(np.asarray(waveforms, dtype=np.float32), np.asarray(labels, dtype=np.int64), strict=False):
        copies = 2 if int(label) in hard_set else 1
        for _ in range(copies):
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift > 0:
                shifted = np.pad(waveform[:-shift], (shift, 0), mode="constant")
            elif shift < 0:
                shifted = np.pad(waveform[-shift:], (0, -shift), mode="constant")
            else:
                shifted = waveform.copy()
            gain = float(rng.uniform(0.90, 1.10))
            noise_scale = float(rng.uniform(0.0, 0.0035 if int(label) in hard_set else 0.0020))
            noisy = np.clip(shifted * gain + rng.normal(0.0, noise_scale, size=shifted.shape), -1.0, 1.0).astype(
                np.float32,
                copy=False,
            )
            augmented_waveforms.append(noisy.reshape(1, -1))
            augmented_labels.append(np.asarray([label], dtype=np.int64))
    return np.concatenate(augmented_waveforms, axis=0), np.concatenate(augmented_labels, axis=0)


def save_realtime_specialist_artifact(
    path: str | Path,
    *,
    model: RealtimeHardWordSpecialist,
    sample_rate: int,
    target_samples: int,
    n_mels: int,
    hidden_dim: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_n_mels = int(getattr(model, "n_mels", n_mels))
    resolved_hidden_dim = int(getattr(model, "hidden_dim", hidden_dim))
    payload = {
        "version": 2,
        "labels": list(HARD_WORD_SPECIALIST_LABELS),
        "sample_rate": int(sample_rate),
        "target_samples": int(target_samples),
        "n_mels": resolved_n_mels,
        "hidden_dim": resolved_hidden_dim,
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, target)
    return target


def load_realtime_specialist_artifact(path: str | Path, *, device: torch.device | str = "cpu") -> LoadedRealtimeSpecialist:
    target = Path(path).expanduser().resolve()
    payload = torch.load(target, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("hidden_dim", 96))
    n_mels = int(payload.get("n_mels", 64))
    model = RealtimeHardWordSpecialist(
        n_mels=n_mels,
        hidden_dim=hidden_dim,
        output_dim=len(payload.get("labels", HARD_WORD_SPECIALIST_LABELS)),
    )
    model.load_state_dict(payload["state_dict"])
    runtime_device = torch.device(device)
    model.to(runtime_device)
    model.eval()
    return LoadedRealtimeSpecialist(
        path=target,
        model=model,
        device=runtime_device,
        sample_rate=int(payload.get("sample_rate", SAMPLE_RATE)),
        target_samples=int(payload.get("target_samples", CLIP_SAMPLES)),
        n_mels=n_mels,
        hidden_dim=hidden_dim,
        labels=tuple(str(label) for label in payload.get("labels", HARD_WORD_SPECIALIST_LABELS)),
        feature_mean=np.asarray(payload.get("feature_mean", np.zeros((1, n_mels, 126), dtype=np.float32)), dtype=np.float32),
        feature_std=np.asarray(payload.get("feature_std", np.ones((1, n_mels, 126), dtype=np.float32)), dtype=np.float32),
    )


def predict_realtime_specialist(
    specialist: LoadedRealtimeSpecialist,
    waveform: np.ndarray,
) -> np.ndarray:
    feature = waveform_to_specialist_feature(
        waveform,
        sample_rate=specialist.sample_rate,
        target_samples=specialist.target_samples,
        n_mels=specialist.n_mels,
    )
    mean = specialist.feature_mean.reshape(1, *specialist.feature_mean.shape)
    std = np.maximum(specialist.feature_std.reshape(1, *specialist.feature_std.shape), 1e-5)
    batch = (feature.reshape(1, *feature.shape) - mean) / std
    with torch.no_grad():
        logits = specialist.model(torch.from_numpy(batch).to(specialist.device))
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return probs


def summarize_realtime_specialist_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=-1).astype(np.int64, copy=False) if probs.size else np.zeros((0,), dtype=np.int64)
    per_label: dict[str, dict[str, object]] = {}
    f1_values: list[float] = []
    hard_precisions: list[float] = []
    hard_recalls: list[float] = []
    for idx, label in enumerate(HARD_WORD_SPECIALIST_LABELS):
        tp = float(np.sum((preds == idx) & (labels == idx)))
        fp = float(np.sum((preds == idx) & (labels != idx)))
        fn = float(np.sum((preds != idx) & (labels == idx)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        support = int(np.sum(labels == idx))
        predicted = int(np.sum(preds == idx))
        confusion = {}
        wrong_mask = labels == idx
        for wrong_idx in preds[wrong_mask]:
            if int(wrong_idx) == idx:
                continue
            wrong_label = HARD_WORD_SPECIALIST_INDEX_TO_LABEL[int(wrong_idx)]
            confusion[wrong_label] = confusion.get(wrong_label, 0) + 1
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "predicted": predicted,
            "top_confusions": [
                {"label": wrong_label, "count": int(count)}
                for wrong_label, count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))[:3]
            ],
        }
        f1_values.append(float(f1))
        if label in HARD_WORD_SPECIALIST_TARGETS:
            hard_precisions.append(float(precision))
            hard_recalls.append(float(recall))
    return {
        "per_label": per_label,
        "macro_f1": float(np.mean(np.asarray(f1_values, dtype=np.float32))) if f1_values else 0.0,
        "hard_word_macro_f1": float(
            np.mean(np.asarray([per_label[label]["f1"] for label in HARD_WORD_SPECIALIST_TARGETS], dtype=np.float32))
        )
        if probs.size
        else 0.0,
        "hard_word_min_precision": float(min(hard_precisions)) if hard_precisions else 0.0,
        "hard_word_min_recall": float(min(hard_recalls)) if hard_recalls else 0.0,
    }


def train_realtime_specialist(
    *,
    train_waveforms: np.ndarray,
    train_labels: np.ndarray,
    valid_waveforms: np.ndarray,
    valid_labels: np.ndarray,
    device: torch.device | str = "cpu",
    sample_rate: int = SAMPLE_RATE,
    target_samples: int = CLIP_SAMPLES,
    n_mels: int = 64,
    hidden_dim: int = 96,
    epochs: int = 14,
    batch_size: int = 32,
    lr: float = 8e-4,
) -> tuple[RealtimeHardWordSpecialist, np.ndarray, np.ndarray]:
    if train_waveforms.size == 0 or train_labels.size == 0:
        raise ValueError("train_waveforms/train_labels must be non-empty")

    augmented_waveforms, augmented_labels = _augment_waveforms(
        np.asarray(train_waveforms, dtype=np.float32),
        np.asarray(train_labels, dtype=np.int64),
        target_samples=int(target_samples),
    )
    train_features = _stack_features(
        augmented_waveforms,
        sample_rate=sample_rate,
        target_samples=target_samples,
        n_mels=n_mels,
    )
    if valid_waveforms.size and valid_labels.size:
        valid_features = _stack_features(
            np.asarray(valid_waveforms, dtype=np.float32),
            sample_rate=sample_rate,
            target_samples=target_samples,
            n_mels=n_mels,
        )
    else:
        valid_features = np.zeros((0, 1, n_mels, train_features.shape[-1]), dtype=np.float32)

    feature_mean = train_features.mean(axis=0, keepdims=False).astype(np.float32, copy=False)
    feature_std = np.maximum(train_features.std(axis=0, keepdims=False), 1e-5).astype(np.float32, copy=False)
    train_x = ((train_features - feature_mean) / feature_std).astype(np.float32, copy=False)
    valid_x = ((valid_features - feature_mean) / feature_std).astype(np.float32, copy=False) if valid_features.size else valid_features
    train_y = np.asarray(augmented_labels, dtype=np.int64)
    valid_y = np.asarray(valid_labels, dtype=np.int64)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    label_counts = np.bincount(train_y, minlength=len(HARD_WORD_SPECIALIST_LABELS)).astype(np.float32)
    class_weights = np.ones((len(HARD_WORD_SPECIALIST_LABELS),), dtype=np.float32)
    nonzero = label_counts > 0
    class_weights[nonzero] = float(train_y.size) / np.maximum(label_counts[nonzero], 1.0)
    class_weights /= max(float(class_weights.mean()), 1e-6)
    other_idx = HARD_WORD_SPECIALIST_LABEL_TO_INDEX["other"]
    class_weights[other_idx] *= 0.75

    sample_weights = class_weights[train_y].astype(np.float64, copy=False)
    hard_set = {
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["on"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["off"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["go"],
        HARD_WORD_SPECIALIST_LABEL_TO_INDEX["up"],
    }
    for idx, label in enumerate(train_y.tolist()):
        if int(label) in hard_set:
            sample_weights[idx] *= 3.0
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(max(sample_weights.shape[0], batch_size * 2)),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), sampler=sampler)

    runtime_device = torch.device(device)
    model = RealtimeHardWordSpecialist(n_mels=n_mels, hidden_dim=hidden_dim).to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(runtime_device))

    best_state = None
    best_score = None
    for _epoch in range(int(epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(runtime_device)
            batch_y = batch_y.to(runtime_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        if not valid_x.size:
            continue
        with torch.no_grad():
            valid_logits = model(torch.from_numpy(valid_x).to(runtime_device))
            valid_probs = F.softmax(valid_logits, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
        metrics = summarize_realtime_specialist_predictions(valid_probs, valid_y)
        score = (
            -float(metrics["hard_word_min_precision"]),
            -float(metrics["hard_word_min_recall"]),
            -float(metrics["hard_word_macro_f1"]),
            -float(metrics["macro_f1"]),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, feature_mean, feature_std
