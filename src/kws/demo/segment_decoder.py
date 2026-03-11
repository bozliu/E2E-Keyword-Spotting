"""Lightweight stream-segment decoder and feature extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kws.constants import TARGET_KEYWORDS_10, UNKNOWN_LABEL


SEGMENT_TARGET_LABELS = tuple(TARGET_KEYWORDS_10)
SEGMENT_DECODER_LABELS = (*SEGMENT_TARGET_LABELS, UNKNOWN_LABEL)
SEGMENT_DECODER_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(SEGMENT_DECODER_LABELS)}
SEGMENT_DECODER_INDEX_TO_LABEL = {idx: label for label, idx in SEGMENT_DECODER_LABEL_TO_INDEX.items()}
SEGMENT_DECODER_HARD_LABELS = {"on", "off", "go", "up"}


def segment_feature_names() -> list[str]:
    names: list[str] = []
    for label in SEGMENT_TARGET_LABELS:
        names.extend(
            [
                f"{label}_peak_prob",
                f"{label}_mean_prob",
                f"{label}_auc_norm",
                f"{label}_count_above_ratio",
                f"{label}_top_ratio",
            ]
        )
    names.extend(
        [
            "duration_seconds",
            "log1p_frame_count",
            "wake_peak",
            "wake_mean",
            "margin_peak",
            "margin_mean",
            "top_prob_peak",
            "top_prob_mean",
        ]
    )
    return names


SEGMENT_FEATURE_NAMES = tuple(segment_feature_names())
SEGMENT_FEATURE_DIM = len(SEGMENT_FEATURE_NAMES)


@dataclass
class SegmentFeatureStats:
    thresholds: np.ndarray = field(default_factory=lambda: np.full((len(SEGMENT_TARGET_LABELS),), 0.18, dtype=np.float32))
    started_at: float | None = None
    ended_at: float | None = None
    frame_count: int = 0
    probs_sum: np.ndarray = field(default_factory=lambda: np.zeros((len(SEGMENT_TARGET_LABELS),), dtype=np.float32))
    probs_auc: np.ndarray = field(default_factory=lambda: np.zeros((len(SEGMENT_TARGET_LABELS),), dtype=np.float32))
    probs_peak: np.ndarray = field(default_factory=lambda: np.zeros((len(SEGMENT_TARGET_LABELS),), dtype=np.float32))
    count_above: np.ndarray = field(default_factory=lambda: np.zeros((len(SEGMENT_TARGET_LABELS),), dtype=np.float32))
    top_count: np.ndarray = field(default_factory=lambda: np.zeros((len(SEGMENT_TARGET_LABELS),), dtype=np.float32))
    wake_sum: float = 0.0
    wake_peak: float = 0.0
    margin_sum: float = 0.0
    margin_peak: float = 0.0
    top_prob_sum: float = 0.0
    top_prob_peak: float = 0.0
    _prev_time: float | None = None

    def update(self, *, now: float, target_probs: np.ndarray, wake_prob: float) -> None:
        probs = np.asarray(target_probs, dtype=np.float32).reshape(len(SEGMENT_TARGET_LABELS))
        thresholds = np.asarray(self.thresholds, dtype=np.float32).reshape(len(SEGMENT_TARGET_LABELS))
        if self.started_at is None:
            self.started_at = float(now)
        dt = max(0.0, float(now) - float(self._prev_time if self._prev_time is not None else now))
        self._prev_time = float(now)
        self.ended_at = float(now)
        self.frame_count += 1

        self.probs_sum += probs
        self.probs_auc += probs * float(dt)
        self.probs_peak = np.maximum(self.probs_peak, probs)
        self.count_above += (probs >= thresholds).astype(np.float32)

        if probs.size:
            top_idx = int(np.argmax(probs))
            self.top_count[top_idx] += 1.0
            top_prob = float(probs[top_idx])
            self.top_prob_sum += top_prob
            self.top_prob_peak = max(self.top_prob_peak, top_prob)
            if probs.size > 1:
                top2 = np.partition(probs, -2)[-2:]
                margin = float(top2[-1] - top2[-2])
            else:
                margin = top_prob
            self.margin_sum += margin
            self.margin_peak = max(self.margin_peak, margin)

        self.wake_sum += float(wake_prob)
        self.wake_peak = max(self.wake_peak, float(wake_prob))

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, float(self.ended_at) - float(self.started_at))

    def as_feature_vector(self) -> np.ndarray:
        frames = max(int(self.frame_count), 1)
        duration = max(self.duration_seconds, 1e-6)
        features: list[float] = []
        for idx in range(len(SEGMENT_TARGET_LABELS)):
            features.extend(
                [
                    float(self.probs_peak[idx]),
                    float(self.probs_sum[idx] / frames),
                    float(self.probs_auc[idx] / duration),
                    float(self.count_above[idx] / frames),
                    float(self.top_count[idx] / frames),
                ]
            )
        features.extend(
            [
                float(self.duration_seconds),
                float(np.log1p(frames)),
                float(self.wake_peak),
                float(self.wake_sum / frames),
                float(self.margin_peak),
                float(self.margin_sum / frames),
                float(self.top_prob_peak),
                float(self.top_prob_sum / frames),
            ]
        )
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape[0] != SEGMENT_FEATURE_DIM:
            raise RuntimeError(f"Unexpected segment feature dimension {vector.shape[0]} != {SEGMENT_FEATURE_DIM}")
        return vector


def feature_stats_from_frames(
    *,
    timestamps: np.ndarray,
    fused_probs: np.ndarray,
    wake_probs: np.ndarray,
    thresholds: Sequence[float],
) -> SegmentFeatureStats:
    stats = SegmentFeatureStats(thresholds=np.asarray(list(thresholds), dtype=np.float32))
    for idx, now in enumerate(np.asarray(timestamps, dtype=np.float32).tolist()):
        stats.update(
            now=float(now),
            target_probs=np.asarray(fused_probs[idx], dtype=np.float32),
            wake_prob=float(wake_probs[idx]),
        )
    return stats


class StreamSegmentDecoder(nn.Module):
    def __init__(self, input_dim: int = SEGMENT_FEATURE_DIM, hidden_dim: int = 64, output_dim: int = len(SEGMENT_DECODER_LABELS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class LoadedSegmentDecoder:
    path: Path
    model: StreamSegmentDecoder
    device: torch.device
    feature_mean: np.ndarray
    feature_std: np.ndarray
    hidden_dim: int
    labels: tuple[str, ...]


def save_segment_decoder_artifact(
    path: str | Path,
    *,
    model: StreamSegmentDecoder,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    hidden_dim: int,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "labels": list(SEGMENT_DECODER_LABELS),
        "input_dim": int(SEGMENT_FEATURE_DIM),
        "hidden_dim": int(hidden_dim),
        "feature_names": list(SEGMENT_FEATURE_NAMES),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, target)
    return target


def load_segment_decoder_artifact(path: str | Path, *, device: torch.device | str = "cpu") -> LoadedSegmentDecoder:
    target = Path(path).expanduser().resolve()
    payload = torch.load(target, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("hidden_dim", 64))
    model = StreamSegmentDecoder(
        input_dim=int(payload.get("input_dim", SEGMENT_FEATURE_DIM)),
        hidden_dim=hidden_dim,
        output_dim=len(payload.get("labels", SEGMENT_DECODER_LABELS)),
    )
    model.load_state_dict(payload["state_dict"])
    runtime_device = torch.device(device)
    model.to(runtime_device)
    model.eval()
    return LoadedSegmentDecoder(
        path=target,
        model=model,
        device=runtime_device,
        feature_mean=np.asarray(payload.get("feature_mean", np.zeros((SEGMENT_FEATURE_DIM,), dtype=np.float32)), dtype=np.float32),
        feature_std=np.asarray(payload.get("feature_std", np.ones((SEGMENT_FEATURE_DIM,), dtype=np.float32)), dtype=np.float32),
        hidden_dim=hidden_dim,
        labels=tuple(str(label) for label in payload.get("labels", SEGMENT_DECODER_LABELS)),
    )


def predict_segment_decoder(decoder: LoadedSegmentDecoder, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    mean = decoder.feature_mean.reshape(1, -1)
    std = np.maximum(decoder.feature_std.reshape(1, -1), 1e-5)
    x_norm = (x - mean) / std
    with torch.no_grad():
        logits = decoder.model(torch.from_numpy(x_norm).to(decoder.device))
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return probs


def _expand_hard_examples(features: np.ndarray, labels: np.ndarray, hard_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    extra_features = []
    extra_labels = []
    for idx in hard_indices:
        extra_features.append(features[idx])
        extra_labels.append(labels[idx])
    if not extra_features:
        return features, labels
    return (
        np.concatenate([features, np.asarray(extra_features, dtype=np.float32)], axis=0),
        np.concatenate([labels, np.asarray(extra_labels, dtype=np.int64)], axis=0),
    )


def _macro_f1_score(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    scores: list[float] = []
    preds_arr = np.asarray(preds, dtype=np.int64).reshape(-1)
    targets_arr = np.asarray(targets, dtype=np.int64).reshape(-1)
    for class_idx in range(int(num_classes)):
        support = int(np.sum(targets_arr == class_idx))
        predicted = int(np.sum(preds_arr == class_idx))
        if support <= 0 and predicted <= 0:
            continue
        true_positive = int(np.sum((targets_arr == class_idx) & (preds_arr == class_idx)))
        precision = float(true_positive / predicted) if predicted > 0 else 0.0
        recall = float(true_positive / support) if support > 0 else 0.0
        denom = precision + recall
        scores.append((2.0 * precision * recall / denom) if denom > 0.0 else 0.0)
    return float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 0.0


def train_segment_decoder(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 1337,
) -> tuple[StreamSegmentDecoder, np.ndarray, np.ndarray]:
    if train_features.size == 0 or train_labels.size == 0:
        raise ValueError("Segment decoder training requires non-empty feature and label arrays.")

    rng = np.random.default_rng(seed)
    base_x = np.asarray(train_features, dtype=np.float32)
    base_y = np.asarray(train_labels, dtype=np.int64)
    hard_indices = [idx for idx, label in enumerate(base_y.tolist()) if SEGMENT_DECODER_INDEX_TO_LABEL[int(label)] in SEGMENT_DECODER_HARD_LABELS]
    train_x, train_y = _expand_hard_examples(base_x, base_y, hard_indices)
    if train_x.shape[0] > 1:
        order = rng.permutation(train_x.shape[0])
        train_x = train_x[order]
        train_y = train_y[order]

    feature_mean = np.mean(train_x, axis=0).astype(np.float32)
    feature_std = np.std(train_x, axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-5)

    x_norm = ((train_x - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=int(max(8, batch_size)), shuffle=True)

    class_counts = np.bincount(train_y, minlength=len(SEGMENT_DECODER_LABELS)).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = float(class_counts.sum()) / class_counts
    class_weights = class_weights / np.mean(class_weights)

    torch.manual_seed(seed)
    model = StreamSegmentDecoder(input_dim=SEGMENT_FEATURE_DIM, hidden_dim=int(hidden_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))

    valid_x = None
    valid_y = None
    if valid_features is not None and valid_labels is not None and np.asarray(valid_features).size and np.asarray(valid_labels).size:
        valid_x = torch.from_numpy(((np.asarray(valid_features, dtype=np.float32) - feature_mean[None, :]) / feature_std[None, :]).astype(np.float32))
        valid_y = torch.from_numpy(np.asarray(valid_labels, dtype=np.int64))

    best_state = None
    best_score = -np.inf
    for _epoch in range(int(max(1, epochs))):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if valid_x is not None and valid_y is not None:
                logits = model(valid_x)
                preds = torch.argmax(logits, dim=-1)
                score = _macro_f1_score(
                    preds.detach().cpu().numpy(),
                    valid_y.detach().cpu().numpy(),
                    len(SEGMENT_DECODER_LABELS),
                )
            else:
                logits = model(torch.from_numpy(x_norm))
                preds = torch.argmax(logits, dim=-1)
                score = _macro_f1_score(
                    preds.detach().cpu().numpy(),
                    train_y,
                    len(SEGMENT_DECODER_LABELS),
                )
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, feature_mean, feature_std
