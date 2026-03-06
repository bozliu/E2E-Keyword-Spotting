"""Training-only teacher utilities for lightweight KWS distillation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch import nn

from kws.constants import KWS12_LABELS, SAMPLE_RATE
from kws.data.audio import load_audio, pad_or_trim


@dataclass
class TeacherTargets:
    command_logits: torch.Tensor
    kws12_logits: torch.Tensor
    projected_embedding: torch.Tensor


class TeacherHeads(nn.Module):
    """Trainable heads on top of frozen SSL teacher features."""

    def __init__(
        self,
        *,
        feature_dim: int,
        student_dim: int,
        num_commands: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(128, student_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.command_head = nn.Linear(hidden_dim, num_commands)
        self.kws12_head = nn.Linear(hidden_dim, len(KWS12_LABELS))
        self.embedding_head = nn.Linear(hidden_dim, student_dim)

    def forward(self, pooled_features: torch.Tensor) -> TeacherTargets:
        x = self.backbone(self.norm(pooled_features))
        return TeacherTargets(
            command_logits=self.command_head(x),
            kws12_logits=self.kws12_head(x),
            projected_embedding=self.embedding_head(x),
        )


class WavLMFeatureCache:
    """Frozen WavLM feature extractor with on-disk pooled embedding cache."""

    def __init__(
        self,
        *,
        model_id: str,
        cache_dir: str | Path,
        device: torch.device,
        clip_samples: int,
        sample_rate: int = SAMPLE_RATE,
        encoder_factory: Optional[Callable[[str], nn.Module]] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.clip_samples = int(max(1, clip_samples))
        self.sample_rate = int(sample_rate)
        self.encoder_factory = encoder_factory

        self._encoder: nn.Module | None = None
        self._feature_dim: int | None = None
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            encoder = self._ensure_encoder()
            hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
            if hidden_size is None:
                raise AttributeError("Teacher encoder must expose config.hidden_size")
            self._feature_dim = int(hidden_size)
        return self._feature_dim

    def reset_stats(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0

    def load_features(self, paths: Iterable[str], *, device: torch.device) -> torch.Tensor:
        features = [self._load_or_compute_one(path) for path in paths]
        if not features:
            return torch.zeros(0, self.feature_dim, dtype=torch.float32, device=device)
        return torch.stack(features, dim=0).to(device)

    def _cache_path(self, audio_path: str) -> Path:
        resolved = str(Path(audio_path).expanduser().resolve())
        digest = hashlib.sha1(f"{self.model_id}|{resolved}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_or_compute_one(self, audio_path: str) -> torch.Tensor:
        cache_path = self._cache_path(audio_path)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            pooled = payload["pooled"] if isinstance(payload, dict) else payload
            self.cache_hits += 1
            return pooled.to(dtype=torch.float32, device=torch.device("cpu"))

        pooled = self._compute_one(audio_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"pooled": pooled.cpu(), "model_id": self.model_id, "path": str(audio_path)}, cache_path)
        self.cache_misses += 1
        return pooled

    def _compute_one(self, audio_path: str) -> torch.Tensor:
        encoder = self._ensure_encoder()
        waveform = load_audio(audio_path, sample_rate=self.sample_rate)
        waveform = pad_or_trim(waveform, target_samples=self.clip_samples)
        with torch.no_grad():
            input_values = waveform.unsqueeze(0).to(self.device)
            outputs = encoder(input_values=input_values)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise AttributeError("Teacher encoder output must expose last_hidden_state")
            pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().to(dtype=torch.float32)
        return pooled

    def _ensure_encoder(self) -> nn.Module:
        if self._encoder is not None:
            return self._encoder

        if self.encoder_factory is not None:
            encoder = self.encoder_factory(self.model_id)
        else:
            try:
                from transformers import AutoModel
            except Exception as exc:  # pragma: no cover - dependency error
                raise ImportError(
                    "transformers is required for teacher distillation. "
                    "Install it or disable training.teacher.enabled."
                ) from exc
            encoder = AutoModel.from_pretrained(self.model_id)

        encoder = encoder.to(self.device)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
        self._encoder = encoder
        return encoder
