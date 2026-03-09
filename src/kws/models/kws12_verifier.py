"""Lightweight KWS12 verifier with learnable PCEN and temporal attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kws.models.common import VerifierOutput


class LearnablePCEN(nn.Module):
    """Per-channel energy normalization over mel features."""

    def __init__(
        self,
        channels: int,
        *,
        eps: float = 1e-6,
        init_s: float = 0.04,
        init_alpha: float = 0.96,
        init_delta: float = 2.0,
        init_r: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = int(max(1, channels))
        self.eps = float(max(1e-8, eps))
        self.logit_s = nn.Parameter(torch.full((self.channels,), _inverse_sigmoid(init_s), dtype=torch.float32))
        self.alpha = nn.Parameter(torch.full((self.channels,), float(init_alpha), dtype=torch.float32))
        self.delta = nn.Parameter(torch.full((self.channels,), float(init_delta), dtype=torch.float32))
        self.log_r = nn.Parameter(torch.log(torch.full((self.channels,), float(init_r), dtype=torch.float32)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [B, C, T] PCEN input, got shape={tuple(features.shape)}")
        x = features.clamp_min(0.0)
        s = torch.sigmoid(self.logit_s).view(1, -1, 1)
        alpha = self.alpha.clamp(0.1, 1.0).view(1, -1, 1)
        delta = self.delta.clamp_min(0.1).view(1, -1, 1)
        r = torch.exp(self.log_r).clamp(0.05, 1.5).view(1, -1, 1)

        smoother = []
        prev = x[:, :, :1]
        smoother.append(prev)
        for t in range(1, x.size(-1)):
            cur = ((1.0 - s) * prev) + (s * x[:, :, t : t + 1])
            smoother.append(cur)
            prev = cur
        m = torch.cat(smoother, dim=-1)
        pcen = torch.pow(x / torch.pow(self.eps + m, alpha) + delta, r) - torch.pow(delta, r)
        return pcen


def _inverse_sigmoid(value: float) -> float:
    value = min(1.0 - 1e-4, max(1e-4, float(value)))
    return float(torch.logit(torch.tensor(value, dtype=torch.float32)).item())


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.project = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.expand(x)
        x = F.gelu(x)
        x = self.project(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class KWS12VerifierNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_commands: int,
        *,
        conv_channels: int = 32,
        num_blocks: int = 3,
        attn_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_commands < 2:
            raise ValueError("Verifier requires at least 2 output classes.")
        self.pcen = LearnablePCEN(n_mels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(conv_channels, dropout=dropout) for _ in range(max(1, num_blocks))]
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_proj = nn.Linear(conv_channels, attn_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(attn_dim)
        self.embedding = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.logits = nn.Linear(attn_dim, num_commands)

    def forward(self, features: torch.Tensor) -> VerifierOutput:
        if features.ndim != 3:
            raise ValueError(f"Expected verifier input [B, n_mels, T], got shape={tuple(features.shape)}")
        x = self.pcen(features)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.freq_pool(x).squeeze(2).transpose(1, 2)
        x = self.temporal_proj(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        pooled = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
        embedding = self.embedding(pooled)
        logits = self.logits(embedding)
        return VerifierOutput(logits=logits, embedding=embedding)
