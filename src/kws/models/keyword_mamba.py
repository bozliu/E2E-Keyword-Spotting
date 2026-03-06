"""KeywordMamba dual-head model."""

from __future__ import annotations

import torch
from torch import nn

from kws.models.common import DualTaskOutput

try:
    from mambapy.mamba import Mamba, MambaConfig
except Exception as exc:  # pragma: no cover
    Mamba = None
    MambaConfig = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class KeywordMambaNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_commands: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if Mamba is None or MambaConfig is None:
            raise ImportError(
                "mambapy is required for KeywordMambaNet. "
                f"Original import error: {_MAMBA_IMPORT_ERROR}"
            )

        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        cfg = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            pscan=True,
            use_cuda=False,
        )
        self.encoder = Mamba(cfg)
        self.norm = nn.LayerNorm(d_model)

        self.embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.command_head = nn.Linear(d_model, num_commands)
        self.wake_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> DualTaskOutput:
        # features: [B, n_mels, T]
        x = self.stem(features)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        emb = self.embedding(pooled)
        command_logits = self.command_head(emb)
        wake_logits = self.wake_head(emb).squeeze(-1)
        return DualTaskOutput(command_logits=command_logits, wake_logits=wake_logits, embedding=emb)
