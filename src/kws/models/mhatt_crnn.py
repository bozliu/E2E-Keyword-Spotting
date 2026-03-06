"""Modernized multi-head attention CRNN baseline."""

from __future__ import annotations

import torch
from torch import nn

from kws.models.common import DualTaskOutput


class MHAttCRNNNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        num_commands: int,
        conv_channels: int = 32,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )

        reduced_mels = max(1, n_mels // 2)
        self.rnn = nn.GRU(
            input_size=conv_channels * reduced_mels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(embed_dim=gru_hidden * 2, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(gru_hidden * 2)
        self.embedding = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.command_head = nn.Linear(gru_hidden * 2, num_commands)
        self.wake_head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, features: torch.Tensor) -> DualTaskOutput:
        # features: [B, n_mels, T]
        x = features.transpose(1, 2)  # [B, T, n_mels]
        x = x.unsqueeze(1)  # [B, 1, T, n_mels]
        x = self.front(x)
        # [B, C, T, F]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.rnn(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(attn_out + x)

        pooled = x.mean(dim=1)
        emb = self.embedding(pooled)
        command_logits = self.command_head(emb)
        wake_logits = self.wake_head(emb).squeeze(-1)
        return DualTaskOutput(command_logits=command_logits, wake_logits=wake_logits, embedding=emb)
