"""Model factory."""

from __future__ import annotations

from typing import Dict

from torch import nn


def create_model(model_cfg: Dict[str, object], n_mels: int, num_commands: int) -> nn.Module:
    name = str(model_cfg["name"]).lower()
    if name == "kws12_verifier":
        from kws.models.kws12_verifier import KWS12VerifierNet

        return KWS12VerifierNet(
            n_mels=n_mels,
            num_commands=num_commands,
            conv_channels=int(model_cfg.get("conv_channels", 32)),
            num_blocks=int(model_cfg.get("num_blocks", 3)),
            attn_dim=int(model_cfg.get("attn_dim", 96)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.15)),
        )
    if name == "keyword_mamba":
        from kws.models.keyword_mamba import KeywordMambaNet

        return KeywordMambaNet(
            n_mels=n_mels,
            num_commands=num_commands,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            d_state=int(model_cfg.get("d_state", 16)),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand_factor=int(model_cfg.get("expand_factor", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "mhatt_crnn":
        from kws.models.mhatt_crnn import MHAttCRNNNet

        return MHAttCRNNNet(
            n_mels=n_mels,
            num_commands=num_commands,
            conv_channels=int(model_cfg.get("conv_channels", 32)),
            gru_hidden=int(model_cfg.get("gru_hidden", 128)),
            gru_layers=int(model_cfg.get("gru_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown model name: {name}")
