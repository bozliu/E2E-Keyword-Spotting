from __future__ import annotations

import pytest
import torch

from kws.models import create_model


@pytest.mark.parametrize("name", ["keyword_mamba", "mhatt_crnn"])
def test_model_forward_shapes(name: str) -> None:
    cfg = {"name": name}
    if name == "keyword_mamba":
        cfg.update({"d_model": 64, "n_layers": 2, "d_state": 8, "d_conv": 4, "expand_factor": 2, "dropout": 0.0})
    else:
        cfg.update({"conv_channels": 16, "gru_hidden": 64, "gru_layers": 1, "num_heads": 4, "dropout": 0.0})

    model = create_model(cfg, n_mels=64, num_commands=31)
    x = torch.randn(4, 64, 126)
    out = model(x)
    assert out.command_logits.shape == (4, 31)
    assert out.wake_logits.shape == (4,)
    assert out.embedding.shape[0] == 4
