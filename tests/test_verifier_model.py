from __future__ import annotations

import torch

from kws.models import create_model
from kws.models.kws12_verifier import LearnablePCEN
from kws.train.verifier import VERIFIER_LABELS


def test_create_verifier_model_forward_shapes() -> None:
    model = create_model(
        {
            "name": "kws12_verifier",
            "conv_channels": 24,
            "num_blocks": 2,
            "attn_dim": 64,
            "num_heads": 4,
            "dropout": 0.1,
        },
        n_mels=80,
        num_commands=len(VERIFIER_LABELS),
    )

    features = torch.rand(3, 80, 126)
    out = model(features)

    assert out.logits.shape == (3, len(VERIFIER_LABELS))
    assert out.embedding.shape == (3, 64)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.embedding).all()


def test_learnable_pcen_backpropagates() -> None:
    pcen = LearnablePCEN(80)
    features = torch.rand(2, 80, 40, requires_grad=True)
    out = pcen(features)
    loss = out.square().mean()
    loss.backward()

    assert features.grad is not None
    assert pcen.alpha.grad is not None
    assert pcen.delta.grad is not None
    assert pcen.log_r.grad is not None
