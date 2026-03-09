from __future__ import annotations

from pathlib import Path

import torch

from kws.demo.verifier_runtime import LoadedVerifier, resolve_calibration_thresholds, verify_keyword


class _DummyVerifierModel(torch.nn.Module):
    def forward(self, features: torch.Tensor):  # noqa: ARG002
        logits = torch.tensor([[0.1, 0.2, 4.0, 0.0]], dtype=torch.float32)
        return type("VerifierOut", (), {"logits": logits})()


def test_resolve_calibration_thresholds_prefers_per_label() -> None:
    calibration = {
        "default": {"min_accept_prob": 0.7, "min_margin": 0.1},
        "per_label": {"yes": {"min_accept_prob": 0.8, "min_margin": 0.2}},
    }

    yes_prob, yes_margin = resolve_calibration_thresholds(calibration, "yes")
    no_prob, no_margin = resolve_calibration_thresholds(calibration, "no")

    assert yes_prob == 0.8
    assert yes_margin == 0.2
    assert no_prob == 0.7
    assert no_margin == 0.1


def test_verify_keyword_uses_nested_per_label_thresholds() -> None:
    verifier = LoadedVerifier(
        checkpoint_path=Path("/tmp/best_kws12_verifier.pt"),
        runtime_device=torch.device("cpu"),
        model=_DummyVerifierModel(),
        labels=("silence", "unknown", "yes", "reject"),
        calibration={
            "default": {"min_accept_prob": 0.5, "min_margin": 0.1},
            "per_label": {"yes": {"min_accept_prob": 0.9, "min_margin": 0.3}},
        },
        min_accept_prob=0.5,
        min_margin=0.1,
    )

    decision = verify_keyword(verifier, torch.zeros(1, 80, 10), candidate_label="yes")

    assert decision is not None
    assert decision.accepted is True
    assert decision.top_label == "yes"
