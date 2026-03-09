from __future__ import annotations

from pathlib import Path

import torch

from kws.config import load_yaml
from kws.constants import COMMAND31_TO_INDEX, IGNORE_INDEX
from kws.train.verifier import (
    VERIFIER_LABELS,
    VERIFIER_REJECT_LABEL,
    VERIFIER_TO_INDEX,
    VerifierTeacherHeads,
    build_verifier_targets,
    load_verifier_checkpoint,
    verifier_cross_entropy,
    verifier_distillation_loss,
    verifier_margin_loss,
)


def test_build_verifier_targets_maps_command31_and_rejects() -> None:
    command_targets = torch.tensor(
        [
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["zero"],
            COMMAND31_TO_INDEX["silence"],
            IGNORE_INDEX,
        ],
        dtype=torch.long,
    )
    reject_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

    targets = build_verifier_targets(command_targets, reject_mask=reject_mask)

    assert targets.labels.tolist() == [
        VERIFIER_TO_INDEX["yes"],
        VERIFIER_TO_INDEX[VERIFIER_REJECT_LABEL],
        VERIFIER_TO_INDEX["silence"],
        IGNORE_INDEX,
    ]
    assert targets.reject_mask.tolist() == [False, True, False, False]


def test_verifier_losses_are_finite() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.1, -0.2],
            [0.1, 2.3, 1.7],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    teacher_logits = torch.tensor(
        [
            [2.5, 0.3, -0.4],
            [0.2, 2.0, 1.4],
        ],
        dtype=torch.float32,
    )

    ce = verifier_cross_entropy(logits, targets, label_smoothing=0.02)
    margin = verifier_margin_loss(logits, targets, margin=0.1)
    distill = verifier_distillation_loss(logits, teacher_logits)

    assert ce.item() > 0.0
    assert margin.item() >= 0.0
    assert distill.item() >= 0.0


def test_verifier_teacher_heads_shape() -> None:
    heads = VerifierTeacherHeads(feature_dim=768, student_dim=96)
    pooled = torch.randn(4, 768)
    out = heads(pooled)

    assert out.verifier_logits.shape == (4, len(VERIFIER_LABELS))
    assert out.projected_embedding.shape == (4, 96)


def test_verifier_config_loads_and_factory_name_is_stable() -> None:
    bundle = load_yaml("configs/demo_mhatt_small_focus_verifier.yaml")

    assert bundle.raw["model"]["name"] == "kws12_verifier"
    assert bundle.raw["run_name"] == "demo_mhatt_small_focus_verifier"
    assert bundle.raw["training"]["verifier"]["labels"][-1] == VERIFIER_REJECT_LABEL


def test_load_verifier_checkpoint_round_trip(tmp_path: Path) -> None:
    payload = {
        "model_state": {"dummy": torch.tensor([1.0])},
        "config": {"model": {"name": "kws12_verifier"}},
    }
    path = tmp_path / "best_kws12_verifier.pt"
    torch.save(payload, path)

    loaded = load_verifier_checkpoint(path)

    assert loaded["config"]["model"]["name"] == "kws12_verifier"
