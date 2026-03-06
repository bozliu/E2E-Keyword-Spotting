from __future__ import annotations

import torch

from kws.data.dataset import Batch
from kws.models import create_model
from kws.train.engine import run_epoch
from kws.train.teacher import TeacherHeads


def test_run_epoch_smoke() -> None:
    model = create_model(
        {"name": "mhatt_crnn", "conv_channels": 16, "gru_hidden": 64, "gru_layers": 1, "num_heads": 4, "dropout": 0.0},
        n_mels=64,
        num_commands=31,
    )

    features = torch.randn(8, 64, 126)
    command_labels = torch.randint(0, 31, (8,))
    wake_labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    batch = Batch(features=features, command_labels=command_labels, wake_labels=wake_labels, paths=["a"] * 8, sources=["x"] * 8)
    loader = [batch, batch]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = run_epoch(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        lambda_command=1.0,
        lambda_kws12=0.3,
        lambda_wake=1.0,
        lambda_aux=0.1,
        lambda_confusion=0.1,
        aux_margin=0.2,
        audio_seconds=1.0,
    )

    assert result.loss >= 0.0
    assert "kws12_acc" in result.metrics
    assert "kws12_target_precision" in result.metrics
    assert "bottom3_keyword_recall" in result.metrics


def test_run_epoch_handles_all_ignore_command_batch() -> None:
    model = create_model(
        {"name": "mhatt_crnn", "conv_channels": 8, "gru_hidden": 16, "gru_layers": 1, "num_heads": 2, "dropout": 0.0},
        n_mels=64,
        num_commands=31,
    )

    features = torch.randn(4, 64, 126)
    command_labels = torch.full((4,), -100, dtype=torch.long)
    wake_labels = torch.randint(0, 2, (4,), dtype=torch.float32)
    batch = Batch(features=features, command_labels=command_labels, wake_labels=wake_labels, paths=["a"] * 4, sources=["x"] * 4)

    result = run_epoch(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
        optimizer=None,
        lambda_command=1.0,
        lambda_kws12=0.3,
        lambda_wake=1.0,
        lambda_aux=0.1,
        lambda_confusion=0.1,
        aux_margin=0.2,
        audio_seconds=1.0,
    )

    assert result.loss >= 0.0


def test_run_epoch_supports_teacher_distillation() -> None:
    class _DummyTeacherCache:
        def load_features(self, paths, *, device):  # noqa: ARG002
            return torch.ones(len(paths), 4, dtype=torch.float32, device=device)

    model = create_model(
        {"name": "mhatt_crnn", "conv_channels": 8, "gru_hidden": 16, "gru_layers": 1, "num_heads": 2, "dropout": 0.0},
        n_mels=64,
        num_commands=31,
    )
    teacher_heads = TeacherHeads(feature_dim=4, student_dim=model.command_head.in_features, num_commands=31, dropout=0.0)

    features = torch.randn(4, 64, 126)
    command_labels = torch.randint(0, 31, (4,))
    wake_labels = torch.randint(0, 2, (4,), dtype=torch.float32)
    batch = Batch(features=features, command_labels=command_labels, wake_labels=wake_labels, paths=["a"] * 4, sources=["x"] * 4)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(teacher_heads.parameters()), lr=1e-3)

    result = run_epoch(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
        optimizer=optimizer,
        lambda_command=1.0,
        lambda_kws12=0.3,
        lambda_wake=1.0,
        lambda_aux=0.1,
        lambda_confusion=0.1,
        aux_margin=0.2,
        audio_seconds=1.0,
        teacher_cache=_DummyTeacherCache(),
        teacher_heads=teacher_heads,
        lambda_distill_logits=0.3,
        lambda_distill_embed=0.1,
    )

    assert result.loss >= 0.0
