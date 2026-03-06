from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from kws.train.teacher import TeacherHeads, WavLMFeatureCache


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.config = type("Cfg", (), {"hidden_size": 4})()

    def forward(self, input_values: torch.Tensor):  # noqa: D401
        self.calls += 1
        hidden = input_values.unsqueeze(-1).repeat(1, 1, 4)
        return type("Out", (), {"last_hidden_state": hidden})()


def test_teacher_cache_hits_after_first_encode(tmp_path: Path) -> None:
    wav_path = tmp_path / "a.wav"
    waveform = torch.zeros(1, 16000, dtype=torch.float32)
    torchaudio.save(str(wav_path), waveform, 16000)

    fake_encoder = _FakeEncoder()
    cache = WavLMFeatureCache(
        model_id="fake/wavlm",
        cache_dir=tmp_path / "cache",
        device=torch.device("cpu"),
        clip_samples=16000,
        encoder_factory=lambda _model_id: fake_encoder,
    )

    first = cache.load_features([str(wav_path)], device=torch.device("cpu"))
    second = cache.load_features([str(wav_path)], device=torch.device("cpu"))

    assert first.shape == (1, 4)
    assert torch.allclose(first, second)
    assert fake_encoder.calls == 1
    assert cache.cache_hits >= 1


def test_teacher_heads_shapes() -> None:
    heads = TeacherHeads(feature_dim=4, student_dim=8, num_commands=31, dropout=0.0)
    pooled = torch.randn(3, 4)
    targets = heads(pooled)
    assert targets.command_logits.shape == (3, 31)
    assert targets.kws12_logits.shape == (3, 12)
    assert targets.projected_embedding.shape == (3, 8)
