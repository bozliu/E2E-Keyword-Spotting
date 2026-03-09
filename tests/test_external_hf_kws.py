from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kws.external import hf_kws


class _DummyExtractor:
    def __call__(self, waveforms, sampling_rate, return_tensors, padding):  # noqa: ARG002
        batch = len(waveforms)
        return {"input_values": torch.zeros(batch, 16000, dtype=torch.float32)}


class _DummyModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, id2label: dict[int, str]) -> None:
        super().__init__()
        self._logits = logits
        self.config = type("Cfg", (), {"id2label": id2label})()

    def to(self, device):  # noqa: D401
        self._device = device
        return self

    def eval(self):  # noqa: D401
        return self

    def forward(self, **kwargs):  # noqa: D401, ARG002
        return type("Out", (), {"logits": self._logits})()


def test_predict_kws12_from_waveforms_maps_direct_superb_labels(monkeypatch) -> None:
    logits = torch.tensor([[6.0, 1.0, 0.0, 0.0]])
    labels = {0: "yes", 1: "_unknown_", 2: "_silence_", 3: "go"}
    monkeypatch.setattr(
        hf_kws,
        "_load_hf_components",
        lambda model_id, device_type: (_DummyExtractor(), _DummyModel(logits, labels), torch.device("cpu")),
    )

    result = hf_kws.predict_kws12_from_waveforms(
        [np.zeros((16000,), dtype=np.float32)],
        model_id=hf_kws.DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
        device="cpu",
    )

    assert result.top_labels == ("yes",)
    assert result.probs.shape == (1, 12)
    assert float(result.probs[0, 2]) > 0.9


def test_predict_kws12_from_waveforms_maps_ast_non_targets_to_unknown(monkeypatch) -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0]])
    labels = {0: "yes", 1: "bed", 2: "up"}
    monkeypatch.setattr(
        hf_kws,
        "_load_hf_components",
        lambda model_id, device_type: (_DummyExtractor(), _DummyModel(logits, labels), torch.device("cpu")),
    )

    result = hf_kws.predict_kws12_from_waveforms(
        [np.zeros((16000,), dtype=np.float32)],
        model_id=hf_kws.DEFAULT_EXTERNAL_AUX_MODEL_ID,
        device="cpu",
    )

    assert result.top_labels == ("unknown",)
    assert float(result.probs[0, 1]) > 0.9


def test_fit_external_verifier_calibration_includes_backend_fields() -> None:
    probs = np.array(
        [
            [0.01, 0.04, 0.92, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.90, 0.03, 0.03, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([2, 1], dtype=np.int64)
    calibration = hf_kws.fit_external_verifier_calibration(
        probs=probs,
        targets_kws12=targets,
        model_id="demo/model",
        backend="external",
        fit_split="valid",
    )

    assert calibration["backend"] == "external"
    assert calibration["model_id"] == "demo/model"
    assert calibration["fit_split"] == "valid"
    assert "yes" in calibration["per_label"]


def test_predict_kws12_from_waveforms_supports_ast_superb_ensemble(monkeypatch) -> None:
    superb_logits = torch.tensor([[2.5, 1.0, 0.0, 2.0]], dtype=torch.float32)
    ast_logits = torch.tensor([[1.0, 0.5, 7.5]], dtype=torch.float32)
    superb_labels = {0: "_silence_", 1: "_unknown_", 2: "yes", 3: "go"}
    ast_labels = {0: "yes", 1: "bed", 2: "go"}

    def _fake_load(model_id, device_type):  # noqa: ARG001
        if model_id == hf_kws.DEFAULT_EXTERNAL_VERIFIER_MODEL_ID:
            return _DummyExtractor(), _DummyModel(superb_logits, superb_labels), torch.device("cpu")
        if model_id == hf_kws.DEFAULT_EXTERNAL_AUX_MODEL_ID:
            return _DummyExtractor(), _DummyModel(ast_logits, ast_labels), torch.device("cpu")
        raise AssertionError(f"unexpected model_id: {model_id}")

    monkeypatch.setattr(hf_kws, "_load_hf_components", _fake_load)

    result = hf_kws.predict_kws12_from_waveforms(
        [np.zeros((16000,), dtype=np.float32)],
        model_id=hf_kws.ENSEMBLE_AST_SUPERB_MODEL_ID,
        device="cpu",
    )

    assert result.model_id == hf_kws.ENSEMBLE_AST_SUPERB_MODEL_ID
    assert result.top_labels == ("go",)
    assert float(result.probs[0, 0]) > 0.2
    assert float(result.probs[0, 11]) > 0.4


def test_external_logit_cache_reuses_disk_cache(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_predict(paths, *, model_id, device, sample_rate, clip_samples):  # noqa: ARG001
        calls["count"] += 1
        probs = np.tile(np.eye(12, dtype=np.float32)[2], (len(paths), 1))
        return hf_kws.ExternalKWSBatchResult(
            model_id=model_id,
            runtime_device="cpu",
            probs=probs,
            top_indices=np.full((len(paths),), 2, dtype=np.int64),
            top_labels=("yes",) * len(paths),
            margins=np.full((len(paths),), 0.9, dtype=np.float32),
        )

    monkeypatch.setattr(hf_kws, "predict_kws12_from_paths", _fake_predict)
    cache = hf_kws.ExternalKWSLogitCache(
        primary_model_id="demo/primary",
        aux_model_id="demo/aux",
        cache_dir=tmp_path / "cache",
        device="cpu",
        clip_samples=16000,
        agreement_weight=0.25,
    )

    first = cache.load_targets(["a.wav", "b.wav"], device=torch.device("cpu"))
    second = cache.load_targets(["a.wav", "b.wav"], device=torch.device("cpu"))

    assert first.probs.shape == (2, 12)
    assert torch.allclose(first.probs, second.probs)
    assert calls["count"] == 2
    assert cache.cache_hits >= 2
