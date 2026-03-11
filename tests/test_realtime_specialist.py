from __future__ import annotations

import numpy as np
import torch

from kws.demo.realtime_specialist import (
    HARD_WORD_SPECIALIST_LABELS,
    HARD_WORD_SPECIALIST_LABEL_TO_INDEX,
    default_realtime_specialist_calibration,
    load_realtime_specialist_artifact,
    predict_realtime_specialist,
    save_realtime_specialist_artifact,
    save_realtime_specialist_calibration,
    summarize_realtime_specialist_predictions,
    train_realtime_specialist,
    waveform_to_specialist_feature,
)


def test_waveform_to_specialist_feature_shape() -> None:
    feature = waveform_to_specialist_feature(np.zeros((16000,), dtype=np.float32))
    assert feature.ndim == 3
    assert feature.shape[0] == 1
    assert feature.shape[1] == 64


def test_train_save_load_predict_realtime_specialist(tmp_path) -> None:
    train_waveforms = np.stack(
        [
            np.zeros((16000,), dtype=np.float32),
            np.full((16000,), 0.05, dtype=np.float32),
            np.full((16000,), 0.15, dtype=np.float32),
            np.full((16000,), 0.20, dtype=np.float32),
            np.full((16000,), 0.25, dtype=np.float32),
        ],
        axis=0,
    )
    train_labels = np.asarray(
        [
            HARD_WORD_SPECIALIST_LABEL_TO_INDEX["other"],
            HARD_WORD_SPECIALIST_LABEL_TO_INDEX["on"],
            HARD_WORD_SPECIALIST_LABEL_TO_INDEX["off"],
            HARD_WORD_SPECIALIST_LABEL_TO_INDEX["go"],
            HARD_WORD_SPECIALIST_LABEL_TO_INDEX["up"],
        ],
        dtype=np.int64,
    )
    model, mean, std = train_realtime_specialist(
        train_waveforms=train_waveforms,
        train_labels=train_labels,
        valid_waveforms=train_waveforms,
        valid_labels=train_labels,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
    )
    artifact = tmp_path / "realtime_specialist.pt"
    save_realtime_specialist_artifact(
        artifact,
        model=model,
        sample_rate=16000,
        target_samples=16000,
        n_mels=64,
        hidden_dim=64,
        feature_mean=mean,
        feature_std=std,
    )
    loaded = load_realtime_specialist_artifact(artifact, device=torch.device("cpu"))
    probs = predict_realtime_specialist(loaded, np.full((16000,), 0.22, dtype=np.float32))
    assert probs.shape[0] == len(HARD_WORD_SPECIALIST_LABELS)
    assert np.isclose(float(probs.sum()), 1.0, atol=1e-5)


def test_specialist_calibration_round_trip(tmp_path) -> None:
    calibration = default_realtime_specialist_calibration()
    target = tmp_path / "realtime_specialist_calibration.json"
    saved = save_realtime_specialist_calibration(target, calibration)
    assert saved.exists()


def test_summarize_realtime_specialist_predictions_reports_hard_word_metrics() -> None:
    probs = np.asarray(
        [
            [0.80, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.80, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.80],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    report = summarize_realtime_specialist_predictions(probs, labels)
    assert report["hard_word_min_precision"] == 1.0
    assert report["hard_word_min_recall"] == 1.0
    assert "on" in report["per_label"]
