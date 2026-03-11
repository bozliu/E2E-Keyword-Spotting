from __future__ import annotations

import numpy as np
import torch

from kws.demo.segment_decoder import (
    SEGMENT_DECODER_LABEL_TO_INDEX,
    SEGMENT_FEATURE_DIM,
    SEGMENT_TARGET_LABELS,
    SegmentFeatureStats,
    load_segment_decoder_artifact,
    predict_segment_decoder,
    save_segment_decoder_artifact,
    train_segment_decoder,
)


def test_segment_feature_stats_shape() -> None:
    stats = SegmentFeatureStats(thresholds=np.full((len(SEGMENT_TARGET_LABELS),), 0.18, dtype=np.float32))
    stats.update(now=0.10, target_probs=np.linspace(0.05, 0.50, num=len(SEGMENT_TARGET_LABELS), dtype=np.float32), wake_prob=0.8)
    stats.update(now=0.20, target_probs=np.linspace(0.10, 0.60, num=len(SEGMENT_TARGET_LABELS), dtype=np.float32), wake_prob=0.9)
    vector = stats.as_feature_vector()
    assert vector.shape == (SEGMENT_FEATURE_DIM,)
    assert float(vector.max()) > 0.0


def test_segment_decoder_train_save_load_smoke(tmp_path) -> None:
    yes_idx = SEGMENT_DECODER_LABEL_TO_INDEX["yes"]
    on_idx = SEGMENT_DECODER_LABEL_TO_INDEX["on"]
    unknown_idx = SEGMENT_DECODER_LABEL_TO_INDEX["unknown"]
    train_features = np.stack(
        [
            np.full((SEGMENT_FEATURE_DIM,), 0.1, dtype=np.float32),
            np.full((SEGMENT_FEATURE_DIM,), 0.3, dtype=np.float32),
            np.full((SEGMENT_FEATURE_DIM,), 0.8, dtype=np.float32),
            np.full((SEGMENT_FEATURE_DIM,), 0.9, dtype=np.float32),
        ],
        axis=0,
    )
    train_labels = np.asarray([unknown_idx, yes_idx, on_idx, on_idx], dtype=np.int64)
    model, mean, std = train_segment_decoder(
        train_features=train_features,
        train_labels=train_labels,
        valid_features=train_features,
        valid_labels=train_labels,
        epochs=4,
        batch_size=2,
    )
    artifact = tmp_path / "segment_decoder.pt"
    save_segment_decoder_artifact(
        artifact,
        model=model,
        feature_mean=mean,
        feature_std=std,
        hidden_dim=64,
    )
    loaded = load_segment_decoder_artifact(artifact, device=torch.device("cpu"))
    probs = predict_segment_decoder(loaded, np.full((SEGMENT_FEATURE_DIM,), 0.85, dtype=np.float32))
    assert probs.shape[0] == len(loaded.labels)
    assert np.isclose(float(probs.sum()), 1.0, atol=1e-5)
