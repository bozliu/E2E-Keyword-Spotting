from __future__ import annotations

import numpy as np

from kws.constants import COMMAND31_TO_INDEX, IGNORE_INDEX
from kws.train.metrics import (
    compute_command_metrics,
    compute_keyword_breakdown,
    compute_kws12_accuracy,
    compute_kws12_breakdown,
    compute_per_class_kws12,
    compute_wake_metrics,
)


def test_metrics_outputs() -> None:
    preds = np.array([0, 1, 2, 3, 4])
    targets = np.array([0, 1, 2, 2, IGNORE_INDEX])

    cm = compute_command_metrics(preds, targets)
    assert 0.0 <= cm["command_acc"] <= 1.0

    kws12 = compute_kws12_accuracy(preds, targets)
    assert 0.0 <= kws12 <= 1.0
    breakdown = compute_kws12_breakdown(preds, targets)
    assert "kws12_target_precision" in breakdown
    assert "kws12_unknown_to_target_errors" in breakdown
    assert "per_class_kws12" in breakdown
    assert "min_kws12_precision" in breakdown
    assert "min_kws12_recall" in breakdown
    keyword_breakdown = compute_keyword_breakdown(preds, targets)
    assert "per_keyword" in keyword_breakdown
    assert "bottom3_keyword_recall" in keyword_breakdown

    wake_scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    wake_targets = np.array([0, 0, 1, 1, 1])
    wake = compute_wake_metrics(wake_scores, wake_targets)
    assert "wake_roc_auc" in wake
    assert "wake_frr_at_1fa_per_hour" in wake


def test_per_class_kws12_tracks_precision_recall_and_confusions() -> None:
    preds = np.array(
        [
            COMMAND31_TO_INDEX["silence"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["left"],
            COMMAND31_TO_INDEX["right"],
            COMMAND31_TO_INDEX["right"],
            COMMAND31_TO_INDEX["up"],
        ]
    )
    targets = np.array(
        [
            COMMAND31_TO_INDEX["silence"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["left"],
            COMMAND31_TO_INDEX["left"],
            COMMAND31_TO_INDEX["right"],
            IGNORE_INDEX,
        ]
    )

    per_class = compute_per_class_kws12(preds, targets)

    assert set(per_class) == {
        "silence",
        "unknown",
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    }
    assert per_class["yes"]["precision"] == 0.5
    assert per_class["yes"]["recall"] == 1.0
    assert per_class["no"]["precision"] == 1.0
    assert np.isclose(per_class["no"]["recall"], 2.0 / 3.0)
    assert per_class["left"]["top_confusions"] == [{"label": "right", "count": 1}]

    breakdown = compute_kws12_breakdown(preds, targets)
    assert breakdown["per_class_kws12"] == per_class
    assert breakdown["min_kws12_precision"] == 0.0
    assert breakdown["min_kws12_recall"] == 0.0
