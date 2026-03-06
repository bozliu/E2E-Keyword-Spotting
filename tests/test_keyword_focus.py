from __future__ import annotations

import numpy as np

from kws.constants import COMMAND31_TO_INDEX
from kws.utils.keyword_focus import build_keyword_focus_report, fit_keyword_calibration


def test_keyword_focus_report_surfaces_weak_keywords() -> None:
    preds = np.array(
        [
            COMMAND31_TO_INDEX["up"],
            COMMAND31_TO_INDEX["up"],
            COMMAND31_TO_INDEX["go"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["yes"],
        ]
    )
    targets = np.array(
        [
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["yes"],
        ]
    )

    report = build_keyword_focus_report(
        preds,
        targets,
        top_k=2,
        focus_keywords=["left", "on", "down"],
        focus_pairs={"left": ["right", "bed", "yes"], "on": ["off", "five", "one"], "down": ["go", "no"]},
    )
    assert "off" in report["weak_keywords"]
    assert report["per_keyword"]["off"]["top_confusions"][0]["label"] == "up"
    assert "bottom3_keyword_recall" in report
    assert report["focus_keywords"] == ["left", "on", "down"]
    assert "focus_keyword_recall_mean" in report
    assert "focus_pair_confusion_rate" in report


def test_keyword_calibration_boosts_weak_keywords() -> None:
    num_commands = 31
    probs = np.zeros((6, num_commands), dtype=np.float32)
    probs[:, COMMAND31_TO_INDEX["off"]] = np.array([0.24, 0.31, 0.62, 0.18, 0.15, 0.20], dtype=np.float32)
    probs[:, COMMAND31_TO_INDEX["yes"]] = np.array([0.10, 0.10, 0.10, 0.70, 0.65, 0.80], dtype=np.float32)
    probs[:, COMMAND31_TO_INDEX["up"]] = np.array([0.55, 0.52, 0.12, 0.05, 0.10, 0.01], dtype=np.float32)
    wake = np.full((6,), 0.9, dtype=np.float32)
    targets = np.array(
        [
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["yes"],
            COMMAND31_TO_INDEX["yes"],
        ]
    )

    calibration = fit_keyword_calibration(
        probs,
        wake,
        targets,
        focus={
            "weak_keywords": ["off"],
            "focus_keywords": ["left", "on", "down"],
            "focus_pairs": {"left": ["right", "bed", "yes"], "on": ["off", "five", "one"], "down": ["go", "no"]},
            "confusable_groups": {"off": ["up", "on"]},
        },
    )

    assert calibration["keywords"]["off"]["prototype_bonus_max"] > calibration["keywords"]["yes"]["prototype_bonus_max"]
    assert calibration["keywords"]["off"]["vote_min_count"] >= 2
    assert calibration["keywords"]["off"]["min_margin"] > calibration["keywords"]["yes"]["min_margin"]
    assert calibration["keywords"]["off"]["highlight_hold_ms"] >= calibration["keywords"]["yes"]["highlight_hold_ms"]
    assert calibration["keywords"]["on"]["min_margin"] >= 0.16
    assert calibration["keywords"]["on"]["vote_window"] >= 6
    assert calibration["keywords"]["on"]["vote_min_count"] >= 4
    assert calibration["keywords"]["left"]["min_margin"] >= 0.12
    assert calibration["keywords"]["down"]["min_margin"] >= 0.14


def test_keyword_calibration_ignores_non_label_confusions() -> None:
    num_commands = 31
    probs = np.zeros((4, num_commands), dtype=np.float32)
    probs[:, COMMAND31_TO_INDEX["left"]] = 0.8
    probs[:, COMMAND31_TO_INDEX["yes"]] = 0.1
    probs[:, COMMAND31_TO_INDEX["right"]] = 0.1
    wake = np.full((4,), 0.9, dtype=np.float32)
    targets = np.array([COMMAND31_TO_INDEX["left"]] * 4)

    calibration = fit_keyword_calibration(
        probs,
        wake,
        targets,
        focus={
            "focus_keywords": ["left"],
            "focus_pairs": {"left": ["right", "bed", True, "yes"]},
            "confusable_groups": {"left": ["right", "bed", True, "yes"]},
        },
    )

    assert calibration["focus_pairs"]["left"] == ["right", "bed", "yes"]
    assert calibration["confusable_groups"]["left"] == ["right", "bed", "yes"]
