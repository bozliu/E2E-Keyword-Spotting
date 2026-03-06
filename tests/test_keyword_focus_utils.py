from __future__ import annotations

import numpy as np
import torch

from kws.constants import COMMAND31_TO_INDEX
from kws.losses.confusion import confusion_aware_embedding_loss
from kws.utils.keyword_focus import build_keyword_focus_report, fit_keyword_calibration


def test_build_keyword_focus_report_mines_weak_keywords() -> None:
    preds = np.array(
        [
            COMMAND31_TO_INDEX["off"],
            COMMAND31_TO_INDEX["up"],
            COMMAND31_TO_INDEX["no"],
            COMMAND31_TO_INDEX["go"],
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
    assert "no" in report["weak_keywords"]
    assert report["per_keyword"]["off"]["top_confusions"][0]["label"] == "up"
    assert report["focus_pair_confusions"]["on"]["support"] >= 0


def test_fit_keyword_calibration_gives_weak_keywords_more_help() -> None:
    command_probs = np.zeros((6, 31), dtype=np.float32)
    off = COMMAND31_TO_INDEX["off"]
    yes = COMMAND31_TO_INDEX["yes"]
    command_probs[:, yes] = 0.1
    command_probs[:, off] = np.array([0.32, 0.28, 0.18, 0.12, 0.15, 0.10], dtype=np.float32)
    wake_probs = np.array([0.82, 0.76, 0.55, 0.40, 0.35, 0.30], dtype=np.float32)
    targets = np.array([
        off,
        off,
        COMMAND31_TO_INDEX["up"],
        COMMAND31_TO_INDEX["up"],
        COMMAND31_TO_INDEX["yes"],
        COMMAND31_TO_INDEX["yes"],
    ])
    focus = {
        "weak_keywords": ["off"],
        "focus_keywords": ["left", "on", "down"],
        "focus_pairs": {"left": ["right", "bed", "yes"], "on": ["off", "five", "one"], "down": ["go", "no"]},
        "confusable_groups": {"off": ["up", "on"]},
    }

    calibration = fit_keyword_calibration(command_probs, wake_probs, targets, focus=focus)
    assert calibration["keywords"]["off"]["prototype_bonus_max"] >= 0.08
    assert calibration["keywords"]["off"]["vote_min_count"] >= 3
    assert calibration["keywords"]["off"]["min_margin"] >= 0.12
    assert calibration["keywords"]["off"]["highlight_hold_ms"] >= 280
    assert "confusable_groups" in calibration
    assert calibration["focus_keywords"] == ["down", "left", "on"]
    assert calibration["keywords"]["on"]["min_margin"] >= 0.16
    assert calibration["keywords"]["down"]["vote_min_count"] >= 3


def test_confusion_aware_embedding_loss_targets_configured_pairs() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.98, 0.02],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([
        COMMAND31_TO_INDEX["off"],
        COMMAND31_TO_INDEX["off"],
        COMMAND31_TO_INDEX["up"],
    ])
    loss = confusion_aware_embedding_loss(
        embeddings,
        labels,
        confusion_groups={"off": ["up"]},
        margin=0.2,
    )
    assert float(loss.item()) > 0.0
