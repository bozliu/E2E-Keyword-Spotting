from __future__ import annotations

import numpy as np

from kws.constants import COMMAND31_LABELS
from kws.demo.realtime import (
    HighlightPreviewState,
    TemporalLabelSmoother,
    _accept_display_candidate,
    _keyword_runtime_params,
    _passes_confusion_guardrail,
)


def test_accept_display_candidate_requires_confidence_and_wake() -> None:
    assert (
        _accept_display_candidate(
            gate_open=True,
            candidate_label="yes",
            command_conf=0.7,
            wake_prob=0.8,
            min_command_conf=0.35,
            min_wake_prob=0.45,
        )
        == "yes"
    )
    assert (
        _accept_display_candidate(
            gate_open=True,
            candidate_label="yes",
            command_conf=0.2,
            wake_prob=0.8,
            min_command_conf=0.35,
            min_wake_prob=0.45,
        )
        is None
    )
    assert (
        _accept_display_candidate(
            gate_open=False,
            candidate_label="yes",
            command_conf=0.7,
            wake_prob=0.8,
            min_command_conf=0.35,
            min_wake_prob=0.45,
        )
        is None
    )


def test_temporal_label_smoother_requires_repeated_votes() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)

    assert smoother.update(now=0.0, candidate_label="up") is None
    assert smoother.update(now=0.2, candidate_label="up") == "up"


def test_temporal_label_smoother_holds_previous_label_briefly() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)

    assert smoother.update(now=0.0, candidate_label="yes") is None
    assert smoother.update(now=0.1, candidate_label="yes") == "yes"
    assert smoother.update(now=0.2, candidate_label=None) == "yes"
    assert smoother.update(now=0.5, candidate_label=None) is None


def test_temporal_label_smoother_supports_keyword_specific_votes() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3, max_window_size=6)

    assert smoother.update(now=0.0, candidate_label="off", min_count_override=3, window_size_override=5) is None
    assert smoother.update(now=0.1, candidate_label="off", min_count_override=3, window_size_override=5) is None
    assert smoother.update(now=0.2, candidate_label="off", min_count_override=3, window_size_override=5) == "off"


def test_keyword_runtime_params_reads_per_keyword_overrides() -> None:
    calibration = {
        "defaults": {
            "command_conf_threshold": 0.35,
            "vote_window": 4,
            "vote_min_count": 2,
            "prototype_bonus_max": 0.04,
            "min_margin": 0.0,
            "highlight_hold_ms": 220,
        },
        "keywords": {
            "off": {
                "command_conf_threshold": 0.28,
                "vote_window": 5,
                "vote_min_count": 3,
                "prototype_bonus_max": 0.08,
                "min_margin": 0.12,
                "highlight_hold_ms": 280,
            }
        },
    }
    params = _keyword_runtime_params(
        calibration,
        "off",
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    assert params == (0.28, 5, 3, 0.08, 0.12, 280.0)


def test_keyword_runtime_params_uses_builtin_guardrails_for_old_calibration() -> None:
    calibration = {
        "weak_keywords": ["on"],
        "defaults": {"command_conf_threshold": 0.35, "vote_window": 4, "vote_min_count": 2, "prototype_bonus_max": 0.04},
        "keywords": {"on": {"command_conf_threshold": 0.15, "vote_window": 5, "vote_min_count": 3, "prototype_bonus_max": 0.08}},
    }
    params = _keyword_runtime_params(
        calibration,
        "on",
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    assert params[4] >= 0.12
    assert params[5] >= 280.0


def test_keyword_runtime_params_prioritizes_focus_keywords() -> None:
    calibration = {
        "focus_keywords": ["left", "on", "down"],
        "defaults": {"command_conf_threshold": 0.35, "vote_window": 4, "vote_min_count": 2, "prototype_bonus_max": 0.04},
        "keywords": {},
    }
    params = _keyword_runtime_params(
        calibration,
        "on",
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    assert params[1] >= 6
    assert params[2] >= 4
    assert params[4] >= 0.16
    assert params[5] >= 320.0


def test_confusion_guardrail_rejects_low_margin_off() -> None:
    probs = np.zeros(len(COMMAND31_LABELS), dtype=np.float32)
    probs[COMMAND31_LABELS.index("off")] = 0.41
    probs[COMMAND31_LABELS.index("on")] = 0.37
    passed, margin = _passes_confusion_guardrail(
        candidate_label="off",
        command_probs=probs,
        command31_labels=COMMAND31_LABELS,
        calibration={"confusable_groups": {"off": ["on", "up"]}},
        min_margin=0.08,
    )
    assert passed is False
    assert margin < 0.08


def test_confusion_guardrail_accepts_clear_margin() -> None:
    probs = np.zeros(len(COMMAND31_LABELS), dtype=np.float32)
    probs[COMMAND31_LABELS.index("on")] = 0.48
    probs[COMMAND31_LABELS.index("off")] = 0.20
    probs[COMMAND31_LABELS.index("five")] = 0.09
    passed, margin = _passes_confusion_guardrail(
        candidate_label="on",
        command_probs=probs,
        command31_labels=COMMAND31_LABELS,
        calibration={"confusable_groups": {"on": ["off", "five", "one"]}},
        min_margin=0.12,
    )
    assert passed is True
    assert margin >= 0.12


def test_highlight_preview_holds_briefly() -> None:
    preview = HighlightPreviewState()

    assert preview.update(now=0.0, candidate_label="off", hold_seconds=0.25) == "off"
    assert preview.update(now=0.1, candidate_label=None, hold_seconds=0.25) == "off"
    assert preview.update(now=0.3, candidate_label=None, hold_seconds=0.25) is None
