from __future__ import annotations

import numpy as np

from kws.constants import COMMAND31_LABELS
from kws.demo.realtime import _active_label_for_wheel, _wheel_labels, aggregate_command_probs_to_kws12, parse_args, resolve_display_candidate
from kws.demo.visuals import resolve_active_index


def test_resolve_active_index() -> None:
    labels = ["a", "b", "c"]
    assert resolve_active_index(labels, "b") == 1
    assert resolve_active_index(labels, "missing") is None
    assert resolve_active_index(labels, None) is None


def test_wheel_labels_command31() -> None:
    wheel = _wheel_labels("command31", COMMAND31_LABELS)
    assert wheel[0] == "silence"
    assert len(wheel) == 31


def test_wheel_labels_kws12() -> None:
    wheel = _wheel_labels("kws12", COMMAND31_LABELS)
    assert wheel[:2] == ["silence", "unknown"]
    assert len(wheel) == 12


def test_active_label_gate_closed() -> None:
    active, display = _active_label_for_wheel("command31", "yes", gate_open=False)
    assert active is None
    assert display == "UNKNOWN"


def test_active_label_target10() -> None:
    active, display = _active_label_for_wheel("target10", "yes", gate_open=True)
    assert active == "yes"
    assert display == "yes"

    active, display = _active_label_for_wheel("target10", "cat", gate_open=True)
    assert active is None
    assert display == "UNKNOWN"


def test_parse_args_defaults_to_kws12(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["realtime"])
    args = parse_args()
    assert args.demo_profile == "accuracy-first"
    assert args.wheel == "kws12"
    assert args.gate_mode == "adaptive"
    assert args.device == "auto"
    assert args.selection_profile == "stable"
    assert args.sensitivity_profile == "strict"


def test_aggregate_command_probs_to_kws12_merges_unknown_mass() -> None:
    probs = np.zeros(len(COMMAND31_LABELS), dtype=np.float32)
    probs[COMMAND31_LABELS.index("cat")] = 0.4
    probs[COMMAND31_LABELS.index("dog")] = 0.3
    probs[COMMAND31_LABELS.index("yes")] = 0.2
    probs[COMMAND31_LABELS.index("silence")] = 0.1

    kws12 = aggregate_command_probs_to_kws12(probs, COMMAND31_LABELS)
    assert np.isclose(float(kws12[1]), 0.7)
    assert np.isclose(float(kws12[2]), 0.2)


def test_resolve_display_candidate_prefers_kws12_target_for_default_wheel() -> None:
    probs = np.zeros(len(COMMAND31_LABELS), dtype=np.float32)
    probs[COMMAND31_LABELS.index("yes")] = 0.31
    probs[COMMAND31_LABELS.index("no")] = 0.29
    probs[COMMAND31_LABELS.index("cat")] = 0.40

    active, display, conf = resolve_display_candidate("kws12", COMMAND31_LABELS, probs, gate_open=True)
    assert active is None
    assert display == "UNKNOWN"
    assert conf >= 0.40
