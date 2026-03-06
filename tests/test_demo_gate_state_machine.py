from __future__ import annotations

import time

from kws.demo.realtime import AdaptiveGateConfig, GateStateMachine


def test_adaptive_gate_calibrates_and_transitions() -> None:
    t0 = time.monotonic()
    gate = GateStateMachine(
        mode="adaptive",
        open_threshold=0.6,
        close_threshold=0.5,
        cmd_conf_threshold=0.35,
        hold_seconds=0.3,
        adaptive=AdaptiveGateConfig(calibration_seconds=1.0),
    )

    open_now, state, _, _ = gate.update(now=t0, wake_prob=0.10, command_conf=0.9)
    assert open_now is False
    assert state == "calibrating"

    open_now, state, open_thr, close_thr = gate.update(now=t0 + 1.2, wake_prob=0.12, command_conf=0.9)
    assert open_now is False
    assert state == "closed"
    assert open_thr >= 0.25
    assert close_thr >= 0.15

    open_now, state, _, _ = gate.update(now=t0 + 1.3, wake_prob=open_thr + 0.05, command_conf=0.9)
    assert open_now is True
    assert state == "open"

    open_now, state, _, _ = gate.update(now=t0 + 1.4, wake_prob=max(0.0, close_thr - 0.1), command_conf=0.9)
    assert open_now is True
    assert state in {"open", "hold"}

    open_now, state, _, _ = gate.update(now=t0 + 1.5, wake_prob=max(0.0, close_thr - 0.1), command_conf=0.9)
    assert state == "hold"
    assert open_now is True

    open_now, state, _, _ = gate.update(now=t0 + 1.9, wake_prob=max(0.0, close_thr - 0.1), command_conf=0.9)
    assert open_now is False
    assert state == "closed"


def test_gate_requires_command_confidence_to_open() -> None:
    t0 = time.monotonic()
    gate = GateStateMachine(
        mode="fixed",
        open_threshold=0.6,
        close_threshold=0.5,
        cmd_conf_threshold=0.4,
        hold_seconds=0.0,
        adaptive=AdaptiveGateConfig(calibration_seconds=0.0),
    )

    open_now, state, _, _ = gate.update(now=t0, wake_prob=0.9, command_conf=0.2)
    assert open_now is False
    assert state == "closed"

    open_now, state, _, _ = gate.update(now=t0 + 0.1, wake_prob=0.9, command_conf=0.8)
    assert open_now is True
    assert state == "open"
