from __future__ import annotations

import time

from kws.demo.realtime import AdaptiveGateConfig, GateStateMachine
from kws.demo.web_runtime import AdaptiveGateConfig as WebAdaptiveGateConfig
from kws.demo.web_runtime import GateStateMachine as WebGateStateMachine


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


def test_adaptive_gate_clamps_thresholds_below_one() -> None:
    t0 = time.monotonic()
    gate = GateStateMachine(
        mode="adaptive",
        open_threshold=0.6,
        close_threshold=0.5,
        cmd_conf_threshold=0.2,
        hold_seconds=0.0,
        adaptive=AdaptiveGateConfig(calibration_seconds=0.1, open_offset=0.06, close_offset=0.03),
    )

    open_now, state, open_thr, close_thr = gate.update(now=t0 + 0.2, wake_prob=0.995, command_conf=0.9)

    assert open_now is True
    assert state == "open"
    assert open_thr <= 0.98
    assert close_thr < open_thr


def test_web_adaptive_gate_clamps_thresholds_below_one() -> None:
    t0 = time.monotonic()
    gate = WebGateStateMachine(
        mode="adaptive",
        open_threshold=0.6,
        close_threshold=0.5,
        cmd_conf_threshold=0.2,
        hold_seconds=0.0,
        adaptive=WebAdaptiveGateConfig(calibration_seconds=0.1, open_offset=0.06, close_offset=0.03),
    )
    gate.reset(now=t0)

    open_now, state, open_thr, close_thr = gate.update(now=t0 + 0.2, wake_prob=0.995, command_conf=0.9)

    assert open_now is True
    assert state == "open"
    assert open_thr <= 0.98
    assert close_thr < open_thr


def test_web_adaptive_gate_uses_more_reachable_web_max_threshold() -> None:
    t0 = time.monotonic()
    gate = WebGateStateMachine(
        mode="adaptive",
        open_threshold=0.28,
        close_threshold=0.16,
        cmd_conf_threshold=0.14,
        hold_seconds=0.0,
        adaptive=WebAdaptiveGateConfig(
            calibration_seconds=0.1,
            open_offset=0.02,
            close_offset=0.01,
            open_floor=0.28,
            close_floor=0.16,
            calibration_score_cap=0.55,
            max_open_threshold=0.72,
        ),
    )
    gate.reset(now=t0)

    open_now, state, open_thr, close_thr = gate.update(now=t0 + 0.2, wake_prob=0.95, command_conf=0.9)

    assert open_now is True
    assert state == "open"
    assert open_thr <= 0.72
    assert close_thr < open_thr


def test_web_adaptive_gate_can_finish_calibration_without_speech_samples() -> None:
    t0 = time.monotonic()
    gate = WebGateStateMachine(
        mode="adaptive",
        open_threshold=0.28,
        close_threshold=0.16,
        cmd_conf_threshold=0.14,
        hold_seconds=0.0,
        adaptive=WebAdaptiveGateConfig(
            calibration_seconds=0.1,
            open_offset=0.02,
            close_offset=0.01,
            open_floor=0.28,
            close_floor=0.16,
            calibration_score_cap=0.55,
            max_open_threshold=0.72,
        ),
    )
    gate.reset(now=t0)

    open_now, state, open_thr, close_thr = gate.update(
        now=t0 + 0.2,
        wake_prob=0.95,
        command_conf=0.4,
        calibration_wake_prob=None,
    )

    assert open_now is True
    assert state == "open"
    assert open_thr == 0.28
    assert close_thr == 0.16
