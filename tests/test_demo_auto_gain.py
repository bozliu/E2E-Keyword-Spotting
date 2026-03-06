from __future__ import annotations

from kws.demo.realtime import compute_auto_gain


def test_auto_gain_boosts_low_rms() -> None:
    gain, gain_db = compute_auto_gain(rms=0.005, target_rms=0.05, max_gain_db=18.0, enabled=True)
    assert gain > 1.0
    assert gain_db > 0.0


def test_auto_gain_respects_max_db() -> None:
    gain, gain_db = compute_auto_gain(rms=1e-5, target_rms=0.05, max_gain_db=6.0, enabled=True)
    assert gain <= 2.0 + 1e-6
    assert gain_db <= 6.0 + 1e-6


def test_auto_gain_disabled_is_noop() -> None:
    gain, gain_db = compute_auto_gain(rms=0.001, target_rms=0.05, max_gain_db=18.0, enabled=False)
    assert gain == 1.0
    assert gain_db == 0.0
