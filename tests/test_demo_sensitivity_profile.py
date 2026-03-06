from __future__ import annotations

from kws.demo.realtime import get_sensitivity_tuning


def test_sensitivity_profile_high_more_sensitive_than_strict() -> None:
    high = get_sensitivity_tuning("high")
    strict = get_sensitivity_tuning("strict")

    assert high.cmd_conf_thr < strict.cmd_conf_thr
    assert high.open_offset < strict.open_offset
    assert high.open_floor < strict.open_floor
    assert high.display_conf_thr < strict.display_conf_thr
    assert high.display_wake_thr < strict.display_wake_thr
    assert high.vote_min_count <= strict.vote_min_count


def test_sensitivity_profile_balanced_exists() -> None:
    bal = get_sensitivity_tuning("balanced")
    assert 0.0 < bal.cmd_conf_thr < 1.0
    assert bal.open_floor > bal.close_floor
    assert 0.0 < bal.display_conf_thr < 1.0
    assert 0.0 < bal.display_wake_thr < 1.0
    assert bal.vote_window >= bal.vote_min_count >= 1
