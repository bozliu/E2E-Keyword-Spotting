from __future__ import annotations

import numpy as np

from kws.demo.realtime import ema_update, ema_update_scalar


def test_ema_update_initializes_from_current() -> None:
    current = np.array([0.2, 0.8], dtype=np.float32)
    out = ema_update(None, current, 0.3)
    assert np.allclose(out, current)


def test_ema_update_is_monotonic_toward_target() -> None:
    prev = np.zeros(2, dtype=np.float32)
    target = np.array([1.0, 0.5], dtype=np.float32)
    history = []
    for _ in range(6):
        prev = ema_update(prev, target, 0.4)
        history.append(prev.copy())
    for idx in range(len(history) - 1):
        assert history[idx + 1][0] >= history[idx][0]
        assert history[idx + 1][1] >= history[idx][1]
    assert history[-1][0] <= target[0]
    assert history[-1][1] <= target[1]


def test_ema_update_scalar_is_monotonic_toward_target() -> None:
    value = 0.0
    history = []
    for _ in range(6):
        value = ema_update_scalar(value, 1.0, 0.5)
        history.append(value)
    for idx in range(len(history) - 1):
        assert history[idx + 1] >= history[idx]
    assert history[-1] <= 1.0
