from __future__ import annotations

from kws.demo.realtime import map_demo_label


def test_map_demo_label() -> None:
    assert map_demo_label("yes", wake_prob=0.9, threshold=0.5) == "yes"
    assert map_demo_label("cat", wake_prob=0.9, threshold=0.5) == "cat"
    assert map_demo_label("yes", wake_prob=0.1, threshold=0.5) == "unknown"
