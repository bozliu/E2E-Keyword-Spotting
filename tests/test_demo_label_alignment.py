from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kws.demo.visuals import apply_theme, build_wheel


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def test_labels_are_centered_on_wedges() -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    apply_theme(fig, ax)

    labels = ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    artists = build_wheel(ax, labels, radius=1.0, ring_width=0.40, place_labels="sector_center")

    for wedge, text in zip(artists.wedges, artists.labels):
        cx, cy = wedge.center
        tx, ty = text.get_position()

        theta_mid = (float(wedge.theta1) + float(wedge.theta2)) / 2.0
        theta_text = math.degrees(math.atan2(ty - cy, tx - cx))
        if theta_text < 0:
            theta_text += 360.0

        assert _angle_diff_deg(theta_mid % 360.0, theta_text) < 3.0

    plt.close(fig)
