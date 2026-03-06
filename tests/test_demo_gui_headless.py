from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kws.demo.visuals import apply_theme, build_wheel, resolve_active_index, update_wheel


def test_headless_wheel_render_cycle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["silence", "yes", "no", "unknown"]
    apply_theme(fig, ax)
    artists = build_wheel(ax, labels)

    active_idx = resolve_active_index(labels, "yes")
    update_wheel(artists, active_idx)
    fig.canvas.draw()

    assert len(artists.wedges) == len(labels)
    assert active_idx == 1
    plt.close(fig)
