from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kws.demo.visuals import apply_theme, build_wheel, create_hud


def _overlap(a, b, pad: float = 1.0) -> bool:
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def test_hud_text_layout_does_not_overlap() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.7), dpi=120)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.94)
    ax.set_position([0.08, 0.12, 0.84, 0.70])

    apply_theme(fig, ax)
    labels = ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    _artists = build_wheel(ax, labels, radius=1.0, ring_width=0.40, fontsize=10, labeldistance=0.80)

    hud = create_hud(fig, ax)
    hud.prompt.set_text("[LISTENING]")
    hud.center.set_text("UNKNOWN")
    hud.status.set_text(
        "mic=RUNNING rms=0.0200 gain=4.0dB clip=0 wake=0.011 "
        "open=0.26 close=0.17 conf=0.12 lat=17.0ms dev=auto->cpu q=0.10"
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    b_title = hud.title.get_window_extent(renderer=renderer)
    b_prompt = hud.prompt.get_window_extent(renderer=renderer)
    b_center = hud.center.get_window_extent(renderer=renderer)
    b_status = hud.status.get_window_extent(renderer=renderer)

    assert not _overlap(b_title, b_prompt)
    assert not _overlap(b_center, b_status)
    assert "Please say" not in hud.prompt.get_text()
    plt.close(fig)
