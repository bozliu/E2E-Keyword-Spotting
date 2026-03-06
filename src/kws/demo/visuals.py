"""Matplotlib visuals for realtime KWS demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.patheffects as pe
except Exception:  # pragma: no cover
    pe = None


BACKGROUND = "#1b1d34"
EDGE = "#858a97"
TEXT = "#eceff4"
ACTIVE_FILL = "#f5f6ff"
ACTIVE_EDGE = "#ffffff"
INACTIVE_FILL = "#242744"


@dataclass
class WheelArtists:
    wedges: Sequence[object]
    labels: Sequence[object]
    base_centers: List[Tuple[float, float]]
    base_fontsizes: List[float]
    label_dirs: List[Tuple[float, float]]
    label_radii: List[float]
    place_labels: str


@dataclass
class HudArtists:
    title: object
    prompt: object
    center: object
    status: object


def resolve_active_index(labels: Sequence[str], active_label: str | None) -> int | None:
    if not active_label:
        return None
    try:
        return list(labels).index(active_label)
    except ValueError:
        return None


def apply_theme(fig, ax) -> None:
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _truncate_label(label: str, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def _display_labels(labels: Sequence[str], abbreviate: bool) -> List[str]:
    if not abbreviate:
        return list(labels)

    max_chars = 7 if len(labels) > 16 else 9
    return [_truncate_label(label, max_chars=max_chars) for label in labels]


def layout_labels_on_wedges(
    wedges: Sequence[object],
    texts: Sequence[object],
    *,
    radius: float,
    ring_width: float | None,
    inward_pad: float = 0.03,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Place each label at the center angle of its wedge."""
    dirs: List[Tuple[float, float]] = []
    radii: List[float] = []

    if ring_width is None:
        label_radius = max(0.2, float(radius) * 0.66)
    else:
        label_radius = float(radius) - float(ring_width) / 2.0 - float(inward_pad)
        label_radius = max(0.12, label_radius)

    for wedge, text in zip(wedges, texts):
        theta = np.deg2rad((float(wedge.theta1) + float(wedge.theta2)) / 2.0)
        ux = float(np.cos(theta))
        uy = float(np.sin(theta))
        dirs.append((ux, uy))
        radii.append(label_radius)
        cx, cy = tuple(getattr(wedge, "center", (0.0, 0.0)))
        text.set_position((cx + label_radius * ux, cy + label_radius * uy))
        text.set_ha("center")
        text.set_va("center")
    return dirs, radii


def create_hud(fig, ax, *, title_text: str = "Google Speech Dataset Demo") -> HudArtists:
    title = fig.text(
        0.5,
        0.975,
        title_text,
        ha="center",
        va="top",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        family="monospace",
    )
    prompt = fig.text(
        0.5,
        0.94,
        "",
        ha="center",
        va="top",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    center = ax.text(
        0.0,
        0.0,
        "LISTENING",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=25,
        fontweight="heavy",
        family="monospace",
    )
    status = fig.text(
        0.5,
        0.022,
        "",
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=10,
        family="monospace",
    )
    return HudArtists(title=title, prompt=prompt, center=center, status=status)


def build_wheel(
    ax,
    labels: Sequence[str],
    *,
    radius: float = 1.0,
    labeldistance: float = 0.78,
    fontsize: int = 8,
    ring_width: float | None = None,
    abbreviate: bool = False,
    place_labels: str = "sector_center",
) -> WheelArtists:
    sizes = np.ones(len(labels), dtype=np.float32)
    shown_labels = _display_labels(labels, abbreviate=abbreviate)

    wedgeprops = {"linewidth": 1.8, "edgecolor": EDGE, "facecolor": INACTIVE_FILL}
    if ring_width is not None:
        wedgeprops["width"] = float(ring_width)

    wedges, texts = ax.pie(
        sizes,
        labels=shown_labels,
        startangle=90,
        counterclock=False,
        radius=radius,
        wedgeprops=wedgeprops,
        textprops={"color": TEXT, "fontsize": fontsize, "fontweight": "bold", "family": "monospace"},
        labeldistance=labeldistance if place_labels != "sector_center" else 1.0,
    )

    for txt in texts:
        txt.set_clip_on(False)

    if place_labels == "sector_center":
        label_dirs, label_radii = layout_labels_on_wedges(
            wedges,
            texts,
            radius=radius,
            ring_width=ring_width,
            inward_pad=0.03,
        )
    else:
        label_dirs = []
        label_radii = []

    base_centers = [tuple(getattr(w, "center", (0.0, 0.0))) for w in wedges]
    base_fontsizes = [float(txt.get_fontsize()) for txt in texts]
    return WheelArtists(
        wedges=wedges,
        labels=texts,
        base_centers=base_centers,
        base_fontsizes=base_fontsizes,
        label_dirs=label_dirs,
        label_radii=label_radii,
        place_labels=place_labels,
    )


def update_wheel(artists: WheelArtists, active_idx: int | None, *, explode: float = 0.04) -> None:
    for idx, wedge in enumerate(artists.wedges):
        is_active = active_idx is not None and idx == active_idx
        wedge.set_facecolor(ACTIVE_FILL if is_active else INACTIVE_FILL)
        wedge.set_edgecolor(ACTIVE_EDGE if is_active else EDGE)
        wedge.set_linewidth(2.2 if is_active else 1.8)
        wedge.set_alpha(1.0 if is_active else 0.92)

        base_center = artists.base_centers[idx]
        if is_active and explode > 0.0:
            theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
            dx = float(explode * np.cos(theta))
            dy = float(explode * np.sin(theta))
            wedge.set_center((base_center[0] + dx, base_center[1] + dy))
        else:
            wedge.set_center(base_center)

    for idx, txt in enumerate(artists.labels):
        is_active = active_idx is not None and idx == active_idx
        txt.set_color("#ffffff" if is_active else TEXT)
        base_size = artists.base_fontsizes[idx] if idx < len(artists.base_fontsizes) else 8.0
        txt.set_fontsize(base_size + 1.5 if is_active else base_size)
        txt.set_fontweight("heavy" if is_active else "bold")
        if artists.place_labels == "sector_center" and idx < len(artists.label_dirs):
            wedge = artists.wedges[idx]
            cx, cy = tuple(getattr(wedge, "center", (0.0, 0.0)))
            ux, uy = artists.label_dirs[idx]
            rr = artists.label_radii[idx]
            txt.set_position((cx + rr * ux, cy + rr * uy))
            txt.set_ha("center")
            txt.set_va("center")
        if pe is not None:
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="#000000")] if is_active else [])
