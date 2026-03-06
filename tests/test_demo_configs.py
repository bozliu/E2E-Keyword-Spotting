from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_NAMES = [
    "demo_mhatt_small.yaml",
    "demo_mhatt_small_focus.yaml",
    "demo_mhatt_small_focus_lod.yaml",
    "demo_mhatt_base.yaml",
    "demo_mamba_tuned.yaml",
]


def test_demo_keyword_focus_configs_preserve_string_labels() -> None:
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    for name in CONFIG_NAMES:
        payload = yaml.safe_load((configs_dir / name).read_text())
        keyword_focus = payload["training"]["keyword_focus"]
        weak_keywords = keyword_focus["weak_keywords"]
        assert all(isinstance(keyword, str) for keyword in weak_keywords), name
        assert "on" in weak_keywords, name
        assert "off" in weak_keywords, name
        assert "no" in weak_keywords, name

        confusion_groups = keyword_focus["confusion_groups"]
        for label in ("on", "off", "no"):
            assert label in confusion_groups, f"{name}: missing {label}"
            assert all(isinstance(entry, str) for entry in confusion_groups[label]), (
                f"{name}: non-string confusion entry for {label}"
            )


def test_lod_focus_pairs_preserve_yes_as_string() -> None:
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    payload = yaml.safe_load((configs_dir / "demo_mhatt_small_focus_lod.yaml").read_text())
    focus_pairs = payload["training"]["keyword_focus"]["focus_pairs"]
    confusion_groups = payload["training"]["keyword_focus"]["confusion_groups"]

    assert focus_pairs["left"] == ["right", "bed", "yes"]
    assert confusion_groups["left"] == ["right", "bed", "yes"]
