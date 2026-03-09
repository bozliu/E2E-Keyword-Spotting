#!/usr/bin/env python
"""Derive a Phase 5 Round 2 detector config from a fused analysis report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from kws.config import load_yaml
from kws.constants import COMMAND31_TO_INDEX, TARGET_KEYWORDS_10
from kws.utils.keyword_focus import DEFAULT_CONFUSION_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a Round 2 detector config from a fused report")
    parser.add_argument("--base-config", type=str, default="configs/demo_mhatt_small_focus_v3.yaml")
    parser.add_argument("--fused-report", type=str, required=True)
    parser.add_argument("--output", type=str, default="configs/demo_mhatt_small_focus_v3_r2.yaml")
    parser.add_argument("--weak-count", type=int, default=5)
    return parser.parse_args()


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    return json.loads(report_path.read_text(encoding="utf-8"))


def _normalize_groups(raw_groups: Any) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    if not isinstance(raw_groups, dict):
        return groups
    for keyword, values in raw_groups.items():
        label = str(keyword).strip()
        if not label:
            continue
        normalized: list[str] = []
        if isinstance(values, (list, tuple)):
            for value in values:
                rival = str(value).strip()
                if rival and rival in COMMAND31_TO_INDEX and rival != label and rival not in normalized:
                    normalized.append(rival)
        groups[label] = normalized
    return groups


def derive_round2_config(base_cfg: dict[str, Any], fused_report: dict[str, Any], *, weak_count: int, output_stem: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fused_per_class = fused_report.get("fused_per_class_kws12", {})
    if not isinstance(fused_per_class, dict):
        fused_per_class = {}

    scored: list[tuple[float, str]] = []
    for label in TARGET_KEYWORDS_10:
        stats = fused_per_class.get(label, {})
        recall = float(stats.get("recall", 0.0)) if isinstance(stats, dict) else 0.0
        scored.append((recall, label))
    scored.sort(key=lambda item: (item[0], item[1]))
    weak_keywords = [label for _recall, label in scored[: max(1, min(int(weak_count), len(scored)))]]

    training_cfg = base_cfg.setdefault("training", {})
    keyword_focus = training_cfg.setdefault("keyword_focus", {})
    existing_groups = _normalize_groups(keyword_focus.get("confusion_groups", {}))
    default_groups = {key: list(values) for key, values in DEFAULT_CONFUSION_GROUPS.items()}
    merged_groups = dict(existing_groups)

    for keyword in weak_keywords:
        stats = fused_per_class.get(keyword, {})
        top_confusions = stats.get("top_confusions", []) if isinstance(stats, dict) else []
        rivals: list[str] = []
        if isinstance(top_confusions, list):
            for item in top_confusions:
                if not isinstance(item, dict):
                    continue
                rival = str(item.get("label", "")).strip()
                if rival and rival in COMMAND31_TO_INDEX and rival != keyword and rival not in rivals:
                    rivals.append(rival)
        if not rivals:
            fallback = existing_groups.get(keyword, default_groups.get(keyword, []))
            for rival in fallback:
                if rival and rival in COMMAND31_TO_INDEX and rival != keyword and rival not in rivals:
                    rivals.append(rival)
        merged_groups[keyword] = rivals

    keyword_focus["weak_keywords"] = weak_keywords
    keyword_focus["confusion_groups"] = merged_groups
    base_cfg["run_name"] = output_stem

    summary = {
        "weak_keywords": weak_keywords,
        "confusion_groups": {keyword: merged_groups.get(keyword, []) for keyword in weak_keywords},
        "source_report": str(fused_report.get("checkpoint", "")),
        "fused_report_weakest": [{"label": label, "recall": recall} for recall, label in scored[: len(weak_keywords)]],
    }
    return base_cfg, summary


def main() -> None:
    args = parse_args()
    base_bundle = load_yaml(args.base_config)
    fused_report = _load_report(args.fused_report)
    output_path = Path(args.output).expanduser().resolve()
    cfg, summary = derive_round2_config(
        dict(base_bundle.raw),
        fused_report,
        weak_count=args.weak_count,
        output_stem=output_path.stem,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(json.dumps({"output_config": str(output_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
