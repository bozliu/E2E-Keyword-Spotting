"""Tune realtime-only runtime calibration for the desktop accuracy-first demo."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from kws.constants import INDEX_TO_COMMAND31, KWS12_LABELS, TARGET_KEYWORDS_10, UNKNOWN_LABEL, command31_to_kws12
from kws.demo.realtime import DEFAULT_DEMO_PROFILE, REALTIME_TUNED_DEMO_PROFILE, ResolvedRealtimeProfile, get_sensitivity_tuning, load_realtime_demo
from kws.demo.replay_realtime import evaluate_trace_manifest
from kws.demo.realtime_trace import replay_trace_segments
from kws.demo.train_realtime_specialist import (
    _calibrate_from_valid_logits,
    _collect_manifest_specialist_samples,
    _collect_specialist_samples,
    _concat_samples,
    _predict_probs,
)
from kws.demo.segment_decoder import (
    SEGMENT_DECODER_LABEL_TO_INDEX,
    SEGMENT_FEATURE_DIM,
    SEGMENT_TARGET_LABELS,
    load_segment_decoder_artifact,
    save_segment_decoder_artifact,
    train_segment_decoder,
)
from kws.demo.realtime_specialist import (
    load_realtime_specialist_artifact,
    save_realtime_specialist_artifact,
    save_realtime_specialist_calibration,
    summarize_realtime_specialist_predictions,
    train_realtime_specialist,
)
from kws.demo.validate_realtime import _manifest_records, evaluate_records
from kws.external import blend_ast_superb_probs, default_external_ensemble_calibration, save_external_ensemble_calibration
from kws.utils.keyword_focus import DEFAULT_EXTERNAL_FORCE_OPEN_CONF_THRESHOLD, save_keyword_calibration


TARGET_PRECISION = 0.95
TARGET_RECALL = 0.95
TARGET_UNKNOWN_RATE = 0.02
LAST_MILE_LABELS = ("off", "go", "on", "down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune realtime-only runtime calibration for the v3 desktop demo.")
    parser.add_argument("--demo-profile", type=str, default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--selection-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--wheel", type=str, default="kws12", choices=["kws12", "target10"])
    parser.add_argument("--runtime-label-backend", type=str, default="")
    parser.add_argument("--external-kws-model", type=str, default="ensemble/ast-superb-kws12")
    parser.add_argument("--external-kws-device", type=str, default="mps")
    parser.add_argument("--trace-manifest", type=str, default="cache/realtime_traces/valid/manifest.json")
    parser.add_argument("--train-trace-manifest", type=str, default="cache/realtime_traces/train/manifest.json")
    parser.add_argument("--test-trace-manifest", type=str, default="cache/realtime_traces/test/manifest.json")
    parser.add_argument("--gate-mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--sensitivity-profile", type=str, default="strict", choices=["high", "balanced", "strict"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--wake-open-thr", type=float, default=0.6)
    parser.add_argument("--wake-close-thr", type=float, default=0.5)
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--cmd-conf-thr", type=float, default=None)
    parser.add_argument("--display-conf-thr", type=float, default=None)
    parser.add_argument("--display-wake-thr", type=float, default=None)
    parser.add_argument("--vote-window", type=int, default=None)
    parser.add_argument("--vote-min-count", type=int, default=None)
    parser.add_argument("--hop-seconds", type=float, default=0.10)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--hold-ms", type=float, default=300.0)
    parser.add_argument("--pre-silence-seconds", type=float, default=0.4)
    parser.add_argument("--post-silence-seconds", type=float, default=0.4)
    parser.add_argument("--match-tail-seconds", type=float, default=0.3)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--seed-report", type=str, default="reports/realtime_accuracy_first_valid.json")
    parser.add_argument("--focus-unknown-quota", type=int, default=200)
    parser.add_argument("--focus-silence-quota", type=int, default=80)
    parser.add_argument("--max-tuned-labels", type=int, default=6)
    parser.add_argument("--output-calibration", type=str, default="")
    parser.add_argument(
        "--output-external-calibration",
        type=str,
        default="",
        help="Output path for external_ensemble_realtime_calibration.json",
    )
    parser.add_argument("--output-report", type=str, default="reports/realtime_tuning_valid.json")
    parser.add_argument("--output-test-report", type=str, default="reports/realtime_tuning_test.json")
    parser.add_argument("--failure-report", type=str, default="reports/realtime_tuning_failure.json")
    parser.add_argument("--output-segment-decoder", type=str, default="")
    parser.add_argument("--output-realtime-specialist", type=str, default="")
    parser.add_argument("--output-realtime-specialist-calibration", type=str, default="")
    parser.add_argument("--output-realtime-specialist-valid-report", type=str, default="reports/realtime_specialist_valid.json")
    parser.add_argument("--output-realtime-specialist-test-report", type=str, default="reports/realtime_specialist_test.json")
    return parser.parse_args()


def _deepcopy_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    calibration = copy.deepcopy(payload)
    defaults = calibration.setdefault("defaults", {})
    defaults.setdefault("command_conf_threshold", 0.35)
    defaults.setdefault("vote_window", 4)
    defaults.setdefault("vote_min_count", 2)
    defaults.setdefault("prototype_bonus_max", 0.04)
    defaults.setdefault("min_margin", 0.0)
    defaults.setdefault("highlight_hold_ms", 220)
    defaults.setdefault("external_force_open_conf_threshold", DEFAULT_EXTERNAL_FORCE_OPEN_CONF_THRESHOLD)
    defaults.setdefault("segment_keepalive_threshold", 0.18)
    defaults.setdefault("segment_min_duration_ms", 160.0)
    defaults.setdefault("segment_accept_prob", 0.55)
    defaults.setdefault("segment_close_grace_ms", 140.0)
    calibration.setdefault("keywords", {})
    calibration.setdefault("weak_keywords", [])
    calibration.setdefault("focus_keywords", [])
    calibration.setdefault("confusable_groups", {})
    calibration.setdefault("segment_decoder_disabled", True)
    return calibration


def _default_keyword_entry(defaults: dict[str, Any], support: int = 0) -> dict[str, Any]:
    return {
        "command_conf_threshold": float(defaults.get("command_conf_threshold", 0.35)),
        "vote_window": int(defaults.get("vote_window", 4)),
        "vote_min_count": int(defaults.get("vote_min_count", 2)),
        "prototype_bonus_max": float(defaults.get("prototype_bonus_max", 0.04)),
        "min_margin": float(defaults.get("min_margin", 0.0)),
        "highlight_hold_ms": int(defaults.get("highlight_hold_ms", 220)),
        "external_force_open_conf_threshold": float(
            defaults.get("external_force_open_conf_threshold", DEFAULT_EXTERNAL_FORCE_OPEN_CONF_THRESHOLD)
        ),
        "segment_keepalive_threshold": float(defaults.get("segment_keepalive_threshold", 0.18)),
        "segment_min_duration_ms": float(defaults.get("segment_min_duration_ms", 160.0)),
        "segment_accept_prob": float(defaults.get("segment_accept_prob", 0.55)),
        "segment_close_grace_ms": float(defaults.get("segment_close_grace_ms", 140.0)),
        "support": int(support),
    }


def _ensure_keyword_entry(entry: dict[str, Any], defaults: dict[str, Any], support: int = 0) -> dict[str, Any]:
    merged = _default_keyword_entry(defaults, support=support)
    merged.update(entry)
    return merged


def _load_seed_report(args: argparse.Namespace) -> dict[str, Any] | None:
    seed_path = Path(str(args.seed_report).strip()).expanduser().resolve() if str(args.seed_report).strip() else None
    if seed_path is None or not seed_path.exists():
        return None
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    return payload if str(payload.get("split")) == "valid" else None


def _report_objective(report: dict[str, Any]) -> tuple[float, float, float, float, float]:
    per_class = report.get("per_class_kws12", {})
    target_labels = [label for label in TARGET_KEYWORDS_10 if label in per_class]
    precision_deficit = sum(max(0.0, TARGET_PRECISION - float(per_class[label]["precision"])) for label in target_labels)
    recall_deficit = sum(max(0.0, TARGET_RECALL - float(per_class[label]["recall"])) for label in target_labels)
    min_precision = min((float(per_class[label]["precision"]) for label in target_labels), default=0.0)
    min_recall = min((float(per_class[label]["recall"]) for label in target_labels), default=0.0)
    unknown_penalty = max(0.0, float(report.get("unknown_to_target_rate", 1.0)) - TARGET_UNKNOWN_RATE)
    return (
        round(precision_deficit, 8),
        round(recall_deficit, 8),
        round(unknown_penalty, 8),
        round(float(report.get("no_match_rate", 1.0)), 8),
        round(-min_recall, 8),
        round(-min_precision, 8),
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _candidate_summary(name: str, report: dict[str, Any], *, specialist_enabled: bool, segment_decoder_active: bool) -> dict[str, Any]:
    return {
        "name": name,
        "objective": list(_report_objective(report)),
        "report": report,
        "specialist_enabled": bool(specialist_enabled),
        "segment_decoder_active": bool(segment_decoder_active),
    }


def _eligible_for_last_mile(report: dict[str, Any]) -> bool:
    return (
        float(report.get("min_kws12_precision", 0.0)) >= TARGET_PRECISION
        and float(report.get("unknown_to_target_rate", 1.0)) <= TARGET_UNKNOWN_RATE
        and not bool(report.get("passed"))
    )


def _build_last_mile_candidates(
    base_calibration: dict[str, Any],
    specialist_calibration: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    defaults = base_calibration.get("defaults", {}) if isinstance(base_calibration.get("defaults", {}), dict) else {}
    candidate_specs = [
        ("last_mile_balanced", 0.03, 0.02, 50.0, 0.03, 0.04, 0.02, 0.04),
        ("last_mile_conservative", 0.02, 0.01, 40.0, 0.02, 0.03, 0.01, 0.03),
        ("last_mile_recall_max", 0.05, 0.03, 80.0, 0.04, 0.06, 0.02, 0.06),
    ]
    candidates: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for name, accept_drop, keepalive_drop, grace_add, force_open_drop, spec_accept_drop, spec_margin_drop, spec_trigger_drop in candidate_specs:
        calibration = _deepcopy_calibration(base_calibration)
        for label in LAST_MILE_LABELS:
            entry = _ensure_keyword_entry(dict(calibration["keywords"].get(label, {})), defaults)
            entry["segment_accept_prob"] = float(entry["segment_accept_prob"]) - accept_drop
            entry["segment_keepalive_threshold"] = float(entry["segment_keepalive_threshold"]) - keepalive_drop
            entry["segment_close_grace_ms"] = float(entry["segment_close_grace_ms"]) + grace_add
            if label in {"on", "off", "go"}:
                entry["external_force_open_conf_threshold"] = float(entry["external_force_open_conf_threshold"]) - force_open_drop
            _clamp_vote(entry)
            calibration["keywords"][label] = entry

        specialist_candidate = None
        if specialist_calibration:
            specialist_candidate = copy.deepcopy(specialist_calibration)
            specialist_candidate["enabled"] = True
            default_entry = specialist_candidate.setdefault("default", {})
            default_entry.setdefault("role", "rescue")
            per_label = specialist_candidate.setdefault("per_label", {})
            for label in ("on", "off", "go"):
                entry = per_label.setdefault(label, {"role": "rescue"})
                entry["accept_prob"] = float(max(0.40, float(entry.get("accept_prob", default_entry.get("accept_prob", 0.66))) - spec_accept_drop))
                entry["min_margin"] = float(max(0.0, float(entry.get("min_margin", default_entry.get("min_margin", 0.08))) - spec_margin_drop))
                entry["trigger_prob"] = float(max(0.08, float(entry.get("trigger_prob", default_entry.get("trigger_prob", 0.24))) - spec_trigger_drop))
                entry["role"] = "rescue"
            up_entry = per_label.setdefault("up", {"role": "guard"})
            up_entry["accept_prob"] = float(max(float(up_entry.get("accept_prob", 0.72)), 0.72))
            up_entry["min_margin"] = float(max(float(up_entry.get("min_margin", 0.10)), 0.10))
            up_entry["trigger_prob"] = float(max(float(up_entry.get("trigger_prob", 0.22)), 0.22))
            up_entry["role"] = "guard"
        candidates.append((name, calibration, specialist_candidate))
    return candidates


def _label_from_record(record) -> str:
    return KWS12_LABELS[command31_to_kws12(INDEX_TO_COMMAND31[int(record.command_label)])]


def _pick_focus_labels(seed_report: dict[str, Any], max_labels: int) -> list[str]:
    per_class = seed_report.get("per_class_kws12", {})
    unknown_fp = {
        str(item.get("label")): int(item.get("count", 0))
        for item in per_class.get("unknown", {}).get("top_confusions", [])
        if str(item.get("label")) in TARGET_KEYWORDS_10
    }
    ranked: list[tuple[float, str]] = []
    for label in TARGET_KEYWORDS_10:
        if label not in per_class:
            continue
        precision = float(per_class[label]["precision"])
        recall = float(per_class[label]["recall"])
        penalty = (TARGET_PRECISION - precision if precision < TARGET_PRECISION else 0.0) + (
            TARGET_RECALL - recall if recall < TARGET_RECALL else 0.0
        )
        penalty += 0.01 * float(unknown_fp.get(label, 0))
        if penalty > 0.0:
            ranked.append((penalty, label))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    labels = [label for _score, label in ranked[: max(1, int(max_labels))]]
    for label in ("on", "off", "go", "up"):
        if label in per_class and label not in labels:
            labels.append(label)
    return labels


def _focus_records(all_records: list[Any], tuned_labels: list[str], *, unknown_quota: int, silence_quota: int) -> list[Any]:
    selected: list[Any] = []
    unknown_seen = 0
    silence_seen = 0
    tuned = set(tuned_labels)
    for record in all_records:
        if record.command_label is None:
            continue
        label = _label_from_record(record)
        if label in tuned:
            selected.append(record)
        elif label == "unknown" and unknown_seen < int(unknown_quota):
            selected.append(record)
            unknown_seen += 1
        elif label == "silence" and silence_seen < int(silence_quota):
            selected.append(record)
            silence_seen += 1
    return selected


def _restrict_report(report: dict[str, Any], tuned_labels: list[str]) -> dict[str, Any]:
    per_class = report.get("per_class_kws12", {})
    keep = {"silence", "unknown", *tuned_labels}
    restricted = {label: payload for label, payload in per_class.items() if label in keep}
    cloned = dict(report)
    cloned["per_class_kws12"] = restricted
    return cloned


def _clamp_vote(entry: dict[str, Any]) -> None:
    entry["vote_window"] = int(max(1, min(int(entry["vote_window"]), 8)))
    entry["vote_min_count"] = int(max(1, min(int(entry["vote_min_count"]), int(entry["vote_window"]))))
    entry["command_conf_threshold"] = float(min(max(float(entry["command_conf_threshold"]), 0.05), 0.98))
    entry["min_margin"] = float(min(max(float(entry["min_margin"]), 0.0), 0.30))
    entry["highlight_hold_ms"] = int(max(120, min(int(entry["highlight_hold_ms"]), 500)))
    entry["external_force_open_conf_threshold"] = float(
        min(max(float(entry["external_force_open_conf_threshold"]), 0.30), 0.98)
    )
    entry["segment_keepalive_threshold"] = float(min(max(float(entry["segment_keepalive_threshold"]), 0.08), 0.90))
    entry["segment_min_duration_ms"] = float(max(60.0, min(float(entry["segment_min_duration_ms"]), 600.0)))
    entry["segment_accept_prob"] = float(min(max(float(entry["segment_accept_prob"]), 0.10), 0.95))
    entry["segment_close_grace_ms"] = float(max(40.0, min(float(entry["segment_close_grace_ms"]), 400.0)))


def _apply_recall_adjustment(entry: dict[str, Any], *, strength: float, unknown_heavy: bool) -> None:
    entry["command_conf_threshold"] = float(entry["command_conf_threshold"]) - (0.05 * strength)
    entry["vote_window"] = int(entry["vote_window"]) - 1
    entry["vote_min_count"] = int(entry["vote_min_count"]) - 1
    entry["min_margin"] = float(entry["min_margin"]) - (0.04 * strength)
    entry["highlight_hold_ms"] = int(entry["highlight_hold_ms"]) - int(round(20 * strength))
    entry["external_force_open_conf_threshold"] = float(entry["external_force_open_conf_threshold"]) - (
        0.12 if unknown_heavy else 0.08
    ) * strength
    entry["segment_keepalive_threshold"] = float(entry["segment_keepalive_threshold"]) - (0.05 * strength)
    entry["segment_min_duration_ms"] = float(entry["segment_min_duration_ms"]) - (30.0 * strength)
    entry["segment_accept_prob"] = float(entry["segment_accept_prob"]) - (0.06 * strength)
    entry["segment_close_grace_ms"] = float(entry["segment_close_grace_ms"]) + (20.0 * strength)
    _clamp_vote(entry)


def _apply_precision_adjustment(entry: dict[str, Any], *, strength: float) -> None:
    entry["command_conf_threshold"] = float(entry["command_conf_threshold"]) + (0.06 * strength)
    entry["vote_window"] = int(entry["vote_window"]) + 1
    entry["vote_min_count"] = int(entry["vote_min_count"]) + 1
    entry["min_margin"] = float(entry["min_margin"]) + (0.05 * strength)
    entry["highlight_hold_ms"] = int(entry["highlight_hold_ms"]) + int(round(20 * strength))
    entry["external_force_open_conf_threshold"] = float(entry["external_force_open_conf_threshold"]) + (0.10 * strength)
    entry["segment_keepalive_threshold"] = float(entry["segment_keepalive_threshold"]) + (0.04 * strength)
    entry["segment_min_duration_ms"] = float(entry["segment_min_duration_ms"]) + (30.0 * strength)
    entry["segment_accept_prob"] = float(entry["segment_accept_prob"]) + (0.06 * strength)
    entry["segment_close_grace_ms"] = float(entry["segment_close_grace_ms"]) - (10.0 * strength)
    _clamp_vote(entry)


def _build_candidate_calibration(
    base_calibration: dict[str, Any],
    seed_report: dict[str, Any],
    tuned_labels: list[str],
    *,
    recall_strength: float,
    precision_strength: float,
) -> dict[str, Any]:
    calibration = _deepcopy_calibration(base_calibration)
    defaults = calibration["defaults"]
    keywords = calibration["keywords"]
    per_class = seed_report.get("per_class_kws12", {})
    unknown_fp = {
        str(item.get("label")): int(item.get("count", 0))
        for item in per_class.get("unknown", {}).get("top_confusions", [])
        if str(item.get("label")) in TARGET_KEYWORDS_10
    }
    for label in tuned_labels:
        metrics = per_class.get(label, {})
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        precision_gap = max(0.0, TARGET_PRECISION - precision)
        recall_gap = max(0.0, TARGET_RECALL - recall)
        entry = _ensure_keyword_entry(
            dict(keywords.get(label, {})),
            defaults,
            support=int(metrics.get("support", 0)),
        )
        keywords[label] = entry
        unknown_heavy = metrics.get("top_confusions", [{}])[0].get("label") == "unknown" if metrics.get("top_confusions") else False
        false_positive_pressure = int(unknown_fp.get(label, 0))

        if false_positive_pressure >= 8 or precision_gap > (recall_gap * 1.5):
            _apply_precision_adjustment(entry, strength=max(0.75, precision_strength))
        elif recall_gap > 0.0:
            _apply_recall_adjustment(entry, strength=max(0.75, recall_strength + recall_gap), unknown_heavy=bool(unknown_heavy))
        elif precision_gap > 0.0:
            _apply_precision_adjustment(entry, strength=max(0.75, precision_strength + precision_gap))
        else:
            continue

        if label in {"on", "off", "go"}:
            extra_strength = max(1.0, recall_strength + recall_gap + (0.25 if label == "on" else 0.15))
            _apply_recall_adjustment(entry, strength=extra_strength, unknown_heavy=True)
        if label == "up":
            _apply_precision_adjustment(entry, strength=max(0.75, precision_strength))
    return calibration


def _build_decoder_post_candidates(
    base_calibration: dict[str, Any],
    tuned_labels: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = [("decoder_default", _deepcopy_calibration(base_calibration))]
    defaults = base_calibration.get("defaults", {}) if isinstance(base_calibration.get("defaults", {}), dict) else {}

    relaxed = _deepcopy_calibration(base_calibration)
    for label in {*(tuned_labels or []), "on", "off", "go", "up"}:
        entry = _ensure_keyword_entry(dict(relaxed["keywords"].get(label, {})), defaults)
        entry["segment_accept_prob"] = float(entry["segment_accept_prob"]) - (0.08 if label != "up" else 0.02)
        entry["min_margin"] = float(entry["min_margin"]) - (0.04 if label != "up" else 0.01)
        entry["segment_min_duration_ms"] = float(entry["segment_min_duration_ms"]) - (25.0 if label != "up" else 10.0)
        entry["segment_close_grace_ms"] = float(entry["segment_close_grace_ms"]) + (25.0 if label != "up" else 10.0)
        _clamp_vote(entry)
        relaxed["keywords"][label] = entry
    candidates.append(("decoder_relaxed", relaxed))

    conservative = _deepcopy_calibration(base_calibration)
    up_entry = _ensure_keyword_entry(dict(conservative["keywords"].get("up", {})), defaults)
    up_entry["segment_accept_prob"] = float(up_entry["segment_accept_prob"]) + 0.04
    up_entry["min_margin"] = float(up_entry["min_margin"]) + 0.03
    up_entry["segment_min_duration_ms"] = float(up_entry["segment_min_duration_ms"]) + 20.0
    _clamp_vote(up_entry)
    conservative["keywords"]["up"] = up_entry
    candidates.append(("decoder_up_guarded", conservative))
    return candidates


def _deepcopy_external_calibration(payload: dict[str, Any] | None) -> dict[str, Any]:
    calibration = copy.deepcopy(payload or default_external_ensemble_calibration(mode="realtime"))
    calibration["version"] = 1
    calibration["model_id"] = "ensemble/ast-superb-kws12"
    calibration["mode"] = "realtime"
    calibration.setdefault("defaults", {})
    defaults = calibration["defaults"]
    defaults.setdefault("silence_weight", 1.0)
    defaults.setdefault("unknown_superb_weight", 1.2)
    defaults.setdefault("unknown_ast_weight", 1.2)
    defaults.setdefault("target_ast_weight", 1.0)
    defaults.setdefault("target_superb_residual_weight", 0.2)
    defaults.setdefault("target_global_scale", 1.0)
    defaults.setdefault("unknown_bias", 0.0)
    calibration.setdefault("per_label_bias", {})
    for label in TARGET_KEYWORDS_10:
        calibration["per_label_bias"].setdefault(label, 0.0)
    return calibration


def _build_candidate_external_calibration(
    base_calibration: dict[str, Any] | None,
    seed_report: dict[str, Any],
    tuned_labels: list[str],
    *,
    recall_strength: float,
    precision_strength: float,
) -> dict[str, Any]:
    calibration = _deepcopy_external_calibration(base_calibration)
    defaults = calibration["defaults"]
    per_class = seed_report.get("per_class_kws12", {})
    unknown_penalty = {
        str(item.get("label")): int(item.get("count", 0))
        for item in per_class.get("unknown", {}).get("top_confusions", [])
        if str(item.get("label")) in TARGET_KEYWORDS_10
    }

    defaults["unknown_superb_weight"] = float(
        max(0.60, min(1.40, float(defaults["unknown_superb_weight"]) - (0.10 * recall_strength)))
    )
    defaults["unknown_ast_weight"] = float(
        max(0.60, min(1.40, float(defaults["unknown_ast_weight"]) - (0.10 * recall_strength)))
    )
    defaults["target_global_scale"] = float(
        max(0.90, min(1.25, float(defaults["target_global_scale"]) + (0.06 * recall_strength)))
    )
    defaults["target_superb_residual_weight"] = float(
        max(0.05, min(0.45, float(defaults["target_superb_residual_weight"]) + (0.05 * recall_strength)))
    )
    defaults["unknown_bias"] = float(max(-0.25, min(0.20, float(defaults["unknown_bias"]) - (0.04 * recall_strength))))

    biases = calibration["per_label_bias"]
    for label in tuned_labels:
        metrics = per_class.get(label, {})
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        precision_gap = max(0.0, TARGET_PRECISION - precision)
        recall_gap = max(0.0, TARGET_RECALL - recall)
        unknown_heavy = int(unknown_penalty.get(label, 0)) >= 8 or (
            metrics.get("top_confusions", [{}])[0].get("label") == "unknown" if metrics.get("top_confusions") else False
        )
        bias = float(biases.get(label, 0.0))
        if recall_gap > 0.0 or unknown_heavy:
            bias += (0.02 + recall_gap * 0.15) * recall_strength
        if precision_gap > 0.0:
            bias -= (0.02 + precision_gap * 0.20) * precision_strength
        if label in {"on", "off", "go"}:
            bias += 0.02 * recall_strength
        if label == "up":
            bias -= 0.05 * precision_strength
        biases[label] = float(max(-0.20, min(0.20, bias)))
    return calibration


def _load_trace_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Realtime trace manifest not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _focus_trace_entries(manifest: dict[str, Any], tuned_labels: list[str], *, unknown_quota: int, silence_quota: int) -> list[dict[str, Any]]:
    tuned = set(tuned_labels)
    unknown_seen = 0
    silence_seen = 0
    selected: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        target_idx = int(entry.get("target_kws12", -1))
        label = KWS12_LABELS[target_idx] if 0 <= target_idx < len(KWS12_LABELS) else "unknown"
        if label in tuned:
            selected.append(entry)
        elif label == "unknown" and unknown_seen < int(unknown_quota):
            selected.append(entry)
            unknown_seen += 1
        elif label == "silence" and silence_seen < int(silence_quota):
            selected.append(entry)
            silence_seen += 1
    return selected


def _evaluate_payload(
    *,
    bundle,
    records: list[Any],
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.split = split
    tuning = get_sensitivity_tuning(eval_args.sensitivity_profile)
    return evaluate_records(bundle=bundle, records=records, args=eval_args, tuning=tuning)


def _evaluate_replay_payload(
    *,
    bundle,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    split: str,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    replay_args = argparse.Namespace(**vars(args))
    replay_args.split = split
    return evaluate_trace_manifest(bundle=bundle, args=replay_args, manifest=manifest, entries_override=entries)


def _resolve_output_calibration_path(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_calibration).strip():
        return Path(str(args.output_calibration).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "keyword_calibration_realtime.json").resolve()


def _resolve_output_external_calibration_path(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_external_calibration).strip():
        return Path(str(args.output_external_calibration).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "external_ensemble_realtime_calibration.json").resolve()


def _resolve_output_segment_decoder_path(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_segment_decoder).strip():
        return Path(str(args.output_segment_decoder).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "segment_decoder_realtime.pt").resolve()


def _resolve_output_realtime_specialist_path(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_realtime_specialist).strip():
        return Path(str(args.output_realtime_specialist).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "realtime_specialist.pt").resolve()


def _resolve_output_realtime_specialist_calibration_path(bundle, args: argparse.Namespace) -> Path:
    if str(args.output_realtime_specialist_calibration).strip():
        return Path(str(args.output_realtime_specialist_calibration).strip()).expanduser().resolve()
    return (bundle.checkpoint_path.parent / "realtime_specialist_calibration.json").resolve()


def _bundle_with_realtime_profile(bundle, **overrides):
    profile = ResolvedRealtimeProfile(
        demo_profile=REALTIME_TUNED_DEMO_PROFILE,
        detector_device_preference=bundle.resolved_profile.detector_device_preference,
        runtime_label_backend=bundle.resolved_profile.runtime_label_backend,
        external_kws_model=bundle.resolved_profile.external_kws_model,
        external_kws_device=bundle.resolved_profile.external_kws_device,
    )
    return replace(bundle, resolved_profile=profile, **overrides)


def _trace_target_probs(trace: dict[str, Any], external_calibration: dict[str, Any]) -> np.ndarray:
    ast_probs = np.asarray(trace.get("ast_probs", []), dtype=np.float32)
    superb_probs = np.asarray(trace.get("superb_probs", []), dtype=np.float32)
    if ast_probs.size and superb_probs.size:
        fused = blend_ast_superb_probs(ast_probs, superb_probs, calibration=external_calibration)
    else:
        detector = np.asarray(trace.get("detector_command_probs", []), dtype=np.float32)
        fused = np.zeros((detector.shape[0], len(KWS12_LABELS)), dtype=np.float32)
        for idx in range(min(detector.shape[1], len(INDEX_TO_COMMAND31))):
            fused[:, command31_to_kws12(INDEX_TO_COMMAND31[idx])] += detector[:, idx]
    return np.asarray(fused[:, 2 : 2 + len(SEGMENT_TARGET_LABELS)], dtype=np.float32)


def _segment_thresholds_from_calibration(keyword_calibration: dict[str, Any]) -> list[float]:
    defaults = keyword_calibration.get("defaults", {}) if isinstance(keyword_calibration.get("defaults", {}), dict) else {}
    keywords = keyword_calibration.get("keywords", {}) if isinstance(keyword_calibration.get("keywords", {}), dict) else {}
    thresholds: list[float] = []
    for label in SEGMENT_TARGET_LABELS:
        entry = keywords.get(label, {}) if isinstance(keywords.get(label, {}), dict) else {}
        thresholds.append(float(entry.get("segment_keepalive_threshold", defaults.get("segment_keepalive_threshold", 0.18))))
    return thresholds


def _trace_to_segment_samples(
    trace: dict[str, Any],
    *,
    bundle,
    args: argparse.Namespace,
    tuning,
    keyword_calibration: dict[str, Any],
    external_calibration: dict[str, Any],
) -> list[tuple[np.ndarray, int]]:
    bundle_view = _bundle_with_realtime_profile(
        bundle,
        keyword_calibration=keyword_calibration,
        external_ensemble_calibration=external_calibration,
        segment_decoder=None,
        segment_decoder_path=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_path=None,
        realtime_specialist_calibration={},
        realtime_specialist_calibration_path=None,
    )
    segments = replay_trace_segments(
        bundle=bundle_view,
        trace=trace,
        args=args,
        tuning=tuning,
    )
    target_idx = int(trace.get("target_kws12", -1))
    target_label = KWS12_LABELS[target_idx] if 0 <= target_idx < len(KWS12_LABELS) else UNKNOWN_LABEL
    window_start = float(trace.get("window_start", 0.0))
    window_end = float(trace.get("window_end", window_start))

    samples: list[tuple[np.ndarray, int]] = []
    for segment in segments:
        start_time = float(segment.get("start_time", window_start))
        end_time = float(segment.get("end_time", start_time))
        overlaps_window = min(end_time, window_end) > max(start_time, window_start)
        if target_label in SEGMENT_DECODER_LABEL_TO_INDEX and overlaps_window:
            sample_label = target_label
        else:
            sample_label = UNKNOWN_LABEL
        samples.append(
            (
                np.asarray(segment["features"], dtype=np.float32),
                int(SEGMENT_DECODER_LABEL_TO_INDEX[sample_label]),
            )
        )
    return samples


def _collect_segment_samples(
    manifest: dict[str, Any] | None,
    *,
    bundle,
    args: argparse.Namespace,
    tuning,
    keyword_calibration: dict[str, Any],
    external_calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if manifest is None:
        return np.zeros((0, SEGMENT_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    from kws.demo.realtime_trace import load_trace  # local import to avoid cycles

    features: list[np.ndarray] = []
    labels: list[int] = []
    for entry in manifest.get("entries", []):
        trace = load_trace(entry["trace_path"])
        samples = _trace_to_segment_samples(
            trace,
            bundle=bundle,
            args=args,
            tuning=tuning,
            keyword_calibration=keyword_calibration,
            external_calibration=external_calibration,
        )
        for feat, label in samples:
            features.append(np.asarray(feat, dtype=np.float32))
            labels.append(int(label))
    if not features:
        return np.zeros((0, SEGMENT_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _worst_target_labels(report: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    per_class = report.get("per_class_kws12", {})
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for label in TARGET_KEYWORDS_10:
        payload = per_class.get(label)
        if not isinstance(payload, dict):
            continue
        recall = float(payload.get("recall", 0.0))
        ranked.append((recall, label, payload))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [
        {
            "label": label,
            "precision": float(payload.get("precision", 0.0)),
            "recall": float(payload.get("recall", 0.0)),
            "top_confusions": payload.get("top_confusions", []),
        }
        for _recall, label, payload in ranked[: max(1, int(limit))]
    ]


def main() -> None:
    args = parse_args()
    tuning = get_sensitivity_tuning(args.sensitivity_profile)
    base_bundle = load_realtime_demo(
        checkpoint=args.checkpoint,
        demo_profile=args.demo_profile,
        detector_device_preference=args.device,
        selection_profile=args.selection_profile,
        keyword_calibration_path="",
        external_ensemble_calibration_path="",
        wheel=args.wheel,
        runtime_label_backend=args.runtime_label_backend,
        external_kws_model=args.external_kws_model,
        external_kws_device=args.external_kws_device,
        ranking_iters=8,
        no_cache_ranking=False,
        rebuild_ranking=False,
        device_auto_bench_iters=6,
    )

    valid_trace_manifest = _load_trace_manifest(args.trace_manifest) if Path(args.trace_manifest).expanduser().exists() else None
    test_trace_manifest = _load_trace_manifest(args.test_trace_manifest) if Path(args.test_trace_manifest).expanduser().exists() else None

    manifest_path = Path("data/processed/manifests") / "local_valid.jsonl"
    all_records = _manifest_records(manifest_path, limit_per_class=int(args.limit_per_class))
    seed_report = _load_seed_report(args)
    if seed_report is None:
        if valid_trace_manifest is not None:
            seed_report = _evaluate_replay_payload(
                bundle=base_bundle,
                manifest=valid_trace_manifest,
                args=args,
                split="valid",
            )
        else:
            seed_report = _evaluate_payload(bundle=base_bundle, records=all_records, args=args, split="valid")

    tuned_labels = _pick_focus_labels(seed_report, int(args.max_tuned_labels))
    focused_records = _focus_records(
        all_records,
        tuned_labels,
        unknown_quota=int(args.focus_unknown_quota),
        silence_quota=int(args.focus_silence_quota),
    )
    focused_trace_entries = (
        _focus_trace_entries(
            valid_trace_manifest,
            tuned_labels,
            unknown_quota=int(args.focus_unknown_quota),
            silence_quota=int(args.focus_silence_quota),
        )
        if valid_trace_manifest is not None
        else None
    )

    candidate_specs = [
        ("moderate", 1.00, 1.00),
        ("recall_relaxed", 1.35, 0.85),
        ("precision_guarded", 0.85, 1.35),
    ]
    best_candidate_name = "baseline"
    best_calibration = _deepcopy_calibration(base_bundle.keyword_calibration)
    best_external_calibration = _deepcopy_external_calibration(base_bundle.external_ensemble_calibration)
    if valid_trace_manifest is not None and focused_trace_entries is not None:
        best_focus_report = _evaluate_replay_payload(
            bundle=_bundle_with_realtime_profile(
                base_bundle,
                keyword_calibration=best_calibration,
                external_ensemble_calibration=best_external_calibration,
            ),
            manifest=valid_trace_manifest,
            entries=focused_trace_entries,
            args=args,
            split="valid",
        )
    else:
        best_focus_report = _restrict_report(seed_report, tuned_labels)
    best_objective = _report_objective(best_focus_report)

    candidate_summaries: list[dict[str, Any]] = [
        _candidate_summary("baseline", best_focus_report, specialist_enabled=False, segment_decoder_active=False)
    ]

    for name, recall_strength, precision_strength in candidate_specs:
        calibration = _build_candidate_calibration(
            base_bundle.keyword_calibration,
            seed_report,
            tuned_labels,
            recall_strength=recall_strength,
            precision_strength=precision_strength,
        )
        external_calibration = _build_candidate_external_calibration(
            base_bundle.external_ensemble_calibration,
            seed_report,
            tuned_labels,
            recall_strength=recall_strength,
            precision_strength=precision_strength,
        )
        candidate_bundle = _bundle_with_realtime_profile(
            base_bundle,
            keyword_calibration=calibration,
            external_ensemble_calibration=external_calibration,
        )
        focus_report = (
            _evaluate_replay_payload(
                bundle=candidate_bundle,
                manifest=valid_trace_manifest,
                entries=focused_trace_entries,
                args=args,
                split="valid",
            )
            if valid_trace_manifest is not None and focused_trace_entries is not None
            else _evaluate_payload(
                bundle=candidate_bundle,
                records=focused_records,
                args=args,
                split="valid",
            )
        )
        objective = _report_objective(focus_report)
        candidate_summaries.append(
            _candidate_summary(name, focus_report, specialist_enabled=False, segment_decoder_active=False)
        )
        if objective < best_objective:
            best_candidate_name = name
            best_calibration = calibration
            best_external_calibration = external_calibration
            best_focus_report = focus_report
            best_objective = objective

    train_trace_manifest = _load_trace_manifest(args.train_trace_manifest) if Path(args.train_trace_manifest).expanduser().exists() else None
    output_calibration = _resolve_output_calibration_path(base_bundle, args)
    output_external_calibration = _resolve_output_external_calibration_path(base_bundle, args)
    output_segment_decoder = _resolve_output_segment_decoder_path(base_bundle, args)
    output_realtime_specialist = _resolve_output_realtime_specialist_path(base_bundle, args)
    output_realtime_specialist_calibration = _resolve_output_realtime_specialist_calibration_path(base_bundle, args)
    best_calibration["segment_decoder_disabled"] = True
    save_keyword_calibration(output_calibration, best_calibration)
    save_external_ensemble_calibration(output_external_calibration, best_external_calibration)

    final_bundle = _bundle_with_realtime_profile(
        base_bundle,
        keyword_calibration=best_calibration,
        external_ensemble_calibration=best_external_calibration,
        segment_decoder=None,
        segment_decoder_path=None,
        segment_decoder_disabled=True,
        realtime_specialist=None,
        realtime_specialist_path=None,
        realtime_specialist_calibration={},
        realtime_specialist_calibration_path=None,
    )
    full_valid_report = (
        _evaluate_replay_payload(
            bundle=final_bundle,
            manifest=valid_trace_manifest,
            args=args,
            split="valid",
        )
        if valid_trace_manifest is not None
        else _evaluate_payload(bundle=final_bundle, records=all_records, args=args, split="valid")
    )
    trained_segment_decoder = False
    trained_realtime_specialist = False
    wrote_realtime_specialist = False
    specialist_enabled = False
    current_specialist_calibration: dict[str, Any] | None = None
    loaded_realtime_specialist = None
    if not bool(full_valid_report.get("passed")) and train_trace_manifest is not None:
        train_features, train_labels = _collect_segment_samples(
            train_trace_manifest,
            bundle=base_bundle,
            args=args,
            tuning=tuning,
            keyword_calibration=best_calibration,
            external_calibration=best_external_calibration,
        )
        valid_features, valid_labels = _collect_segment_samples(
            valid_trace_manifest,
            bundle=base_bundle,
            args=args,
            tuning=tuning,
            keyword_calibration=best_calibration,
            external_calibration=best_external_calibration,
        ) if valid_trace_manifest is not None else (np.zeros((0, SEGMENT_FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        if train_features.size and train_labels.size:
            model, feature_mean, feature_std = train_segment_decoder(
                train_features=train_features,
                train_labels=train_labels,
                valid_features=valid_features,
                valid_labels=valid_labels,
            )
            save_segment_decoder_artifact(
                output_segment_decoder,
                model=model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                hidden_dim=64,
            )
            loaded_decoder = load_segment_decoder_artifact(output_segment_decoder, device=base_bundle.runtime_device)
            decoder_candidate_summaries: list[dict[str, Any]] = []
            best_decoder_choice = ("decoder_default", _deepcopy_calibration(best_calibration), full_valid_report)
            for candidate_name, candidate_calibration in _build_decoder_post_candidates(best_calibration, tuned_labels):
                candidate_calibration["segment_decoder_disabled"] = False
                candidate_bundle = _bundle_with_realtime_profile(
                    base_bundle,
                    keyword_calibration=candidate_calibration,
                    external_ensemble_calibration=best_external_calibration,
                    segment_decoder=loaded_decoder,
                    segment_decoder_path=output_segment_decoder,
                    segment_decoder_disabled=False,
                )
                candidate_report = (
                    _evaluate_replay_payload(
                        bundle=candidate_bundle,
                        manifest=valid_trace_manifest,
                        args=args,
                        split="valid",
                    )
                    if valid_trace_manifest is not None
                    else _evaluate_payload(bundle=candidate_bundle, records=all_records, args=args, split="valid")
                )
                objective = _report_objective(candidate_report)
                decoder_candidate_summaries.append(
                    _candidate_summary(
                        candidate_name,
                        candidate_report,
                        specialist_enabled=False,
                        segment_decoder_active=True,
                    )
                )
                if objective < _report_objective(best_decoder_choice[2]):
                    best_decoder_choice = (candidate_name, candidate_calibration, candidate_report)
            best_candidate_name = f"{best_candidate_name}+{best_decoder_choice[0]}"
            best_calibration = best_decoder_choice[1]
            best_calibration["segment_decoder_disabled"] = False
            save_keyword_calibration(output_calibration, best_calibration)
            final_bundle = _bundle_with_realtime_profile(
                base_bundle,
                keyword_calibration=best_calibration,
                external_ensemble_calibration=best_external_calibration,
                segment_decoder=loaded_decoder,
                segment_decoder_path=output_segment_decoder,
                segment_decoder_disabled=False,
            )
            full_valid_report = best_decoder_choice[2]
            candidate_summaries.extend(decoder_candidate_summaries)
            trained_segment_decoder = True
    if not bool(full_valid_report.get("passed")) and train_trace_manifest is not None and valid_trace_manifest is not None:
        train_manifest_waveforms, train_manifest_labels = _collect_manifest_specialist_samples(
            "train",
            clip_samples=int(base_bundle.clip_samples),
            other_target_quota=180,
            unknown_quota=600,
            silence_quota=180,
        )
        valid_manifest_waveforms, valid_manifest_labels = _collect_manifest_specialist_samples(
            "valid",
            clip_samples=int(base_bundle.clip_samples),
            other_target_quota=90,
            unknown_quota=300,
            silence_quota=90,
        )
        train_trace_waveforms, train_trace_labels = _collect_specialist_samples(train_trace_manifest, bundle=final_bundle, args=args)
        valid_trace_waveforms, valid_trace_labels = _collect_specialist_samples(valid_trace_manifest, bundle=final_bundle, args=args)
        train_waveforms, train_specialist_labels = _concat_samples(
            (train_manifest_waveforms, train_manifest_labels),
            (train_trace_waveforms, train_trace_labels),
        )
        valid_waveforms, valid_specialist_labels = _concat_samples(
            (valid_manifest_waveforms, valid_manifest_labels),
            (valid_trace_waveforms, valid_trace_labels),
        )
        if train_waveforms.size and train_specialist_labels.size and valid_waveforms.size and valid_specialist_labels.size:
            specialist_model, specialist_feature_mean, specialist_feature_std = train_realtime_specialist(
                train_waveforms=train_waveforms,
                train_labels=train_specialist_labels,
                valid_waveforms=valid_waveforms,
                valid_labels=valid_specialist_labels,
                device=base_bundle.runtime_device,
                sample_rate=int(base_bundle.sample_rate),
                target_samples=int(base_bundle.clip_samples),
                hidden_dim=96,
                epochs=14,
                batch_size=32,
            )
            save_realtime_specialist_artifact(
                output_realtime_specialist,
                model=specialist_model,
                sample_rate=int(base_bundle.sample_rate),
                target_samples=int(base_bundle.clip_samples),
                n_mels=64,
                hidden_dim=96,
                feature_mean=specialist_feature_mean,
                feature_std=specialist_feature_std,
            )
            wrote_realtime_specialist = True
            specialist_valid_probs = _predict_probs(
                specialist_model,
                valid_waveforms,
                device=base_bundle.runtime_device,
                sample_rate=int(base_bundle.sample_rate),
                target_samples=int(base_bundle.clip_samples),
                n_mels=64,
                feature_mean=specialist_feature_mean,
                feature_std=specialist_feature_std,
            )
            specialist_valid_metrics = summarize_realtime_specialist_predictions(specialist_valid_probs, valid_specialist_labels)
            specialist_calibration = _calibrate_from_valid_logits(specialist_valid_probs, valid_specialist_labels)
            specialist_valid_report = {
                "num_valid_samples": int(valid_specialist_labels.shape[0]),
                "num_valid_manifest_samples": int(valid_manifest_labels.shape[0]),
                "num_valid_trace_samples": int(valid_trace_labels.shape[0]),
                "specialist_path": str(output_realtime_specialist),
                "calibration_path": str(output_realtime_specialist_calibration),
                "labels": ["on", "off", "go", "up", "other"],
                **specialist_valid_metrics,
            }
            _write_json(args.output_realtime_specialist_valid_report, specialist_valid_report)
            if test_trace_manifest is not None:
                test_manifest_waveforms, test_manifest_labels = _collect_manifest_specialist_samples(
                    "test",
                    clip_samples=int(base_bundle.clip_samples),
                    other_target_quota=90,
                    unknown_quota=300,
                    silence_quota=90,
                )
                test_trace_waveforms, test_trace_labels = _collect_specialist_samples(
                    test_trace_manifest,
                    bundle=final_bundle,
                    args=args,
                )
                test_waveforms, test_specialist_labels = _concat_samples(
                    (test_manifest_waveforms, test_manifest_labels),
                    (test_trace_waveforms, test_trace_labels),
                )
                specialist_test_report = {
                    "num_test_samples": int(test_specialist_labels.shape[0]),
                    "num_test_manifest_samples": int(test_manifest_labels.shape[0]),
                    "num_test_trace_samples": int(test_trace_labels.shape[0]),
                    "specialist_path": str(output_realtime_specialist),
                    "calibration_path": str(output_realtime_specialist_calibration),
                    "labels": ["on", "off", "go", "up", "other"],
                }
                if test_waveforms.size and test_specialist_labels.size:
                    specialist_test_probs = _predict_probs(
                        specialist_model,
                        test_waveforms,
                        device=base_bundle.runtime_device,
                        sample_rate=int(base_bundle.sample_rate),
                        target_samples=int(base_bundle.clip_samples),
                        n_mels=64,
                        feature_mean=specialist_feature_mean,
                        feature_std=specialist_feature_std,
                    )
                    specialist_test_report.update(
                        summarize_realtime_specialist_predictions(specialist_test_probs, test_specialist_labels)
                    )
                _write_json(args.output_realtime_specialist_test_report, specialist_test_report)
            loaded_specialist = load_realtime_specialist_artifact(output_realtime_specialist, device=base_bundle.runtime_device)
            specialist_bundle = _bundle_with_realtime_profile(
                base_bundle,
                keyword_calibration=best_calibration,
                external_ensemble_calibration=best_external_calibration,
                segment_decoder=final_bundle.segment_decoder,
                segment_decoder_path=final_bundle.segment_decoder_path,
                segment_decoder_disabled=final_bundle.segment_decoder_disabled,
                realtime_specialist=loaded_specialist,
                realtime_specialist_path=output_realtime_specialist,
                realtime_specialist_calibration=specialist_calibration,
                realtime_specialist_calibration_path=output_realtime_specialist_calibration,
            )
            specialist_report = _evaluate_replay_payload(
                bundle=specialist_bundle,
                manifest=valid_trace_manifest,
                args=args,
                split="valid",
            )
            candidate_summaries.append(
                {
                    **_candidate_summary(
                        "hard_word_specialist",
                        specialist_report,
                        specialist_enabled=True,
                        segment_decoder_active=not bool(final_bundle.segment_decoder_disabled),
                    ),
                    "specialist_valid_metrics": specialist_valid_metrics,
                }
            )
            if _report_objective(specialist_report) < _report_objective(full_valid_report):
                specialist_calibration["enabled"] = True
                save_realtime_specialist_calibration(output_realtime_specialist_calibration, specialist_calibration)
                final_bundle = specialist_bundle
                full_valid_report = specialist_report
                best_candidate_name = f"{best_candidate_name}+hard_word_specialist"
                trained_realtime_specialist = True
                specialist_enabled = True
                current_specialist_calibration = copy.deepcopy(specialist_calibration)
                loaded_realtime_specialist = loaded_specialist
            else:
                specialist_calibration["enabled"] = False
                save_realtime_specialist_calibration(output_realtime_specialist_calibration, specialist_calibration)
                current_specialist_calibration = copy.deepcopy(specialist_calibration)
                specialist_enabled = False
                loaded_realtime_specialist = loaded_specialist
    if _eligible_for_last_mile(full_valid_report):
        best_last_mile_choice = None
        for candidate_name, candidate_calibration, specialist_candidate in _build_last_mile_candidates(
            best_calibration,
            current_specialist_calibration,
        ):
            segment_decoder_disabled = bool(candidate_calibration.get("segment_decoder_disabled", True))
            candidate_bundle = _bundle_with_realtime_profile(
                base_bundle,
                keyword_calibration=candidate_calibration,
                external_ensemble_calibration=best_external_calibration,
                segment_decoder=final_bundle.segment_decoder,
                segment_decoder_path=final_bundle.segment_decoder_path,
                segment_decoder_disabled=segment_decoder_disabled,
                realtime_specialist=loaded_realtime_specialist if specialist_candidate else None,
                realtime_specialist_path=output_realtime_specialist if specialist_candidate else None,
                realtime_specialist_calibration=specialist_candidate or {},
                realtime_specialist_calibration_path=output_realtime_specialist_calibration if specialist_candidate else None,
            )
            candidate_report = _evaluate_replay_payload(
                bundle=candidate_bundle,
                manifest=valid_trace_manifest,
                args=args,
                split="valid",
            ) if valid_trace_manifest is not None else _evaluate_payload(
                bundle=candidate_bundle,
                records=all_records,
                args=args,
                split="valid",
            )
            candidate_summaries.append(
                _candidate_summary(
                    candidate_name,
                    candidate_report,
                    specialist_enabled=bool(specialist_candidate and specialist_candidate.get("enabled", True)),
                    segment_decoder_active=not segment_decoder_disabled,
                )
            )
            if _report_objective(candidate_report) < _report_objective(full_valid_report):
                best_last_mile_choice = (
                    candidate_name,
                    candidate_calibration,
                    specialist_candidate,
                    candidate_bundle,
                    candidate_report,
                )
        if best_last_mile_choice is not None:
            candidate_name, candidate_calibration, specialist_candidate, candidate_bundle, candidate_report = best_last_mile_choice
            best_candidate_name = f"{best_candidate_name}+{candidate_name}"
            best_calibration = candidate_calibration
            final_bundle = candidate_bundle
            full_valid_report = candidate_report
            save_keyword_calibration(output_calibration, best_calibration)
            if specialist_candidate is not None:
                current_specialist_calibration = copy.deepcopy(specialist_candidate)
                specialist_enabled = bool(specialist_candidate.get("enabled", True))
                save_realtime_specialist_calibration(output_realtime_specialist_calibration, specialist_candidate)
    full_valid_report["tuned_profile"] = REALTIME_TUNED_DEMO_PROFILE
    full_valid_report["keyword_calibration_path"] = str(output_calibration)
    full_valid_report["external_ensemble_calibration_path"] = str(output_external_calibration)
    full_valid_report["segment_decoder_path"] = str(output_segment_decoder) if trained_segment_decoder else ""
    full_valid_report["segment_decoder_disabled"] = bool(best_calibration.get("segment_decoder_disabled", True))
    full_valid_report["realtime_specialist_path"] = str(output_realtime_specialist) if wrote_realtime_specialist else ""
    full_valid_report["realtime_specialist_calibration_path"] = str(output_realtime_specialist_calibration) if wrote_realtime_specialist else ""
    full_valid_report["source_demo_profile"] = args.demo_profile
    full_valid_report["focused_labels"] = tuned_labels
    full_valid_report["selected_candidate"] = best_candidate_name
    full_valid_report["selected_candidate_specialist_enabled"] = bool(specialist_enabled)
    full_valid_report["selected_candidate_segment_decoder_active"] = not bool(best_calibration.get("segment_decoder_disabled", True))
    full_valid_report["tuning_mode"] = (
        "trace_replay+segment_decoder+hard_word_specialist"
        if trained_realtime_specialist
        else ("trace_replay+segment_decoder" if trained_segment_decoder else ("trace_replay" if valid_trace_manifest is not None else "live_validation"))
    )
    full_valid_report["focused_candidate_summaries"] = [
        {
            "name": item["name"],
            "objective": item["objective"],
            "min_kws12_precision": float(item["report"]["min_kws12_precision"]),
            "min_kws12_recall": float(item["report"]["min_kws12_recall"]),
            "unknown_to_target_rate": float(item["report"]["unknown_to_target_rate"]),
            "specialist_enabled": bool(item.get("specialist_enabled", False)),
            "segment_decoder_active": bool(item.get("segment_decoder_active", False)),
        }
        for item in candidate_summaries
    ]

    output_report = Path(args.output_report).expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(full_valid_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if test_trace_manifest is not None:
        full_test_report = _evaluate_replay_payload(
            bundle=final_bundle,
            manifest=test_trace_manifest,
            args=args,
            split="test",
        )
        full_test_report["tuned_profile"] = REALTIME_TUNED_DEMO_PROFILE
        full_test_report["keyword_calibration_path"] = str(output_calibration)
        full_test_report["external_ensemble_calibration_path"] = str(output_external_calibration)
        full_test_report["segment_decoder_path"] = str(output_segment_decoder) if trained_segment_decoder else ""
        full_test_report["segment_decoder_disabled"] = bool(best_calibration.get("segment_decoder_disabled", True))
        full_test_report["realtime_specialist_path"] = str(output_realtime_specialist) if wrote_realtime_specialist else ""
        full_test_report["realtime_specialist_calibration_path"] = str(output_realtime_specialist_calibration) if wrote_realtime_specialist else ""
        full_test_report["source_demo_profile"] = args.demo_profile
        full_test_report["selected_candidate"] = best_candidate_name
        full_test_report["selected_candidate_specialist_enabled"] = bool(specialist_enabled)
        full_test_report["selected_candidate_segment_decoder_active"] = not bool(best_calibration.get("segment_decoder_disabled", True))
        output_test_report = Path(args.output_test_report).expanduser().resolve()
        output_test_report.parent.mkdir(parents=True, exist_ok=True)
        output_test_report.write_text(json.dumps(full_test_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not bool(full_valid_report.get("passed")):
        failure_payload = {
            "status": "failed_to_meet_realtime_goal_on_valid",
            "selected_candidate": best_candidate_name,
            "keyword_calibration_path": str(output_calibration),
            "external_ensemble_calibration_path": str(output_external_calibration),
            "segment_decoder_path": str(output_segment_decoder) if trained_segment_decoder else "",
            "segment_decoder_disabled": bool(best_calibration.get("segment_decoder_disabled", True)),
            "realtime_specialist_path": str(output_realtime_specialist) if wrote_realtime_specialist else "",
            "realtime_specialist_calibration_path": str(output_realtime_specialist_calibration) if wrote_realtime_specialist else "",
            "selected_candidate_specialist_enabled": bool(specialist_enabled),
            "selected_candidate_segment_decoder_active": not bool(best_calibration.get("segment_decoder_disabled", True)),
            "focused_labels": tuned_labels,
            "baseline_seed_report": seed_report,
            "best_focused_report": best_focus_report,
            "best_valid_report": full_valid_report,
            "worst_valid_labels": _worst_target_labels(full_valid_report),
        }
        failure_report = Path(args.failure_report).expanduser().resolve()
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        failure_report.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(full_valid_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
