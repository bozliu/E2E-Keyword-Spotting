#!/usr/bin/env python
"""Collect Phase 5 round artifacts into a summary and optional failure package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kws.constants import TARGET_KEYWORDS_10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase 5 round summary artifacts")
    parser.add_argument("--analyze-report", type=str, required=True)
    parser.add_argument("--eval-report", type=str, required=True)
    parser.add_argument("--hi-mia-status", type=str, default="data/processed/manifests/hi_mia_status.json")
    parser.add_argument("--verifier-metrics", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failure-output", type=str, default="")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--target-unknown-rate", type=float, default=0.02)
    parser.add_argument("--target-latency-ms", type=float, default=30.0)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    return json.loads(target.read_text(encoding="utf-8"))


def _worst_fused_labels(analyze_report: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    fused = analyze_report.get("fused_per_class_kws12", {})
    if not isinstance(fused, dict):
        fused = {}
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for label in TARGET_KEYWORDS_10:
        stats = fused.get(label, {})
        if not isinstance(stats, dict):
            stats = {}
        scored.append((float(stats.get("recall", 0.0)), label, stats))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [
        {
            "label": label,
            "recall": float(stats.get("recall", 0.0)),
            "precision": float(stats.get("precision", 0.0)),
            "top_confusions": stats.get("top_confusions", []),
        }
        for _recall, label, stats in scored[: max(1, min(limit, len(scored)))]
    ]


def main() -> None:
    args = parse_args()
    analyze = _load_json(args.analyze_report)
    eval_report = _load_json(args.eval_report)
    hi_status = _load_json(args.hi_mia_status)
    verifier_metrics = _load_json(args.verifier_metrics) if str(args.verifier_metrics).strip() else {}

    detector_metrics = analyze.get("metrics", {})
    latency_cpu = float((analyze.get("latency_ms", {}) or {}).get("cpu", 0.0))
    summary = {
        "checkpoint": str(analyze.get("checkpoint", "")),
        "verifier_checkpoint": str(
            eval_report.get("verifier_checkpoint")
            or analyze.get("verifier_checkpoint")
            or ""
        ),
        "decision_profile": str(eval_report.get("decision_profile") or analyze.get("decision_profile") or "stable"),
        "detector": {
            "min_kws12_precision": float(analyze.get("min_kws12_precision", detector_metrics.get("min_kws12_precision", 0.0))),
            "min_kws12_recall": float(analyze.get("min_kws12_recall", detector_metrics.get("min_kws12_recall", 0.0))),
            "kws12_unknown_to_target_rate": float(detector_metrics.get("kws12_unknown_to_target_rate", 0.0)),
            "latency_ms_cpu": latency_cpu,
        },
        "fused": {
            "fused_min_kws12_precision": float(eval_report.get("fused_min_kws12_precision", analyze.get("fused_min_kws12_precision", 0.0))),
            "fused_min_kws12_recall": float(eval_report.get("fused_min_kws12_recall", analyze.get("fused_min_kws12_recall", 0.0))),
            "fused_unknown_to_target_rate": float(eval_report.get("fused_unknown_to_target_rate", analyze.get("fused_unknown_to_target_rate", 0.0))),
            "verify_rate": float(eval_report.get("verify_rate", analyze.get("verify_rate", 0.0))),
            "verifier_accept_rate": float(eval_report.get("verifier_accept_rate", analyze.get("verifier_accept_rate", 0.0))),
        },
        "verifier": {
            "verifier_macro_f1": float(verifier_metrics.get("verifier_macro_f1", 0.0)),
            "min_verifier_precision": float(verifier_metrics.get("min_verifier_precision", 0.0)),
            "min_verifier_recall": float(verifier_metrics.get("min_verifier_recall", 0.0)),
        },
        "hi_mia_status": hi_status,
        "worst5_fused_recall_labels": _worst_fused_labels(analyze, limit=5),
        "acceptance": {
            "target_precision": float(args.target_precision),
            "target_recall": float(args.target_recall),
            "target_unknown_rate": float(args.target_unknown_rate),
            "target_latency_ms": float(args.target_latency_ms),
        },
    }

    failed_checks: list[str] = []
    if summary["fused"]["fused_min_kws12_precision"] < float(args.target_precision):
        failed_checks.append("fused_min_kws12_precision")
    if summary["fused"]["fused_min_kws12_recall"] < float(args.target_recall):
        failed_checks.append("fused_min_kws12_recall")
    if summary["fused"]["fused_unknown_to_target_rate"] > float(args.target_unknown_rate):
        failed_checks.append("fused_unknown_to_target_rate")
    if summary["detector"]["latency_ms_cpu"] > float(args.target_latency_ms):
        failed_checks.append("latency_ms_cpu")
    summary["acceptance"]["passed"] = not failed_checks
    summary["acceptance"]["failed_checks"] = failed_checks

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.failure_output and failed_checks:
        failure_payload = {
            "status": "failed_acceptance",
            "checkpoint": summary["checkpoint"],
            "failed_checks": failed_checks,
            "worst5_fused_recall_labels": summary["worst5_fused_recall_labels"],
            "verifier_accept_rate": summary["fused"]["verifier_accept_rate"],
            "cpu_latency_ms": summary["detector"]["latency_ms_cpu"],
            "hi_mia_status": hi_status,
        }
        failure_path = Path(args.failure_output).expanduser().resolve()
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
