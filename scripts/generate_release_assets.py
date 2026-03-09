#!/usr/bin/env python3
"""Generate public-release figures and cleaned metric summaries from local reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DATA_DIR = ASSETS_DIR / "data"
KWS12_ORDER = ["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(*names: str) -> tuple[dict[str, Any] | None, str]:
    for name in names:
        path = REPORTS_DIR / name
        if path.exists():
            return _load_json(path), name
    return None, ""


def _as_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _min_keyword_metric(per_class: dict[str, Any], key: str, *, ignore: set[str] | None = None) -> float:
    labels = [label for label in per_class.keys() if label not in (ignore or set())]
    if not labels:
        return 0.0
    return min(float(per_class[label][key]) for label in labels)


def _extract_v2_detector_summary() -> dict[str, Any]:
    stable = _load_json(REPORTS_DIR / "demo_model_ranking_stable.json")
    chosen = stable["chosen"]
    return {
        "version": "v2",
        "label": "v2 stable detector-only",
        "split": "selection validation",
        "min_kws12_precision": float(chosen["min_kws12_precision"]),
        "min_kws12_recall": float(chosen["min_kws12_recall"]),
        "unknown_to_target_rate": float(chosen["kws12_unknown_to_target_rate"]),
        "latency_ms": float(chosen["latency_ms"]),
        "device": str(chosen["runtime_device"]),
        "checkpoint_run": str(chosen["run"]),
    }


def _extract_v3_acceptance_summary() -> dict[str, Any]:
    acceptance = _load_json(REPORTS_DIR / "accuracy_first_local_acceptance.json")
    return {
        "version": "v3",
        "label": "v3 accuracy-first external ensemble",
        "protocol": str(acceptance["protocol"]),
        "model_id": str(acceptance["model_id"]),
        "valid": acceptance["valid"],
        "test": acceptance["test"],
        "overall_passed": bool(acceptance["overall_passed"]),
        "hi_mia_status_snapshot": acceptance.get("hi_mia_status_snapshot", {}),
    }


def _build_version_rows(v2: dict[str, Any], v3: dict[str, Any]) -> list[dict[str, Any]]:
    v3_quality = min(float(v3["valid"]["min_kws12_precision"]), float(v3["valid"]["min_kws12_recall"]))
    return [
        {
            "version": "v1 (2021 legacy)",
            "core_stack": "archived end-to-end KWS prototype",
            "primary_demo_mode": "historical local script",
            "training_setup": "legacy codebase, not reproduced in this repo",
            "eval_protocol": "not rebenchmarked in March 2026 repo",
            "min_per_class_precision": None,
            "min_per_class_recall": None,
            "unknown_to_target_rate": None,
            "latency_ms": None,
            "status": "historical reference only",
        },
        {
            "version": "v2 (branch + v2.0.0 tag)",
            "core_stack": "MHAtt-CRNN detector-only",
            "primary_demo_mode": "CPU desktop demo",
            "training_setup": "Speech Commands + HI-MIA negatives, stable selection profile",
            "eval_protocol": "selection-time validation on current local protocol",
            "min_per_class_precision": float(v2["min_kws12_precision"]),
            "min_per_class_recall": float(v2["min_kws12_recall"]),
            "unknown_to_target_rate": float(v2["unknown_to_target_rate"]),
            "latency_ms": float(v2["latency_ms"]),
            "status": "public-safe CPU baseline",
        },
        {
            "version": "v3 (main / v3.0.0)",
            "core_stack": "detector gate + AST/SUPERB external ensemble",
            "primary_demo_mode": "accuracy-first desktop demo on MPS",
            "training_setup": "imported HF verifier/teacher + realtime validator + public release assets",
            "eval_protocol": "full valid/test local command-label protocol",
            "min_per_class_precision": float(v3["valid"]["min_kws12_precision"]),
            "min_per_class_recall": float(v3["valid"]["min_kws12_recall"]),
            "unknown_to_target_rate": float(v3["valid"]["unknown_to_target_rate"]),
            "latency_ms": float(v3["valid"]["avg_ms_mps_1x1s"]),
            "status": f"offline acceptance passed (quality={v3_quality:.4f})",
        },
    ]


def _build_self_benchmark_rows(v2: dict[str, Any], v3: dict[str, Any]) -> list[dict[str, Any]]:
    superb = _load_json(REPORTS_DIR / "benchmark_superb_wav2vec2_base_superb_ks_test.json")
    ast = _load_json(REPORTS_DIR / "benchmark_mit_ast_finetuned_speech_commands_v2_test.json")
    ensemble = _load_json(REPORTS_DIR / "benchmark_ensemble_ast_superb_kws12_test.json")
    return [
        {
            "label": "v2 stable detector-only",
            "split": "selection validation",
            "metric_scope": "min per-class precision/recall",
            "min_kws12_precision": float(v2["min_kws12_precision"]),
            "min_kws12_recall": float(v2["min_kws12_recall"]),
            "unknown_to_target_rate": float(v2["unknown_to_target_rate"]),
            "latency_ms": float(v2["latency_ms"]),
            "device": str(v2["device"]),
        },
        {
            "label": "SUPERB imported model",
            "split": "test",
            "metric_scope": "min per-class precision/recall",
            "min_kws12_precision": float(superb["min_kws12_precision"]),
            "min_kws12_recall": float(superb["min_kws12_recall"]),
            "unknown_to_target_rate": float(superb["unknown_to_target_rate"]),
            "latency_ms": float(superb["avg_ms_mps_1x1s"]),
            "device": "mps",
        },
        {
            "label": "MIT AST imported model",
            "split": "test",
            "metric_scope": "min per-class precision/recall",
            "min_kws12_precision": float(ast["min_kws12_precision"]),
            "min_kws12_recall": float(ast["min_kws12_recall"]),
            "unknown_to_target_rate": float(ast["unknown_to_target_rate"]),
            "latency_ms": float(ast["avg_ms_mps_1x1s"]),
            "device": "mps",
        },
        {
            "label": "v3 AST+SUPERB ensemble",
            "split": "test",
            "metric_scope": "min per-class precision/recall",
            "min_kws12_precision": float(ensemble["min_kws12_precision"]),
            "min_kws12_recall": float(ensemble["min_kws12_recall"]),
            "unknown_to_target_rate": float(ensemble["unknown_to_target_rate"]),
            "latency_ms": float(ensemble["avg_ms_mps_1x1s"]),
            "device": "mps",
        },
        {
            "label": "v3 acceptance summary",
            "split": "valid + test",
            "metric_scope": "goal pass/fail snapshot",
            "min_kws12_precision": min(float(v3["valid"]["min_kws12_precision"]), float(v3["test"]["min_kws12_precision"])),
            "min_kws12_recall": min(float(v3["valid"]["min_kws12_recall"]), float(v3["test"]["min_kws12_recall"])),
            "unknown_to_target_rate": max(float(v3["valid"]["unknown_to_target_rate"]), float(v3["test"]["unknown_to_target_rate"])),
            "latency_ms": max(float(v3["valid"]["avg_ms_mps_1x1s"]), float(v3["test"]["avg_ms_mps_1x1s"])),
            "device": "mps",
        },
    ]


def _build_sota_rows(v2: dict[str, Any], v3: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": "Ours v3 accuracy-first external ensemble",
            "dataset_protocol": "Current local command-label protocol (valid + test)",
            "headline_metric": "Min per-class precision / recall",
            "value": f"{_as_percent(min(v3['valid']['min_kws12_precision'], v3['test']['min_kws12_precision'])):.2f}% / {_as_percent(min(v3['valid']['min_kws12_recall'], v3['test']['min_kws12_recall'])):.2f}%",
            "source_type": "Reproduced in this repo",
            "comparability": "Directly comparable only inside this repo",
        },
        {
            "method": "Ours v2 stable detector-only",
            "dataset_protocol": "Current local protocol selection validation",
            "headline_metric": "Min per-class precision / recall",
            "value": f"{_as_percent(v2['min_kws12_precision']):.2f}% / {_as_percent(v2['min_kws12_recall']):.2f}%",
            "source_type": "Reproduced in this repo",
            "comparability": "Directly comparable only inside this repo",
        },
        {
            "method": "MatchboxNet-3x2x64",
            "dataset_protocol": "Speech Commands V2 12-class closed-set classification",
            "headline_metric": "Accuracy",
            "value": "98.19%",
            "source_type": "Reported in prior paper benchmarks",
            "comparability": "Not directly comparable",
        },
        {
            "method": "BC-ResNet",
            "dataset_protocol": "Speech Commands V2 12-class closed-set classification",
            "headline_metric": "Accuracy",
            "value": "98.0%",
            "source_type": "Reported in prior paper benchmarks",
            "comparability": "Not directly comparable",
        },
        {
            "method": "EdgeSpot (ICASSP 2026)",
            "dataset_protocol": "Few-shot KWS at fixed FAR",
            "headline_metric": "10-shot accuracy @ 1% FAR",
            "value": "82.0%",
            "source_type": "Reported in paper abstract",
            "comparability": "Not directly comparable",
        },
    ]


def _realtime_status_payload() -> dict[str, Any]:
    valid, valid_name = _optional_json("realtime_accuracy_first_valid.json", "realtime_accuracy_first_valid_smoke.json")
    test, test_name = _optional_json("realtime_accuracy_first_test.json", "realtime_accuracy_first_test_smoke.json")
    if valid and test and valid_name.endswith(".json") and test_name.endswith(".json") and not valid_name.endswith("_smoke.json") and not test_name.endswith("_smoke.json"):
        status = "full"
    elif valid or test:
        status = "smoke_only"
    else:
        status = "missing"
    return {
        "status": status,
        "valid_source": valid_name,
        "test_source": test_name,
        "valid": valid,
        "test": test,
    }


def _plot_per_class(valid: dict[str, Any], test: dict[str, Any], output_path: Path) -> None:
    labels = KWS12_ORDER
    valid_precision = np.asarray([float(valid["per_class_kws12"][label]["precision"]) for label in labels], dtype=np.float64)
    valid_recall = np.asarray([float(valid["per_class_kws12"][label]["recall"]) for label in labels], dtype=np.float64)
    test_precision = np.asarray([float(test["per_class_kws12"][label]["precision"]) for label in labels], dtype=np.float64)
    test_recall = np.asarray([float(test["per_class_kws12"][label]["recall"]) for label in labels], dtype=np.float64)
    x = np.arange(len(labels))
    width = 0.19

    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("#f7f4ed")
    for ax in axes:
        ax.set_facecolor("#fffaf0")
        ax.grid(axis="y", color="#d9d4c7", alpha=0.7, linewidth=0.8)
        ax.axhline(0.95, color="#b7410e", linestyle="--", linewidth=1.4, label="95% target")
        ax.set_ylim(0.90, 1.005)

    axes[0].bar(x - width * 1.5, valid_precision, width=width, color="#0b6e4f", label="Valid precision")
    axes[0].bar(x - width * 0.5, test_precision, width=width, color="#5ec576", label="Test precision")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower right")

    axes[1].bar(x + width * 0.5, valid_recall, width=width, color="#1768ac", label="Valid recall")
    axes[1].bar(x + width * 1.5, test_recall, width=width, color="#4fc3f7", label="Test recall")
    axes[1].set_ylabel("Recall")
    axes[1].legend(loc="lower right")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("v3 accuracy-first per-class metrics on the local KWS12 protocol", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _plot_latency_accuracy(rows: list[dict[str, Any]], output_path: Path) -> None:
    points = []
    for row in rows:
        min_score = min(float(row["min_kws12_precision"]), float(row["min_kws12_recall"]))
        points.append(
            {
                "label": row["label"],
                "latency_ms": float(row["latency_ms"]),
                "quality": min_score,
                "device": row["device"],
            }
        )

    colors = {
        "v2 stable detector-only": "#8c5e58",
        "SUPERB imported model": "#d95f02",
        "MIT AST imported model": "#7570b3",
        "v3 AST+SUPERB ensemble": "#1b9e77",
        "v3 acceptance summary": "#0b6e4f",
    }

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f7f4ed")
    ax.set_facecolor("#fffaf0")
    ax.grid(color="#d9d4c7", alpha=0.7, linewidth=0.8)
    ax.axhline(0.95, color="#b7410e", linestyle="--", linewidth=1.4, label="95% target")
    ax.axvline(30.0, color="#555555", linestyle=":", linewidth=1.2, label="30 ms CPU target")

    for point in points:
        ax.scatter(point["latency_ms"], point["quality"], s=140, color=colors.get(point["label"], "#333333"), edgecolors="black", linewidths=0.6)
        ax.annotate(
            point["label"],
            (point["latency_ms"], point["quality"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Minimum per-class precision/recall")
    ax.set_title("Latency vs. worst-class quality on the current local protocol", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_assets() -> dict[str, Any]:
    v2 = _extract_v2_detector_summary()
    v3 = _extract_v3_acceptance_summary()
    valid = _load_json(REPORTS_DIR / "benchmark_ensemble_ast_superb_kws12_valid.json")
    test = _load_json(REPORTS_DIR / "benchmark_ensemble_ast_superb_kws12_test.json")
    version_rows = _build_version_rows(v2, v3)
    self_rows = _build_self_benchmark_rows(v2, v3)
    sota_rows = _build_sota_rows(v2, v3)
    realtime = _realtime_status_payload()

    summary = {
        "headline": {
            "model": str(v3["model_id"]),
            "protocol": str(v3["protocol"]),
            "overall_passed": bool(v3["overall_passed"]),
        },
        "version_rows": version_rows,
        "self_benchmark_rows": self_rows,
        "sota_rows": sota_rows,
        "realtime_status": realtime,
        "release_claim": (
            "full_realtime_validated"
            if realtime["status"] == "full"
            and bool(realtime["valid"].get("passed"))
            and bool(realtime["test"].get("passed"))
            else "offline_validated_realtime_smoke_only"
        ),
    }

    _write_json(ASSETS_DATA_DIR / "release_summary_v3.json", summary)
    _write_json(ASSETS_DATA_DIR / "version_comparison_v1_v2_v3.json", version_rows)
    _write_json(ASSETS_DATA_DIR / "self_benchmark_v3.json", self_rows)
    _write_json(ASSETS_DATA_DIR / "sota_reference_table.json", sota_rows)
    _write_json(ASSETS_DATA_DIR / "realtime_status.json", realtime)
    _plot_per_class(valid, test, ASSETS_DIR / "per_class_valid_test.png")
    _plot_latency_accuracy(self_rows, ASSETS_DIR / "latency_vs_accuracy.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate public release assets from local benchmark reports.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    summary = build_assets()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
