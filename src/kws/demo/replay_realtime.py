"""Replay cached full-stream realtime traces with stream-tuned calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kws.demo.realtime import DEFAULT_DEMO_PROFILE, DEFAULT_RUNTIME_LABEL_BACKEND, get_sensitivity_tuning, load_realtime_demo
from kws.demo.realtime_trace import load_trace, replay_clip_trace
from kws.train.metrics import compute_kws12_breakdown_from_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay cached realtime traces and score the stream decision path.")
    parser.add_argument("--demo-profile", type=str, default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--selection-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--keyword-calibration-path", type=str, default="")
    parser.add_argument("--external-ensemble-calibration-path", type=str, default="")
    parser.add_argument("--wheel", type=str, default="kws12", choices=["kws12", "target10"])
    parser.add_argument("--runtime-label-backend", type=str, default="")
    parser.add_argument("--external-kws-model", type=str, default="ensemble/ast-superb-kws12")
    parser.add_argument("--external-kws-device", type=str, default="mps")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
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
    parser.add_argument("--trace-manifest", type=str, default="")
    parser.add_argument("--cache-root", type=str, default="cache/realtime_traces")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _manifest_path(args: argparse.Namespace) -> Path:
    if str(args.trace_manifest).strip():
        return Path(args.trace_manifest).expanduser().resolve()
    return (Path(args.cache_root).expanduser().resolve() / str(args.split) / "manifest.json").resolve()


def evaluate_trace_manifest(*, bundle, args: argparse.Namespace, manifest: dict[str, object], entries_override=None) -> dict[str, object]:
    entries = list(entries_override) if entries_override is not None else list(manifest.get("entries", []))
    tuning = get_sensitivity_tuning(args.sensitivity_profile)

    preds: list[int] = []
    targets: list[int] = []
    no_match_count = 0
    latencies_ms: list[float] = []
    for entry in entries:
        trace = load_trace(entry["trace_path"])
        pred, no_match, latency_ms = replay_clip_trace(bundle=bundle, trace=trace, args=args, tuning=tuning)
        preds.append(int(pred))
        targets.append(int(trace["target_kws12"]))
        no_match_count += int(no_match)
        if latency_ms is not None:
            latencies_ms.append(float(latency_ms))

    preds_arr = np.asarray(preds, dtype=np.int64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    metrics = compute_kws12_breakdown_from_indices(preds_arr, targets_arr)
    return {
        "split": str(args.split),
        "num_eval_samples": int(targets_arr.size),
        "runtime_label_backend": bundle.resolved_profile.runtime_label_backend,
        "external_kws_model_id": (
            bundle.resolved_profile.external_kws_model if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND else ""
        ),
        "external_kws_device": (
            bundle.resolved_profile.external_kws_device if bundle.resolved_profile.runtime_label_backend == DEFAULT_RUNTIME_LABEL_BACKEND else ""
        ),
        "trace_manifest": str(_manifest_path(args)),
        "per_class_kws12": metrics.get("per_class_kws12", {}),
        "min_kws12_precision": float(metrics.get("min_kws12_precision", 0.0)),
        "min_kws12_recall": float(metrics.get("min_kws12_recall", 0.0)),
        "unknown_to_target_rate": float(metrics.get("kws12_unknown_to_target_rate", 0.0)),
        "no_match_rate": float(no_match_count / max(int(targets_arr.size), 1)),
        "avg_match_latency_ms": float(np.mean(np.asarray(latencies_ms, dtype=np.float64))) if latencies_ms else None,
        "p95_match_latency_ms": float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 95.0)) if latencies_ms else None,
        "goal": {
            "min_kws12_precision": 0.95,
            "min_kws12_recall": 0.95,
            "unknown_to_target_rate_max": 0.02,
        },
        "passed": bool(
            float(metrics.get("min_kws12_precision", 0.0)) >= 0.95
            and float(metrics.get("min_kws12_recall", 0.0)) >= 0.95
            and float(metrics.get("kws12_unknown_to_target_rate", 1.0)) <= 0.02
        ),
    }


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    bundle = load_realtime_demo(
        checkpoint=args.checkpoint,
        demo_profile=args.demo_profile,
        detector_device_preference=args.device,
        selection_profile=args.selection_profile,
        keyword_calibration_path=args.keyword_calibration_path,
        external_ensemble_calibration_path=args.external_ensemble_calibration_path,
        wheel=args.wheel,
        runtime_label_backend=args.runtime_label_backend,
        external_kws_model=args.external_kws_model,
        external_kws_device=args.external_kws_device,
        ranking_iters=8,
        no_cache_ranking=False,
        rebuild_ranking=False,
        device_auto_bench_iters=6,
    )
    manifest = json.loads(_manifest_path(args).read_text(encoding="utf-8"))
    return evaluate_trace_manifest(bundle=bundle, args=args, manifest=manifest)


def main() -> None:
    args = parse_args()
    payload = run_replay(args)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (Path.cwd() / "reports" / f"replay_{args.demo_profile}_{args.split}.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
