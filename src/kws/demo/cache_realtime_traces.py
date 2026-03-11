"""Cache full-stream realtime traces so tuning can replay decisions quickly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kws.constants import IGNORE_INDEX, INDEX_TO_COMMAND31, command31_to_kws12
from kws.data.manifest import read_manifest
from kws.demo.realtime import DEFAULT_DEMO_PROFILE, load_realtime_demo
from kws.demo.realtime_trace import collect_clip_trace, save_trace, trace_manifest_payload, write_trace_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache per-hop realtime traces for replay-based tuning.")
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
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
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
    parser.add_argument("--cache-root", type=str, default="cache/realtime_traces")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _records_for_split(split: str, *, limit_per_class: int) -> list:
    records = read_manifest(Path("data/processed/manifests") / f"local_{split}.jsonl")
    if int(limit_per_class) <= 0:
        return [rec for rec in records if rec.command_label is not None and int(rec.command_label) != IGNORE_INDEX]
    per_class: dict[int, int] = {}
    selected = []
    for rec in records:
        if rec.command_label is None or int(rec.command_label) == IGNORE_INDEX:
            continue
        label = int(rec.command_label)
        seen = per_class.get(label, 0)
        if seen >= int(limit_per_class):
            continue
        per_class[label] = seen + 1
        selected.append(rec)
    return selected


def main() -> None:
    args = parse_args()
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
    records = _records_for_split(args.split, limit_per_class=int(args.limit_per_class))
    split_root = Path(args.cache_root).expanduser().resolve() / str(args.split)
    split_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for idx, record in enumerate(records):
        trace_path = split_root / f"{idx:06d}.npz"
        if trace_path.exists() and not bool(args.overwrite):
            entry = {
                "trace_path": str(trace_path),
                "record_path": str(Path(record.path).resolve()),
                "target_kws12": int(command31_to_kws12(INDEX_TO_COMMAND31[int(record.command_label)])),
            }
            entries.append(entry)
            continue
        trace = collect_clip_trace(bundle=bundle, record=record, args=args)
        save_trace(trace_path, trace)
        entries.append(
            {
                "trace_path": str(trace_path),
                "record_path": str(Path(record.path).resolve()),
                "target_kws12": int(trace["target_kws12"]),
            }
        )
    manifest = trace_manifest_payload(split=args.split, bundle=bundle, args=args, entries=entries)
    manifest_path = write_trace_manifest(split_root / "manifest.json", manifest)
    print(json.dumps({"cache_root": str(split_root), "manifest": str(manifest_path), "num_eval_samples": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
