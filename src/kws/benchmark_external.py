"""Benchmark imported external HF KWS models on the local protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kws.constants import INDEX_TO_COMMAND31, IGNORE_INDEX, command31_to_kws12
from kws.external import benchmark_external_latency_ms, predict_kws12_from_paths, slugify_model_id
from kws.train.metrics import compute_kws12_breakdown_from_indices
from kws.data.manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an imported external HF KWS model")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--manifests-dir", type=str, default="data/processed/manifests")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _targets_to_kws12(records) -> tuple[list[str], np.ndarray]:
    paths: list[str] = []
    targets: list[int] = []
    for rec in records:
        if rec.command_label is None or int(rec.command_label) == IGNORE_INDEX:
            continue
        paths.append(str(rec.path))
        targets.append(command31_to_kws12(INDEX_TO_COMMAND31[int(rec.command_label)]))
    return paths, np.asarray(targets, dtype=np.int64)


def main() -> None:
    args = parse_args()
    manifests_dir = Path(args.manifests_dir).expanduser().resolve()
    manifest_path = manifests_dir / f"local_{args.split}.jsonl"
    records = read_manifest(manifest_path)
    paths, targets = _targets_to_kws12(records)

    all_probs = []
    for start in range(0, len(paths), max(1, int(args.batch_size))):
        batch_paths = paths[start : start + max(1, int(args.batch_size))]
        result = predict_kws12_from_paths(
            batch_paths,
            model_id=args.model_id,
            device=args.device,
        )
        all_probs.append(result.probs)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 12), dtype=np.float32)
    preds = probs.argmax(axis=1).astype(np.int64, copy=False) if probs.size else np.zeros((0,), dtype=np.int64)
    metrics = compute_kws12_breakdown_from_indices(preds, targets)

    payload = {
        "model_id": str(args.model_id),
        "split": args.split,
        "device": args.device,
        "num_eval_samples": int(targets.size),
        "per_class_kws12": metrics.get("per_class_kws12", {}),
        "min_kws12_precision": float(metrics.get("min_kws12_precision", 0.0)),
        "min_kws12_recall": float(metrics.get("min_kws12_recall", 0.0)),
        "unknown_to_target_rate": float(metrics.get("kws12_unknown_to_target_rate", 0.0)),
        "avg_ms_cpu_1x1s": float(benchmark_external_latency_ms(model_id=args.model_id, device="cpu")),
        "avg_ms_mps_1x1s": float(benchmark_external_latency_ms(model_id=args.model_id, device="mps")) if torch.backends.mps.is_available() else None,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / "reports" / f"benchmark_{slugify_model_id(args.model_id)}_{args.split}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
