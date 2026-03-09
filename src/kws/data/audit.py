"""Manifest audit helpers and CLI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from kws.data.manifest import read_manifest


def _resolve_manifest_paths(manifests_dir: Path, manifest_names: Iterable[str] | None) -> List[Path]:
    names = [name for name in manifest_names or [] if name]
    if names:
        resolved: List[Path] = []
        for name in names:
            path = Path(name).expanduser()
            if not path.is_absolute():
                path = manifests_dir / path
            resolved.append(path.resolve())
        return resolved
    return sorted(manifests_dir.glob("*.jsonl"))


def audit_manifests(
    manifests_dir: str | Path,
    *,
    manifest_names: Iterable[str] | None = None,
    sample_missing_limit: int = 10,
) -> dict:
    """Audit manifest presence and referenced audio paths."""
    manifests_root = Path(manifests_dir).expanduser().resolve()
    manifest_paths = _resolve_manifest_paths(manifests_root, manifest_names)

    report = {
        "manifests_dir": str(manifests_root),
        "missing_manifest_files": [],
        "manifests": {},
        "aggregate": {
            "record_count": 0,
            "missing_file_count": 0,
            "source_counts": {},
            "split_counts": {},
        },
        "is_clean": True,
    }

    if not manifest_paths:
        report["missing_manifest_files"].append(str(manifests_root / "*.jsonl"))
        report["is_clean"] = False
        return report

    aggregate_sources: Counter[str] = Counter()
    aggregate_splits: Counter[str] = Counter()

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            report["missing_manifest_files"].append(str(manifest_path))
            report["is_clean"] = False
            continue

        records = read_manifest(manifest_path)
        source_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        missing_files: List[str] = []
        for record in records:
            source_counts[record.source] += 1
            split_counts[record.split] += 1
            if not Path(record.path).exists():
                missing_files.append(record.path)

        aggregate_sources.update(source_counts)
        aggregate_splits.update(split_counts)
        report["aggregate"]["record_count"] += len(records)
        report["aggregate"]["missing_file_count"] += len(missing_files)

        report["manifests"][manifest_path.name] = {
            "record_count": len(records),
            "missing_file_count": len(missing_files),
            "source_counts": dict(sorted(source_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "missing_file_examples": missing_files[:sample_missing_limit],
        }
        if missing_files:
            report["is_clean"] = False

    report["aggregate"]["source_counts"] = dict(sorted(aggregate_sources.items()))
    report["aggregate"]["split_counts"] = dict(sorted(aggregate_splits.items()))
    if report["missing_manifest_files"]:
        report["is_clean"] = False
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset manifests and referenced audio files")
    parser.add_argument("--manifests-dir", type=str, default="data/processed/manifests")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--sample-missing-limit", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_manifests(
        args.manifests_dir,
        manifest_names=args.manifest,
        sample_missing_limit=args.sample_missing_limit,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not report["is_clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
