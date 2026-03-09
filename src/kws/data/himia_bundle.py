"""Prepare and import full HI-MIA bundle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backports import zstd

from kws.data.audit import audit_manifests
from kws.data.hi_mia import (
    HI_MIA_OPTIONAL_SOURCE_SPLITS,
    HI_MIA_REQUIRED_SOURCE_SPLITS,
    build_himia_manifests_with_status,
)
from kws.env import ensure_repo_import


HI_MIA_MANIFEST_NAMES: tuple[str, ...] = (
    "hi_mia_train.jsonl",
    "hi_mia_valid.jsonl",
    "hi_mia_test.jsonl",
)
HI_MIA_REPORT_FILES: tuple[str, ...] = (
    "reports/hi_mia_remote_audit.json",
    "reports/hi_mia_import_audit.json",
)


def _resolve_project_root(value: str | Path | None = None) -> Path:
    root = Path(value or Path.cwd()).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root.resolve()


def _resolve_path(project_root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path.resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hi_mia_paths(project_root: Path) -> dict[str, Path]:
    manifests_root = (project_root / "data" / "processed" / "manifests").resolve()
    return {
        "project_root": project_root,
        "data_root": (project_root / "data" / "external" / "hi_mia").resolve(),
        "manifests_root": manifests_root,
        "status_path": (manifests_root / "hi_mia_status.json").resolve(),
        "stats_path": (manifests_root / "hi_mia_stats.json").resolve(),
    }


def load_himia_status(project_root: Path) -> dict:
    paths = hi_mia_paths(project_root)
    status_path = paths["status_path"]
    if not status_path.exists():
        raise FileNotFoundError(f"HI-MIA status file not found: {status_path}")
    return _load_json(status_path)


def is_himia_full(status: dict) -> bool:
    if bool(status.get("reduced_data_mode", True)):
        return False
    if str(status.get("build_mode", "")).strip() != "official":
        return False
    present = {str(item) for item in status.get("present_source_splits", [])}
    return all(split in present for split in HI_MIA_REQUIRED_SOURCE_SPLITS)


def assert_himia_full(status: dict) -> None:
    if not is_himia_full(status):
        raise RuntimeError(
            "HI-MIA restore is not full. "
            f"reduced_data_mode={status.get('reduced_data_mode')} "
            f"present_source_splits={status.get('present_source_splits')} "
            f"build_mode={status.get('build_mode')}"
        )


def _bundle_members(project_root: Path) -> list[Path]:
    members = [
        project_root / "data" / "external" / "hi_mia" / split
        for split in HI_MIA_REQUIRED_SOURCE_SPLITS
    ]
    for split in HI_MIA_OPTIONAL_SOURCE_SPLITS:
        optional_dir = project_root / "data" / "external" / "hi_mia" / split
        if optional_dir.exists():
            members.append(optional_dir)
    members.extend((project_root / "data" / "processed" / "manifests" / name) for name in HI_MIA_MANIFEST_NAMES)
    members.append(project_root / "data" / "processed" / "manifests" / "hi_mia_status.json")
    members.append(project_root / "data" / "processed" / "manifests" / "hi_mia_stats.json")
    members.append(project_root / "reports" / "hi_mia_remote_audit.json")

    resolved: list[Path] = []
    for path in members:
        candidate = path.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Required HI-MIA bundle member not found: {candidate}")
        resolved.append(candidate)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_himia_stats(manifests_root: Path, manifests: dict[str, list], status: dict) -> dict:
    stats = {
        "manifest_counts": {split: len(records) for split, records in manifests.items()},
        "reduced_data_mode": bool(status.get("reduced_data_mode", False)),
        "present_source_splits": list(status.get("present_source_splits", [])),
        "source_file_counts": dict(status.get("source_file_counts", {})),
        "build_mode": str(status.get("build_mode", "")),
    }
    _write_json(manifests_root / "hi_mia_stats.json", stats)
    return stats


def describe_himia_status(project_root: Path) -> dict:
    status = load_himia_status(project_root)
    return {
        "project_root": str(project_root),
        "status": status,
        "is_full": bool(is_himia_full(status)),
    }


def prepare_himia_bundle(
    project_root: str | Path,
    *,
    bundle_path: str | Path,
    manifest_path: str | Path,
    audit_output_path: str | Path,
) -> dict:
    root = _resolve_project_root(project_root)
    ensure_repo_import(Path(__file__).resolve().parents[3])
    status = load_himia_status(root)
    assert_himia_full(status)

    manifests_root = (root / "data" / "processed" / "manifests").resolve()
    audit = audit_manifests(manifests_root, manifest_names=HI_MIA_MANIFEST_NAMES)
    if not bool(audit.get("is_clean", False)):
        raise RuntimeError("HI-MIA manifest audit must be clean before bundling.")
    audit_output = _resolve_path(root, audit_output_path, "reports/hi_mia_remote_audit.json")
    _write_json(audit_output, audit)

    bundle = _resolve_path(root, bundle_path, "artifacts/hi_mia_full_bundle.tar.zst")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        bundle.unlink()

    members = _bundle_members(root)
    with zstd.open(bundle, "wb") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for member in members:
                tar.add(member, arcname=member.relative_to(root).as_posix(), recursive=True)

    manifest = {
        "bundle_path": str(bundle),
        "bundle_size_bytes": int(bundle.stat().st_size),
        "bundle_sha256": _sha256(bundle),
        "project_root": str(root),
        "status": status,
        "audit_is_clean": bool(audit.get("is_clean", False)),
        "audit_output_path": str(audit_output),
        "included_members": [member.relative_to(root).as_posix() for member in members],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    manifest_out = _resolve_path(root, manifest_path, "artifacts/hi_mia_full_bundle_manifest.json")
    _write_json(manifest_out, manifest)
    return manifest


def _extract_bundle(bundle_path: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    with zstd.open(bundle_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as tar:
            for member in tar:
                member_path = (destination_root / member.name).resolve()
                if not member_path.is_relative_to(destination_root.resolve()):
                    raise ValueError(f"Refusing to extract outside destination root: {member.name}")
                if member.islnk() or member.issym():
                    raise ValueError(f"Refusing to extract symlink member: {member.name}")
                tar.extract(member, path=destination_root, filter="data")


def _move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _backup_current_himia_tree(project_root: Path, backup_root: Path) -> None:
    paths = hi_mia_paths(project_root)
    _move_if_exists(paths["data_root"], backup_root / "data" / "external" / "hi_mia")
    for name in (*HI_MIA_MANIFEST_NAMES, "hi_mia_status.json", "hi_mia_stats.json"):
        _move_if_exists(paths["manifests_root"] / name, backup_root / "data" / "processed" / "manifests" / name)
    for rel_path in HI_MIA_REPORT_FILES:
        _move_if_exists(project_root / rel_path, backup_root / rel_path)


def import_himia_bundle(
    project_root: str | Path,
    *,
    bundle_path: str | Path,
    backup_root: str | Path | None = None,
    audit_output_path: str | Path = "reports/hi_mia_import_audit.json",
) -> dict:
    root = _resolve_project_root(project_root)
    ensure_repo_import(Path(__file__).resolve().parents[3])
    bundle = _resolve_path(root, bundle_path, "")
    if not bundle.exists():
        raise FileNotFoundError(f"HI-MIA bundle not found: {bundle}")

    cache_root = (root / ".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="himia_bundle_", dir=str(cache_root)) as tmpdir:
        staging_root = Path(tmpdir).resolve()
        _extract_bundle(bundle, staging_root)

        staged_status_path = staging_root / "data" / "processed" / "manifests" / "hi_mia_status.json"
        if not staged_status_path.exists():
            raise FileNotFoundError(f"Bundle does not contain hi_mia_status.json: {bundle}")
        staged_status = _load_json(staged_status_path)
        assert_himia_full(staged_status)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = _resolve_path(root, backup_root, f"backups/hi_mia_import_{timestamp}")
        _backup_current_himia_tree(root, backup_dir)

        staged_himia_root = staging_root / "data" / "external" / "hi_mia"
        if not staged_himia_root.exists():
            raise FileNotFoundError(f"Bundle does not contain data/external/hi_mia: {bundle}")
        target_himia_root = root / "data" / "external" / "hi_mia"
        target_himia_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_himia_root), str(target_himia_root))

        staged_reports_root = staging_root / "reports"
        staged_manifests_root = staging_root / "data" / "processed" / "manifests"
        target_manifests_root = root / "data" / "processed" / "manifests"
        target_manifests_root.mkdir(parents=True, exist_ok=True)
        for name in (*HI_MIA_MANIFEST_NAMES, "hi_mia_status.json", "hi_mia_stats.json"):
            staged_file = staged_manifests_root / name
            if staged_file.exists():
                shutil.copy2(staged_file, target_manifests_root / name)
        staged_remote_audit = staged_reports_root / "hi_mia_remote_audit.json"
        if staged_remote_audit.exists():
            reports_root = root / "reports"
            reports_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_remote_audit, reports_root / "hi_mia_remote_audit.json")

    manifests, status = build_himia_manifests_with_status(
        root / "data" / "external" / "hi_mia",
        root / "data" / "processed" / "manifests",
        limit_per_split=None,
    )
    assert_himia_full(status)
    stats = _write_himia_stats(root / "data" / "processed" / "manifests", manifests, status)

    audit_output = _resolve_path(root, audit_output_path, "reports/hi_mia_import_audit.json")
    audit = audit_manifests(root / "data" / "processed" / "manifests")
    _write_json(audit_output, audit)
    if not bool(audit.get("is_clean", False)):
        raise RuntimeError("Local manifest audit must be clean after HI-MIA import.")

    return {
        "bundle_path": str(bundle),
        "backup_root": str(backup_dir),
        "audit_output_path": str(audit_output),
        "status": status,
        "stats": stats,
        "audit_is_clean": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or import a full HI-MIA bundle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Inspect the current HI-MIA restore status")
    status_parser.add_argument("--project-root", type=str, default=".")
    status_parser.add_argument("--output", type=str, default="")
    status_parser.add_argument("--require-full", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="Package the current full HI-MIA restore into a tar.zst bundle")
    prepare_parser.add_argument("--project-root", type=str, default=".")
    prepare_parser.add_argument("--bundle-path", type=str, default="artifacts/hi_mia_full_bundle.tar.zst")
    prepare_parser.add_argument("--manifest-path", type=str, default="artifacts/hi_mia_full_bundle_manifest.json")
    prepare_parser.add_argument("--audit-output", type=str, default="reports/hi_mia_remote_audit.json")

    import_parser = subparsers.add_parser("import", help="Import a full HI-MIA bundle into the current repo")
    import_parser.add_argument("--project-root", type=str, default=".")
    import_parser.add_argument("--bundle-path", type=str, required=True)
    import_parser.add_argument("--backup-root", type=str, default="")
    import_parser.add_argument("--audit-output", type=str, default="reports/hi_mia_import_audit.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _resolve_project_root(args.project_root)
    repo_root = Path(__file__).resolve().parents[3]
    ensure_repo_import(repo_root)

    if args.command == "status":
        payload = describe_himia_status(project_root)
        if args.output:
            _write_json(_resolve_path(project_root, args.output, ""), payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.require_full and not bool(payload.get("is_full", False)):
            raise SystemExit(1)
        return

    if args.command == "prepare":
        payload = prepare_himia_bundle(
            project_root,
            bundle_path=args.bundle_path,
            manifest_path=args.manifest_path,
            audit_output_path=args.audit_output,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "import":
        payload = import_himia_bundle(
            project_root,
            bundle_path=args.bundle_path,
            backup_root=args.backup_root or None,
            audit_output_path=args.audit_output,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
