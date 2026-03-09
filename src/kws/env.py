"""Environment alignment helpers for repo-local execution."""

from __future__ import annotations

from collections.abc import Iterable
import sys
from pathlib import Path

import torch

import kws
from kws.data.audit import audit_manifests


def active_kws_file() -> Path:
    path = getattr(kws, "__file__", "")
    return Path(path).expanduser().resolve()


def ensure_repo_import(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    expected_src = root / "src"
    actual = active_kws_file()
    if expected_src in actual.parents:
        return actual

    path_preview = "\n".join(f"- {entry}" for entry in sys.path[:12])
    raise RuntimeError(
        "The active 'kws' import does not resolve to the current repo.\n"
        f"expected under: {expected_src}\n"
        f"actual module: {actual}\n"
        "Fix the environment before running training or evaluation.\n"
        f"sys.path preview:\n{path_preview}"
    )


def ensure_mps_available() -> bool:
    available = bool(torch.backends.mps.is_available())
    if not available:
        raise RuntimeError("MPS is required for Phase 3 training, but torch.backends.mps.is_available() is False.")
    return available


def ensure_teacher_model_loadable(model_id: str, *, cache_dir: str | Path | None = None) -> str:
    model_id = str(model_id).strip()
    if not model_id:
        raise ValueError("teacher model id must be a non-empty string")
    try:
        from transformers import AutoModel
    except Exception as exc:  # pragma: no cover - dependency error
        raise ImportError(
            "transformers is required for SSL teacher preflight. Install the pinned environment first."
        ) from exc

    kwargs = {"use_safetensors": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    try:
        AutoModel.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the SSL teacher during preflight. "
            f"Model id: {model_id}. Resolve this before training or final evaluation."
        ) from exc
    return model_id


def ensure_manifest_audit_clean(
    manifests_dir: str | Path,
    *,
    manifest_names: Iterable[str] | None = None,
) -> dict:
    report = audit_manifests(manifests_dir, manifest_names=manifest_names)
    if not bool(report.get("is_clean", False)):
        raise RuntimeError(
            "Manifest audit failed during preflight. "
            f"See manifests under {Path(manifests_dir).expanduser().resolve()}."
        )
    return report


def run_repo_preflight(
    project_root: str | Path,
    *,
    manifests_dir: str | Path,
    manifest_names: Iterable[str] | None = None,
    teacher_model_id: str | None = None,
    teacher_cache_dir: str | Path | None = None,
    require_mps: bool = False,
) -> dict:
    root = Path(project_root).expanduser().resolve()
    active = ensure_repo_import(root)
    manifests_root = Path(manifests_dir).expanduser()
    if not manifests_root.is_absolute():
        manifests_root = (root / manifests_root).resolve()
    else:
        manifests_root = manifests_root.resolve()

    report = {
        "active_kws_file": str(active),
        "project_root": str(root),
        "mps_available": bool(torch.backends.mps.is_available()),
        "manifest_audit": ensure_manifest_audit_clean(manifests_root, manifest_names=manifest_names),
        "teacher_model_id": None,
    }
    if require_mps:
        ensure_mps_available()
    if teacher_model_id:
        report["teacher_model_id"] = ensure_teacher_model_loadable(
            teacher_model_id,
            cache_dir=teacher_cache_dir,
        )
    return report
