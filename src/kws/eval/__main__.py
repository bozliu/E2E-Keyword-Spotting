"""CLI entrypoint for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kws.constants import IGNORE_INDEX, KWS12_LABELS, command31_to_kws12
from kws.data.pipeline import create_dataloaders, prepare_data
from kws.demo.verifier_runtime import load_runtime_verifier
from kws.env import ensure_repo_import, run_repo_preflight
from kws.eval.fusion import compute_fused_payload, compute_hybrid_fused_payload
from kws.external import DEFAULT_EXTERNAL_VERIFIER_MODEL_ID, collect_external_probs_from_loader, fit_external_verifier_calibration
from kws.models import create_model
from kws.train.engine import pick_device, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-task KWS checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--verifier-checkpoint", type=str, default="auto", help="Path to verifier checkpoint, 'auto', or empty to disable")
    parser.add_argument("--verifier-backend", type=str, default="internal", choices=["internal", "external", "hybrid"])
    parser.add_argument("--external-verifier-model", type=str, default=DEFAULT_EXTERNAL_VERIFIER_MODEL_ID)
    parser.add_argument("--external-verifier-device", type=str, default="auto")
    parser.add_argument("--decision-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def _build_manifest_names(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    external_cfg = data_cfg.get("external", {}) if isinstance(data_cfg, dict) else {}
    manifests = ["local_train.jsonl", "local_valid.jsonl", "local_test.jsonl"]
    hi_cfg = external_cfg.get("hi_mia", {})
    if isinstance(hi_cfg, dict) and bool(hi_cfg.get("enabled", False)):
        manifests.extend(["hi_mia_train.jsonl", "hi_mia_valid.jsonl", "hi_mia_test.jsonl"])
    mswc_cfg = external_cfg.get("mswc", {})
    if isinstance(mswc_cfg, dict) and bool(mswc_cfg.get("enabled", False)):
        manifests.extend(["mswc_train.jsonl", "mswc_valid.jsonl", "mswc_test.jsonl"])
    return manifests


def _resolve_verifier_arg(value: str | None) -> str | Path | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "off", "disable", "disabled"}:
        return None
    if raw.lower() == "auto":
        return None
    return Path(raw).expanduser().resolve()


def _targets_to_kws12_indices(targets: np.ndarray, label_set: list[str]) -> np.ndarray:
    mapped: list[int] = []
    for idx in targets:
        if int(idx) == IGNORE_INDEX:
            continue
        mapped.append(command31_to_kws12(str(label_set[int(idx)])))
    return np.asarray(mapped, dtype=np.int64)


def _collect_external_verifier_artifacts(
    *,
    loaders,
    split: str,
    model_id: str,
    device: str,
    sample_rate: int,
    clip_samples: int,
    label_set: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    split_payload = collect_external_probs_from_loader(
        getattr(loaders, split),
        model_id=model_id,
        device=device,
        sample_rate=sample_rate,
        clip_samples=clip_samples,
    )
    valid_payload = collect_external_probs_from_loader(
        loaders.valid,
        model_id=model_id,
        device=device,
        sample_rate=sample_rate,
        clip_samples=clip_samples,
    )
    valid_targets = np.asarray(valid_payload["targets"], dtype=np.int64)
    valid_mask = valid_targets != IGNORE_INDEX
    calibration = fit_external_verifier_calibration(
        probs=np.asarray(valid_payload["probs"], dtype=np.float32)[valid_mask],
        targets_kws12=_targets_to_kws12_indices(valid_targets[valid_mask], label_set) if valid_mask.any() else np.zeros((0,), dtype=np.int64),
        model_id=model_id,
        backend="external",
        fit_split="valid",
    )
    return np.asarray(split_payload["probs"], dtype=np.float32), calibration


def _collect_outputs(
    *,
    model: torch.nn.Module,
    verifier,
    loader,
    device: torch.device,
    num_commands: int,
) -> dict:
    training = model.training
    verifier_model = None if verifier is None else verifier.model
    verifier_training = False if verifier_model is None else verifier_model.training
    model.eval()
    if verifier_model is not None:
        verifier_model.eval()

    all_command_probs = []
    all_targets = []
    all_verifier_probs = []
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device)
            out = model(features)
            all_command_probs.append(torch.softmax(out.command_logits, dim=-1).detach().cpu().numpy())
            all_targets.append(batch.command_labels.detach().cpu().numpy())
            if verifier_model is not None:
                vout = verifier_model(features.to(verifier.runtime_device))
                all_verifier_probs.append(torch.softmax(vout.logits, dim=-1).detach().cpu().numpy())
    model.train(mode=training)
    if verifier_model is not None:
        verifier_model.train(mode=verifier_training)
    return {
        "command_probs": np.concatenate(all_command_probs, axis=0) if all_command_probs else np.zeros((0, num_commands), dtype=np.float32),
        "targets": np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64),
        "verifier_probs": np.concatenate(all_verifier_probs, axis=0) if all_verifier_probs else None,
    }


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Recover project root from output_dir in config.
    output_dir = ckpt_path.parent
    project_root = output_dir.parent.parent.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_repo_import(repo_root)

    prepare_data(cfg, project_root)
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    manifests_dir_value = str(data_cfg.get("manifests_dir", "data/processed/manifests")) if isinstance(data_cfg, dict) else "data/processed/manifests"
    manifests_dir = (project_root / manifests_dir_value).resolve()
    teacher_cfg = cfg.get("training", {}).get("teacher", {})
    run_repo_preflight(
        project_root,
        manifests_dir=manifests_dir,
        manifest_names=_build_manifest_names(cfg),
        teacher_model_id=str(teacher_cfg.get("model_id", "")).strip() if isinstance(teacher_cfg, dict) and bool(teacher_cfg.get("enabled", False)) else None,
        teacher_cache_dir=(project_root / str(teacher_cfg.get("cache_dir", ""))).resolve() if isinstance(teacher_cfg, dict) and bool(teacher_cfg.get("enabled", False)) and str(teacher_cfg.get("cache_dir", "")).strip() else None,
        require_mps=False,
    )
    loaders = create_dataloaders(cfg, project_root)

    device = pick_device(args.device)
    model = create_model(
        cfg["model"],
        n_mels=int(cfg["features"]["n_mels"]),
        num_commands=len(checkpoint["label_set"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    loader = getattr(loaders, args.split)
    verifier = None
    if args.verifier_backend in {"internal", "hybrid"}:
        verifier = load_runtime_verifier(
            ckpt_path,
            device=device,
            verifier_path=_resolve_verifier_arg(args.verifier_checkpoint),
        )
    result = run_epoch(
        model=model,
        loader=loader,
        device=device,
        optimizer=None,
        lambda_command=1.0,
        lambda_kws12=float(cfg.get("training", {}).get("loss_weights", {}).get("kws12", 0.0)),
        lambda_wake=1.0,
        lambda_aux=0.0,
        lambda_confusion=0.0,
        aux_margin=0.2,
        audio_seconds=float(cfg["features"].get("audio_seconds", 1.0)),
    )
    collected = _collect_outputs(
        model=model,
        verifier=verifier,
        loader=loader,
        device=device,
        num_commands=len(checkpoint["label_set"]),
    )
    external_probs = None
    external_calibration: dict[str, object] = {}
    if args.verifier_backend in {"external", "hybrid"}:
        external_probs, external_calibration = _collect_external_verifier_artifacts(
            loaders=loaders,
            split=args.split,
            model_id=args.external_verifier_model,
            device=args.external_verifier_device,
            sample_rate=int(cfg["features"].get("sample_rate", 16_000)),
            clip_samples=int(round(float(cfg["features"].get("audio_seconds", 1.0)) * int(cfg["features"].get("sample_rate", 16_000)))),
            label_set=checkpoint["label_set"],
        )
    valid_mask = collected["targets"] != IGNORE_INDEX
    targets_kws12 = _targets_to_kws12_indices(collected["targets"][valid_mask], checkpoint["label_set"])
    if args.verifier_backend == "hybrid":
        fused = compute_hybrid_fused_payload(
            command_probs=collected["command_probs"][valid_mask],
            command31_labels=checkpoint["label_set"],
            targets_kws12=targets_kws12,
            internal_verifier_probs=None if collected["verifier_probs"] is None else collected["verifier_probs"][valid_mask],
            internal_verifier_labels=None if verifier is None else verifier.labels,
            internal_verifier_calibration=None if verifier is None else verifier.calibration,
            external_verifier_probs=None if external_probs is None else external_probs[valid_mask],
            external_verifier_labels=KWS12_LABELS,
            external_verifier_calibration=external_calibration,
            decision_profile=args.decision_profile,
        )
    else:
        active_probs = None
        active_labels = None
        active_calibration = None
        if args.verifier_backend == "external":
            active_probs = None if external_probs is None else external_probs[valid_mask]
            active_labels = KWS12_LABELS
            active_calibration = external_calibration
        else:
            active_probs = None if collected["verifier_probs"] is None else collected["verifier_probs"][valid_mask]
            active_labels = None if verifier is None else verifier.labels
            active_calibration = None if verifier is None else verifier.calibration
        fused = compute_fused_payload(
            command_probs=collected["command_probs"][valid_mask],
            command31_labels=checkpoint["label_set"],
            targets_kws12=targets_kws12,
            verifier_probs=active_probs,
            verifier_labels=active_labels,
            verifier_calibration=active_calibration,
            decision_profile=args.decision_profile,
        )

    payload = {
        "split": args.split,
        "loss": result.loss,
        "metrics": result.metrics,
        "per_class_kws12": result.metrics.get("per_class_kws12", {}),
        "min_kws12_precision": float(result.metrics.get("min_kws12_precision", 0.0)),
        "min_kws12_recall": float(result.metrics.get("min_kws12_recall", 0.0)),
        "verifier_backend": args.verifier_backend,
        "verifier_checkpoint": str(verifier.checkpoint_path) if verifier is not None else "",
        "external_verifier_model_id": str(args.external_verifier_model) if args.verifier_backend in {"external", "hybrid"} else "",
        "external_verifier_device": str(args.external_verifier_device) if args.verifier_backend in {"external", "hybrid"} else "",
        "decision_profile": args.decision_profile,
        "fused_metrics": fused["fused_metrics"],
        "fused_per_class_kws12": fused["fused_metrics"].get("per_class_kws12", {}),
        "fused_min_kws12_precision": float(fused["fused_metrics"].get("min_kws12_precision", 0.0)),
        "fused_min_kws12_recall": float(fused["fused_metrics"].get("min_kws12_recall", 0.0)),
        "fused_unknown_to_target_rate": float(fused["fused_metrics"].get("kws12_unknown_to_target_rate", 0.0)),
        "verify_rate": float(fused.get("verify_rate", 0.0)),
        "verifier_accept_rate": float(fused.get("verifier_accept_rate", 0.0)),
        "external_verify_rate": float(fused.get("verify_rate", 0.0)) if args.verifier_backend in {"external", "hybrid"} else 0.0,
        "external_verifier_accept_rate": float(fused.get("external_verifier_accept_rate", fused.get("verifier_accept_rate", 0.0))) if args.verifier_backend in {"external", "hybrid"} else 0.0,
        "external_verifier_calibration": external_calibration if args.verifier_backend in {"external", "hybrid"} else {},
    }

    out_path = Path(args.output).expanduser().resolve() if args.output else output_dir / f"eval_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
