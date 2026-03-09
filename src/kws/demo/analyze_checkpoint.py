"""Offline evaluation + latency report for demo checkpoint selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from kws.constants import IGNORE_INDEX, KWS12_LABELS, command31_to_kws12
from kws.data.audio import MelFrontend, load_audio, pad_or_trim
from kws.data.dataset import load_manifests
from kws.data.pipeline import create_dataloaders, prepare_data
from kws.demo.rank_checkpoints import benchmark_latency_ms
from kws.demo.verifier_runtime import load_runtime_verifier
from kws.env import ensure_repo_import, run_repo_preflight
from kws.eval.fusion import compute_fused_payload, compute_hybrid_fused_payload
from kws.eval.stress import run_stress_eval
from kws.external import DEFAULT_EXTERNAL_VERIFIER_MODEL_ID, collect_external_probs_from_loader, fit_external_verifier_calibration
from kws.models import create_model
from kws.train.engine import pick_device, run_epoch
from kws.utils.keyword_focus import build_keyword_focus_report, compute_accent_slices, fit_keyword_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a checkpoint for demo use.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--verifier-checkpoint", type=str, default="auto", help="Path to verifier checkpoint, 'auto', or empty to disable")
    parser.add_argument("--verifier-backend", type=str, default="internal", choices=["internal", "external", "hybrid"])
    parser.add_argument("--external-verifier-model", type=str, default=DEFAULT_EXTERNAL_VERIFIER_MODEL_ID)
    parser.add_argument("--external-verifier-device", type=str, default="auto")
    parser.add_argument("--decision-profile", type=str, default="stable", choices=["stable", "balanced", "fast"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def _baseline_unknown_guardrail(project_root: Path) -> float:
    metrics_path = (project_root / "outputs" / "quick_mhatt" / "metrics_history.jsonl").resolve()
    baseline = 1.0
    if not metrics_path.exists():
        return baseline + 0.02
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            valid = row.get("valid_metrics", {})
            baseline = min(baseline, float(valid.get("kws12_unknown_to_target_rate", 1.0)))
    return float(baseline + 0.02)


def _build_manifest_names(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    external_cfg = data_cfg.get("external", {}) if isinstance(data_cfg, dict) else {}
    manifests = ["local_train.jsonl", "local_valid.jsonl", "local_test.jsonl"]
    hi_cfg = external_cfg.get("hi_mia", {})
    if isinstance(hi_cfg, dict) and bool(hi_cfg.get("enabled", False)):
        manifests.extend(["hi_mia_train.jsonl", "hi_mia_valid.jsonl", "hi_mia_test.jsonl"])
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


def _collect_outputs(model: torch.nn.Module, loader, device: torch.device, *, num_commands: int) -> Dict[str, np.ndarray]:
    training = model.training
    model.eval()
    all_probs = []
    all_wake = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device)
            targets = batch.command_labels.to(device)
            out = model(features)
            all_probs.append(torch.softmax(out.command_logits, dim=-1).detach().cpu().numpy())
            all_wake.append(torch.sigmoid(out.wake_logits).detach().cpu().numpy())
            all_preds.append(out.command_logits.detach().argmax(dim=-1).cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    model.train(mode=training)
    return {
        "command_probs": np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_commands), dtype=np.float32),
        "wake_probs": np.concatenate(all_wake, axis=0) if all_wake else np.zeros((0,), dtype=np.float32),
        "preds": np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,), dtype=np.int64),
        "targets": np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64),
    }


def _collect_verifier_probs(verifier, loader) -> np.ndarray | None:
    if verifier is None:
        return None
    model = verifier.model
    training = model.training
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(verifier.runtime_device)
            out = model(features)
            all_probs.append(torch.softmax(out.logits, dim=-1).detach().cpu().numpy())
    model.train(mode=training)
    return np.concatenate(all_probs, axis=0) if all_probs else None


def _predict_records(
    records,
    *,
    model: torch.nn.Module,
    frontend: MelFrontend,
    device: torch.device,
    audio_seconds: float,
    num_commands: int,
) -> Dict[str, np.ndarray]:
    command_probs = []
    wake_probs = []
    preds = []
    targets = []
    target_samples = int(round(frontend.sample_rate * audio_seconds))
    training = model.training
    model.eval()
    with torch.no_grad():
        for rec in records:
            if rec.command_label is None:
                continue
            waveform = load_audio(rec.path, sample_rate=frontend.sample_rate)
            waveform = pad_or_trim(waveform, target_samples=target_samples)
            feat = frontend(waveform)
            feat = (feat - feat.mean()) / feat.std().clamp(min=1e-5)
            out = model(feat.unsqueeze(0).to(device))
            command_probs.append(torch.softmax(out.command_logits, dim=-1).squeeze(0).cpu().numpy())
            wake_probs.append(torch.sigmoid(out.wake_logits).squeeze(0).cpu().numpy())
            preds.append(int(out.command_logits.argmax(dim=-1).item()))
            targets.append(int(rec.command_label))
    model.train(mode=training)
    return {
        "command_probs": np.asarray(command_probs, dtype=np.float32).reshape(-1, num_commands) if command_probs else np.zeros((0, num_commands), dtype=np.float32),
        "wake_probs": np.asarray(wake_probs, dtype=np.float32).reshape(-1) if wake_probs else np.zeros((0,), dtype=np.float32),
        "preds": np.asarray(preds, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    project_root = ckpt_path.parent.parent.parent.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_repo_import(repo_root)

    prepare_data(cfg, project_root)
    manifests_dir = (project_root / cfg["data"]["manifests_dir"]).resolve()
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
    loader = getattr(loaders, args.split)
    manifests_root = (project_root / cfg["data"]["manifests_dir"]).resolve()
    use_hi_mia = bool(cfg["data"]["external"]["hi_mia"].get("enabled", False))
    manifest_paths = [manifests_root / f"local_{args.split}.jsonl"]
    if use_hi_mia:
        manifest_paths.append(manifests_root / f"hi_mia_{args.split}.jsonl")
    records = load_manifests(manifest_paths)

    eval_device = pick_device("cpu")
    model = create_model(
        cfg["model"],
        n_mels=int(cfg["features"]["n_mels"]),
        num_commands=len(checkpoint["label_set"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(eval_device)
    verifier = None
    if args.verifier_backend in {"internal", "hybrid"}:
        verifier = load_runtime_verifier(
            ckpt_path,
            device=eval_device,
            verifier_path=_resolve_verifier_arg(args.verifier_checkpoint),
        )
    frontend = MelFrontend(
        sample_rate=int(cfg["features"]["sample_rate"]),
        n_fft=int(cfg["features"]["n_fft"]),
        hop_length=int(cfg["features"]["hop_length"]),
        n_mels=int(cfg["features"]["n_mels"]),
        f_min=float(cfg["features"].get("f_min", 20.0)),
        f_max=float(cfg["features"].get("f_max", 7600.0)),
    )

    result = run_epoch(
        model=model,
        loader=loader,
        device=eval_device,
        optimizer=None,
        lambda_command=1.0,
        lambda_kws12=float(cfg.get("training", {}).get("loss_weights", {}).get("kws12", 0.0)),
        lambda_wake=1.0,
        lambda_aux=0.0,
        lambda_confusion=0.0,
        aux_margin=float(cfg.get("training", {}).get("aux_margin", 0.2)),
        audio_seconds=float(cfg["features"].get("audio_seconds", 1.0)),
    )
    collected = _collect_outputs(model, loader, eval_device, num_commands=len(checkpoint["label_set"]))
    verifier_probs = _collect_verifier_probs(verifier, loader)
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
    keyword_focus_cfg = cfg.get("training", {}).get("keyword_focus", {})
    if not isinstance(keyword_focus_cfg, dict):
        keyword_focus_cfg = {}
    keyword_focus = build_keyword_focus_report(
        collected["preds"],
        collected["targets"],
        top_k=int(keyword_focus_cfg.get("top_k", 5)),
        focus_keywords=keyword_focus_cfg.get("focus_keywords"),
        focus_pairs=keyword_focus_cfg.get("focus_pairs"),
    )
    keyword_calibration = fit_keyword_calibration(
        collected["command_probs"],
        collected["wake_probs"],
        collected["targets"],
        focus=keyword_focus,
    )
    accent_slices = compute_accent_slices(records, collected["preds"], collected["targets"])
    l2_cfg = cfg.get("data", {}).get("external", {}).get("l2_arctic_eval", {})
    if isinstance(l2_cfg, dict) and bool(l2_cfg.get("enabled", False)):
        l2_manifest = manifests_root / "l2_arctic_eval_test.jsonl"
        l2_records = load_manifests([l2_manifest])
        if l2_records:
            l2_outputs = _predict_records(
                l2_records,
                model=model,
                frontend=frontend,
                device=eval_device,
                audio_seconds=float(cfg["features"].get("audio_seconds", 1.0)),
                num_commands=len(checkpoint["label_set"]),
            )
            accent_slices.update(compute_accent_slices(l2_records, l2_outputs["preds"], l2_outputs["targets"]))

    latency = {
        "cpu": benchmark_latency_ms(checkpoint, torch.device("cpu")),
    }
    if torch.backends.mps.is_available():
        latency["mps"] = benchmark_latency_ms(checkpoint, torch.device("mps"))

    stress_eval = run_stress_eval(
        records=records,
        model=model,
        frontend=frontend,
        device=eval_device,
        audio_seconds=float(cfg["features"].get("audio_seconds", 1.0)),
    )
    fp_guardrail = _baseline_unknown_guardrail(project_root)
    valid_mask = collected["targets"] != IGNORE_INDEX
    targets_kws12 = _targets_to_kws12_indices(collected["targets"][valid_mask], checkpoint["label_set"])
    if args.verifier_backend == "hybrid":
        fused = compute_hybrid_fused_payload(
            command_probs=collected["command_probs"][valid_mask],
            command31_labels=checkpoint["label_set"],
            targets_kws12=targets_kws12,
            internal_verifier_probs=None if verifier_probs is None else verifier_probs[valid_mask],
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
            active_probs = None if verifier_probs is None else verifier_probs[valid_mask]
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
        "checkpoint": str(ckpt_path),
        "split": args.split,
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
        "per_keyword": keyword_focus["per_keyword"],
        "bottom3_keyword_recall": keyword_focus["bottom3_keyword_recall"],
        "keyword_balance_gap": keyword_focus["keyword_balance_gap"],
        "focus_keyword_recall_mean": keyword_focus["focus_keyword_recall_mean"],
        "focus_pair_confusions": keyword_focus["focus_pair_confusions"],
        "focus_pair_confusion_rate": keyword_focus["focus_pair_confusion_rate"],
        "weak_keywords": keyword_focus["weak_keywords"],
        "keyword_focus": keyword_focus,
        "keyword_calibration": keyword_calibration,
        "latency_ms": latency,
        "selection_passed_fp_guardrail": bool(float(result.metrics.get("kws12_unknown_to_target_rate", 1.0)) <= fp_guardrail),
        "fp_guardrail_threshold": fp_guardrail,
        "accent_slices": accent_slices,
        "stress_eval": stress_eval,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else ckpt_path.parent / "demo_analysis.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (ckpt_path.parent / "keyword_focus.json").write_text(json.dumps(keyword_focus, indent=2, ensure_ascii=False), encoding="utf-8")
    (ckpt_path.parent / "keyword_calibration.json").write_text(
        json.dumps(keyword_calibration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.verifier_backend in {"external", "hybrid"}:
        (ckpt_path.parent / "external_verifier_calibration.json").write_text(
            json.dumps(external_calibration, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
