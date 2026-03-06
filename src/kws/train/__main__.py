"""CLI entrypoint for training dual-task KWS models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from kws.config import load_yaml
from kws.constants import COMMAND31_LABELS
from kws.data.pipeline import create_dataloaders, prepare_data
from kws.models import create_model
from kws.train.engine import pick_device, run_epoch
from kws.train.teacher import TeacherHeads, WavLMFeatureCache
from kws.utils.keyword_focus import DEFAULT_CONFUSION_GROUPS, DEFAULT_WEAK_KEYWORDS, build_keyword_focus_report, fit_keyword_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-task KWS models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_validation_outputs(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
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
        "command_probs": np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(COMMAND31_LABELS)), dtype=np.float32),
        "wake_probs": np.concatenate(all_wake, axis=0) if all_wake else np.zeros((0,), dtype=np.float32),
        "preds": np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,), dtype=np.int64),
        "targets": np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    bundle = load_yaml(args.config)
    cfg = bundle.raw
    project_root = bundle.path.parent.parent.resolve()

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))
    set_seed(seed)

    device = pick_device(str(cfg["training"].get("device", "auto")))

    stats = prepare_data(cfg, project_root)
    dataloaders = create_dataloaders(cfg, project_root)

    model = create_model(
        cfg["model"],
        n_mels=int(cfg["features"]["n_mels"]),
        num_commands=len(COMMAND31_LABELS),
    ).to(device)

    teacher_cfg = cfg.get("training", {}).get("teacher", {})
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    teacher_cache = None
    teacher_heads = None
    if teacher_enabled:
        student_dim = int(getattr(getattr(model, "command_head", None), "in_features", 0))
        if student_dim <= 0:
            raise AttributeError("Could not infer student embedding dim from model.command_head.in_features")
        teacher_cache = WavLMFeatureCache(
            model_id=str(teacher_cfg.get("model_id", "microsoft/wavlm-base-plus")),
            cache_dir=(project_root / teacher_cfg.get("cache_dir", "cache/wavlm_teacher")).resolve(),
            device=device,
            clip_samples=int(round(float(cfg["features"].get("audio_seconds", 1.0)) * int(cfg["features"]["sample_rate"]))),
        )
        teacher_heads = TeacherHeads(
            feature_dim=teacher_cache.feature_dim,
            student_dim=student_dim,
            num_commands=len(COMMAND31_LABELS),
            dropout=float(teacher_cfg.get("dropout", 0.1)),
        ).to(device)

    optimizer_name = str(cfg["training"].get("optimizer", "adamw")).lower()
    lr = float(cfg["training"].get("lr", 1e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    trainable_params = list(model.parameters()) + (list(teacher_heads.parameters()) if teacher_heads is not None else [])
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["training"].get("lr_gamma", 0.5)),
        patience=int(cfg["training"].get("lr_patience", 2)),
    )

    run_name = cfg.get("run_name") or f"{cfg['model']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (project_root / cfg["training"].get("output_dir", "outputs") / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved config and stats.
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    epochs = int(cfg["training"].get("epochs", 10))
    loss_w = cfg["training"].get("loss_weights", {"command": 1.0, "kws12": 0.0, "wake": 1.0, "aux": 0.0})
    lambda_command = float(loss_w.get("command", 1.0))
    lambda_kws12 = float(loss_w.get("kws12", 0.0))
    lambda_wake = float(loss_w.get("wake", 1.0))
    lambda_aux = float(loss_w.get("aux", 0.0))
    lambda_confusion = float(loss_w.get("confusion", 0.0))
    lambda_distill_logits = float(loss_w.get("distill_logits", 0.0))
    lambda_distill_embed = float(loss_w.get("distill_embed", 0.0))
    aux_margin = float(cfg["training"].get("aux_margin", 0.2))
    confusion_margin = float(cfg["training"].get("confusion_margin", 0.2))
    audio_seconds = float(cfg["features"].get("audio_seconds", 1.0))
    keyword_focus_cfg = cfg.get("training", {}).get("keyword_focus", {})
    if not isinstance(keyword_focus_cfg, dict):
        keyword_focus_cfg = {}
    weak_keywords = keyword_focus_cfg.get("weak_keywords", list(DEFAULT_WEAK_KEYWORDS))
    confusion_groups = keyword_focus_cfg.get("confusion_groups", DEFAULT_CONFUSION_GROUPS)
    ce_weight = float(keyword_focus_cfg.get("ce_weight", 1.5))
    keyword_ce_weights = {str(keyword): ce_weight for keyword in weak_keywords}

    best_kws12 = -1.0
    best_frr = float("inf")
    history_path = output_dir / "metrics_history.jsonl"

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            model=model,
            loader=dataloaders.train,
            device=device,
            optimizer=optimizer,
            lambda_command=lambda_command,
            lambda_kws12=lambda_kws12,
            lambda_wake=lambda_wake,
            lambda_aux=lambda_aux,
            lambda_confusion=lambda_confusion,
            aux_margin=aux_margin,
            confusion_margin=confusion_margin,
            audio_seconds=audio_seconds,
            teacher_cache=teacher_cache,
            teacher_heads=teacher_heads,
            lambda_distill_logits=lambda_distill_logits,
            lambda_distill_embed=lambda_distill_embed,
            keyword_ce_weights=keyword_ce_weights,
            confusion_groups=confusion_groups,
        )
        valid_result = run_epoch(
            model=model,
            loader=dataloaders.valid,
            device=device,
            optimizer=None,
            lambda_command=lambda_command,
            lambda_kws12=lambda_kws12,
            lambda_wake=lambda_wake,
            lambda_aux=lambda_aux,
            lambda_confusion=lambda_confusion,
            aux_margin=aux_margin,
            confusion_margin=confusion_margin,
            audio_seconds=audio_seconds,
            teacher_cache=None,
            teacher_heads=None,
            lambda_distill_logits=0.0,
            lambda_distill_embed=0.0,
            keyword_ce_weights=keyword_ce_weights,
            confusion_groups=confusion_groups,
        )

        scheduler.step(float(valid_result.metrics.get("kws12_acc", 0.0)))

        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "valid_loss": valid_result.loss,
            "train_metrics": train_result.metrics,
            "valid_metrics": valid_result.metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "metrics": valid_result.metrics,
            "device": str(device),
            "label_set": COMMAND31_LABELS,
            "teacher_state": teacher_heads.state_dict() if teacher_heads is not None else None,
        }

        torch.save(checkpoint, output_dir / "last.pt")

        kws12_acc = float(valid_result.metrics.get("kws12_acc", 0.0))
        frr = float(valid_result.metrics.get("wake_frr_at_1fa_per_hour", 1.0))
        if kws12_acc > best_kws12:
            best_kws12 = kws12_acc
            collected = _collect_validation_outputs(model, dataloaders.valid, device)
            keyword_focus = build_keyword_focus_report(
                collected["preds"],
                collected["targets"],
                top_k=int(keyword_focus_cfg.get("top_k", 5)),
                focus_keywords=keyword_focus_cfg.get("focus_keywords"),
                focus_pairs=keyword_focus_cfg.get("focus_pairs"),
            )
            calibration = fit_keyword_calibration(
                collected["command_probs"],
                collected["wake_probs"],
                collected["targets"],
                focus=keyword_focus,
            )
            checkpoint["keyword_focus"] = keyword_focus
            checkpoint["keyword_calibration"] = calibration
            torch.save(checkpoint, output_dir / "best_kws12.pt")
            (output_dir / "keyword_focus.json").write_text(json.dumps(keyword_focus, indent=2, ensure_ascii=False), encoding="utf-8")
            (output_dir / "keyword_calibration.json").write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
        if frr < best_frr:
            best_frr = frr
            torch.save(checkpoint, output_dir / "best_wake_frr.pt")

        print(
            f"[epoch {epoch:02d}] train_loss={train_result.loss:.4f} "
            f"valid_loss={valid_result.loss:.4f} "
            f"kws12={kws12_acc:.4f} wake_frr@1fa={frr:.4f}"
        )

    print(f"Training complete. Outputs at: {output_dir}")


if __name__ == "__main__":
    main()
