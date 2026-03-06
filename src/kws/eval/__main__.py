"""CLI entrypoint for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kws.data.pipeline import create_dataloaders, prepare_data
from kws.models import create_model
from kws.train.engine import pick_device, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-task KWS checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Recover project root from output_dir in config.
    output_dir = ckpt_path.parent
    project_root = output_dir.parent.parent.resolve()

    prepare_data(cfg, project_root)
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

    payload = {
        "split": args.split,
        "loss": result.loss,
        "metrics": result.metrics,
    }

    out_path = Path(args.output).expanduser().resolve() if args.output else output_dir / f"eval_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
