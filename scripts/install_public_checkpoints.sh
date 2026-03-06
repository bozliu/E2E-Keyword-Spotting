#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/quick_mhatt outputs/demo_mhatt_small_focus outputs/demo_mhatt_small_focus_lod

ln -sfn ../../checkpoints/quick_mhatt_best_kws12.pt outputs/quick_mhatt/best_kws12.pt
ln -sfn ../../checkpoints/demo_mhatt_small_focus_best_kws12.pt outputs/demo_mhatt_small_focus/best_kws12.pt
ln -sfn ../../checkpoints/demo_mhatt_small_focus_lod_best_kws12.pt outputs/demo_mhatt_small_focus_lod/best_kws12.pt

echo "Installed public checkpoint symlinks under outputs/."
echo "You can now run: python scripts/select_demo_checkpoint.py"
