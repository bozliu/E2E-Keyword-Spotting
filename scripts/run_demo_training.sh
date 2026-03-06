#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
ENV_NAME="${KWS_ENV_NAME:-dl}"

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
  configs=(
    "configs/demo_mhatt_small.yaml"
    "configs/demo_mhatt_base.yaml"
    "configs/demo_mamba_tuned.yaml"
  )
fi

mkdir -p reports

for config in "${configs[@]}"; do
  run_name="$(basename "${config%.yaml}")"
  checkpoint_path="outputs/${run_name}/best_kws12.pt"
  analysis_path="reports/${run_name}_analysis.json"

  echo "==> Training ${config}"
  conda run --no-capture-output -n "${ENV_NAME}" python -m kws.train --config "${config}"

  if [ -f "${checkpoint_path}" ]; then
    echo "==> Analyzing ${checkpoint_path}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -m kws.demo.analyze_checkpoint \
      --checkpoint "${checkpoint_path}" \
      --split test \
      --output "${analysis_path}"
  else
    echo "Warning: ${checkpoint_path} not found, skipping analysis" >&2
  fi
done

echo "==> Refreshing demo ranking"
conda run --no-capture-output -n "${ENV_NAME}" python scripts/select_demo_checkpoint.py
