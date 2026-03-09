#!/usr/bin/env bash
set -euo pipefail

# Disable Xet mode to reduce unauthenticated rate-limit failures on some networks.
export HF_HUB_DISABLE_XET=1

conda run -n dl python -m kws.data.download_external \
  --dataset hi_mia \
  --root data/external/hi_mia \
  --manifests-dir data/processed/manifests
