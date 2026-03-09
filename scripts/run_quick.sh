#!/usr/bin/env bash
set -euo pipefail

conda run -n dl python -m kws.train --config configs/quick_mamba.yaml
conda run -n dl python -m kws.train --config configs/quick_mhatt.yaml
