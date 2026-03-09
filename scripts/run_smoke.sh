#!/usr/bin/env bash
set -euo pipefail

conda run -n dl python -m kws.train --config configs/smoke_mamba.yaml
conda run -n dl python -m kws.train --config configs/smoke_mhatt.yaml
