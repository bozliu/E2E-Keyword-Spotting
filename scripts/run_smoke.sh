#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${KWS_ENV_NAME:-dl}"

conda run -n "${ENV_NAME}" python -m kws.train --config configs/smoke_mamba.yaml
conda run -n "${ENV_NAME}" python -m kws.train --config configs/smoke_mhatt.yaml
