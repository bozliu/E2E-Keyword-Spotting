#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${KWS_ENV_NAME:-dl}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

conda run -n "${ENV_NAME}" python -m pip install -e .
