#!/usr/bin/env bash
set -euo pipefail

export KWS_BOOTSTRAP_REQUIRE_MPS="${KWS_BOOTSTRAP_REQUIRE_MPS:-1}"

conda run -n dl python -m pip uninstall -y e2e-keyword-spotting >/dev/null 2>&1 || true
conda env update -n dl -f environment.yml --prune
conda run -n dl python -m pip install -e .
conda run -n dl python - <<'PY'
import os
from pathlib import Path

from kws.env import run_repo_preflight

report = run_repo_preflight(
    Path.cwd(),
    manifests_dir="data/processed/manifests",
    teacher_model_id="microsoft/wavlm-base-plus",
    require_mps=os.environ.get("KWS_BOOTSTRAP_REQUIRE_MPS", "1") not in {"0", "false", "False"},
)
print(report["active_kws_file"])
print(report["teacher_model_id"])
print(report["mps_available"])
print(report["manifest_audit"]["is_clean"])
PY
