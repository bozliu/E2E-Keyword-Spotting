#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_ID="${1:-${HF_SPACE_ID:-bozliu/e2e-keyword-spotting-demo}}"

fail() {
  echo "[release_public_demo] $1" >&2
  exit 1
}

command -v conda >/dev/null 2>&1 || fail "conda is required"
conda env list | awk '{print $1}' | grep -qx 'dl' || fail "conda env 'dl' is missing"

cd "$ROOT"

echo "[release_public_demo] Verifying dl Python version"
PYV="$(conda run -n dl python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYV" == "3.12" ]] || fail "dl must use Python 3.12, got $PYV"

echo "[release_public_demo] Installing package and Space dependencies into dl"
conda run -n dl python -m pip install -e . >/dev/null
conda run -n dl python -m pip install -r requirements-space.txt >/dev/null

echo "[release_public_demo] Syntax checks"
conda run -n dl python -m py_compile app.py scripts/deploy_hf_space.py src/kws/demo/web.py
conda run --no-capture-output -n dl python -c "import app; print('import_ok')"

echo "[release_public_demo] Running targeted tests"
conda run --no-capture-output -n dl pytest -q tests/test_demo_web.py tests/test_demo*.py tests/test_keyword_focus*.py

echo "[release_public_demo] Running web smoke inference"
PYTHONPATH=src conda run --no-capture-output -n dl python - <<'PY'
import numpy as np
from kws.demo.web import load_web_demo, predict_web_clip

bundle = load_web_demo('auto')
audio = np.zeros(16000, dtype=np.float32)
out = predict_web_clip(bundle, (16000, audio))
assert hasattr(out, 'label') and hasattr(out, 'status_message') and hasattr(out, 'keyword_scores')
print('smoke_ok', out.label, out.status_message)
PY

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[release_public_demo] Verifying Hugging Face login"
  huggingface-cli whoami >/dev/null 2>&1 || fail "HF login missing. Run 'huggingface-cli login' or set HF_TOKEN."
fi

echo "[release_public_demo] Deploying to Space: $SPACE_ID"
conda run -n dl python scripts/deploy_hf_space.py --space-id "$SPACE_ID"

echo "[release_public_demo] Public demo URL: https://huggingface.co/spaces/$SPACE_ID"
