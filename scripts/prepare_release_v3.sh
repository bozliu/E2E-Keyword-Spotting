#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_LONG_VALIDATION="${KWS_RELEASE_RUN_LONG_VALIDATION:-0}"

fail() {
  echo "[prepare_release_v3] $1" >&2
  exit 1
}

command -v conda >/dev/null 2>&1 || fail "conda is required"
conda env list | awk '{print $1}' | grep -qx 'dl' || fail "conda env 'dl' is missing"

cd "$ROOT"

echo "[prepare_release_v3] Git summary"
git rev-parse --show-toplevel >/dev/null 2>&1 || fail "not inside a git repository"
echo "  branch: $(git branch --show-current || true)"
echo "  remotes:"
git remote -v || true
echo "  status:"
git status --short

echo "[prepare_release_v3] Installing package into dl"
conda run -n dl python -m pip install -e . >/dev/null

echo "[prepare_release_v3] Syntax checks"
conda run -n dl python -m py_compile \
  app.py \
  scripts/generate_release_assets.py \
  src/kws/benchmark_external.py \
  src/kws/demo/realtime.py \
  src/kws/demo/validate_realtime.py

echo "[prepare_release_v3] Running release test subset"
conda run --no-capture-output -n dl pytest -q \
  tests/test_external_hf_kws.py \
  tests/test_demo_accuracy_first.py \
  tests/test_validate_realtime.py \
  tests/test_demo_analyze_checkpoint.py \
  tests/test_eval_cli.py \
  tests/test_demo_web.py \
  tests/test_training_smoke.py

echo "[prepare_release_v3] Generating public release assets"
conda run --no-capture-output -n dl python scripts/generate_release_assets.py

if [[ "$RUN_LONG_VALIDATION" == "1" ]]; then
  echo "[prepare_release_v3] Running full realtime validation on valid"
  conda run --no-capture-output -n dl python -m kws.demo.validate_realtime \
    --demo-profile accuracy-first \
    --checkpoint auto \
    --device mps \
    --external-kws-device mps \
    --wheel kws12 \
    --split valid \
    --output reports/realtime_accuracy_first_valid.json

  echo "[prepare_release_v3] Running full realtime validation on test"
  conda run --no-capture-output -n dl python -m kws.demo.validate_realtime \
    --demo-profile accuracy-first \
    --checkpoint auto \
    --device mps \
    --external-kws-device mps \
    --wheel kws12 \
    --split test \
    --output reports/realtime_accuracy_first_test.json

  echo "[prepare_release_v3] Refreshing release assets after full validation"
  conda run --no-capture-output -n dl python scripts/generate_release_assets.py
else
  echo "[prepare_release_v3] Skipping long realtime validation. Set KWS_RELEASE_RUN_LONG_VALIDATION=1 to enable it."
fi

cat <<'EOF'
[prepare_release_v3] Next git/release steps
1. Ensure the public topology is legacy-v1 / v2 / main
2. Keep v2 as a read-only branch that matches the v2.0.0 snapshot
3. Remove legacy public aliases such as v2.0-public after checkpoint URLs are migrated
4. Regenerate docs/assets and commit the release-doc updates
5. Let core-release-checks and smoke-tests pass on main
6. Keep v3.0.0 as the tag for the current public main line
EOF
