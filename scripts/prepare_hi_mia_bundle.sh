#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_SECONDS="${KWS_HI_MIA_WAIT_SECONDS:-360}"
MAX_ATTEMPTS="${KWS_HI_MIA_MAX_ATTEMPTS:-0}"
SKIP_BOOTSTRAP="${KWS_SKIP_BOOTSTRAP:-0}"
BUNDLE_PATH="${KWS_HI_MIA_BUNDLE_PATH:-artifacts/hi_mia_full_bundle.tar.zst}"
MANIFEST_PATH="${KWS_HI_MIA_BUNDLE_MANIFEST_PATH:-artifacts/hi_mia_full_bundle_manifest.json}"
AUDIT_PATH="${KWS_HI_MIA_REMOTE_AUDIT_PATH:-reports/hi_mia_remote_audit.json}"

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  export KWS_BOOTSTRAP_REQUIRE_MPS=0
  bash scripts/bootstrap_env.sh
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[prepare_hi_mia_bundle] restore attempt ${attempt}"

  set +e
  conda run --no-capture-output -n dl python -m kws.data.download_external \
    --dataset hi_mia \
    --root data/external/hi_mia \
    --manifests-dir data/processed/manifests
  restore_status=$?
  set -e

  if conda run -n dl python -m kws.data.himia_bundle status --project-root "$ROOT_DIR" --require-full >/dev/null; then
    break
  fi

  if [ "$MAX_ATTEMPTS" != "0" ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[prepare_hi_mia_bundle] restore never reached full official status after ${attempt} attempts" >&2
    exit 1
  fi

  echo "[prepare_hi_mia_bundle] current restore is still partial; sleeping ${WAIT_SECONDS}s before retry"
  sleep "$WAIT_SECONDS"
done

conda run --no-capture-output -n dl python -m kws.data.himia_bundle prepare \
  --project-root "$ROOT_DIR" \
  --bundle-path "$BUNDLE_PATH" \
  --manifest-path "$MANIFEST_PATH" \
  --audit-output "$AUDIT_PATH"
