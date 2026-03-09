#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_PATH="${1:-${KWS_HI_MIA_BUNDLE_PATH:-artifacts/hi_mia_full_bundle.tar.zst}}"
SKIP_BOOTSTRAP="${KWS_SKIP_BOOTSTRAP:-0}"
AUDIT_PATH="${KWS_HI_MIA_IMPORT_AUDIT_PATH:-reports/hi_mia_import_audit.json}"
BACKUP_ROOT="${KWS_HI_MIA_IMPORT_BACKUP_ROOT:-}"

if [ ! -f "$BUNDLE_PATH" ]; then
  echo "[import_hi_mia_bundle] bundle not found: $BUNDLE_PATH" >&2
  exit 1
fi

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  bash scripts/bootstrap_env.sh
fi

args=(
  --project-root "$ROOT_DIR"
  --bundle-path "$BUNDLE_PATH"
  --audit-output "$AUDIT_PATH"
)
if [ -n "$BACKUP_ROOT" ]; then
  args+=(--backup-root "$BACKUP_ROOT")
fi

conda run --no-capture-output -n dl python -m kws.data.himia_bundle import "${args[@]}"
