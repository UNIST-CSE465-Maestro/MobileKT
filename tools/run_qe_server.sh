#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${MOBILEKT_QE_HOST:-0.0.0.0}"
PORT="${MOBILEKT_QE_PORT:-8091}"
DEVICE="${MOBILEKT_QE_DEVICE:-cuda}"
FEATURE_MODE="${MOBILEKT_QE_FEATURE_MODE:-harrier}"
EXPORT_DIR="${MOBILEKT_QE_EXPORT_DIR:-export}"
ALLOW_MODEL_DOWNLOAD="${MOBILEKT_QE_ALLOW_MODEL_DOWNLOAD:-0}"

EXTRA_ARGS=()
if [ "$ALLOW_MODEL_DOWNLOAD" = "1" ]; then
  EXTRA_ARGS+=(--allow_model_download)
fi

exec python3 -m server.app \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --feature_mode "$FEATURE_MODE" \
  --export_dir "$EXPORT_DIR" \
  "${EXTRA_ARGS[@]}"
