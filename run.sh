#!/usr/bin/env bash
# Start Showroom Pulse at http://localhost:${PORT:-8000} (local only: binds 127.0.0.1).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  PY=/opt/homebrew/bin/python3.12
  command -v "$PY" >/dev/null 2>&1 || PY=python3.12
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -e .
fi
# --ws-max-size 4194304: live JPEG frames are up to 2 MB (main.WS_MAX_SIZE)
exec .venv/bin/uvicorn pulse.app.main:app --host 127.0.0.1 --port "${PORT:-8000}" --ws-max-size 4194304
