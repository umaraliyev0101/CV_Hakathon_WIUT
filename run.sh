#!/usr/bin/env bash
# Showroom Coach launcher (bash): sets up .venv and the model weights on first run, then starts the web app.
# Extra arguments are passed to "showroom-coach serve", e.g.  ./run.sh --port 8080
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

MODELS_ROOT="${SHOWROOM_MODELS:-models}"

# The venv's interpreter: .venv/bin on Linux and macOS, .venv/Scripts on Windows (Git Bash).
venv_python() {
  local p
  for p in .venv/bin/python .venv/Scripts/python.exe .venv/Scripts/python; do
    if [ -x "$p" ]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

# A Python 3.11-3.13 for creating the venv (the package requires >=3.11,<3.14).
find_python() {
  local c win=""
  if [ -n "${LOCALAPPDATA:-}" ] && command -v cygpath >/dev/null 2>&1; then
    win="$(cygpath -u "$LOCALAPPDATA")/Programs/Python/Python312/python.exe"
  fi
  for c in python3.12 python3.13 python3.11 "$win" python3 python; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

if ! VENV_PY="$(venv_python)"; then
  echo "Showroom Coach: creating the virtual environment in .venv (first run only)"
  if ! PY="$(find_python)"; then
    echo "ERROR: Python 3.11, 3.12 or 3.13 is required but was not found. Install Python 3.12 and run ./run.sh again." >&2
    exit 1
  fi
  "$PY" -m venv .venv || { echo "ERROR: could not create .venv with $PY." >&2; exit 1; }
  if ! VENV_PY="$(venv_python)"; then
    echo "ERROR: .venv was created but contains no python executable." >&2
    exit 1
  fi
fi

if ! "$VENV_PY" -c "import showroom_coach" >/dev/null 2>&1; then
  echo "Showroom Coach: installing the package and its dependencies (first run only, needs internet)"
  "$VENV_PY" -m pip install -e . || { echo "ERROR: pip install failed. Check the internet connection and run ./run.sh again." >&2; exit 1; }
fi

need_models=0
for m in yolox_nano rtmpose_s dinov2_small; do
  [ -f "$MODELS_ROOT/$m/$m.onnx" ] || need_models=1
done
if [ "$need_models" = 1 ]; then
  echo "Showroom Coach: downloading the model weights (first run only, needs internet)"
  "$VENV_PY" -m showroom_coach fetch-models yolox_nano rtmpose_s dinov2_small \
    || { echo "ERROR: the model download failed. Check the internet connection and run ./run.sh again." >&2; exit 1; }
fi

echo "Showroom Coach: open http://localhost:8000 (Ctrl+C to stop)"
exec "$VENV_PY" -m showroom_coach serve --host 127.0.0.1 --port 8000 "$@"
