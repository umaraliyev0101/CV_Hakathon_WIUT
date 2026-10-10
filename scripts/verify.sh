#!/usr/bin/env bash
# Showroom Pulse verification chain (spec §9). Runs each step in order, prints its output,
# exits non-zero at the first failure.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
step() {
  local name="$1"; shift
  echo; echo "=== [verify] $name ==="; echo "\$ $*"
  "$@"; local rc=$?
  echo "=== [verify] $name -> exit $rc ==="
  if [ $rc -ne 0 ]; then echo "VERIFY FAILED at: $name"; exit $rc; fi
}
step "pytest (default)" $PY -m pytest -q
step "pytest (slow)" $PY -m pytest -q -m slow
# --- later milestones insert steps here ---
step "git status clean" bash -c 'out=$(git status --short); echo "$out"; test -z "$out"'
step "no git remote" bash -c 'out=$(git remote -v); echo "$out"; test -z "$out"'
echo; echo "VERIFY OK"
