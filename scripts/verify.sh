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
MEVA_DEV=$($PY -c "import json;print(json.load(open('labels/split.json'))['dev'][0]['video'])" 2>/dev/null || ls data/meva/videos/*.avi | head -1)
SHOW=$(ls data/stock/*.mp4 data/stock/*.mov data/stock/*.webm data/synthetic/share/*.mp4 2>/dev/null | head -1)
step "analyze MEVA dev clip" .venv/bin/pulse analyze "$MEVA_DEV" --out runs/verify/meva --preset dealership
step "analyze showroom clip" .venv/bin/pulse analyze "$SHOW" --out runs/verify/stock --preset short
step "check outputs" $PY scripts/check_outputs.py runs/verify/meva runs/verify/stock
step "git status clean" bash -c 'out=$(git status --short); echo "$out"; test -z "$out"'
step "no git remote" bash -c 'out=$(git remote -v); echo "$out"; test -z "$out"'
echo; echo "VERIFY OK"
