#!/bin/bash
# Full deck build: JSON -> HTML -> PDF -> figure PNGs -> PPTX
# Run from the deck directory (the one containing slides/ and build/).
#   ./build/build-all.sh            -> out/deck.pdf, out/deck.pptx
#   NAME=clio-teardown ./build/build-all.sh
set -e
cd "$(dirname "$0")/.."
NAME="${NAME:-deck}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PY="${PY:-python3}"   # needs python-pptx; a venv path works: PY=/tmp/deckvenv/bin/python
mkdir -p out build/shots

echo "== validate =="
node build/validate.mjs slides || { echo "VALIDATION FAILED — fix before building"; exit 1; }

echo "== html =="
node build/render-html.mjs slides build/deck.html

echo "== pdf =="
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="out/$NAME.pdf" build/deck.html 2>/dev/null

echo "== figure shots =="
FIGS=$(node build/indices.mjs slides --figures | node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>console.log(JSON.parse(d).map(x=>x.n).join(",")))')
if [ -n "$FIGS" ]; then node build/shoot.mjs build/deck.html build/shots "$FIGS"; fi

echo "== pptx =="
$PY build/build-pptx.py slides "out/$NAME.pptx" build/shots

echo; ls -lh out/
