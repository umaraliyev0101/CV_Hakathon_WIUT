#!/usr/bin/env bash
# Build and start the Showroom Suite stack (Coach + Pulse + suite page behind nginx on 127.0.0.1:${WEB_PORT:-8090}).
#   deploy/up.sh            local only
#   deploy/up.sh --public   also start the ngrok tunnel and print the public HTTPS URL
# Settings come from deploy/.env (copy deploy/.env.example). See deploy/README.md.
set -euo pipefail
cd "$(dirname "$0")"

public=0
[[ "${1:-}" == "--public" ]] && public=1

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

# Optional password for the whole site (strongly recommended before going public).
if [[ -n "${DEMO_USER:-}" && -n "${DEMO_PASS:-}" ]]; then
  printf '%s:%s\n' "$DEMO_USER" "$(openssl passwd -apr1 "$DEMO_PASS")" > htpasswd
  export AUTH_REALM="Showroom demo"
else
  : > htpasswd
  export AUTH_REALM=off
fi
mkdir -p demo

# Our fine-tuned person detector (models/showroom_detector_ft.onnx, mounted at /opt/ft): used for people in both
# apps whenever the file is there and deploy/.env does not choose otherwise. Cars stay on Pulse's COCO detector,
# because the fine-tuned head knows only "person".
if [[ -f ../models/showroom_detector_ft.onnx ]]; then
  : "${SHOWROOM_RFDETR_FT:=/opt/ft/showroom_detector_ft.onnx}" "${PULSE_RFDETR_FT:=/opt/ft/showroom_detector_ft.onnx}"
  : "${SHOWROOM_DETECTOR:=rfdetr_ft@2}" "${SHOWROOM_LIVE_DETECTOR:=rfdetr_ft}"
  export SHOWROOM_RFDETR_FT PULSE_RFDETR_FT SHOWROOM_DETECTOR SHOWROOM_LIVE_DETECTOR
  echo "people: fine-tuned detector (coach ${SHOWROOM_DETECTOR}, pulse ${PULSE_RFDETR_FT:-off})"
fi

compose=(docker compose -f docker-compose.yml)
"${compose[@]}" build
"${compose[@]}" up -d coach pulse web

# Pulse downloads its models on first use; fetch them now so the first visitor does not wait.
"${compose[@]}" exec -T pulse python -c "
from pulse.config import get_config
from pulse.perception.models import ensure_model
c = get_config()
for name in (c.detector, c.vehicle_detector, c.pose_model, 'dinov2_small'):
    print(name, ensure_model(name, '/app/models'))
"

port="${WEB_PORT:-8090}"
for _ in $(seq 60); do
  curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1 && break
  sleep 1
done
echo "local:  http://127.0.0.1:${port}/"

if (( public )); then
  # Token order: deploy/.env NGROK_AUTHTOKEN, then NGROK_TOKEN in the repo-root .env, then the local ngrok config.
  if [[ -z "${NGROK_AUTHTOKEN:-}" && -f ../.env ]]; then
    NGROK_AUTHTOKEN="$(sed -n 's/^NGROK_TOKEN=//p' ../.env | tail -1 | tr -d '[:space:]"'"'")"
    export NGROK_AUTHTOKEN
  fi
  if [[ -z "${NGROK_AUTHTOKEN:-}" && -f "$HOME/.config/ngrok/ngrok.yml" ]]; then
    NGROK_AUTHTOKEN="$(sed -n 's/^ *authtoken: *//p' "$HOME/.config/ngrok/ngrok.yml" | head -1)"
    export NGROK_AUTHTOKEN
  fi
  [[ -n "${NGROK_AUTHTOKEN:-}" ]] || { echo "no ngrok authtoken (set NGROK_AUTHTOKEN in deploy/.env)" >&2; exit 1; }
  [[ "$AUTH_REALM" == off ]] && echo "WARNING: going public WITHOUT a password (set DEMO_USER/DEMO_PASS in deploy/.env)" >&2
  "${compose[@]}" --profile public up -d ngrok
  url=""
  for _ in $(seq 30); do
    url="$("${compose[@]}" --profile public logs ngrok 2>/dev/null | grep -o 'url=https://[^ ]*' | tail -1 | cut -d= -f2 || true)"
    [[ -n "$url" ]] && break
    sleep 1
  done
  echo "public: ${url:-<not found yet: docker compose -f deploy/docker-compose.yml --profile public logs ngrok>}"
fi
