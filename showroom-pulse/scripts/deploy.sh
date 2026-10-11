#!/usr/bin/env bash
# NOT RUN — local-only hackathon rule
#
# On-prem deployment of Showroom Pulse to an Ubuntu box: service user, python3.12 venv, models,
# hardened systemd unit, nginx reverse-proxy notes. Written for a later, supervised rollout.
# It refuses to do anything unless BOTH hold:
#   1. the explicit flag --i-know-this-deploys is given
#   2. the OS is Linux (so an accidental run on a Mac does nothing)
#
# Usage (on the target box, as root, from a checkout of this repo):
#   sudo bash scripts/deploy.sh --i-know-this-deploys [--models-from /path/to/models] [--host pulse.example.lan]
set -euo pipefail

CONFIRMED=0
MODELS_FROM=""
PUBLIC_HOST="localhost"
while [ $# -gt 0 ]; do
  case "$1" in
    --i-know-this-deploys) CONFIRMED=1 ;;
    --models-from) MODELS_FROM="${2:?--models-from needs a path}"; shift ;;
    --host) PUBLIC_HOST="${2:?--host needs a hostname}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$CONFIRMED" -ne 1 ]; then
  echo "Refusing: this script deploys. Pass --i-know-this-deploys to proceed." >&2
  exit 1
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "Refusing: deploy.sh only runs on Linux (this is $(uname -s))." >&2
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "Refusing: run as root (sudo)." >&2
  exit 1
fi

command -v python3.12 >/dev/null || { echo "python3.12 not found (apt install python3.12 python3.12-venv)" >&2; exit 1; }
APP_USER=pulse
APP_DIR=/opt/showroom-pulse
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 1. service user (no login shell, no home login)
id "$APP_USER" >/dev/null 2>&1 || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"

# 2. code + venv (python3.12)
mkdir -p "$APP_DIR" "$APP_DIR/models" "$APP_DIR/jobs"
cp -R "$SRC_DIR/src" "$SRC_DIR/pyproject.toml" "$SRC_DIR/labels" "$APP_DIR/"
python3.12 -m venv "$APP_DIR/.venv"
# editable: the app finds models/ and labels/ relative to the source tree under $APP_DIR
"$APP_DIR/.venv/bin/pip" install --no-cache-dir -e "$APP_DIR"

# 3. models (not in git; licence records travel with MANIFEST.json)
if [ -n "$MODELS_FROM" ]; then
  cp -R "$MODELS_FROM"/. "$APP_DIR/models/"
else
  cp "$SRC_DIR/models/MANIFEST.json" "$APP_DIR/models/" 2>/dev/null || true
  echo "No --models-from given: models download on first use (needs internet once) or copy them to $APP_DIR/models" >&2
fi
chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

# 4. systemd unit, hardened; only jobs/ and models/ (first-use model downloads) are writable
cat > /etc/systemd/system/showroom-pulse.service <<UNIT
[Unit]
Description=Showroom Pulse (anonymous showroom analytics)
After=network.target

[Service]
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PULSE_ALLOWED_HOSTS=$PUBLIC_HOST
ExecStart=$APP_DIR/.venv/bin/uvicorn pulse.app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 4194304
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true
ReadWritePaths=$APP_DIR/jobs $APP_DIR/models
# Uploads/job data are deleted after 24 h by a cleanup that runs only at app start, so restart daily.
# A RuntimeMaxSec timeout counts as a failure, so Restart=on-failure brings the service back.
RuntimeMaxSec=86400

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now showroom-pulse.service

# 5. nginx reverse proxy: NOT configured by this script; add this server block yourself (+ TLS).
cat <<NOTES

nginx notes (put TLS in front; the app itself only listens on 127.0.0.1:8000):

  server {
      listen 443 ssl;
      server_name $PUBLIC_HOST;
      client_max_body_size 1g;          # matches the app upload limit (MAX_UPLOAD = 1 GiB)
      location / {
          proxy_pass http://127.0.0.1:8000;
          proxy_set_header Host \$host;  # must match PULSE_ALLOWED_HOSTS
          proxy_http_version 1.1;
          proxy_set_header Upgrade \$http_upgrade;     # WebSocket (/ws/live)
          proxy_set_header Connection "upgrade";
          proxy_read_timeout 300s;
      }
  }

Reminders:
  - Uploads and job data are deleted after 24 h by a cleanup that runs at app start; the unit's
    RuntimeMaxSec=86400 restarts the service daily so that deletion actually happens.
  - Signage is required: customers must be told that camera analytics is in use. This is the
    operator's legal duty; the tool does not replace it.
NOTES
