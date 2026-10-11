#!/usr/bin/env bash
# EXAMPLE ONLY — NOT RUN. DO NOT RUN WITHOUT THE OWNER'S APPROVAL.
#
# How one would build the Showroom Coach image and run it on an on-premise Linux box (a computer in the showroom
# or the dealership's own server room). It was written during the hackathon and never executed: Docker was not
# running on the build machine, so neither the Dockerfile nor this script has been tried. It has no cloud commands
# and no credentials, and nothing here leaves the machine it runs on.
#
# The guard below stops it from doing anything if it is run by accident. To use it for real: read every line,
# adjust the settings, get the owner's approval, then delete the guard line.
set -euo pipefail

echo "This is an example; edit and remove this line to use it."; exit 1

# ---------------------------------------------------------------------------------------------------- settings
IMAGE="showroom-coach:local"
NAME="showroom-coach"
DATA="/srv/showroom-coach"          # uploaded clips, results and the perception cache live here
BIND="127.0.0.1"                    # 127.0.0.1 = this computer only; see the HTTPS note below before using 0.0.0.0
PORT="8000"

# Run from the repository root (the folder with the Dockerfile).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---------------------------------------------------------------------------------------------------- build
# Needs internet once: pip packages and the three model files (about 50 MB) are fetched while the image is built.
docker build -t "$IMAGE" .

# ---------------------------------------------------------------------------------------------------- run
# The container runs as a non-root user (uid 10001), so the data folder must belong to it.
mkdir -p "$DATA"
chown 10001:10001 "$DATA"

# Replace an older container of the same name, if there is one. The data folder is kept.
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -p "$BIND:$PORT:8000" \
  -v "$DATA:/data" \
  "$IMAGE"

# ---------------------------------------------------------------------------------------------------- check
echo "Showroom Coach should now answer at http://$BIND:$PORT (the first request can take a few seconds)."
echo "Logs:    docker logs -f $NAME"
echo "Health:  docker inspect --format '{{.State.Health.Status}}' $NAME"
echo "Stop:    docker stop $NAME"

# HTTPS note: the live camera page only works when the browser and the server are on the same machine
# (http://localhost) or the page is served over HTTPS, because browsers refuse camera access otherwise. Uploading
# and analysing clips works over plain HTTP on the local network. For live mode from other computers, put a
# reverse proxy with a TLS certificate in front of 127.0.0.1:$PORT instead of binding to 0.0.0.0.
#
# Privacy note: videos and results stay in $DATA on this machine; the app deletes jobs older than 24 hours when it
# starts. Staff must be told that analytics is in use (see the notice in the app and the README).
