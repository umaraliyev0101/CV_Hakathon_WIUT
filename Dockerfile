# Written for later use; not built during the hackathon (Docker was not running).
# Live mode needs the browser and server on the same machine or HTTPS: browsers only allow camera access on
# localhost or over HTTPS, so put a TLS reverse proxy in front of this when it runs on another computer.
#
#   docker build -t showroom-coach .
#   docker run -p 127.0.0.1:8000:8000 -v showroom-data:/data showroom-coach      # http://localhost:8000
#
# Licence decision needed before building or distributing this image: it contains the FFmpeg executable bundled in
# imageio-ffmpeg, which is a GPL-3.0-or-later build (libx264, libx265, libxvid; README, licences, note 1). It runs as a
# separate process and is not linked into this code, but shipping the image ships that binary, so the GPL source-offer
# terms and H.264 patent licensing apply to whoever distributes it. Alternatives: an LGPL FFmpeg build, or have the
# customer run `pip install` themselves. Keeping the image on one's own machines is not distribution.
#
# CPU only (onnxruntime, no GPU). The three models the app uses are downloaded while the image is built, so the
# build needs internet and about 50 MB of disk for them; the running container needs no network. Everything the
# app writes (uploaded clips, results, perception cache) goes to /data; jobs older than 24 hours are deleted when
# the app starts.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHOWROOM_MODELS=/opt/showroom/models \
    SHOWROOM_JOBS=/data/jobs \
    SHOWROOM_CACHE=/data/cache

# opencv-python needs these two shared libraries on a slim image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

# Model weights (permissive licences, sha256-pinned in the registry): fetched at build time, not shipped in the repo.
RUN showroom-coach fetch-models yolox_nano rtmpose_s dinov2_small

RUN useradd --create-home --uid 10001 showroom \
 && mkdir -p /data/jobs /data/cache \
 && chown -R showroom:showroom /data /opt/showroom
USER showroom

VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/presets', timeout=4)"]
CMD ["showroom-coach", "serve", "--host", "0.0.0.0", "--port", "8000"]
