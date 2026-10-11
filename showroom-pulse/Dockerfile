# Showroom Pulse - local/on-prem image. Build: docker build -t showroom-pulse:local .
#
# Models are NOT baked in: they are large (hundreds of MB) and each one has a licence record in
# models/MANIFEST.json that must travel with the files. Mount them as a volume at /app/models.
# Run:
#   docker run --rm -p 127.0.0.1:8000:8000 \
#     -v "$PWD/models:/app/models" -v pulse-jobs:/app/jobs \
#     -e PULSE_ALLOWED_HOSTS=localhost,127.0.0.1 showroom-pulse:local
# PULSE_ALLOWED_HOSTS: extra hostnames accepted in the Host header (comma separated) besides
# localhost/127.0.0.1; add the real hostname when serving behind a reverse proxy or on a LAN.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PULSE_ALLOWED_HOSTS=localhost,127.0.0.1

# opencv-python-headless needs libglib2.0-0 at import time; no libgl1 (headless build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY labels ./labels
# Editable install on purpose: the app resolves models/ and labels/ relative to the source tree
# (/app), which a site-packages install would move.
RUN pip install --no-cache-dir -e .

# ORT_GPU=1 (deploy/docker-compose.yml): swap in onnxruntime-gpu with its CUDA/cuDNN pip wheels; the app uses it
# when ORT_DEVICE=cuda and a GPU is passed in, else it stays on the CPU provider.
ARG ORT_GPU=0
RUN if [ "$ORT_GPU" = 1 ]; then pip uninstall -y onnxruntime \
    && pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]==1.23.2"; fi

RUN useradd --create-home --uid 10001 pulse \
    && mkdir -p /app/models /app/jobs \
    && chown -R pulse:pulse /app/jobs /app/models
USER pulse

VOLUME ["/app/models", "/app/jobs"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# 0.0.0.0 is the container's own interface; publish the port on 127.0.0.1 only (see docker run above).
# --ws-max-size 4194304: live JPEG frames are up to 2 MB (main.WS_MAX_SIZE).
CMD ["uvicorn", "pulse.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "4194304"]
