"""Model registry, retrying downloads with sha256 checks, and the committed MANIFEST.

Downloaded files are untrusted data: this module only streams them to disk, hashes them and
(for zip archives) copies out the single named member. Nothing is executed or unpickled, and
archive members are never extracted by path, so a hostile archive cannot write elsewhere.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from . import __version__
from .config import repo_root

log = logging.getLogger(__name__)

MIN_FREE_BYTES = 3 * 1024**3  # stop downloading when free disk would drop below 3 GB
MAX_MEMBER_BYTES = 1024**3  # sanity cap for a member copied out of an (untrusted) archive
CHUNK_BYTES = 1024 * 1024
TIMEOUT_S = 60


class ModelUnavailable(RuntimeError):
    """A model could not be fetched (network, bad archive, hash mismatch, low disk)."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    filename: str  # name of the .onnx file kept under models/<name>/
    licence: str
    trained_on: str
    source_page: str
    archive_member: str | None = None  # zip member (matched by basename) to keep
    sha256: str | None = None  # expected sha256 of the kept .onnx file
    note: str = ""


_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
_YOLOX_PAGE = "https://github.com/Megvii-BaseDetection/YOLOX"

MODELS: dict[str, ModelSpec] = {
    "yolox_nano": ModelSpec(
        "yolox_nano", f"{_YOLOX}/yolox_nano.onnx", "yolox_nano.onnx",
        "Apache-2.0", "COCO 2017", _YOLOX_PAGE, note="person detector; input 416x416",
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "yolox_tiny": ModelSpec(
        "yolox_tiny", f"{_YOLOX}/yolox_tiny.onnx", "yolox_tiny.onnx",
        "Apache-2.0", "COCO 2017", _YOLOX_PAGE, note="person detector; input 416x416",
        sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
    "yolox_s": ModelSpec(
        "yolox_s", f"{_YOLOX}/yolox_s.onnx", "yolox_s.onnx",
        "Apache-2.0", "COCO 2017", _YOLOX_PAGE, note="person detector; input 640x640",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
    "rtmpose_s": ModelSpec(
        "rtmpose_s",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        "rtmpose_s.onnx",
        "Apache-2.0",
        "Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18)",
        "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        archive_member="end2end.onnx",
        note="17-keypoint COCO pose; input 192x256 (WxH)",
        sha256="9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d"),
    "dinov2_small": ModelSpec(
        "dinov2_small",
        "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx",
        "dinov2_small.onnx",
        "Apache-2.0", "LVD-142M (self-supervised, Meta)",
        "https://huggingface.co/Xenova/dinov2-small",
        note="appearance embedding for outfit similarity; int8-quantized ONNX export",
        sha256="3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917"),
    "rfdetr_nano": ModelSpec(
        "rfdetr_nano",
        "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_n_v142_384x384.onnx",
        "rfdetr_nano.onnx",
        "Apache-2.0", "COCO 2017",
        "https://huggingface.co/saifkhichi96/opendetect",
        note="ONNX export by a third party of Roboflow RF-DETR (Apache-2.0); "
             "input 384x384; benchmark candidate only",
        sha256="aceebff282d046fae831217136c4c0ce91fb28a6b834dfd48f3a3f46fddc1d9e"),
}


def models_dir() -> Path:
    env = os.environ.get("SHOWROOM_MODELS")
    return Path(env) if env else repo_root() / "models"


class _BadDownload(Exception):
    """The bytes we received are unusable; worth a retry."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_free_disk(path: Path, incoming: int = 0) -> None:
    free = shutil.disk_usage(path).free
    if free - incoming < MIN_FREE_BYTES:
        raise ModelUnavailable(
            f"free disk < {MIN_FREE_BYTES / 1024**3:g} GB "
            f"({free / 1024**3:.1f} GB free, {incoming / 1024**2:.0f} MB incoming); not downloading"
        )


def _copy(src: BinaryIO, dst: Path, limit: int | None = None) -> str:
    """Stream src to dst and return the sha256 hex digest of what was written."""
    h = hashlib.sha256()
    written = 0
    with open(dst, "wb") as out:
        while chunk := src.read(CHUNK_BYTES):
            written += len(chunk)
            if limit is not None and written > limit:
                raise _BadDownload(f"more than {limit} bytes; refusing")
            h.update(chunk)
            out.write(chunk)
    return h.hexdigest()


def _extract_member(archive: Path, member: str, dst: Path) -> str:
    """Copy the zip member whose basename is `member` to dst (never extracting by path)."""
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if PurePosixPath(info.filename.replace("\\", "/")).name != member:
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise _BadDownload(f"{member} is {info.file_size} bytes; refusing")
            with zf.open(info) as src:
                return _copy(src, dst, limit=MAX_MEMBER_BYTES)
    raise _BadDownload(f"{member} not found in archive")


def _download(spec: ModelSpec, dest_dir: Path, target: Path) -> None:
    """One download attempt: fetch, (extract), verify, then atomically move into place."""
    part = dest_dir / (spec.filename + ".part")
    archive = dest_dir / (spec.filename + ".archive.part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": f"showroom-coach/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            length = resp.headers.get("Content-Length")
            if length and length.isdigit():
                _check_free_disk(dest_dir, int(length))
            if spec.archive_member is None:
                digest = _copy(resp, part)
            else:
                _copy(resp, archive)
                digest = _extract_member(archive, spec.archive_member, part)
        if spec.sha256 and digest != spec.sha256.lower():
            raise _BadDownload(f"sha256 mismatch: got {digest}, expected {spec.sha256}")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)


def ensure_model(name: str, retries: int = 3, backoff: float = 2.0) -> Path:
    """Return the path of the model's .onnx file, downloading it first if it is missing.

    `retries` is the number of attempts; the wait between attempts doubles from `backoff`
    seconds. Raises ModelUnavailable when every attempt fails or free disk is below 3 GB.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
    spec = MODELS[name]
    dest_dir = models_dir() / name
    target = dest_dir / spec.filename
    if target.is_file():
        if spec.sha256 is None or _sha256_file(target) == spec.sha256.lower():
            return target
        log.warning("%s: sha256 differs from the registry; downloading again", name)
        target.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    attempts = max(1, retries)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        _check_free_disk(dest_dir)
        try:
            _download(spec, dest_dir, target)
            return target
        except (OSError, http.client.HTTPException, zipfile.BadZipFile, _BadDownload) as exc:
            last = exc
            log.warning("%s: attempt %d/%d failed: %s", name, attempt, attempts, exc)
            if attempt < attempts and backoff > 0:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise ModelUnavailable(f"{name}: download failed after {attempts} attempt(s): {last}") from last


def write_manifest() -> Path:
    """Write models/MANIFEST.json with one entry per model file that is present on disk."""
    root = models_dir()
    entries = []
    for name, spec in MODELS.items():
        f = root / name / spec.filename
        if not f.is_file():
            continue
        entry = {
            "name": name,
            "file": f"{name}/{spec.filename}",
            "url": spec.url,
            "sha256": _sha256_file(f),
            "bytes": f.stat().st_size,
            "licence": spec.licence,
            "trained_on": spec.trained_on,
            "source_page": spec.source_page,
        }
        if spec.archive_member:
            entry["archive_member"] = spec.archive_member
        if spec.note:
            entry["note"] = spec.note
        entries.append(entry)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "MANIFEST.json"
    path.write_text(json.dumps({"models": entries}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path
