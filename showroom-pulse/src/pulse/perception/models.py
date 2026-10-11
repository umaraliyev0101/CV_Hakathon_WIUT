"""Model registry + downloader. Downloads are untrusted: each model lives in its own folder
``models/<name>/``; zips are extracted only after rejecting absolute paths and ``..``."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

_YX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
_RF = "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr"
_RTM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
_COCO = "COCO 2017"
_RF_TRAIN = "COCO 2017 (Objects365 pretrain)"
_BODY7 = "Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18)"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    file: str            # file name inside models/<name>/ (for zips: the member basename to find)
    kind: str            # yolox | rfdetr | rtmpose | embed
    licence: str
    trained_on: str
    input_size: int | tuple
    fallback: str | None
    sha256: str | None = None


def _s(name, url, file, kind, lic, trained, size, fb):
    return ModelSpec(name, url, file, kind, lic, trained, size, fb)


REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    _s("yolox_nano", f"{_YX}/yolox_nano.onnx", "yolox_nano.onnx", "yolox", "Apache-2.0", _COCO, 416, None),
    _s("yolox_tiny", f"{_YX}/yolox_tiny.onnx", "yolox_tiny.onnx", "yolox", "Apache-2.0", _COCO, 416, "yolox_nano"),
    _s("yolox_s", f"{_YX}/yolox_s.onnx", "yolox_s.onnx", "yolox", "Apache-2.0", _COCO, 640, "yolox_tiny"),
    _s("yolox_m", f"{_YX}/yolox_m.onnx", "yolox_m.onnx", "yolox", "Apache-2.0", _COCO, 640, "yolox_s"),
    _s("rfdetr_n", f"{_RF}/rfdetr_n_v142_384x384.onnx", "rfdetr_n_v142_384x384.onnx", "rfdetr",
       "Apache-2.0", _RF_TRAIN, 384, "yolox_tiny"),
    _s("rfdetr_s", f"{_RF}/rfdetr_s_v142_512x512.onnx", "rfdetr_s_v142_512x512.onnx", "rfdetr",
       "Apache-2.0", _RF_TRAIN, 512, "yolox_tiny"),
    _s("rtmpose_s", f"{_RTM}/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
       "end2end.onnx", "rtmpose", "Apache-2.0", _BODY7, (192, 256), None),
    _s("rtmpose_m", f"{_RTM}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
       "end2end.onnx", "rtmpose", "Apache-2.0", _BODY7, (192, 256), "rtmpose_s"),
    _s("dinov2_small", "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx",
       "model.onnx", "embed", "Apache-2.0 (facebook/dinov2-small; ONNX export by Xenova)",
       "LVD-142M", 224, None),
]}

UA = {"User-Agent": "showroom-pulse/0.1 (+local research; model fetch)"}


def _log(msg: str) -> None:
    print(f"[models] {msg}", file=sys.stderr)


def default_downloader(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def safe_extract(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract zip into out_dir, rejecting absolute paths and '..' components."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            p = Path(n.replace("\\", "/"))
            if n.startswith(("/", "\\")) or p.is_absolute() or ".." in p.parts:
                raise ValueError(f"unsafe path in zip: {n!r}")
            if root not in (root / p).resolve().parents and (root / p).resolve() != root:
                raise ValueError(f"zip member escapes target: {n!r}")
        zf.extractall(out_dir)
        return [out_dir / n for n in zf.namelist()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(models_dir: Path) -> Path:
    return Path(models_dir) / "MANIFEST.json"


def _read_manifest(models_dir: Path) -> dict:
    p = _manifest_path(models_dir)
    return json.loads(p.read_text()) if p.exists() else {"models": []}


def _record(models_dir: Path, spec: ModelSpec, path: Path) -> None:
    man = _read_manifest(models_dir)
    man["models"] = [m for m in man["models"] if m["name"] != spec.name]
    man["models"].append({
        "name": spec.name, "url": spec.url,
        "file": str(path.relative_to(models_dir)),
        "sha256": _sha256(path), "bytes": path.stat().st_size,
        "licence": spec.licence, "trained_on": spec.trained_on,
        "downloaded": date.today().isoformat(),
    })
    man["models"].sort(key=lambda m: m["name"])
    _manifest_path(models_dir).write_text(json.dumps(man, indent=2) + "\n")


def _find(folder: Path, fname: str) -> Path | None:
    direct = folder / fname
    if direct.exists():
        return direct
    hits = sorted(folder.rglob(fname))
    return hits[0] if hits else None


def _fetch(spec: ModelSpec, models_dir: Path, downloader, sleep, retries: int = 3) -> Path:
    folder = Path(models_dir) / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    is_zip = spec.url.endswith(".zip")
    target = folder / (f"{spec.name}.zip" if is_zip else spec.file)
    last: Exception | None = None
    for i in range(retries):
        try:
            downloader(spec.url, target)
            if is_zip:
                safe_extract(target, folder)
                found = _find(folder, spec.file)
                if found is None:
                    raise FileNotFoundError(f"{spec.file} not in {target.name}")
                return found
            return target
        except Exception as e:  # noqa: BLE001 - retry on any network/format error
            last = e
            _log(f"{spec.name}: attempt {i + 1}/{retries} failed: {e}")
            if i < retries - 1:
                sleep(2 ** i)
    raise RuntimeError(f"{spec.name}: download failed after {retries} attempts: {last}")


def ensure_model(name: str, models_dir="models", *, downloader: Callable | None = None,
                 sleep: Callable = time.sleep) -> Path:
    """Return the local model file, downloading if needed (3 retries, then fallback chain)."""
    models_dir = Path(models_dir)
    downloader = downloader or default_downloader
    spec = REGISTRY[name]
    while True:
        folder = models_dir / spec.name
        have = _find(folder, spec.file) if folder.exists() else None
        if have is not None:
            if not any(m["name"] == spec.name for m in _read_manifest(models_dir)["models"]):
                _record(models_dir, spec, have)
            return have
        try:
            path = _fetch(spec, models_dir, downloader, sleep)
        except RuntimeError as e:
            if spec.fallback is None:
                raise
            _log(f"{e}; falling back to {spec.fallback}")
            spec = REGISTRY[spec.fallback]
            continue
        _record(models_dir, spec, path)
        return path
