"""Optional fine-tuned person detector (our own RF-DETR Nano ONNX export).

Set ``PULSE_RFDETR_FT`` to the .onnx path to make person detection use it; vehicle detection keeps
``Config.detector`` (the fine-tuned head knows only 'person'). Unset or empty means no change.
Kept free of heavy imports so the cache key can use it.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV = "PULSE_RFDETR_FT"
NAME = "rfdetr_ft"
SHA_CHARS = 12

_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def path_from_env() -> Path | None:
    """The configured weights path, or None when the variable is unset, empty or whitespace."""
    raw = os.environ.get(ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def require_file(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{ENV}={path}: fine-tuned detector weights not found")
    return Path(path)


def short_sha(path: Path) -> str:
    """First 12 hex chars of the file's sha256 (memoised per path, size and mtime)."""
    path = require_file(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()[:SHA_CHARS]
    return _SHA_CACHE[key]


def tag(path: Path) -> str:
    """'rfdetr_ft-<sha12>': names the exact weights in cache keys and metadata."""
    return f"{NAME}-{short_sha(path)}"


def tag_from_env() -> str | None:
    """The tag for the configured weights, or None when the variable is unset/empty."""
    p = path_from_env()
    return tag(p) if p is not None else None
