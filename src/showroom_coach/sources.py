"""Video frame source: probe, sample, resize and trim.

Frames are decoded with OpenCV, sampled at ``sample_fps`` on the source timeline (frame ``i`` is at
``i / src_fps``; the container's own timestamps are not trusted), resized to at most ``max_width`` pixels
wide (even dimensions) and cut off at ``max_seconds``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

_EPS = 1e-6
_MAX_FPS = 240.0           # containers with no real frame rate report 1000 or 90000
_MIN_READ_FRACTION = 0.9   # fewer decoded frames than this share of the expected count means a damaged file


class VideoError(Exception):
    """The input cannot be decoded as video (missing, corrupt, not a video, zero frames, bad fps)."""


@dataclass
class VideoInfo:
    path: str
    src_width: int
    src_height: int
    src_fps: float
    src_frames: int
    duration: float          # seconds of the processed part (after trimming)
    src_duration: float
    width: int               # processed size (max width 960, even numbers)
    height: int
    scale: float             # processed / source
    sample_fps: float
    trimmed: bool
    max_seconds: float


def _even(x: float) -> int:
    return max(2, 2 * int(round(x / 2.0)))


def _target_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Processed size: never upscaled, at most ``max_width`` wide, aspect kept, both sides even."""
    tw = min(width, int(max_width))
    tw = max(2, tw - tw % 2)
    th = _even(height * tw / width)
    return tw, th


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink ``frame`` to at most ``max_width`` px wide, keeping the aspect ratio, with even dimensions."""
    h, w = frame.shape[:2]
    tw, th = _target_size(w, h, max_width)
    if (tw, th) == (w, h):
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def decode_jpeg(data: bytes, max_width: int = 960) -> np.ndarray:
    """Decode JPEG/PNG bytes (e.g. a live-camera frame) to a resized BGR image; ``VideoError`` on bad bytes."""
    if not data:
        raise VideoError("cannot decode image: empty data")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise VideoError("cannot decode image") from exc
    if img is None or img.size == 0:
        raise VideoError("cannot decode image")
    return resize_max_width(img, max_width)


def _open(path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"cannot read video: {Path(str(path)).name}")
    return cap


def _fps(cap: cv2.VideoCapture, path) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0 or fps > _MAX_FPS:
        raise VideoError(f"cannot read video: {Path(str(path)).name}")
    return fps


def _frame_count(cap: cv2.VideoCapture) -> int:
    """The container's frame count, or 0 when it reports none (or nonsense)."""
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    return int(n) if math.isfinite(n) and n > 0 else 0


def probe(path, max_width: int = 960, sample_fps: float = 5.0, max_seconds: float = 300.0) -> VideoInfo:
    """Read size, fps and length of ``path``; ``VideoError`` if it is unreadable or has no frames."""
    name = Path(str(path)).name
    cap = _open(path)
    try:
        fps = _fps(cap, path)
        ok, first = cap.read()
        if not ok or first is None or first.size == 0:
            raise VideoError(f"cannot read video: {name}")
        src_h, src_w = first.shape[:2]
        n_frames = _frame_count(cap)
        if n_frames <= 0:                        # some containers do not report a count: count by grabbing
            n_frames = 1
            while cap.grab():
                n_frames += 1
    finally:
        cap.release()
    if n_frames <= 0:
        raise VideoError(f"cannot read video: {name}")
    src_duration = n_frames / fps
    tw, th = _target_size(src_w, src_h, max_width)
    return VideoInfo(
        path=str(path), src_width=src_w, src_height=src_h, src_fps=fps, src_frames=n_frames,
        duration=min(src_duration, float(max_seconds)), src_duration=src_duration,
        width=tw, height=th, scale=tw / src_w, sample_fps=float(sample_fps),
        trimmed=src_duration > max_seconds + _EPS, max_seconds=float(max_seconds),
    )


def iter_frames(path, sample_fps: float = 5.0, max_width: int = 960,
                max_seconds: float = 300.0) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(src_frame_idx, t_sec, frame_bgr_resized)`` at about ``sample_fps``, for ``t < max_seconds``.

    Frame ``i`` is kept when ``i / src_fps >= next_t - 1e-6`` and ``next_t`` then advances by ``1 / sample_fps``.
    Skipped frames are only ``grab()``-ed (no decode to an array); kept ones are ``retrieve()``-d.
    ``VideoError`` is raised on the first ``next()`` if the file cannot be read, and after the last readable
    frame if decoding stops before ``max_seconds`` having read under 90% of the container's frame count (damaged
    or cut-off file); a short file whose count matches what was read is fine.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    cap = _open(path)
    try:
        fps = _fps(cap, path)
        n_frames = _frame_count(cap)
        expected = min(n_frames, max_seconds * fps) if n_frames > 0 else 0.0
        step = 1.0 / sample_fps
        next_t = 0.0
        i = 0
        while cap.grab():
            t = i / fps
            if t >= max_seconds:
                break
            if t >= next_t - _EPS:
                next_t += step
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield i, t, resize_max_width(frame, max_width)
            i += 1
        else:                                    # grab() failed: end of file, or a decode error that looks like it
            if expected > 0 and i < _MIN_READ_FRACTION * expected:
                raise VideoError(
                    f"video stops early at {i / fps:.1f} s of ~{expected / fps:.1f} s: the file may be damaged")
    finally:
        cap.release()
