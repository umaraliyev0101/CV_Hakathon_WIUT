"""Video sources: probe, sequential frame sampling, JPEG decode. Shared unit with Showroom Coach."""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FALLBACK_FPS = 25.0


class VideoError(Exception):
    pass


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    fps_assumed: bool = False  # True when the container fps was 0/NaN and 25 was used


def _unreadable(path) -> VideoError:
    return VideoError(f"Unreadable or unsupported video: {path}")


def _src_fps(cap) -> tuple[float, bool]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        return FALLBACK_FPS, True
    return float(fps), False


def probe(path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise _unreadable(path)
        ok, frame = cap.read()  # actually decode one frame
        if not ok or frame is None:
            raise _unreadable(path)
        fps, assumed = _src_fps(cap)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
        return VideoInfo(w, h, fps, n, n / fps if n > 0 else 0.0, assumed)
    finally:
        cap.release()


def sampled_frame_size(info: VideoInfo, max_width: int = 960) -> tuple[int, int]:
    if info.width <= max_width:
        return info.width, info.height
    return max_width, int(round(info.height * max_width / info.width))


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)


def iter_frames(path, sample_fps: float = 5.0, max_width: int = 960,
                max_duration: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise _unreadable(path)
        src_fps, _ = _src_fps(cap)
        pos = -1  # index of the last frame consumed (grabbed)
        k = 0
        while True:
            t = round(k / sample_fps, 3)
            if max_duration is not None and t >= max_duration:
                return
            target = int(round(k * src_fps / sample_fps))
            if target <= pos:  # source slower than sample rate: no new frame for this sample
                k += 1
                continue
            ok = True
            while pos < target - 1:
                if not cap.grab():
                    ok = False
                    break
                pos += 1
            if not ok:
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            pos += 1
            yield t, _resize(frame, max_width)
            k += 1
    finally:
        cap.release()


def decode_jpeg(data: bytes, max_width: int = 960) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return _resize(img, max_width)
