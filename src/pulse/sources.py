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
    return pick_fps(cap.get(cv2.CAP_PROP_FPS))


def pick_fps(fps) -> tuple[float, bool]:
    """Container fps, or (25, True) when it is 0/NaN/negative/missing."""
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


MAX_JPEG_PIXELS = 16_000_000
MAX_JPEG_SIDE = 8192


def _jpeg_dims(data: bytes) -> tuple[int, int]:
    """(width, height) from the first SOF marker, without decoding. ValueError if not a JPEG."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG image")
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0x01, 0xD8) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9 or m == 0xDA:                     # EOI / start of scan: no SOF before it
            break
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                break
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + max(seg, 2)
    raise ValueError("not a decodable image")


def decode_jpeg(data: bytes, max_width: int = 960) -> np.ndarray:
    if not data:
        raise ValueError("empty image data")
    w, h = _jpeg_dims(data)                            # reject decompression bombs before decoding
    if w < 1 or h < 1 or w > MAX_JPEG_SIDE or h > MAX_JPEG_SIDE or w * h > MAX_JPEG_PIXELS:
        raise ValueError("image dimensions too large")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as e:
        raise ValueError(f"not a decodable image: {e}") from e
    if img is None:
        raise ValueError("not a decodable image")
    return _resize(img, max_width)
