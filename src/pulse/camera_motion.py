"""Camera motion / scene-cut detection on small grayscale frames (ORB + RANSAC homography).

Used to find stable segments (zones are only valid while the camera does not move) and the keyframe.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .datatypes import StableSegment

GRAY_WIDTH = 320
MIN_MATCHES = 15
HIST_CORR_CUT = 0.5
SAME_VIEW_THR = 0.02
_MATCH_DIST = 64


def to_gray_small(frame_bgr: np.ndarray, width: int = GRAY_WIDTH) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if w <= width:
        return g
    return cv2.resize(g, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def _hist(gray: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([gray], [0], None, [64], [0, 256])
    return cv2.normalize(h, h).flatten()


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, bool]:
    """(mean corner displacement as a fraction of width, cut). A cut reports displacement 1.0."""
    if prev_gray.shape == gray.shape and np.array_equal(prev_gray, gray):
        return 0.0, False
    if float(cv2.compareHist(_hist(prev_gray), _hist(gray), cv2.HISTCMP_CORREL)) < HIST_CORR_CUT:
        return 1.0, True
    orb = cv2.ORB_create(500)
    k1, d1 = orb.detectAndCompute(prev_gray, None)
    k2, d2 = orb.detectAndCompute(gray, None)
    if d1 is None or d2 is None:
        return 1.0, True
    matches = [m for m in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
               if m.distance < _MATCH_DIST]
    if len(matches) < MIN_MATCHES:
        return 1.0, True
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or not np.all(np.isfinite(H)):
        return 1.0, True
    h, w = prev_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H)
    disp = float(np.mean(np.linalg.norm((moved - corners).reshape(-1, 2), axis=1))) / w
    return disp, False


def motion_flags(grays: list[np.ndarray], move_thr: float = 0.01) -> list[bool]:
    """moving[i]: frame i moved (or cut) relative to frame i-1; frame 0 copies frame 1."""
    n = len(grays)
    flags = [False] * n
    for i in range(1, n):
        d, cut = motion_score(grays[i - 1], grays[i])
        flags[i] = cut or d > move_thr
    if n > 1:
        flags[0] = flags[1]
    return flags


def pick_key_idx(flags: list[bool]) -> int:
    """Middle frame of the longest run of stable frames (whole video middle if none is stable)."""
    best, best_len, i, n = None, 0, 0, len(flags)
    while i < n:
        j = i
        while j + 1 < n and flags[j + 1] == flags[i]:
            j += 1
        if not flags[i] and j - i + 1 > best_len:
            best, best_len = (i, j), j - i + 1
        i = j + 1
    return n // 2 if best is None else (best[0] + best[1]) // 2


def segments(times, grays=None, key_idx: int | None = 0, move_thr: float = 0.01,
             flags: list[bool] | None = None) -> list[StableSegment]:
    """Segments of consecutive frames with the same moving/stable state.

    Segment ``t1`` is the start of the next segment (contiguous cover); the last ends at the last frame.
    Convenience: ``segments(video_path, cfg)`` reads the frames itself and uses the auto keyframe.
    """
    if isinstance(times, (str, Path)):
        return _segments_from_video(times, grays)
    n = len(times)
    if n == 0:
        return []
    flags = motion_flags(grays, move_thr) if flags is None else flags
    key_idx = pick_key_idx(flags) if key_idx is None else key_idx
    out: list[StableSegment] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and flags[j + 1] == flags[i]:
            j += 1
        t1 = times[j + 1] if j + 1 < n else times[j]
        same = False
        if not flags[i]:
            d, cut = motion_score(grays[key_idx], grays[(i + j) // 2])
            same = (not cut) and d < SAME_VIEW_THR
        out.append(StableSegment(float(times[i]), float(t1), not flags[i], same))
        i = j + 1
    return out


def _segments_from_video(path, cfg) -> list[StableSegment]:
    from .sources import iter_frames
    times, grays = [], []
    for t, f in iter_frames(path, cfg.sample_fps, cfg.max_width, cfg.max_duration_s):
        times.append(t)
        grays.append(to_gray_small(f))
    return segments(times, grays, key_idx=None)
