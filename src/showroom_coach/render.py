"""Rendering helpers. This file holds only what the perception pipeline needs for privacy: the face region of a
person and the blur that hides it. Every rendered frame and every saved thumbnail goes through `blur_faces`; faces are
never recognised, matched or stored. (Annotated video, timelines and contact sheets are added in M3.)
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

KPT_THR = 0.3                 # a face keypoint must be at least this confident to place the face
FACE_KPTS = 5                 # COCO order: nose, left eye, right eye, left ear, right ear
FACE_EXPAND = 1.8             # the keypoint box is grown by this factor around its centre ...
FACE_MIN_SIDE = 0.5           # ... and is never narrower / shorter than this fraction of the person box width
FALLBACK_HEAD_FRACTION = 0.2  # without usable face keypoints: the top 20 % of the person box, full width
PIXELATE_DIV = 6              # blur: the region is first averaged down to 1/6 of its size (blocks of about 6 px)
_EPS = 1e-6                   # slack so exact box edges survive float arithmetic when rounded outward


def _floor(v: float) -> int:
    return int(math.floor(v + _EPS))


def _ceil(v: float) -> int:
    return int(math.ceil(v - _EPS))


def _face_points(kpts, kpt_thr: float) -> np.ndarray | None:
    """(K, 2) coordinates of the confident nose / eye / ear keypoints, or None when there are none."""
    if kpts is None:
        return None
    try:
        a = np.asarray(kpts, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] < FACE_KPTS or a.shape[1] < 3:
        return None
    a = a[:FACE_KPTS]
    ok = np.isfinite(a[:, :3]).all(axis=1) & (a[:, 2] >= kpt_thr)
    return a[ok, :2] if ok.any() else None


def face_region(bbox, kpts, kpt_thr: float = KPT_THR) -> tuple[int, int, int, int]:
    """Where to blur one person's face, as integer pixel corners (x1, y1, x2, y2) in the frame of `bbox`.

    With confident nose / eye / ear keypoints: their bounding box grown x1.8 about its centre, at least 0.5 x the
    person box width on each side. Without them, or when that region's centre falls outside the person's box (a
    stray keypoint): the top 20 % of the box, full width. The result may extend past the frame; `blur_faces` clips it.
    A box with non-finite coordinates yields an empty region (0, 0, 0, 0).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return 0, 0, 0, 0
    w = x2 - x1
    pts = _face_points(kpts, kpt_thr)
    if pts is not None:
        fx1, fy1 = pts.min(axis=0)
        fx2, fy2 = pts.max(axis=0)
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            half_w = max((fx2 - fx1) * FACE_EXPAND, FACE_MIN_SIDE * w) / 2.0
            half_h = max((fy2 - fy1) * FACE_EXPAND, FACE_MIN_SIDE * w) / 2.0
            return _floor(cx - half_w), _floor(cy - half_h), _ceil(cx + half_w), _ceil(cy + half_h)
    return _floor(x1), _floor(y1), _ceil(x2), _ceil(y1 + FALLBACK_HEAD_FRACTION * (y2 - y1))


def _obscure(roi: np.ndarray) -> np.ndarray:
    """Pixelate-then-blur: block averaging throws the detail away for good, the Gaussian smooths the blocks."""
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // PIXELATE_DIV), max(1, h // PIXELATE_DIV)), interpolation=cv2.INTER_AREA)
    blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(max(w, h) * 0.5) | 1)
    return cv2.GaussianBlur(blocks, (k, k), 0)


def blur_faces(frame: np.ndarray, bboxes: Sequence, kpts_list: Sequence | None = None) -> np.ndarray:
    """A copy of `frame` (BGR) with every person's face region blurred hard.

    `bboxes` are person boxes in the frame's pixels; `kpts_list[i]` is that person's (17, 3) keypoints or None
    (people past the end of a short `kpts_list` count as None, so nobody is left unblurred). Regions are clipped
    to the frame; the rest of the image is untouched and `frame` itself is never modified.
    """
    out = frame.copy()
    fh, fw = out.shape[:2]
    kpts_list = kpts_list if kpts_list is not None else ()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = face_region(bbox, kpts_list[i] if i < len(kpts_list) else None)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, fw), min(y2, fh)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out[y1:y2, x1:x2] = _obscure(out[y1:y2, x1:x2])
    return out
