"""Appearance descriptors for identity: person / torso crops, HSV histograms and the DINOv2-small embedder.

Everything here describes body and clothing appearance, never faces; nothing recognises or stores who anyone
is. Crops are BGR uint8 arrays cut from a frame; an unusable crop (None, empty) yields a zero descriptor,
which `identity` treats as "no appearance information".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..models import ensure_model
from . import rtm_onnx as R

KPT_CONF = 0.3          # keypoint confidence needed to place the torso box (same bar as geometry.KPT_THR)
MIN_CROP_PX = 2         # a crop with a side shorter than this is useless
TORSO_PAD = 0.15
TORSO_ROWS = (0.20, 0.55)   # fallback torso: these fractions of the person box height, full box width
_L_SHO, _R_SHO, _L_HIP, _R_HIP = 5, 6, 11, 12
_EPS = 1e-12


# --------------------------------------------------------------------------- crops


def _cut(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Copy of frame[y1:y2, x1:x2] with the (rounded) box clipped to the frame; None if it is too small."""
    if frame is None or frame.ndim < 2 or frame.size == 0:
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    h, w = frame.shape[:2]
    xa, xb = max(0, int(round(x1))), min(w, int(round(x2)))
    ya, yb = max(0, int(round(y1))), min(h, int(round(y2)))
    if xb - xa < MIN_CROP_PX or yb - ya < MIN_CROP_PX:
        return None
    return frame[ya:yb, xa:xb].copy()


def person_crop(frame: np.ndarray, bbox, pad: float = 0.05) -> np.ndarray | None:
    """The person's box plus `pad` (fraction of the box size on every side), clipped to the frame.
    None for a degenerate box or one outside the frame."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if not (w > 0 and h > 0):
        return None
    return _cut(frame, x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h)


def _torso_box(kpts) -> tuple[float, float, float, float] | None:
    """Shoulders-to-hips box (x1, y1, x2, y2) from confident keypoints, or None when it cannot be placed."""
    if kpts is None:
        return None
    try:
        a = np.asarray(kpts, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] != 17 or a.shape[1] < 3:
        return None
    idx = (_L_SHO, _R_SHO, _L_HIP, _R_HIP)
    pts = a[list(idx)]
    if not (np.isfinite(pts).all() and (pts[:, 2] >= KPT_CONF).all()):
        return None
    sho, hip = pts[:2], pts[2:]
    if hip[:, 1].mean() <= sho[:, 1].mean():          # hips above the shoulders: the pose is nonsense
        return None
    x1, x2 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y1, y2 = float(sho[:, 1].min()), float(hip[:, 1].max())
    px, py = TORSO_PAD * (x2 - x1), TORSO_PAD * (y2 - y1)
    return x1 - px, y1 - py, x2 + px, y2 + py


def torso_crop(frame: np.ndarray, bbox, kpts) -> np.ndarray | None:
    """The clothing region: shoulders to hips (+15% pad) when those four keypoints are confident, otherwise
    rows 20%-55% of the person box. None when even the box fallback is degenerate."""
    box = _torso_box(kpts)
    if box is not None:
        crop = _cut(frame, *box)
        if crop is not None:
            return crop
    x1, y1, x2, y2 = (float(v) for v in bbox)
    h = y2 - y1
    if not (x2 - x1 > 0 and h > 0):
        return None
    return _cut(frame, x1, y1 + TORSO_ROWS[0] * h, x2, y1 + TORSO_ROWS[1] * h)


# --------------------------------------------------------------------------- HSV histogram


def usable_crop(crop) -> bool:
    """Whether `crop` is a non-empty H x W x 3 image that can be histogrammed or embedded."""
    return crop is not None and getattr(crop, "size", 0) > 0 and crop.ndim == 3 and crop.shape[2] == 3


def hsv_hist(crop_bgr: np.ndarray, bins=(8, 4, 4)) -> np.ndarray:
    """Joint H-S-V colour histogram of a crop, flattened (8*4*4 = 128-d) and L2-normalised.
    A missing or empty crop gives zeros (of the same length)."""
    out_len = int(np.prod(bins))
    if not usable_crop(crop_bgr):
        return np.zeros(out_len, np.float32)
    if crop_bgr.dtype != np.uint8:
        crop_bgr = np.clip(crop_bgr, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [int(b) for b in bins], [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float64)
    norm = float(np.linalg.norm(hist))
    if norm < _EPS:
        return np.zeros(out_len, np.float32)
    return (hist / norm).astype(np.float32)


# --------------------------------------------------------------------------- DINOv2


class Embedder:
    """DINOv2-small appearance embedding (384-d, L2-normalised) of BGR crops.

    Crops are resized to 224x224, converted to RGB and normalised with the ImageNet mean/std. The embedding is
    `pooler_output` when the ONNX export has one, else the CLS token of `last_hidden_state` (the Xenova int8
    export has only the latter, which equals the pooled output). Crops go through the model in batches of
    `batch`; a model with a fixed batch axis is fed that many at a time (padded) instead. Rows for unusable crops
    are zeros.
    """

    def __init__(self, name: str = "dinov2_small", batch: int = 8, threads: int | None = None):
        self.name = name
        self.session = R.make_session(ensure_model(name), threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        self._fixed_batch = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else None
        self.batch = self._fixed_batch or max(1, int(batch))
        self.size = tuple(int(s) if isinstance(s, int) and s > 0 else 224 for s in shape[2:4])  # (H, W)
        outs = self.session.get_outputs()
        names = [o.name for o in outs]
        self._out_name = "pooler_output" if "pooler_output" in names else (
            "last_hidden_state" if "last_hidden_state" in names else names[0])
        last = list(outs[names.index(self._out_name)].shape)[-1]
        self.dim = int(last) if isinstance(last, int) and last > 0 else 384

    def _blob(self, crops: list[np.ndarray]) -> np.ndarray:
        h, w = self.size
        imgs = []
        for c in crops:
            if c.dtype != np.uint8:
                c = np.clip(c, 0, 255).astype(np.uint8)
            interp = cv2.INTER_AREA if c.shape[0] * c.shape[1] > h * w else cv2.INTER_LINEAR
            imgs.append(cv2.cvtColor(cv2.resize(c, (w, h), interpolation=interp), cv2.COLOR_BGR2RGB))
        x = np.stack(imgs).astype(np.float32) / 255.0
        x = (x - R.IMAGENET_MEAN) / R.IMAGENET_STD
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def _run(self, crops: list[np.ndarray]) -> np.ndarray:
        n = len(crops)
        if self._fixed_batch and n < self._fixed_batch:
            crops = crops + [crops[-1]] * (self._fixed_batch - n)
        (out,) = self.session.run([self._out_name], {self.input_name: self._blob(crops)})
        out = np.asarray(out, dtype=np.float32)
        vec = out[:, 0] if out.ndim == 3 else out
        return vec[:n]

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """`crops` -> (N, dim) float32, rows L2-normalised (zeros for crops that cannot be embedded)."""
        out = np.zeros((len(crops), self.dim), np.float32)
        todo = [i for i, c in enumerate(crops) if usable_crop(c)]
        for s in range(0, len(todo), self.batch):
            chunk = todo[s:s + self.batch]
            vec = self._run([crops[i] for i in chunk])
            vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), _EPS)
            out[chunk] = vec
        return out
