"""Appearance embeddings: HSV histogram of the torso + (optional) DINOv2-small CLS token.

Only ever compared inside one clip (identity merge); never used to recognise people across visits.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from pulse.datatypes import TrackObs

log = logging.getLogger("pulse.embed")

HSV_BINS = (16, 4)
HSV_DIM = HSV_BINS[0] * HSV_BINS[1]
DINO_DIM = 384
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def hsv_hist(crop_bgr: np.ndarray, bins=HSV_BINS) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256]).flatten().astype(np.float64)
    return _unit(h)


def torso_crop(frame: np.ndarray, obs: TrackObs, kp_thr: float) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = obs.bbox
    kp = obs.keypoints
    box = None
    if kp is not None and all(kp[i][2] >= kp_thr for i in (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)):
        xs = [kp[i][0] for i in (_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)]
        sy = (kp[_L_SHOULDER][1] + kp[_R_SHOULDER][1]) / 2
        hy = (kp[_L_HIP][1] + kp[_R_HIP][1]) / 2
        if hy > sy:
            box = (min(xs), sy, max(xs), hy)
    if box is None:
        bh = y2 - y1
        box = (x1, y1 + 0.2 * bh, x2, y1 + 0.6 * bh)
    a, b = int(max(0, round(box[0]))), int(max(0, round(box[1])))
    c, d = int(min(W, round(box[2]))), int(min(H, round(box[3])))
    if c - a < 4 or d - b < 4:
        return None
    return frame[b:d, a:c]


class Embedder:
    def __init__(self, use_dino: bool = True, models_dir="models", providers=("CPUExecutionProvider",)):
        self.session = None
        self.dim = HSV_DIM
        if use_dino:
            try:
                import onnxruntime as ort
                from pulse.perception.models import ensure_model
                path = ensure_model("dinov2_small", models_dir)
                self.session = ort.InferenceSession(str(path), providers=list(providers))
                self.dim = HSV_DIM + DINO_DIM
            except Exception as e:  # noqa: BLE001 - any failure -> HSV only
                log.warning("DINOv2 unavailable (%s); using HSV-only embeddings", e)

    @property
    def uses_dino(self) -> bool:
        return self.session is not None

    def dino(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            x = np.stack([
                (cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224),
                            interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - _MEAN) / _STD
                for c in crops[i:i + batch]]).transpose(0, 3, 1, 2)
            res = self.session.run(None, {"pixel_values": np.ascontiguousarray(x)})[0]
            out.append(res[:, 0])
        return np.concatenate(out) if out else np.zeros((0, DINO_DIM), np.float32)

    def embed_many(self, groups: dict, dino_keys=None) -> dict:
        """groups: key -> list of crops. Keys not in ``dino_keys`` (default: all) get HSV only
        (zero DINO part, so every vector has the same length). Empty groups are dropped."""
        keys = [k for k, c in groups.items() if c]
        use = set(keys if dino_keys is None else dino_keys) if self.uses_dino else set()
        flat = [c for k in keys if k in use for c in groups[k]]
        feats = self.dino(flat) if flat else None
        res, pos = {}, 0
        for k in keys:
            crops = groups[k]
            hsv = _unit(np.mean([hsv_hist(c) for c in crops], axis=0))
            if not self.uses_dino:
                res[k] = hsv
                continue
            if k in use:
                f = _unit(feats[pos:pos + len(crops)].astype(np.float64).mean(axis=0))
                pos += len(crops)
            else:
                f = np.zeros(DINO_DIM)
            res[k] = _unit(np.concatenate([0.5 * hsv, 0.5 * f]))
        return res

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        r = self.embed_many({0: list(crops)})
        return r[0] if r else np.zeros(self.dim)
