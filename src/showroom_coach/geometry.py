"""Calibration-free pinhole geometry: floor position, distance and facing from one camera.

Everything here is a pure function of a person's box and COCO-17 keypoints. A person's real height H is assumed
(``CameraConfig.person_height_m``); the apparent body height in pixels then gives depth Z = f * H / h.
When the body is cut off or sitting, a shorter body segment (torso) or the box height is used instead and the
position is flagged accordingly (``Position.source``). An optional 4-point floor homography overrides all of it.

Floor coordinates: X to the right of the optical axis, Z away from the camera, both in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import CameraConfig

NOSE_ANKLE_RATIO = 0.89  # nose height 0.93 H - ankle height 0.04 H
SHOULDER_HIP_RATIO = 0.30  # shoulder 0.82 H - hip 0.52 H
SHOULDER_WIDTH_RATIO = 0.23  # biacromial width ~ 0.39 m for H = 1.70 m
STANDING_MIN_RATIO = 2.4  # (ankle_y - nose_y) / torso_px above this => standing
KPT_THR = 0.3

# COCO-17 indices
_NOSE = 0
_FACE = (1, 2, 3, 4)  # eyes and ears
_L_SHO, _R_SHO = 5, 6
_L_HIP, _R_HIP = 11, 12
_ANKLES = (15, 16)

_EPS = 1e-9
_FACING_AWAY_FZ = 0.9  # |fz| at or above which the heading is "straight", so the nose is not needed for its sign


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov_deg: float = 70.0
    person_height_m: float = 1.70
    floor_h: np.ndarray | None = field(default=None, compare=False, repr=False)  # image (u, v, 1) -> floor (X, Z, 1)

    @property
    def f(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @classmethod
    def from_config(cls, width: int, height: int, cfg: CameraConfig) -> "Camera":
        """Build a camera; four ``(u, v, X, Z)`` floor points in ``cfg`` add the image-to-floor homography."""
        floor_h = None
        pts = cfg.floor_points
        if pts is not None and len(pts) == 4:
            import cv2  # only needed here

            arr = np.asarray(pts, dtype=np.float64)
            if arr.shape == (4, 4) and np.isfinite(arr).all():
                try:
                    h, _ = cv2.findHomography(arr[:, :2], arr[:, 2:])
                except cv2.error:
                    h = None
                if h is not None and np.isfinite(h).all():
                    floor_h = np.asarray(h, dtype=np.float64)
        return cls(width, height, cfg.hfov_deg, cfg.person_height_m, floor_h)


@dataclass(frozen=True)
class Position:
    x: float
    z: float
    source: str  # "full" | "torso" | "box" | "floor"

    @property
    def approx(self) -> bool:
        return self.source == "box"

    @property
    def xz(self) -> tuple[float, float]:
        return (self.x, self.z)


# ----------------------------------------------------------------------------------------------- keypoints

def _as_kpts(kpts) -> np.ndarray | None:
    """(17, 3) float array from an array/nested list, or None when absent or malformed."""
    if kpts is None:
        return None
    try:
        a = np.asarray(kpts, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] != 17 or a.shape[1] < 3:
        return None
    return a


def _ok(a: np.ndarray, i: int) -> bool:
    return bool(a[i, 2] >= KPT_THR and np.isfinite(a[i, 0]) and np.isfinite(a[i, 1]))


def _head_y(a: np.ndarray) -> float | None:
    if _ok(a, _NOSE):
        return float(a[_NOSE, 1])
    ys = [a[i, 1] for i in _FACE if _ok(a, i)]
    return float(np.mean(ys)) if ys else None


def _ankles(a: np.ndarray) -> tuple[float, float] | None:
    """Mean (x, y) of the confident ankles."""
    pts = [a[i, :2] for i in _ANKLES if _ok(a, i)]
    if not pts:
        return None
    m = np.mean(pts, axis=0)
    return float(m[0]), float(m[1])


def _torso_px(a: np.ndarray) -> float | None:
    """Shoulder-to-hip distance in px: mid-shoulders to mid-hips when both sides are confident, else one side."""
    if all(_ok(a, i) for i in (_L_SHO, _R_SHO, _L_HIP, _R_HIP)):
        d = (a[_L_SHO, :2] + a[_R_SHO, :2]) / 2.0 - (a[_L_HIP, :2] + a[_R_HIP, :2]) / 2.0
    else:
        for s, h in ((_L_SHO, _L_HIP), (_R_SHO, _R_HIP)):
            if _ok(a, s) and _ok(a, h):
                d = a[s, :2] - a[h, :2]
                break
        else:
            return None
    t = float(math.hypot(d[0], d[1]))
    return t if t > _EPS else None


def _box_height(bbox) -> float:
    return max(float(bbox[3]) - float(bbox[1]), 1.0)


def body_height_px(bbox, kpts) -> tuple[float, str]:
    """Equivalent full-body height in px and where it came from: "full" | "torso" | "box"."""
    a = _as_kpts(kpts)
    if a is not None:
        torso = _torso_px(a)
        head_y = _head_y(a)
        ankles = _ankles(a)
        if head_y is not None and ankles is not None:
            span = ankles[1] - head_y
            if span > _EPS and (torso is None or span / torso >= STANDING_MIN_RATIO - _EPS):
                return span / NOSE_ANKLE_RATIO, "full"
        if torso is not None:
            return torso / SHOULDER_HIP_RATIO, "torso"
    return _box_height(bbox), "box"


# ----------------------------------------------------------------------------------------------- position

def _to_floor(cam: Camera, u: float, v: float) -> tuple[float, float] | None:
    p = cam.floor_h @ np.array([u, v, 1.0])
    w = p[2]
    if not np.isfinite(w) or abs(w) < 1e-12:
        return None
    x, z = float(p[0] / w), float(p[1] / w)
    return (x, z) if math.isfinite(x) and math.isfinite(z) else None


def position_m(bbox, kpts, cam: Camera) -> Position:
    """Floor position (X right, Z away from the camera, metres) of a person."""
    a = _as_kpts(kpts)
    if cam.floor_h is not None:
        ank = _ankles(a) if a is not None else None
        foot = ank if ank is not None else ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
        xz = _to_floor(cam, float(foot[0]), float(foot[1]))
        if xz is not None:
            return Position(xz[0], xz[1], "floor")

    h, src = body_height_px(bbox, kpts)
    z = cam.f * cam.person_height_m / h
    if a is not None and _ok(a, _L_HIP) and _ok(a, _R_HIP):
        u = float((a[_L_HIP, 0] + a[_R_HIP, 0]) / 2.0)
    else:
        u = (float(bbox[0]) + float(bbox[2])) / 2.0
    x = (u - cam.cx) * z / cam.f
    return Position(float(x), float(z), src)


def _xz(p) -> tuple[float, float]:
    return p.xz if isinstance(p, Position) else (float(p[0]), float(p[1]))


def distance_m(a, b) -> float:
    """Floor distance in metres between two ``Position`` objects or ``(x, z)`` pairs."""
    ax, az = _xz(a)
    bx, bz = _xz(b)
    return math.hypot(ax - bx, az - bz)


# ----------------------------------------------------------------------------------------------- facing

def facing_vector(bbox, kpts, cam: Camera) -> tuple[float, float] | None:
    """Unit floor direction (dx, dz) the person faces, or None when it cannot be told.

    The left/right shoulder order gives the sign of dz (toward or away from the camera) and the apparent shoulder
    width relative to its expected width gives its size; the nose offset from the shoulder midpoint gives the
    sign of dx. Needs both shoulders. With the nose hidden the heading is only known when the person is nearly
    front- or back-on (|dz| >= 0.9), in which case dx is taken as 0.
    """
    a = _as_kpts(kpts)
    if a is None or not (_ok(a, _L_SHO) and _ok(a, _R_SHO)):
        return None
    h, _ = body_height_px(bbox, a)
    exp_dx = SHOULDER_WIDTH_RATIO * h
    fz = float(np.clip(-(a[_L_SHO, 0] - a[_R_SHO, 0]) / exp_dx, -1.0, 1.0))
    fx_mag = math.sqrt(max(0.0, 1.0 - fz * fz))

    sign = 0.0
    if _ok(a, _NOSE):
        off = float(a[_NOSE, 0] - (a[_L_SHO, 0] + a[_R_SHO, 0]) / 2.0)
        if abs(off) > _EPS:
            sign = 1.0 if off > 0 else -1.0
    if sign == 0.0:
        if abs(fz) < _FACING_AWAY_FZ - _EPS:
            return None
        fx_mag = 0.0
    fx = sign * fx_mag
    n = math.hypot(fx, fz)
    return (fx / n, fz / n)


def faces(a_xz, a_facing, b_xz, max_deg: float = 60.0) -> bool | None:
    """Whether A (at ``a_xz`` heading ``a_facing``) faces B within ``max_deg``; None when unknown."""
    if a_facing is None:
        return None
    fx, fz = float(a_facing[0]), float(a_facing[1])
    dx, dz = float(b_xz[0]) - float(a_xz[0]), float(b_xz[1]) - float(a_xz[1])
    nf, nd = math.hypot(fx, fz), math.hypot(dx, dz)
    if nf < _EPS or nd < _EPS:
        return None
    cos = (fx * dx + fz * dz) / (nf * nd)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= max_deg + _EPS


def at_frame_edge(bbox, width, height, margin: float = 0.01) -> bool:
    """True when the box touches (within ``margin`` of the frame size) any image border."""
    mx, my = margin * width, margin * height
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return bool(x1 <= mx + _EPS or y1 <= my + _EPS or x2 >= width - mx - _EPS or y2 >= height - my - _EPS)
