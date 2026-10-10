"""Pinhole-camera geometry shared with Showroom Coach (spec 4.3): pixel -> metres, facing, homography.

Ground coordinates are (x, z) in metres: x to the right of the optical axis, z forward (depth).
Dependencies kept to numpy + cv2.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import cv2
import numpy as np

from pulse.datatypes import TrackObs

NOSE_TO_ANKLE_FRAC = 0.88
TORSO_FRAC = 0.30  # shoulder-mid to hip-mid as a fraction of body height

NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 11, 12, 15, 16


def focal_px(frame_w: int, hfov_deg: float) -> float:
    return frame_w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def _conf(keypoints, i: int, kp_thr: float) -> bool:
    return keypoints is not None and len(keypoints) > i and keypoints[i][2] >= kp_thr


def foot_point(bbox, keypoints, kp_thr: float = 0.3) -> tuple[float, float]:
    """Mean of confident ankles, else bottom-centre of the box (pixels)."""
    ank = [keypoints[i] for i in (L_ANK, R_ANK) if _conf(keypoints, i, kp_thr)]
    if ank:
        return (float(np.mean([a[0] for a in ank])), float(np.mean([a[1] for a in ank])))
    return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


@dataclass
class PosEstimate:
    x: float
    z: float
    method: str


def position_m(obs: TrackObs, frame_size: tuple[int, int], hfov_deg: float = 70.0,
               person_h: float = 1.70, kp_thr: float = 0.3,
               floor_h: np.ndarray | None = None) -> PosEstimate:
    kps = obs.keypoints
    u_foot, v_foot = foot_point(obs.bbox, kps, kp_thr)
    if not (math.isfinite(u_foot) and math.isfinite(v_foot)):
        return PosEstimate(math.nan, math.nan, "invalid")
    if floor_h is not None:
        x, y = to_floor(floor_h, (u_foot, v_foot))
        if not (math.isfinite(x) and math.isfinite(y)):
            return PosEstimate(math.nan, math.nan, "invalid")
        return PosEstimate(x, y, "floor")

    w = frame_size[0]
    f = focal_px(w, hfov_deg)
    method, h_px = "box", None
    ankles = [kps[i][1] for i in (L_ANK, R_ANK) if _conf(kps, i, kp_thr)]
    full_px = None
    if _conf(kps, NOSE, kp_thr) and ankles:
        # same ankle estimate as the foot point (mean of confident ankles)
        full_px = (float(np.mean(ankles)) - kps[NOSE][1]) / NOSE_TO_ANKLE_FRAC
    if full_px is not None and full_px > 0:
        h_px = full_px
        method = "full"
    elif all(_conf(kps, i, kp_thr) for i in (L_SH, R_SH, L_HIP, R_HIP)):
        sh_y = (kps[L_SH][1] + kps[R_SH][1]) / 2.0
        hip_y = (kps[L_HIP][1] + kps[R_HIP][1]) / 2.0
        if hip_y > sh_y:
            h_px = (hip_y - sh_y) / TORSO_FRAC
            method = "torso"
    if h_px is None or not math.isfinite(h_px) or h_px <= 0:
        h_px = obs.bbox[3] - obs.bbox[1]
        method = "box"
    if not math.isfinite(h_px) or h_px <= 0:
        return PosEstimate(math.nan, math.nan, "invalid")
    z = f * person_h / h_px
    x = (u_foot - w / 2.0) * z / f
    return PosEstimate(float(x), float(z), method)


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    if not all(math.isfinite(v) for v in (*a, *b)):
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])


def facing_dir(keypoints, kp_thr: float = 0.3) -> tuple[float, float] | None:
    """Coarse facing direction on the ground plane (x, z); None when unknown."""
    if not (_conf(keypoints, L_SH, kp_thr) and _conf(keypoints, R_SH, kp_thr)):
        return None
    l_sh, r_sh = keypoints[L_SH], keypoints[R_SH]
    sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
    sw = abs(r_sh[0] - l_sh[0])
    if _conf(keypoints, L_HIP, kp_thr) and _conf(keypoints, R_HIP, kp_thr):
        hip_mid = ((keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2.0,
                   (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2.0)
        th = math.hypot(hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1])
    else:
        # hips missing: fall back to box/3, estimated from the visible keypoints' vertical extent
        ys = [k[1] for k in keypoints if k[2] >= kp_thr]
        th = (max(ys) - min(ys)) / 3.0 if ys else 0.0
    if sw < 0.25 * th and _conf(keypoints, NOSE, kp_thr):
        d = keypoints[NOSE][0] - sh_mid[0]
        return (float((d > 0) - (d < 0)), 0.0)
    if l_sh[0] > r_sh[0]:
        return (0, -1)
    return (0, 1)


def faces(a_obs: TrackObs, a_pos, b_pos, max_angle_deg: float = 60) -> bool | None:
    d = facing_dir(a_obs.keypoints)
    if d is None:
        return None
    ax, az = (a_pos.x, a_pos.z) if hasattr(a_pos, "x") else a_pos
    bx, bz = (b_pos.x, b_pos.z) if hasattr(b_pos, "x") else b_pos
    vx, vz = bx - ax, bz - az
    nv, nd = math.hypot(vx, vz), math.hypot(*d)
    if nv == 0 or nd == 0:
        return None
    cos = max(-1.0, min(1.0, (d[0] * vx + d[1] * vz) / (nv * nd)))
    return math.degrees(math.acos(cos)) <= max_angle_deg


def fit_homography(img_pts, floor_pts) -> np.ndarray:
    if len(img_pts) < 4 or len(floor_pts) < 4 or len(img_pts) != len(floor_pts):
        raise ValueError("fit_homography needs at least 4 matching image/floor point pairs "
                         f"(got {len(img_pts)} and {len(floor_pts)})")
    h, _ = cv2.findHomography(np.float32(img_pts), np.float32(floor_pts), 0)
    if h is None:
        raise ValueError("homography could not be fitted from the given points")
    return h


def to_floor(h: np.ndarray, pt) -> tuple[float, float]:
    v = np.asarray(h, dtype=np.float64) @ np.array([pt[0], pt[1], 1.0])
    if v[2] == 0 or not np.all(np.isfinite(v)):
        return (math.nan, math.nan)
    return (float(v[0] / v[2]), float(v[1] / v[2]))
