"""Image <-> showroom floor geometry for a fixed, calibrated camera.

Convention (same as sim/ gt.json `camera`): x_cam = R @ x_world + t, OpenCV axes, pixel origin at
the top-left corner, world z up, floor at z = 0, distances in metres. No lens distortion.

A person's floor position is taken from the bottom-centre of their box (≈ feet). This is exact
when the feet are visible; when the lower body is hidden (car, desk, frame edge) the point lands
too close to the camera, so callers should prefer boxes whose bottom edge is not occluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world->camera rotation
    t: np.ndarray  # 3 world->camera translation

    @classmethod
    def from_gt(cls, cam: dict) -> "Camera":
        return cls(np.asarray(cam["K"], float), np.asarray(cam["R"], float), np.asarray(cam["t"], float))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def pixel_to_floor(self, uv) -> np.ndarray:
        """(N,2) pixels -> (N,2) floor xy in metres. NaN where the ray does not hit the floor."""
        return self.pixel_to_plane(uv, 0.0)

    def pixel_to_plane(self, uv, z: float) -> np.ndarray:
        """(N,2) pixels -> (N,2) world xy where the pixel ray crosses the horizontal plane at height z."""
        uv = np.atleast_2d(np.asarray(uv, float))
        rays = np.linalg.inv(self.K) @ np.vstack([uv.T, np.ones(len(uv))])
        d = self.R.T @ rays  # ray directions in world
        c = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (z - c[2]) / d[2]
        xy = (c[:2, None] + lam * d[:2]).T
        xy[~(lam > 0)] = np.nan  # plane is behind the camera along this ray
        return xy

    def height_above(self, xy, v_top: float, u: float) -> float:
        """Height (m) of the point straight above floor point xy that images at row v_top (column u)."""
        c = self.center
        ray = self.R.T @ np.linalg.inv(self.K) @ np.array([u, v_top, 1.0])
        # closest approach of the camera ray to the vertical line through xy, measured on that line
        h = np.asarray(xy, float) - c[:2]
        s = (ray[:2] @ h) / (ray[:2] @ ray[:2])
        return float(c[2] + s * ray[2])

    def floor_polygon_to_pixel(self, poly, near_z: float = 0.1) -> np.ndarray | None:
        """Floor polygon -> pixel polygon, clipped to the part in front of the camera (z_cam >= near_z).

        Projecting vertices behind the camera directly flips them across the image; clipping first keeps
        the outline correct for footprints that are only partly in view. None if fully behind.
        """
        v = np.asarray(poly, float)
        pc = (self.R @ np.vstack([v.T, np.zeros(len(v))]) + self.t[:, None]).T  # camera coords
        out = []
        for a, b in zip(pc, np.roll(pc, -1, axis=0)):  # Sutherland-Hodgman against the near plane
            ina, inb = a[2] >= near_z, b[2] >= near_z
            if ina:
                out.append(a)
            if ina != inb:
                out.append(a + (near_z - a[2]) / (b[2] - a[2]) * (b - a))
        if len(out) < 3:
            return None
        p = (self.K @ np.array(out).T)
        return (p[:2] / p[2]).T

    def floor_to_pixel(self, xy) -> np.ndarray:
        """(N,2) floor xy in metres -> (N,2) pixels."""
        xy = np.atleast_2d(np.asarray(xy, float))
        xw = np.vstack([xy.T, np.zeros(len(xy))])
        p = self.K @ (self.R @ xw + self.t[:, None])
        return (p[:2] / p[2]).T


def foot_point(box_xyxy) -> tuple[float, float]:
    x1, _, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, float(y2)


PERSON_HEIGHT_M = 1.7
MIN_PLAUSIBLE_HEIGHT_M = 1.4


def person_floor_xy(cam: Camera, box_xyxy, assumed_height_m: float = PERSON_HEIGHT_M,
                    min_height_m: float = MIN_PLAUSIBLE_HEIGHT_M) -> tuple[np.ndarray, str]:
    """Floor xy of a person from their box, robust to a hidden lower body.

    Uses the feet (box bottom-centre) when the box implies a plausible standing height; otherwise the
    bottom edge is an occluder (car, desk) and the head (box top-centre) is projected onto the plane at
    `assumed_height_m`. Returns (xy, "feet" | "head").
    """
    x1, y1, x2, _ = box_xyxy
    u = (x1 + x2) / 2.0
    xy = cam.pixel_to_floor([foot_point(box_xyxy)])[0]
    if not np.any(np.isnan(xy)) and cam.height_above(xy, float(y1), u) >= min_height_m:
        return xy, "feet"
    return cam.pixel_to_plane([(u, float(y1))], assumed_height_m)[0], "head"


def distance_to_polygon(xy, poly) -> float:
    """Euclidean distance (m) from point to a polygon footprint; 0 inside."""
    p = np.asarray(xy, float)
    v = np.asarray(poly, float)
    if _inside(p, v):
        return 0.0
    a, b = v, np.roll(v, -1, axis=0)
    ab = b - a
    s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + s[:, None] * ab - p, axis=1)))


def _inside(p: np.ndarray, v: np.ndarray) -> bool:
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(v, np.roll(v, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
