"""Pinhole camera matching Blender's camera (sensor_fit HORIZONTAL) and world_to_camera_view (pure).

World: x east, y north, z up (metres). Camera frame (OpenCV): x right, y down, z forward.
Pixel (u, v) continuous with the image origin at the top-left corner, so Blender's
normalised (x, y) maps to u = x * W, v = (1 - y) * H.
"""
import math

import numpy as np


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def blender_euler(yaw_deg, pitch_deg):
    """Blender camera rotation_euler (XYZ) looking along heading yaw, tilted by pitch (neg = down)."""
    return (math.radians(90.0 + pitch_deg), 0.0, math.radians(yaw_deg - 90.0))


def blender_lens_mm(hfov_deg, sensor_width_mm=36.0):
    return (sensor_width_mm / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


class Camera:
    def __init__(self, K, R, t, width, height, params=None):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)
        self.width, self.height = int(width), int(height)
        self.params = params or {}

    @classmethod
    def from_params(cls, p):
        W, H = p["width_px"], p["height_px"]
        f = (W / 2.0) / math.tan(math.radians(p["hfov_deg"]) / 2.0)
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        ex, ey, ez = blender_euler(p["yaw_deg"], p["pitch_deg"])
        Rb = _rz(ez) @ _ry(ey) @ _rx(ex)          # camera -> world (Blender camera axes)
        D = np.diag([1.0, -1.0, -1.0])             # Blender cam axes -> OpenCV axes
        R = D @ Rb.T
        C = np.asarray(p["position_m"], float)
        return cls(K, R, -R @ C, W, H, dict(p))

    def center(self):
        return -self.R.T @ self.t

    def forward(self):
        return self.R[2]

    def to_cam(self, Xw):
        return self.R @ np.asarray(Xw, float) + self.t

    def project(self, Xw):
        Xc = self.to_cam(Xw)
        z = Xc[2]
        u = self.K[0, 0] * Xc[0] / z + self.K[0, 2]
        v = self.K[1, 1] * Xc[1] / z + self.K[1, 2]
        return float(u), float(v), float(z)

    def in_view(self, Xw, near=0.1):
        u, v, z = self.project(Xw)
        return z > near and 0 <= u < self.width and 0 <= v < self.height

    def to_dict(self):
        p = self.params
        return {"hfov_deg": p.get("hfov_deg"), "K": self.K.tolist(), "R": self.R.tolist(),
                "t": self.t.tolist(), "height_m": p.get("position_m", [0, 0, 0])[2],
                "pitch_deg": p.get("pitch_deg"), "yaw_deg": p.get("yaw_deg"),
                "position_m": list(p.get("position_m", [])),
                "convention": "x_cam = R @ x_world + t; OpenCV axes; pixel origin top-left corner"}


def floor_polygon_to_image(cam, poly_xy, near=0.1, z=0.0):
    """Project a floor polygon, clipping it to the camera's near plane first (Sutherland-Hodgman)."""
    pts = [cam.to_cam([x, y, z]) for x, y in poly_xy]
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            s = (near - a[2]) / (b[2] - a[2])
            out.append(a + s * (b - a))
    res = []
    for X in out:
        u = cam.K[0, 0] * X[0] / X[2] + cam.K[0, 2]
        v = cam.K[1, 1] * X[1] / X[2] + cam.K[1, 2]
        res.append([round(float(u), 2), round(float(v), 2)])
    return res if len(res) >= 3 else []
