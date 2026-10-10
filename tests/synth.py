"""Synthetic pinhole people and scripted timelines for tests."""
import math
import numpy as np
from showroom_coach.geometry import Camera

# (height fraction of H, lateral metres (+ = person's left), forward metres) in COCO order
STAND = [
    (0.930, 0.00, 0.09), (0.945, 0.03, 0.07), (0.945, -0.03, 0.07), (0.935, 0.075, 0.0), (0.935, -0.075, 0.0),
    (0.820, 0.195, 0.0), (0.820, -0.195, 0.0), (0.630, 0.22, 0.0), (0.630, -0.22, 0.0),
    (0.480, 0.22, 0.03), (0.480, -0.22, 0.03), (0.520, 0.10, 0.0), (0.520, -0.10, 0.0),
    (0.285, 0.10, 0.0), (0.285, -0.10, 0.0), (0.040, 0.10, 0.0), (0.040, -0.10, 0.0),
]
SIT = [
    (0.680, 0.00, 0.09), (0.695, 0.03, 0.07), (0.695, -0.03, 0.07), (0.685, 0.075, 0.0), (0.685, -0.075, 0.0),
    (0.570, 0.195, 0.0), (0.570, -0.195, 0.0), (0.420, 0.22, 0.05), (0.420, -0.22, 0.05),
    (0.330, 0.20, 0.25), (0.330, -0.20, 0.25), (0.270, 0.10, 0.0), (0.270, -0.10, 0.0),
    (0.300, 0.10, 0.45), (0.300, -0.10, 0.45), (0.040, 0.10, 0.25), (0.040, -0.10, 0.25),
]
# Seated feet are tucked back (0.25 m ahead of the hips, knees still 0.45 m): with the feet 0.45 m ahead, perspective
# toward the camera pushed (ankle_y - nose_y) / torso_px to 2.41-2.45 at 4.4-5.4 m, over STANDING_MIN_RATIO (2.4),
# so a seated person read as standing. Tucked feet keep the worst case at 2.30 for z in 3-10 m.
CAM_HEIGHT = 1.6   # metres above the floor; optical axis horizontal


def project_person(cam: Camera, x: float, z: float, facing=(0.0, -1.0), H: float = 1.70, sitting: bool = False):
    """Return (bbox, kpts (17,3)) for a person standing/sitting at floor point (x, z)."""
    fx, fz = facing
    lx, lz = -fz, fx                      # person's left
    joints = SIT if sitting else STAND
    facing_camera = fz < 0.3
    k = np.zeros((17, 3))
    for i, (hf, lat, fwd) in enumerate(joints):
        X = x + lat * lx + fwd * fx
        Z = z + lat * lz + fwd * fz
        Y = CAM_HEIGHT - hf * H
        u, v = cam.cx + cam.f * X / Z, cam.cy + cam.f * Y / Z
        conf = 0.9
        if i <= 2 and not facing_camera:
            conf = 0.05
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            conf = 0.0
        k[i] = (u, v, conf)
    top_h = (0.75 if sitting else 1.0) * H
    us = [cam.cx + cam.f * (x + s * 0.25) / z for s in (-1, 1)]
    v_top = cam.cy + cam.f * (CAM_HEIGHT - top_h) / z
    v_bot = cam.cy + cam.f * CAM_HEIGHT / z
    bbox = (max(0.0, us[0]), max(0.0, v_top), min(cam.width - 1.0, us[1]), min(cam.height - 1.0, v_bot))
    return bbox, k


def unit(deg: float):
    """Floor direction from an angle in degrees: 0° = +X (image right), 90° = +Z (away from camera)."""
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
