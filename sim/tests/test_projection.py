import json
from pathlib import Path

import numpy as np

from sim import projection

FIX = Path(__file__).parent / "fixtures" / "blender_projection_ref.json"


def test_matches_blender_world_to_camera_view():
    """Reference values written by Blender's bpy_extras.object_utils.world_to_camera_view
    (sim/blender/cameras.py --write-fixture); pixel v = (1 - y) * H.

    Blender evaluates in float32, so off-image points close to the camera carry ~1e-3 px of
    rounding; 0.01 px is still far below anything that could move a label."""
    tol_px = 0.01
    ref = json.loads(FIX.read_text())
    assert ref["blender_version"].startswith("4.")
    for cam in ref["cameras"]:
        P = projection.Camera.from_params(cam["params"])
        for item in cam["points"]:
            u, v, depth = P.project(np.array(item["world"]))
            nx, ny, nz = item["ndc"]
            assert abs(depth - nz) < 1e-4
            assert abs(u - nx * P.width) < tol_px, (item, u)
            assert abs(v - (1.0 - ny) * P.height) < tol_px, (item, v)


def test_center_ray(cfg):
    cam = projection.Camera.from_params(cfg["camera"])
    c = np.array(cfg["camera"]["position_m"], float)
    fwd = cam.forward()
    u, v, z = cam.project(c + 5.0 * fwd)
    assert abs(u - cam.width / 2) < 1e-6 and abs(v - cam.height / 2) < 1e-6 and abs(z - 5.0) < 1e-9
    assert abs(np.degrees(np.arcsin(fwd[2])) - cfg["camera"]["pitch_deg"]) < 1e-9


def test_floor_polygon_clipped_to_front(cfg):
    cam = projection.Camera.from_params(cfg["camera"])
    behind = [[-5, -5], [-4, -5], [-4, -4], [-5, -4]]
    assert projection.floor_polygon_to_image(cam, behind) == []
    pts = projection.floor_polygon_to_image(cam, [[2, 2], [6, 2], [6, 6], [2, 6]])
    assert len(pts) >= 3
