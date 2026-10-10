"""Create the Blender camera from config; write the projection test fixture.

Fixture: blender -b --factory-startup --python sim/blender/cameras.py -- --write-fixture
writes sim/tests/fixtures/blender_projection_ref.json using bpy_extras world_to_camera_view.
"""
import json
import math
import random
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.projection import blender_euler, blender_lens_mm  # noqa: E402


def make_camera(p, name="cam"):
    data = bpy.data.cameras.new(name)
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = blender_lens_mm(p["hfov_deg"], 36.0)
    data.clip_start, data.clip_end = 0.05, 200.0
    data.shift_x = data.shift_y = 0.0
    ob = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(ob)
    ob.location = p["position_m"]
    ob.rotation_mode = "XYZ"
    ob.rotation_euler = blender_euler(p["yaw_deg"], p["pitch_deg"])
    scene = bpy.context.scene
    scene.camera = ob
    scene.render.resolution_x, scene.render.resolution_y = p["width_px"], p["height_px"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    return ob


def write_fixture(out_path):
    from bpy_extras.object_utils import world_to_camera_view
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = random.Random(7)
    cams = [
        {"position_m": [0.5, 0.5, 3.0], "yaw_deg": 45.0, "pitch_deg": -35.0, "hfov_deg": 74.0, "width_px": 1280, "height_px": 720},
        {"position_m": [19.2, 13.1, 3.4], "yaw_deg": -140.0, "pitch_deg": -48.0, "hfov_deg": 92.0, "width_px": 1280, "height_px": 720},
        {"position_m": [10.0, 0.4, 2.7], "yaw_deg": 92.0, "pitch_deg": -26.0, "hfov_deg": 61.0, "width_px": 960, "height_px": 540},
    ]
    out = {"blender_version": bpy.app.version_string, "cameras": []}
    for i, p in enumerate(cams):
        ob = make_camera(p, f"fixture_cam{i}")
        bpy.context.view_layer.update()
        pts = []
        for _ in range(40):
            w = [rng.uniform(0, 20), rng.uniform(0, 14), rng.uniform(0, 2.0)]
            v = world_to_camera_view(bpy.context.scene, ob, __import__("mathutils").Vector(w))
            if v.z > 0.2:
                pts.append({"world": w, "ndc": [v.x, v.y, v.z]})
        out["cameras"].append({"params": p, "points": pts})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
    print("fixture written", out_path, sum(len(c["points"]) for c in out["cameras"]), "points")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--write-fixture" in argv:
        write_fixture(REPO / "sim" / "tests" / "fixtures" / "blender_projection_ref.json")
