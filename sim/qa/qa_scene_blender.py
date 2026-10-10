"""Runs INSIDE Blender on a rendered scene.blend and writes measurements as JSON (no pass/fail here).

blender -b <scene_dir>/scene.blend --python sim/qa/qa_scene_blender.py -- --config C --timeline T --out J
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sim import schema  # noqa: E402


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def mesh_z_range(objs, dg):
    zmin, zmax = math.inf, -math.inf
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        for v in me.vertices:
            z = (mw @ v.co).z
            zmin, zmax = min(zmin, z), max(zmax, z)
        ev.to_mesh_clear()
    return zmin, zmax


def pose_signature(arm):
    if arm is None or arm.type != "ARMATURE":
        return []
    out = []
    for pb in arm.pose.bones:
        q = pb.matrix_basis.to_quaternion()
        out += [q.w, q.x, q.y, q.z]
    return out


def main():
    a = _args()
    cfg = schema.load_scene(a.config)
    tl = json.loads(Path(a.timeline).read_text())
    sc = bpy.context.scene
    frames = schema.output_scene_frames(cfg)
    res = {"persons": {}, "cars": {}, "images": {}}
    # cars: horizontal extent along the car's own axis (world bbox rotated by -yaw)
    for car in cfg["cars"]:
        objs = [o for o in bpy.data.objects if o.name.startswith(f"car{car['id']}_") and o.type == "MESH"]
        dg = bpy.context.evaluated_depsgraph_get()
        yaw = math.radians(car["yaw_deg"])
        c, s = math.cos(-yaw), math.sin(-yaw)
        us = []
        for ob in objs:
            mw = ob.matrix_world
            for v in ob.data.vertices:
                w = mw @ v.co
                us.append(c * w.x - s * w.y)
        res["cars"][str(car["id"])] = {"n_objects": len(objs), "length_m": (max(us) - min(us)) if us else None}
    # images: missing files
    for img in bpy.data.images:
        if img.source != "FILE":
            continue
        path = bpy.path.abspath(img.filepath)
        packed = img.packed_file is not None
        ok = (packed or os.path.exists(path)) and img.size[0] > 0 and img.size[1] > 0
        res["images"][img.name] = {"path": path, "packed": packed, "exists": ok}
    # people
    for p in cfg["persons"]:
        pid = p["id"]
        meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.get("sim_person") == pid]
        arm = bpy.data.objects.get(f"P{pid}_arm")
        fr = tl["agents"][str(pid)]["frames"]
        sample = [f for f in frames[::5] if fr[f]["rendered"]]
        lows, height = [], None
        sigs = {}
        for f in sample:
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            zmin, zmax = mesh_z_range(meshes, dg)
            lows.append({"f": f, "zmin": zmin, "action": fr[f]["action"], "anim": fr[f]["anim_clip"]})
            if height is None and fr[f]["action"] in ("idle", "talk") and not fr[f]["anim_clip"].startswith("turn"):
                height = zmax - zmin
        for f in frames:
            if fr[f]["rendered"] and f % 30 == 0:
                sc.frame_set(f)
                sigs[f] = pose_signature(arm) if arm else [o.matrix_world.translation.z for o in meshes]
        res["persons"][str(pid)] = {"kind": "rocketbox" if arm else "mannequin", "standing_height_m": height,
                                    "feet": lows, "pose_sig": {str(k): v for k, v in sigs.items()}}
    Path(a.out).write_text(json.dumps(res))
    print("QA_SCENE_DONE", a.out)


if __name__ == "__main__":
    main()
