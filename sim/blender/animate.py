"""Keyframe people from the precomputed timeline (runs inside Blender; no behaviour logic here).

Mannequins: joint angles from sim.behaviour.mannequin_pose. Rocketbox characters: clip choice,
clip frames and blend weights come from sim.anim_plan (pure, tested); here we only sample the
baked actions and write keys (fast path: keyframe_points.add + foreach_set).
"""
import math

import numpy as np

from sim import anim_plan
from sim.behaviour import mannequin_pose

HIDDEN_Z = -100.0


def _key(ob, path, frame):
    ob.keyframe_insert(data_path=path, frame=frame)


def animate_person(joints, frames, height_m, keep=None):
    root, pelvis = joints["root"], joints["pelvis"]
    for f, st in enumerate(frames):
        if keep is not None and f not in keep:
            continue
        x, y = st["pos_m"]
        root.location = (x, y, 0.0 if st["rendered"] else HIDDEN_Z)
        root.rotation_euler = (0.0, 0.0, math.radians(st["yaw_deg"]))
        pose = mannequin_pose(st, height_m)
        J = pose["joints"]
        pelvis.location.z = pose["pelvis_z"]
        for side in ("l", "r"):
            th, kn = J[f"{side}_hip"], J[f"{side}_knee"]
            joints[f"{side}_hip"].rotation_euler = (0.0, math.radians(-th), 0.0)
            joints[f"{side}_knee"].rotation_euler = (0.0, math.radians(kn), 0.0)
            joints[f"{side}_ankle"].rotation_euler = (0.0, math.radians(th - kn), 0.0)
            joints[f"{side}_shoulder"].rotation_euler = (0.0, math.radians(-J[f"{side}_shoulder"]), 0.0)
            joints[f"{side}_elbow"].rotation_euler = (0.0, math.radians(-J[f"{side}_elbow"]), 0.0)
        joints["head"].rotation_euler = (0.0, math.radians(J["head"]), 0.0)
        _key(root, "location", f)
        _key(root, "rotation_euler", f)
        _key(pelvis, "location", f)
        for name in ("l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
                     "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "head"):
            _key(joints[name], "rotation_euler", f)
    for ob in joints.values():
        if ob.animation_data and ob.animation_data.action:
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


# ---------------------------------------------------------------- Rocketbox (mocap clips)

def sample_action(action):
    """{(data_path, index): np.array over clip frames 1..N}; baked actions have a key per frame."""
    chans = {}
    n = int(round(action.frame_range[1]))
    for fc in action.fcurves:
        dp = fc.data_path
        if not (dp.endswith("rotation_quaternion") or dp in ("location", "rotation_euler")):
            continue
        k = len(fc.keyframe_points)
        co = np.empty(2 * k)
        fc.keyframe_points.foreach_get("co", co)
        xs, ys = co[0::2], co[1::2]
        grid = np.arange(1, n + 1, dtype=float)
        chans[(dp, fc.array_index)] = np.interp(grid, xs, ys)
    return {"n": n, "chans": chans}


def _at(arr, f):
    i = f - 1.0
    i0 = int(math.floor(i))
    i1 = min(i0 + 1, len(arr) - 1)
    u = i - i0
    return (1.0 - u) * arr[min(i0, len(arr) - 1)] + u * arr[i1]


def _fcurves(ob, specs, action_name):
    import bpy
    ob.animation_data_create()
    act = bpy.data.actions.new(action_name)
    ob.animation_data.action = act
    return {s: act.fcurves.new(s[0], index=s[1]) for s in specs}


def _write(fc, frames, values):
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames))
    co[0::2], co[1::2] = frames, values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [1] * len(frames))   # LINEAR
    fc.update()


def animate_rocketbox(rig, frames, plan, sampled, src_height_m, height_m, keep, gender):
    """rig: import_character() result; plan: anim_plan.plan(...) per scene frame; keep: scene
    frames to key (the rendered ones)."""
    keep = sorted(keep)
    root, arm = rig["root"], rig["arm"]
    s = height_m / src_height_m
    root.scale = (s, s, s)
    rf = _fcurves(root, [("location", 0), ("location", 1), ("location", 2), ("rotation_euler", 2)], root.name + "_act")
    xs = [frames[f]["pos_m"][0] for f in keep]
    ys = [frames[f]["pos_m"][1] for f in keep]
    zs = [0.0 if frames[f]["rendered"] else HIDDEN_Z for f in keep]
    yaws = [math.radians(frames[f]["yaw_deg"] + 90.0) for f in keep]
    yaws = np.unwrap(np.asarray(yaws)).tolist()
    for spec, vals in zip(rf, (xs, ys, zs, yaws)):
        _write(rf[spec], keep, vals)
    # channel set: union over clips used (all clips share the skeleton)
    any_clip = next(iter(sampled.values()))
    specs = sorted(any_clip["chans"].keys())
    af = _fcurves(arm, specs, arm.name + "_act")
    vals = {sp: [] for sp in specs}
    quat_paths = sorted({sp[0] for sp in specs if sp[0].endswith("rotation_quaternion")})
    for f in keep:
        entries = plan[f]
        ref = sampled[entries[-1][0]]
        for dp in quat_paths:
            acc = np.zeros(4)
            q0 = None
            for clip, cf, w in entries:
                ch = sampled[clip]["chans"]
                q = np.array([_at(ch[(dp, i)], cf) for i in range(4)])
                if q0 is None:
                    q0 = q
                elif np.dot(q, q0) < 0:
                    q = -q
                acc += w * q
            acc /= np.linalg.norm(acc)
            for i in range(4):
                vals[(dp, i)].append(acc[i])
        for sp in specs:
            dp, i = sp
            if dp.endswith("rotation_quaternion"):
                continue
            if dp == "location" and i in (0, 1):
                vals[sp].append(0.0)                      # in place: the root Empty carries the path
                continue
            v = 0.0
            for clip, cf, w in entries:
                ch = sampled[clip]["chans"]
                if dp == "rotation_euler" and i == 2 and "turn_" in clip:
                    v += w * ch[sp][0]                    # turn clips: yaw comes from the timeline
                else:
                    v += w * _at(ch[sp], cf)
            vals[sp].append(v)
        del ref
    for sp in specs:
        _write(af[sp], keep, vals[sp])


def plan_for(frames, person, gender, src_height_m, fps):
    return anim_plan.plan(frames, gender, person["role"], person["height_m"], src_height_m, fps, person["id"])
