"""Keyframe mannequins from the precomputed timeline (runs inside Blender; no behaviour logic here)."""
import math

from sim.behaviour import mannequin_pose

HIDDEN_Z = -100.0


def _key(ob, path, frame):
    ob.keyframe_insert(data_path=path, frame=frame)


def animate_person(joints, frames, height_m):
    root, pelvis = joints["root"], joints["pelvis"]
    for f, st in enumerate(frames):
        x, y = st["pos_m"]
        root.location = (x, y, 0.0 if st["present"] else HIDDEN_Z)
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
