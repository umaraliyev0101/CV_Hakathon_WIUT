"""Procedural capsule mannequins (stand-ins while assets/mixamo/ is empty). Runs inside Blender.

Hierarchy per person: root (floor x, y; yaw) -> pelvis (z) -> torso/neck/head/arms, hips -> thigh ->
knee -> shin -> ankle -> foot. Local +x is forward, +y is left. Joint angles come from
sim.behaviour.mannequin_pose so the pure tests cover them (feet on floor, no frozen limbs).
"""
import math

import bpy

from sim.behaviour import PROPORTIONS
from sim.blender.build_scene import material

PERSON_TAG = "sim_person"


def _empty(name, parent, loc):
    ob = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(ob)
    ob.parent = parent
    ob.location = loc
    ob.rotation_mode = "XYZ"
    return ob


def _ellipsoid(name, parent, loc, radii, mat, pid):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1.0, location=(0, 0, 0))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = radii
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.shade_smooth()
    ob.parent = parent
    ob.location = loc
    ob.data.materials.append(mat)
    ob[PERSON_TAG] = pid
    return ob


def _limb(name, parent, length, radius, mat, pid):
    """Capsule hanging down from the parent's origin (along -z)."""
    return _ellipsoid(name, parent, (0, 0, -length / 2), (radius, radius, length / 2 + radius * 0.4), mat, pid)


def outfit_materials(person, uniform):
    pid = person["id"]
    skin = material(f"skin_{pid}", (0.55, 0.38, 0.28), rough=0.6)
    shoes = material(f"shoes_{pid}", (0.03, 0.03, 0.03), rough=0.4)
    if person["outfit"] == "uniform":
        top = material(f"top_{pid}", tuple(uniform["top"]), rough=0.7)
        vest = material(f"vest_{pid}", tuple(uniform["vest"]), rough=0.6)
        legs = material(f"legs_{pid}", tuple(uniform["trousers"]), rough=0.7)
    else:
        top = material(f"top_{pid}", tuple(person["colors"]["top"]), rough=0.7)
        vest = None
        legs = material(f"legs_{pid}", tuple(person["colors"]["trousers"]), rough=0.75)
    return dict(skin=skin, shoes=shoes, top=top, vest=vest, legs=legs)


def build_mannequin(person, uniform):
    pid, h = person["id"], person["height_m"]
    P = {k: v * h for k, v in PROPORTIONS.items()}
    M = outfit_materials(person, uniform)
    j = {}
    root = _empty(f"P{pid}_root", None, (0, 0, 0))
    pelvis = _empty(f"P{pid}_pelvis", root, (0, 0, P["thigh"] + P["shin"] + P["foot_h"]))
    j["root"], j["pelvis"] = root, pelvis
    torso_len = P["torso"]
    _ellipsoid(f"P{pid}_hips", pelvis, (0, 0, 0.03 * h), (0.09 * h, 0.12 * h, 0.07 * h), M["legs"], pid)
    _ellipsoid(f"P{pid}_torso", pelvis, (0, 0, 0.06 * h + torso_len / 2),
               (0.085 * h, 0.125 * h, torso_len / 2 + 0.01 * h), M["top"], pid)
    if M["vest"] is not None:
        _ellipsoid(f"P{pid}_vest", pelvis, (0.003 * h, 0, 0.05 * h + torso_len * 0.42),
                   (0.092 * h, 0.13 * h, torso_len * 0.43), M["vest"], pid)
    neck_z = 0.06 * h + torso_len
    _ellipsoid(f"P{pid}_neck", pelvis, (0, 0, neck_z + P["neck"] / 2), (0.03 * h, 0.03 * h, P["neck"]), M["skin"], pid)
    head = _empty(f"P{pid}_headpivot", pelvis, (0, 0, neck_z + P["neck"]))
    j["head"] = head
    _ellipsoid(f"P{pid}_head", head, (0, 0, P["head_r"]), (P["head_r"] * 0.95, P["head_r"] * 0.85, P["head_r"] * 1.1),
               M["skin"], pid)
    _ellipsoid(f"P{pid}_nose", head, (P["head_r"] * 0.9, 0, P["head_r"] * 0.9), (0.015 * h, 0.01 * h, 0.012 * h),
               M["skin"], pid)
    upper, fore = 0.17 * h, 0.16 * h
    for side, sy in (("l", 1), ("r", -1)):
        sh = _empty(f"P{pid}_{side}_shoulder", pelvis, (0, sy * (0.135 * h), neck_z - 0.025 * h))
        _limb(f"P{pid}_{side}_upperarm", sh, upper, 0.028 * h, M["top"], pid)
        el = _empty(f"P{pid}_{side}_elbow", sh, (0, 0, -upper))
        _limb(f"P{pid}_{side}_forearm", el, fore * 0.75, 0.024 * h, M["top"], pid)
        _ellipsoid(f"P{pid}_{side}_hand", el, (0, 0, -fore * 0.85), (0.022 * h, 0.014 * h, 0.035 * h), M["skin"], pid)
        hip = _empty(f"P{pid}_{side}_hip", pelvis, (0, sy * P["hip_w"], 0))
        _limb(f"P{pid}_{side}_thigh", hip, P["thigh"], 0.045 * h, M["legs"], pid)
        knee = _empty(f"P{pid}_{side}_knee", hip, (0, 0, -P["thigh"]))
        _limb(f"P{pid}_{side}_shin", knee, P["shin"], 0.035 * h, M["legs"], pid)
        ankle = _empty(f"P{pid}_{side}_ankle", knee, (0, 0, -P["shin"]))
        _ellipsoid(f"P{pid}_{side}_foot", ankle, (0.04 * h, 0, -P["foot_h"] / 2),
                   (0.07 * h, 0.03 * h, P["foot_h"] / 2), M["shoes"], pid)
        j[f"{side}_shoulder"], j[f"{side}_elbow"] = sh, el
        j[f"{side}_hip"], j[f"{side}_knee"], j[f"{side}_ankle"] = hip, knee, ankle
    return j


def person_meshes(pid):
    return [ob for ob in bpy.data.objects if ob.type == "MESH" and ob.get(PERSON_TAG) == pid]
