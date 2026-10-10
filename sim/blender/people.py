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


# ---------------------------------------------------------------- Rocketbox characters (real assets)

import os  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHAR_DIR = REPO / "assets" / "people" / "characters"
ACTIONS_BLEND = REPO / "assets" / "people" / "anims" / "rocketbox_actions_retargeted.blend"


def rocketbox_available(name):
    return name is not None and (CHAR_DIR / name / "Export" / f"{name}.fbx").exists() and ACTIONS_BLEND.exists()


def _recolour_nodes(nt, tex_out, rgb):
    """Uniform: recolour dark suit fabric (HSV value < 0.28) to `rgb`, keeping the texture's shading."""
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    sep.mode = "HSV"
    nt.links.new(tex_out, sep.inputs["Color"])
    lt = nt.nodes.new("ShaderNodeMath")
    lt.operation = "LESS_THAN"
    lt.inputs[1].default_value = 0.28
    nt.links.new(sep.outputs[2], lt.inputs[0])
    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.use_clamp = False
    gain.inputs[1].default_value = 1.0 / 0.12
    nt.links.new(sep.outputs[2], gain.inputs[0])
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    nt.links.new(gain.outputs[0], tint.inputs[6])
    tint.inputs[7].default_value = (*rgb, 1.0)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(lt.outputs[0], mix.inputs["Factor"])
    nt.links.new(tex_out, mix.inputs[6])
    nt.links.new(tint.outputs[2], mix.inputs[7])
    return mix.outputs[2]


def fix_materials(meshes, texdir, recolour=None):
    """FBX texture paths point at the author's PC: relink by basename; opacity maps drive alpha."""
    for o in meshes:
        for s in o.material_slots:
            m = s.material
            if m is None or m.get("sim_fixed"):
                continue
            files = {os.path.basename(n.image.filepath.replace("\\", "/"))
                     for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}
            nt = m.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            b = nt.nodes.new("ShaderNodeBsdfPrincipled")
            nt.links.new(b.outputs[0], out.inputs[0])
            b.inputs["Roughness"].default_value = 0.6

            def tex(fn, nc=False):
                t = nt.nodes.new("ShaderNodeTexImage")
                t.image = bpy.data.images.load(str(texdir / fn), check_existing=True)
                if nc:
                    t.image.colorspace_settings.name = "Non-Color"
                return t
            col = sorted(f for f in files if f.endswith("_color.tga"))
            nor = sorted(f for f in files if f.endswith("_normal.tga"))
            if col:
                t = tex(col[0])
                c_out = t.outputs["Color"]
                if recolour is not None and "_body_" in col[0]:
                    c_out = _recolour_nodes(nt, c_out, recolour)
                nt.links.new(c_out, b.inputs["Base Color"])
                if "opacity" in col[0]:
                    nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
                    m.blend_method = "HASHED"
            if nor:
                t = tex(nor[0], True)
                nm = nt.nodes.new("ShaderNodeNormalMap")
                nt.links.new(t.outputs["Color"], nm.inputs["Color"])
                nt.links.new(nm.outputs[0], b.inputs["Normal"])
            m["sim_fixed"] = True


def import_character(person, recolour=None):
    """Import a Rocketbox FBX, parent its armature to an Empty `P<id>_root` (the Empty is moved by
    the timeline; the armature keeps its import transform, which the actions key)."""
    name, pid = person["character"], person["id"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(CHAR_DIR / name / "Export" / f"{name}.fbx"))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    meshes = [o for o in new if o.type == "MESH"]
    for o in new:
        if o.type == "EMPTY":
            o.hide_render = True
            o.hide_viewport = True
    if arm.animation_data:
        arm.animation_data.action = None
    fix_materials(meshes, CHAR_DIR / name / "Textures", recolour)
    root = bpy.data.objects.new(f"P{pid}_root", None)
    bpy.context.collection.objects.link(root)
    root.rotation_mode = "XYZ"
    arm.parent = root
    arm.name = f"P{pid}_arm"
    for o in meshes:
        o[PERSON_TAG] = pid
        o.name = f"P{pid}_mesh_{o.name}"
    return {"root": root, "arm": arm, "meshes": meshes}


def load_actions(names):
    with bpy.data.libraries.load(str(ACTIONS_BLEND)) as (src, dst):
        dst.actions = [n for n in src.actions if n in names]
    return {a.name: a for a in dst.actions}
