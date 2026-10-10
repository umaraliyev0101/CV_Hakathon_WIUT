"""Static showroom: room, glass facade, roller door, desk, props, real car models, lighting variants.

Runs inside Blender 4.2 (EEVEE Next). API used by render.py:
    build(lay, cfg)              clear the scene and build everything static (calls set_lighting once)
    set_lighting(lighting, lay)  switch 'day' | 'evening_dim' | 'glare' in the built scene (idempotent)

Tags: every non-person object gets ob[NON_PERSON_TAG] = True; facade glass panes also get
ob[GLASS_TAG] = True (hidden in the mask pass). Coordinates: metres, x east, y north, z up, floor z = 0.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

NON_PERSON_TAG = "sim_static"
GLASS_TAG = "sim_glass"

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CARS_JSON = REPO / "sim" / "configs" / "assets_cars.json"
LIGHTINGS = ("day", "evening_dim", "glare")

# Per-model import fixes (ASSETS_REPORT section 5.7). Ground lift / centring is generic (bbox based).
CAR_MODELS = {
    "ferrari_458": {"file": "assets/cars/ferrari_458.glb", "scale": 1.0, "paint": "Body_Color"},
    "khronos_carconcept": {"file": "assets/cars/khronos_carconcept.glb", "scale": 1.0, "paint": "Paint 1 Carmine"},
    "lamborghini_urus": {"file": "assets/cars/lamborghini_urus.glb", "scale": 0.01, "paint": "WhiteCar"},
    "porsche_930_turbo": {"file": "assets/cars/porsche_930_turbo.glb", "scale": 1.0, "paint": "paint"},
}


# ----------------------------------------------------------------------------------------------- helpers
def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves):
        for block in list(coll):
            coll.remove(block)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def material(name, rgb, rough=0.5, metallic=0.0, emission=None, strength=0.0, coat=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*rgb, 1.0)
    return m


def _tag(ob, glass=False):
    ob[NON_PERSON_TAG] = True
    if glass:
        ob[GLASS_TAG] = True
    return ob


def box(name, center, size, mat, rot_z_deg=0.0, bevel=0.0, tag=NON_PERSON_TAG):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    ob.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments = bevel, 3
    ob.data.materials.append(mat)
    ob[tag] = True
    return ob


def cylinder(name, center, radius, depth, mat, rot=(0, 0, 0), verts=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.materials.append(mat)
    ob[NON_PERSON_TAG] = True
    bpy.ops.object.shade_smooth()
    return ob


def _rot2(dx, dy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * dx - s * dy, s * dx + c * dy


def _image(path, non_color=False):
    img = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def pbr_material(name, tex_dir, repeat_m, rough_mul=1.0, rough_add=0.0, normal_strength=1.0,
                 tint=(1.0, 1.0, 1.0), projection="FLAT", coat=0.0, outdoor=False):
    """Poly Haven diff/rough/nor_gl set, mapped by world position at real-world scale (`repeat_m`)."""
    tex_dir = Path(tex_dir)
    files = {p.name.split("_2k")[0].rsplit("_", 1)[-1]: p for p in tex_dir.glob("*_2k.jpg")}
    diff = files.get("diff") or files.get("diffuse")
    rough = files.get("rough")
    nor = next(p for p in tex_dir.glob("*_nor_gl_2k.jpg"))
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / repeat_m,) * 3
    nt.links.new(geo.outputs["Position"], mp.inputs["Vector"])

    def tex(path, nc):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _image(path, nc)
        t.projection = projection
        if projection == "BOX":
            t.projection_blend = 0.2
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t

    td = tex(diff, False)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(td.outputs["Color"], mix.inputs["A"])
    mix.inputs["B"].default_value = (*tint, 1.0)
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    tr = tex(rough, True)
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    ma.inputs[1].default_value, ma.inputs[2].default_value = rough_mul, rough_add
    ma.use_clamp = True
    nt.links.new(tr.outputs["Color"], ma.inputs[0])
    nt.links.new(ma.outputs[0], bsdf.inputs["Roughness"])
    tn = tex(nor, True)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal_strength
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    if outdoor:
        _outdoor_emission(nt, bsdf, mix.outputs["Result"])
    m.diffuse_color = (*[0.6 * t for t in tint], 1.0)
    return m


def _outdoor_emission(nt, bsdf, colour_socket):
    """Outdoor surfaces glow with their own albedo (strength set per lighting in set_lighting): fakes
    full-strength daylight outside while the HDRI lighting strength is kept low for the interior."""
    v = nt.nodes.new("ShaderNodeValue")
    v.name = "outdoor_emit"
    v.outputs[0].default_value = 1.0
    if isinstance(colour_socket, tuple):
        bsdf.inputs["Emission Color"].default_value = (*colour_socket, 1.0)
    else:
        nt.links.new(colour_socket, bsdf.inputs["Emission Color"])
    nt.links.new(v.outputs[0], bsdf.inputs["Emission Strength"])


def glass_material(name="facade_glass", tint=(0.9, 0.95, 0.95)):
    """Clear glazing: transparent + Fresnel-weighted glossy reflection (no refraction cost)."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (*tint, 1.0)
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.02
    fr = nt.nodes.new("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.25  # weaker reflections than real 1.5: keeps people behind glass visible
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fr.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    m.use_transparent_shadow = True
    m.diffuse_color = (0.7, 0.8, 0.85, 0.3)
    return m


# ----------------------------------------------------------------------------------------------- room
def build_room(lay):
    r = lay["room"]
    W, D, H, T = r["width_m"], r["depth_m"], r["height_m"], r["wall_thickness_m"]
    floor_m = pbr_material("floor_marble", ASSETS / "textures" / "marble_01", 1.5, rough_mul=0.35, rough_add=0.04,
                           normal_strength=0.25, tint=(0.92, 0.91, 0.9))
    wall_m = pbr_material("wall_plaster", ASSETS / "textures" / "beige_wall_001", 3.0, rough_mul=1.0,
                          normal_strength=0.4, tint=(1.3, 1.28, 1.25), projection="BOX")
    ceil_m = material("ceiling", (0.78, 0.78, 0.77), rough=0.9)
    pave_m = pbr_material("pavement", ASSETS / "textures" / "concrete_floor_worn_001", 3.0, normal_strength=0.6,
                          tint=(0.85, 0.85, 0.85), outdoor=True)
    road_m = pbr_material("road", ASSETS / "textures" / "concrete_floor_worn_001", 4.0, normal_strength=0.8,
                          tint=(0.22, 0.22, 0.23), rough_add=0.15, outdoor=True)
    skirt_m = material("skirting", (0.05, 0.05, 0.055), rough=0.5)

    box("floor", (W / 2, D / 2, -0.05), (W + 2 * T, D + 2 * T, 0.1), floor_m)
    box("ceiling", (W / 2, D / 2, H + 0.05), (W + 2 * T, D + 2 * T, 0.1), ceil_m)
    # outside: pavement to y=-8 (people spawn at y~-2.6), then a road out to the horizon
    box("pavement", (W / 2, -T - 4.0, -0.05), (W + 80.0, 8.0, 0.1), pave_m)
    box("road", (W / 2, -T - 8.0 - 70.0, -0.07), (W + 240.0, 140.0, 0.1), road_m)
    kerb_m = material("kerb", (0.5, 0.5, 0.48), rough=0.8)
    _outdoor_emission(kerb_m.node_tree, kerb_m.node_tree.nodes["Principled BSDF"], (0.5, 0.5, 0.48))
    box("kerb", (W / 2, -T - 8.0, 0.0), (W + 80.0, 0.25, 0.14), kerb_m)

    # opaque walls (north, west, east-with-roller-door)
    box("wall_north", (W / 2, D + T / 2, H / 2), (W + 2 * T, T, H), wall_m)
    box("wall_west", (-T / 2, D / 2, H / 2), (T, D, H), wall_m)
    box("skirt_north", (W / 2, D - 0.01, 0.05), (W, 0.02, 0.1), skirt_m)
    box("skirt_west", (0.01, D / 2, 0.05), (0.02, D, 0.1), skirt_m)
    x_ = lay.get("exit")
    if x_:
        cy, ew, eh = x_["center_y_m"], x_["width_m"], x_["height_m"]
        y0, y1 = cy - ew / 2, cy + ew / 2
        box("wall_east_s", (W + T / 2, y0 / 2, H / 2), (T, y0, H), wall_m)
        box("wall_east_n", (W + T / 2, (y1 + D) / 2, H / 2), (T, D - y1, H), wall_m)
        box("wall_east_lintel", (W + T / 2, cy, (eh + H) / 2), (T, ew, H - eh), wall_m)
        box("skirt_east_s", (W - 0.01, y0 / 2, 0.05), (0.02, y0, 0.1), skirt_m)
        box("skirt_east_n", (W - 0.01, (y1 + D) / 2, 0.05), (0.02, D - y1, 0.1), skirt_m)
        build_roller_door(W, T, cy, ew, eh)
    else:
        box("wall_east", (W + T / 2, D / 2, H / 2), (T, D, H), wall_m)
    build_facade(lay)


def build_roller_door(W, T, cy, ew, eh):
    m = bpy.data.materials.new("roller_door")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.42, 0.44, 0.46, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.85
    bsdf.inputs["Roughness"].default_value = 0.38
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type, wave.bands_direction = "BANDS", "Z"
    wave.inputs["Scale"].default_value = 1.0
    mapz = nt.nodes.new("ShaderNodeMath")
    mapz.operation = "MULTIPLY"
    mapz.inputs[1].default_value = 1.0 / 0.1  # one slat per 10 cm
    nt.links.new(sep.outputs["Z"], mapz.inputs[0])
    sin = nt.nodes.new("ShaderNodeMath")
    sin.operation = "SINE"
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    k.inputs[1].default_value = 2 * math.pi
    nt.links.new(mapz.outputs[0], k.inputs[0])
    nt.links.new(k.outputs[0], sin.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(sin.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.nodes.remove(wave)
    m.diffuse_color = (0.42, 0.44, 0.46, 1.0)
    frame = material("door_frame", (0.06, 0.06, 0.065), rough=0.4, metallic=0.6)
    box("roller_door", (W + T * 0.6, cy, eh / 2), (0.05, ew, eh), m)
    box("roller_rail_s", (W + 0.04, cy - ew / 2 - 0.04, eh / 2), (0.08, 0.08, eh), frame)
    box("roller_rail_n", (W + 0.04, cy + ew / 2 + 0.04, eh / 2), (0.08, 0.08, eh), frame)
    box("roller_head", (W + 0.04, cy, eh + 0.06), (0.08, ew + 0.16, 0.12), frame)
    box("roller_floor_strip", (W - 0.08, cy, 0.002), (0.12, ew, 0.004),
        material("hazard_strip", (0.35, 0.27, 0.03), rough=0.6))


def build_facade(lay):
    """South wall: floor-to-ceiling glazing with dark mullions and an open entrance."""
    r = lay["room"]
    W, H, T = r["width_m"], r["height_m"], r["wall_thickness_m"]
    e = lay["entrance"]
    ex0, ex1 = e["center_x_m"] - e["width_m"] / 2, e["center_x_m"] + e["width_m"] / 2
    eh = e["height_m"]
    glass = glass_material()
    mull = material("mullion", (0.035, 0.035, 0.04), rough=0.35, metallic=0.7)
    y = -T / 2
    mw, md = 0.07, T  # mullion width (x) and depth (y)
    # mullion positions: room corners, both door jambs, and ~2.5 m spacing in between
    xs = {0.0, W, ex0, ex1}
    for a, b in ((0.0, ex0), (ex1, W)):
        n = max(1, round((b - a) / 2.5))
        xs.update(a + (b - a) * i / n for i in range(1, n))
    xs = sorted(xs)
    for i, x in enumerate(xs):
        box(f"mullion_{i}", (x, y, H / 2), (mw, md, H), mull)
    box("facade_head", (W / 2, y, H - 0.06), (W, md, 0.12), mull)
    box("facade_sill_a", (ex0 / 2, y, 0.03), (ex0, md, 0.06), mull)
    box("facade_sill_b", ((ex1 + W) / 2, y, 0.03), (W - ex1, md, 0.06), mull)
    box("door_transom", (e["center_x_m"], y, eh + 0.05), (e["width_m"] + mw, md + 0.04, 0.1), mull)
    k = 0
    for a, b in zip(xs[:-1], xs[1:]):
        w = b - a - mw
        if w <= 0.05:
            continue
        if a >= ex0 - 1e-6 and b <= ex1 + 1e-6:  # door opening: glass only above the transom
            p = box(f"facade_glass_{k}", ((a + b) / 2, y, (eh + 0.1 + H - 0.12) / 2), (w, 0.02, H - 0.12 - eh - 0.1),
                    glass)
        else:
            p = box(f"facade_glass_{k}", ((a + b) / 2, y, (0.06 + H - 0.12) / 2), (w, 0.02, H - 0.18), glass)
        p.visible_shadow = False
        _tag(p, glass=True)
        k += 1
    # entrance: open automatic sliding doors parked beside the opening on the outside
    dw = e["width_m"] / 2
    for side, xc in (("w", ex0 - dw / 2 + 0.02), ("e", ex1 + dw / 2 - 0.02)):
        fr = box(f"door_leaf_frame_{side}", (xc, y - T / 2 - 0.06, eh / 2), (dw, 0.05, eh), mull)
        fr.modifiers.new("wire", "WIREFRAME").thickness = 0.06
        g = box(f"door_leaf_glass_{side}", (xc, y - T / 2 - 0.06, eh / 2), (dw - 0.08, 0.015, eh - 0.08), glass)
        g.visible_shadow = False
        _tag(g, glass=True)
    box("door_mat", (e["center_x_m"], 0.75, 0.003), (e["width_m"], 1.3, 0.006),
        material("door_mat", (0.03, 0.03, 0.032), rough=0.95))


# ----------------------------------------------------------------------------------------------- desk/props
def build_desk(desk):
    wood = material("desk_top", (0.32, 0.2, 0.11), rough=0.35)
    panel = material("desk_panel", (0.85, 0.85, 0.84), rough=0.4)
    dark = material("desk_dark", (0.02, 0.02, 0.022), rough=0.4, metallic=0.3)
    screen = material("monitor_screen", (0.01, 0.01, 0.012), rough=0.15, emission=(0.35, 0.45, 0.6), strength=0.35)
    seat = material("chair_seat", (0.04, 0.04, 0.045), rough=0.7)
    cx, cy = desk["center_m"]
    L, Wd, Hh = desk["length_m"], desk["width_m"], desk["height_m"]
    yaw = desk["yaw_deg"]

    def at(dx, dy, z):
        ox, oy = _rot2(dx, dy, yaw)
        return (cx + ox, cy + oy, z)

    box("desk_top", at(0, 0, Hh - 0.02), (L, Wd, 0.04), wood, yaw, bevel=0.005)
    box("desk_body", at(0, 0.05 * Wd, (Hh - 0.04) / 2), (L * 0.97, Wd * 0.75, Hh - 0.04), panel, yaw)
    box("desk_kick", at(0, 0.05 * Wd, 0.04), (L * 0.95, Wd * 0.7, 0.08), dark, yaw)
    # monitor towards the -y (local) edge, facing +y side... both sides plausible for CCTV
    box("monitor_stand", at(0.2 * L, 0.15 * Wd, Hh + 0.12), (0.05, 0.05, 0.24), dark, yaw)
    box("monitor_base", at(0.2 * L, 0.15 * Wd, Hh + 0.01), (0.22, 0.18, 0.02), dark, yaw)
    box("monitor", at(0.2 * L, 0.12 * Wd, Hh + 0.35), (0.6, 0.03, 0.36), screen, yaw, bevel=0.005)
    box("keyboard", at(0.2 * L, -0.15 * Wd, Hh + 0.01), (0.42, 0.14, 0.015), dark, yaw)
    # two chairs tucked under the long sides (stay inside the desk apron zone)
    for side, sgn in (("a", 1.0), ("b", -1.0)):
        dy = sgn * (Wd / 2 - 0.02)
        box(f"chair_{side}_seat", at(0.1 * L * sgn, dy, 0.46), (0.46, 0.44, 0.07), seat, yaw, bevel=0.02)
        box(f"chair_{side}_back", at(0.1 * L * sgn, dy + sgn * 0.24, 0.78), (0.44, 0.05, 0.52), seat, yaw, bevel=0.02)
        cylinder(f"chair_{side}_post", at(0.1 * L * sgn, dy, 0.24), 0.025, 0.44, dark)
        cylinder(f"chair_{side}_foot", at(0.1 * L * sgn, dy, 0.02), 0.27, 0.04, dark)


def build_props(lay):
    """Plants in free corners and a backlit logo panel on the north wall (cheap)."""
    from sim.layout import dist_to_polygons
    r = lay["room"]
    W, D, H = r["width_m"], r["depth_m"], r["height_m"]
    zones = list(lay["zones"].values())
    pot = material("plant_pot", (0.75, 0.74, 0.72), rough=0.3)
    leaf = material("plant_leaf", (0.04, 0.16, 0.035), rough=0.6)
    k = 0
    for (px, py) in ((0.55, D - 0.55), (W - 0.55, D - 0.55), (0.55, 0.65), (W - 0.55, 0.65), (W / 2, D - 0.55)):
        if zones and dist_to_polygons((px, py), zones) < 0.45:
            continue
        cylinder(f"plant_{k}_pot", (px, py, 0.3), 0.26, 0.6, pot, verts=20)
        for j, (dx, dy, z, s) in enumerate(((0, 0, 1.05, 0.42), (0.12, 0.05, 1.4, 0.32), (-0.1, -0.06, 1.3, 0.3),
                                            (0.02, -0.1, 0.8, 0.33))):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=s, location=(px + dx, py + dy, z))
            ob = bpy.context.active_object
            ob.name = f"plant_{k}_leaves_{j}"
            ob.scale = (1.0, 1.0, 1.25)
            disp = ob.modifiers.new("d", "DISPLACE")
            tex = bpy.data.textures.get("leaf_noise") or bpy.data.textures.new("leaf_noise", "CLOUDS")
            tex.noise_scale = 0.12
            disp.texture, disp.strength = tex, 0.12
            ob.data.materials.append(leaf)
            bpy.ops.object.shade_smooth()
            _tag(ob)
        k += 1
    # logo panel
    panel = material("logo_panel", (0.02, 0.02, 0.025), rough=0.3, metallic=0.4)
    glow = material("logo_glow", (1, 1, 1), emission=(1.0, 0.95, 0.85), strength=4.0)
    pw = min(6.0, 0.4 * W)
    box("logo_panel", (W / 2, D - 0.03, H - 1.2), (pw, 0.04, 0.9), panel)
    box("logo_stripe", (W / 2, D - 0.055, H - 1.52), (pw * 0.9, 0.01, 0.03), glow)
    bpy.ops.object.text_add(location=(W / 2, D - 0.06, H - 1.22), rotation=(math.pi / 2, 0, 0))
    t = bpy.context.active_object
    t.name = "logo_text"
    t.data.body = "AUTO GALLERY"
    t.data.align_x, t.data.align_y = "CENTER", "CENTER"
    t.data.size = 0.42
    t.data.extrude = 0.005
    t.data.materials.append(glow)
    bpy.ops.object.convert(target="MESH")  # mask pass colours MESH objects only
    _tag(bpy.context.active_object)


# ----------------------------------------------------------------------------------------------- cars
def _car_registry():
    reg = {k: dict(v) for k, v in CAR_MODELS.items()}
    if CARS_JSON.exists():
        for c in json.loads(CARS_JSON.read_text()).get("cars", []):
            if c.get("model") in reg:
                reg[c["model"]].update(file=c.get("file", reg[c["model"]]["file"]),
                                       paint=c.get("paint_material", reg[c["model"]]["paint"]))
    return reg


def _mesh_world_coords(ob):
    me = ob.data
    n = len(me.vertices)
    if n == 0:
        return np.zeros((0, 3))
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def world_bbox(objs):
    pts = [_mesh_world_coords(o) for o in objs if o.type == "MESH"]
    pts = np.concatenate([p for p in pts if len(p)])
    return pts.min(0), pts.max(0)


def _delete_faces(ob, pred_face):
    """Delete faces of a mesh object where pred_face(bm_face, world_matrix) is True."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    M = ob.matrix_world
    dead = [f for f in bm.faces if pred_face(f, M)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()


def _fix_porsche(objs):
    by = {o.name: o for o in objs}
    m2 = by.get("Mesh_2")  # 16 m ground disc (Material.001) + 4 traffic cones (DefaultMaterial); keep plates
    if m2:
        mats = m2.data.materials
        _delete_faces(m2, lambda f, M: mats[f.material_index].name.split(".")[0] != "plate")
    m0 = by.get("Mesh_0")  # radio antenna sticks up to 1.87 m: drop islands reaching above the roof
    if m0:
        bm = bmesh.new()
        bm.from_mesh(m0.data)
        M = m0.matrix_world
        seen, dead = set(), []
        for v in bm.verts:
            if v.index in seen:
                continue
            comp, st = [], [v]
            seen.add(v.index)
            while st:
                x = st.pop()
                comp.append(x)
                for e in x.link_edges:
                    w = e.other_vert(x)
                    if w.index not in seen:
                        seen.add(w.index)
                        st.append(w)
            if max((M @ x.co).z for x in comp) > 1.45:
                dead += comp
        bmesh.ops.delete(bm, geom=dead, context="VERTS")
        bm.to_mesh(m0.data)
        bm.free()


def _fix_materials(model, objs, paint_name):
    mats = {s.material for o in objs if o.type == "MESH" for s in o.material_slots if s.material}
    if model == "ferrari_458":  # importer leaves everything metallic 0.2 / rough 0.8
        for m in mats:
            b = m.node_tree.nodes.get("Principled BSDF")
            if not b:
                continue
            nm = m.name
            met, rough = 0.0, 0.5
            if nm.startswith("metal_chrome"):
                met, rough = 1.0, 0.08
            elif nm.startswith("metal_gray"):
                met, rough = 1.0, 0.3
            elif nm.startswith("Tires"):
                rough = 0.85
            elif nm.startswith(("Taillight", "Projector", "Turn_Signal")):
                rough = 0.1
            elif nm.startswith(("Leather", "Carpet", "Interior")):
                rough = 0.7
            b.inputs["Metallic"].default_value = met
            b.inputs["Roughness"].default_value = rough
            if nm.startswith("Glass_Gray"):
                b.inputs["Base Color"].default_value = (0.02, 0.022, 0.025, 1.0)
                b.inputs["Roughness"].default_value = 0.03
                b.inputs["Alpha"].default_value = 0.55
                m.surface_render_method = "BLENDED"
    for m in mats:
        if m.name.split(".")[0] == paint_name:
            return m
    return None


def import_car_template(model):
    """Import one car model with fixes. Returns (root_empty, objects, paint_material, (L, W, H)).

    The root empty normalises the car: wheels on z=0, bbox centred on the origin in x/y, long axis along +x.
    """
    reg = _car_registry()
    spec = reg[model]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(REPO / spec["file"]))
    objs = [o for o in bpy.data.objects if o not in before]
    if model == "porsche_930_turbo":
        _fix_porsche(objs)
    paint = _fix_materials(model, objs, spec["paint"])
    root = bpy.data.objects.new(f"tmpl_{model}", None)
    bpy.context.scene.collection.objects.link(root)
    for o in objs:
        if o.parent is None:
            o.parent = root
    s = spec["scale"]
    root.scale = (s, s, s)
    bpy.context.view_layer.update()
    mn, mx = world_bbox(objs)
    ext = mx - mn
    if ext[1] > ext[0]:  # long axis along y -> rotate onto x
        root.rotation_euler.z = math.pi / 2
        bpy.context.view_layer.update()
        mn, mx = world_bbox(objs)
    c = (mn + mx) / 2
    root.location = (-c[0], -c[1], -mn[2])
    bpy.context.view_layer.update()
    mn, mx = world_bbox(objs)
    dims = tuple(float(v) for v in (mx - mn))
    return root, objs, paint, dims


def _instance_car(car, tmpl, k):
    root, objs, paint, dims = tmpl
    L0 = dims[0]
    inst = bpy.data.objects.new(f"car{car['id']}", None)
    bpy.context.scene.collection.objects.link(inst)
    _tag(inst)
    pm = None
    if paint is not None:
        pm = paint.copy()
        pm.name = f"paint_car{car['id']}"
        b = pm.node_tree.nodes.get("Principled BSDF")
        if b:
            bc = b.inputs["Base Color"]
            for ln in list(bc.links):
                pm.node_tree.links.remove(ln)
            bc.default_value = (*car["paint"], 1.0)
            b.inputs["Metallic"].default_value = 0.45
            b.inputs["Roughness"].default_value = 0.28
            b.inputs["Coat Weight"].default_value = 1.0
            b.inputs["Coat Roughness"].default_value = 0.03
        pm.diffuse_color = (*car["paint"], 1.0)
    mapping = {root: root.copy()}
    for o in objs:
        mapping[o] = o.copy()  # linked mesh data
    for old, new in mapping.items():
        bpy.context.scene.collection.objects.link(new)
        new.parent = mapping.get(old.parent, inst) if old is not root else inst
        _tag(new)
        new.name = f"car{car['id']}_{old.name}"
        if pm is not None and new.type == "MESH":
            for slot in new.material_slots:
                if slot.material is paint:
                    slot.link = "OBJECT"
                    slot.material = pm
    s = car["length_m"] / L0
    inst.scale = (s, s, s)
    inst.rotation_euler = (0.0, 0.0, math.radians(car["yaw_deg"]))
    inst.location = (car["center_m"][0], car["center_m"][1], 0.0)
    return inst


def build_car(car):
    """Box stand-in car (fallback when no model is given)."""
    L, Wd, Hh = car["length_m"], car["width_m"], car["height_m"]
    cx, cy = car["center_m"]
    yaw = car["yaw_deg"]
    k = car["id"]
    paint = material(f"paint_{k}", tuple(car["paint"]), rough=0.25, metallic=0.6, coat=1.0)
    glass = material("car_glass", (0.02, 0.025, 0.03), rough=0.05)
    tyre = material("tyre", (0.015, 0.015, 0.015), rough=0.7)
    wheel_r = 0.33
    body_h = 0.62

    def at(dx, dy, z):
        ox, oy = _rot2(dx, dy, yaw)
        return (cx + ox, cy + oy, z)

    parts = [box(f"car{k}_body", at(0, 0, wheel_r * 0.6 + body_h / 2), (L, Wd, body_h), paint, yaw, bevel=0.12)]
    cab_h = Hh - (wheel_r * 0.6 + body_h)
    parts.append(box(f"car{k}_cabin", at(-0.1 * L, 0, Hh - cab_h / 2), (0.52 * L, 0.86 * Wd, cab_h), glass, yaw,
                     bevel=0.08))
    for dx in (-0.33 * L, 0.33 * L):
        for dy in (-Wd / 2 + 0.12, Wd / 2 - 0.12):
            parts.append(cylinder(f"car{k}_wheel", at(dx, dy, wheel_r), wheel_r, 0.24, tyre,
                                  rot=(math.pi / 2, 0, math.radians(yaw))))
    return parts


def build_cars(lay):
    templates = {}
    for k, car in enumerate(lay["cars"]):
        model = car.get("model")
        if not model or model not in CAR_MODELS:
            build_car(car)
            continue
        if model not in templates:
            templates[model] = import_car_template(model)
        _instance_car(car, templates[model], k)
    for root, objs, _, _ in templates.values():  # templates are only data holders now
        for o in objs + [root]:
            bpy.data.objects.remove(o, do_unlink=True)


# ----------------------------------------------------------------------------------------------- lighting
LIGHT_PRESETS = {
    # ceil_w_m2: ceiling area-light watts per m2 of floor; panel: emission of the visible light panels;
    # world: HDRI strength for lighting/reflections; bg: HDRI strength seen directly by the camera (through glass)
    "day": dict(ceil_w_m2=6.0, ceil_rgb=(1.0, 0.97, 0.92), panel=6.0, hdri="wide_street_01_2k.hdr",
                world=0.2, bg=1.0, outdoor=2.0, hdri_rot=0.0, sun=0.0),
    "evening_dim": dict(ceil_w_m2=2.5, ceil_rgb=(1.0, 0.78, 0.55), panel=2.0, hdri="sunset_jhbcentral_2k.hdr",
                        world=0.05, bg=0.2, outdoor=0.06, hdri_rot=0.0, sun=0.0),
    "glare": dict(ceil_w_m2=2.5, ceil_rgb=(1.0, 0.95, 0.88), panel=4.0, hdri="sunset_jhbcentral_2k.hdr",
                  world=0.12, bg=1.0, outdoor=0.3, hdri_rot=0.0, sun=70.0, sun_elev=11.0, sun_az=-25.0,
                  sun_rgb=(1.0, 0.82, 0.62)),
}


def build_lights(lay):
    r = lay["room"]
    W, D, H = r["width_m"], r["depth_m"], r["height_m"]
    panel_m = material("light_panel", (1, 1, 1), emission=(1.0, 0.97, 0.92), strength=8.0)
    nx, ny = max(2, round(W / 5.0)), max(2, round(D / 5.0))
    for i in range(nx):
        for j in range(ny):
            x, y = W * (i + 0.5) / nx, D * (j + 0.5) / ny
            ld = bpy.data.lights.new(f"ceiling_light_{i}{j}", "AREA")
            ld.shape, ld.size, ld.size_y = "RECTANGLE", 2.4, 1.0
            L = bpy.data.objects.new(f"ceiling_light_{i}{j}", ld)
            L.location = (x, y, H - 0.03)
            bpy.context.scene.collection.objects.link(L)
            _tag(L)
            box(f"light_panel_{i}{j}", (x, y, H - 0.01), (2.4, 1.0, 0.02), panel_m)
    sd = bpy.data.lights.new("sun_glare", "SUN")
    sd.angle = math.radians(0.6)
    sun = bpy.data.objects.new("sun_glare", sd)
    bpy.context.scene.collection.objects.link(sun)
    _tag(sun)
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.name = "env_map"
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.name = "env"
    bg_l = nt.nodes.new("ShaderNodeBackground")
    bg_l.name = "bg_light"
    bg_c = nt.nodes.new("ShaderNodeBackground")
    bg_c.name = "bg_cam"
    lp = nt.nodes.new("ShaderNodeLightPath")
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg_l.inputs["Color"])
    nt.links.new(env.outputs["Color"], bg_c.inputs["Color"])
    nt.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
    nt.links.new(bg_l.outputs[0], mx.inputs[1])
    nt.links.new(bg_c.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])


def set_lighting(lighting, lay=None):
    """Switch the built scene to one lighting variant. Idempotent; call any number of times."""
    if lighting not in LIGHT_PRESETS:
        raise ValueError(f"unknown lighting {lighting!r}; expected one of {LIGHTINGS}")
    p = LIGHT_PRESETS[lighting]
    lights = [o for o in bpy.data.objects if o.type == "LIGHT" and o.name.startswith("ceiling_light_")]
    if lay is not None:
        r = lay["room"]
        area = r["width_m"] * r["depth_m"]
    else:
        fl = bpy.data.objects.get("floor")
        area = (fl.dimensions.x * fl.dimensions.y) if fl else 300.0
    per = p["ceil_w_m2"] * area / max(1, len(lights))
    for L in lights:
        L.data.energy = per
        L.data.color = p["ceil_rgb"]
    pm = bpy.data.materials.get("light_panel")
    if pm:
        b = pm.node_tree.nodes["Principled BSDF"]
        b.inputs["Emission Strength"].default_value = p["panel"]
        b.inputs["Emission Color"].default_value = (*p["ceil_rgb"], 1.0)
    sun = bpy.data.objects.get("sun_glare")
    if sun:
        on = p["sun"] > 0
        sun.hide_render = sun.hide_viewport = not on
        sun.data.energy = p["sun"]
        if on:
            sun.data.color = p["sun_rgb"]
            # sun lamp shines along its local -Z: tilt towards +y (north) by (90 - elevation), azimuth about z
            sun.rotation_euler = (math.radians(90.0 - p["sun_elev"]), 0.0, math.radians(p["sun_az"]))
    for m in bpy.data.materials:
        node = m.node_tree.nodes.get("outdoor_emit") if m.use_nodes else None
        if node:
            node.outputs[0].default_value = p["outdoor"]
    world = bpy.context.scene.world
    nt = world.node_tree
    env = nt.nodes["env"]
    env.image = _image(ASSETS / "hdri" / p["hdri"])
    nt.nodes["env_map"].inputs["Rotation"].default_value = (0.0, 0.0, math.radians(p["hdri_rot"]))
    nt.nodes["bg_light"].inputs["Strength"].default_value = p["world"]
    nt.nodes["bg_cam"].inputs["Strength"].default_value = p["bg"]
    bpy.context.scene["sim_lighting"] = lighting


# ----------------------------------------------------------------------------------------------- entry
def build(lay, cfg=None):
    """Build the static scene. `cfg` is the scene config (or, for backwards compat, a lighting name)."""
    if isinstance(cfg, str):
        lighting = cfg
    else:
        cfg = cfg or {}
        lighting = cfg.get("lighting") or (cfg.get("lightings") or ["day"])[0]
    clear_scene()
    build_room(lay)
    build_lights(lay)
    build_cars(lay)
    build_desk(lay["desk"])
    build_props(lay)
    for ob in bpy.data.objects:
        ob[NON_PERSON_TAG] = True
    set_lighting(lighting, lay)
