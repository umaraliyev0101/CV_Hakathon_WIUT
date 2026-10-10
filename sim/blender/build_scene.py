"""Room, materials, lights, stand-in box cars and desk (runs inside Blender)."""
import math

import bpy

NON_PERSON_TAG = "sim_static"


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            coll.remove(block)


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


def build_room(lay):
    r = lay["room"]
    W, D, H, T = r["width_m"], r["depth_m"], r["height_m"], r["wall_thickness_m"]
    floor_m = material("floor_tile", (0.22, 0.22, 0.23), rough=0.22)
    wall_m = material("wall", (0.62, 0.6, 0.56), rough=0.85)
    ceil_m = material("ceiling", (0.5, 0.5, 0.5), rough=0.9)
    box("floor", (W / 2, D / 2 - 1.5, -0.05), (W + 2 * T, D + 3.0 + 2 * T, 0.1), floor_m)
    box("ceiling", (W / 2, D / 2, H + 0.05), (W + 2 * T, D + 2 * T, 0.1), ceil_m)
    box("wall_north", (W / 2, D + T / 2, H / 2), (W + 2 * T, T, H), wall_m)
    box("wall_west", (-T / 2, D / 2, H / 2), (T, D, H), wall_m)
    box("wall_east", (W + T / 2, D / 2, H / 2), (T, D, H), wall_m)
    e = lay["entrance"]
    x0, x1 = e["center_x_m"] - e["width_m"] / 2, e["center_x_m"] + e["width_m"] / 2
    box("wall_south_a", (x0 / 2, -T / 2, H / 2), (x0, T, H), wall_m)
    box("wall_south_b", ((x1 + W) / 2, -T / 2, H / 2), (W - x1, T, H), wall_m)
    box("wall_south_lintel", (e["center_x_m"], -T / 2, (e["height_m"] + H) / 2),
        (e["width_m"], T, H - e["height_m"]), wall_m)
    # outside the doorway: a bright vestibule so the entrance does not read as a black hole
    box("vestibule_back", (e["center_x_m"], -3.0, H / 2), (e["width_m"] + 2, 0.1, H),
        material("vestibule", (0.7, 0.72, 0.75), rough=0.9))


def build_lights(lay, lighting):
    if lighting != "day":
        raise NotImplementedError("quick test renders 'day' only")
    r = lay["room"]
    W, D, H = r["width_m"], r["depth_m"], r["height_m"]
    panel_m = material("light_panel", (1, 1, 1), emission=(1.0, 0.97, 0.92), strength=6.0)
    for i, x in enumerate((W * 0.2, W * 0.5, W * 0.8)):
        for j, y in enumerate((D * 0.25, D * 0.75)):
            bpy.ops.object.light_add(type="AREA", location=(x, y, H - 0.05))
            L = bpy.context.active_object
            L.name = f"ceiling_light_{i}{j}"
            L.data.shape, L.data.size, L.data.size_y = "RECTANGLE", 3.0, 1.2
            L.data.energy = 900.0
            L.data.color = (1.0, 0.97, 0.92)
            box(f"light_panel_{i}{j}", (x, y, H - 0.01), (3.0, 1.2, 0.02), panel_m)
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.62, 0.72, 1.0)
    bg.inputs["Strength"].default_value = 0.6


def build_car(car):
    L, Wd, Hh = car["length_m"], car["width_m"], car["height_m"]
    cx, cy = car["center_m"]
    yaw = car["yaw_deg"]
    k = car["id"]
    paint = material(f"paint_{k}", tuple(car["paint"]), rough=0.25, metallic=0.6, coat=1.0)
    glass = material("car_glass", (0.02, 0.025, 0.03), rough=0.05)
    tyre = material("tyre", (0.015, 0.015, 0.015), rough=0.7)
    wheel_r = 0.33
    body_h = 0.62
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

    def at(dx, dy, z):
        return (cx + c * dx - s * dy, cy + s * dx + c * dy, z)

    parts = [box(f"car{k}_body", at(0, 0, wheel_r * 0.6 + body_h / 2), (L, Wd, body_h), paint, yaw, bevel=0.12)]
    cab_h = Hh - (wheel_r * 0.6 + body_h)
    parts.append(box(f"car{k}_cabin", at(-0.1 * L, 0, Hh - cab_h / 2), (0.52 * L, 0.86 * Wd, cab_h), glass, yaw,
                     bevel=0.08))
    for dx in (-0.33 * L, 0.33 * L):
        for dy in (-Wd / 2 + 0.12, Wd / 2 - 0.12):
            parts.append(cylinder(f"car{k}_wheel", at(dx, dy, wheel_r), wheel_r, 0.24, tyre,
                                  rot=(math.pi / 2, 0, math.radians(yaw))))
    return parts


def build_desk(desk):
    wood = material("desk_top", (0.32, 0.2, 0.11), rough=0.4)
    panel = material("desk_panel", (0.12, 0.12, 0.13), rough=0.6)
    cx, cy = desk["center_m"]
    L, Wd, Hh = desk["length_m"], desk["width_m"], desk["height_m"]
    yaw = desk["yaw_deg"]
    box("desk_top", (cx, cy, Hh - 0.02), (L, Wd, 0.04), wood, yaw)
    box("desk_body", (cx, cy, (Hh - 0.04) / 2), (L * 0.96, Wd * 0.8, Hh - 0.04), panel, yaw)


def build(lay, lighting):
    clear_scene()
    build_room(lay)
    build_lights(lay, lighting)
    for car in lay["cars"]:
        build_car(car)
    build_desk(lay["desk"])
