"""Build one scene from config + timeline and render, per camera: the instance-mask pass and one
solo-mask pass per person (Workbench, shared by all lighting variants), then the RGB pass (EEVEE)
for every requested lighting variant.

blender -b --factory-startup --python sim/blender/render.py -- \
    --config sim/configs/scene_003.json --timeline <scene_dir>/timeline.json --scene-dir <scene_dir> \
    --clips-root data/synthetic/clips [--cams 1,2] [--lightings day,glare] [--passes rgb,mask,solo] [--max-frames N]

Writes <scene_dir>/cam<k>/mask/NNNN.png, <scene_dir>/cam<k>/solo/p<id>/NNNN.png,
<clips_root>/<clip_id>/rgb/NNNN.png, <scene_dir>/render_stats.json, <scene_dir>/scene.blend.
NNNN = output frame index. Glass is hidden in the mask passes, so people seen through the glass
facade are labelled visible, as they appear in RGB.
"""
import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import layout, schema  # noqa: E402
from sim.blender import animate, build_scene, cameras, people  # noqa: E402
from sim.labels import person_colour  # noqa: E402

SRC_HEIGHTS = {c["name"]: (c["height_m"], c["gender"]) for c in
               json.loads((REPO / "sim" / "configs" / "assets_people.json").read_text())["characters"]}


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--clips-root", required=True)
    ap.add_argument("--cams", default="")
    ap.add_argument("--lightings", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--passes", default="rgb,mask,solo")
    ap.add_argument("--samples", type=int, default=16)
    return ap.parse_args(argv)


def setup_rgb(scene, samples=16):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    ee = scene.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.use_raytracing = True
    ee.ray_tracing_options.resolution_scale = "2"
    ee.use_gtao = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.render.dither_intensity = 1.0
    scene.render.film_transparent = False
    _png(scene)
    for ob in bpy.data.objects:
        if ob.get("sim_glass"):
            ob.hide_render = False


def setup_mask(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_object_outline = False
    sh.show_specular_highlight = False
    sh.show_xray = False
    sh.use_dof = False
    scene.display.render_aa = "OFF"
    scene.world.color = (0.0, 0.0, 0.0)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = False
    _png(scene)
    for ob in bpy.data.objects:
        if ob.get("sim_glass"):
            ob.hide_render = True
        if ob.type == "MESH":
            pid = ob.get(people.PERSON_TAG)
            ob.color = (*[c / 255.0 for c in person_colour(pid)], 1.0) if pid else (0.0, 0.0, 0.0, 1.0)


def _png(scene):
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "RGB"
    s.color_depth = "8"
    s.compression = 15


def dedupe_exif(path):
    """Blender 4.2 writes the PNG eXIf chunk twice; libpng then warns on every read. Keep the
    first copy (lossless: chunk-level rewrite, pixel data untouched)."""
    data = Path(path).read_bytes()
    out, i, seen = [data[:8]], 8, False
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        chunk = data[i:i + 12 + n]
        if chunk[4:8] == b"eXIf":
            if seen:
                i += 12 + n
                continue
            seen = True
        out.append(chunk)
        i += 12 + n
    Path(path).write_bytes(b"".join(out))


def render_frames(scene, frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = []
    for k, f in enumerate(frames):
        scene.frame_set(f)
        scene.render.filepath = str(out_dir / f"{k:04d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(time.perf_counter() - t0)
        dedupe_exif(scene.render.filepath)
    return times


def _stats(times):
    steady = times[1:] if len(times) > 1 else times
    return {"n": len(times), "first_s": round(times[0], 4), "mean_s": round(statistics.mean(steady), 4),
            "median_s": round(statistics.median(steady), 4), "total_s": round(sum(times), 2),
            "note": "mean/median exclude the first frame (shader compilation)"}


def build_people(cfg, tl, keep):
    rigs = {}
    actions = None
    sampled = {}
    for p in cfg["persons"]:
        fr = tl["agents"][str(p["id"])]["frames"]
        name = p.get("character")
        if people.rocketbox_available(name):
            src_h, gender = SRC_HEIGHTS[name]
            recol = cfg["uniform"].get("suit_recolour") if p["outfit"] == "uniform" else None
            rig = people.import_character(p, recolour=recol)
            plan = animate.plan_for(fr, p, gender, src_h, cfg["fps_scene"])
            needed = {c for f in keep for c, _, _ in plan[f]}
            names = {f"RB_{c}__{gender}" for c in needed}
            if actions is None:
                actions = {}
            missing = names - set(actions)
            if missing:
                actions.update(people.load_actions(missing))
            for c in needed:
                if (c, gender) not in sampled:
                    sampled[(c, gender)] = animate.sample_action(actions[f"RB_{c}__{gender}"])
            samp = {c: sampled[(c, gender)] for c in needed}
            animate.animate_rocketbox(rig, fr, plan, samp, src_h, p["height_m"], keep, gender)
            rig["kind"] = "rocketbox"
        else:
            rig = people.build_mannequin(p, cfg["uniform"])
            animate.animate_person(rig, fr, p["height_m"], keep=set(keep))
            rig = {"kind": "mannequin", **rig}
        rigs[p["id"]] = rig
    return rigs


def main():
    a = _args()
    cfg = schema.load_scene(a.config)
    tl = json.loads(Path(a.timeline).read_text())
    sdir, croot = Path(a.scene_dir), Path(a.clips_root)
    lay = layout.build_layout(cfg)
    t_build = time.perf_counter()
    build_scene.build(lay, cfg)
    frames = schema.output_scene_frames(cfg)
    if a.max_frames:
        frames = frames[: a.max_frames]
    rigs = build_people(cfg, tl, frames)
    t_build = time.perf_counter() - t_build
    scene = bpy.context.scene
    scene.render.fps = cfg["fps_scene"]
    scene.frame_start, scene.frame_end = 0, schema.n_scene_frames(cfg) - 1
    scene.frame_step = cfg["frame_step"]
    cams = {c["id"]: cameras.make_camera(c, f"cam{c['id']}") for c in cfg["cameras"]}
    cam_ids = [int(x) for x in a.cams.split(",")] if a.cams else sorted(cams)
    lightings = a.lightings.split(",") if a.lightings else cfg["lightings"]
    passes = a.passes.split(",")
    stats = {"blender": bpy.app.version_string, "gpu": None, "frames": len(frames), "build_s": round(t_build, 2),
             "resolution": [scene.render.resolution_x, scene.render.resolution_y],
             "people": {str(k): v["kind"] for k, v in rigs.items()}, "cams": {}}
    try:
        import gpu
        stats["gpu"] = gpu.platform.renderer_get()
    except Exception:  # noqa: BLE001 - informational only
        pass
    sdir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(sdir / "scene.blend"))
    meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
    for cid in cam_ids:
        scene.camera = cams[cid]
        cs = stats["cams"].setdefault(str(cid), {})
        if "mask" in passes or "solo" in passes:
            setup_mask(scene)
        if "mask" in passes:
            cs["mask_workbench"] = _stats(render_frames(scene, frames, sdir / f"cam{cid}" / "mask"))
        if "solo" in passes:
            solo_times = []
            for pid in rigs:
                for ob in meshes:
                    ob.hide_render = ob.get(people.PERSON_TAG) != pid
                solo_times += render_frames(scene, frames, sdir / f"cam{cid}" / "solo" / f"p{pid}")
            for ob in meshes:
                ob.hide_render = False
            cs["solo_workbench"] = _stats(solo_times)
    if "rgb" in passes:
        for lt in lightings:
            build_scene.set_lighting(lt, lay)
            setup_rgb(scene, a.samples)
            for cid in cam_ids:
                scene.camera = cams[cid]
                cid_s = schema.clip_id(cfg, cid, lt)
                t = render_frames(scene, frames, croot / cid_s / "rgb")
                stats["cams"].setdefault(str(cid), {})[f"rgb_eevee_{lt}"] = _stats(t)
    (sdir / "render_stats.json").write_text(json.dumps(stats, indent=1))
    print("RENDER_STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
