"""Build the scene from config + timeline and render RGB (EEVEE) and instance masks (Workbench).

blender -b --factory-startup --python sim/blender/render.py -- \
    --config sim/configs/scene_000.json --timeline <clip>/timeline.json --out <clip> [--max-frames N]

Writes <out>/rgb/NNNN.png, <out>/mask/NNNN.png, <out>/solo/p<id>/NNNN.png (each person rendered
alone, for visible_frac), <out>/render_stats.json and <out>/scene.blend. NNNN = output frame index.
"""
import argparse
import json
import statistics
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


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--passes", default="rgb,mask,solo")
    return ap.parse_args(argv)


def setup_rgb(scene):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    ee = scene.eevee
    ee.taa_render_samples = 16
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
        if ob.type == "MESH":
            pid = ob.get(people.PERSON_TAG)
            ob.color = (*[c / 255.0 for c in person_colour(pid)], 1.0) if pid else (0.0, 0.0, 0.0, 1.0)


def _png(scene):
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "RGB"
    s.color_depth = "8"
    s.compression = 15


def render_frames(scene, frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    times = []
    for k, f in enumerate(frames):
        scene.frame_set(f)
        scene.render.filepath = str(out_dir / f"{k:04d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(time.perf_counter() - t0)
    return times


def _stats(times):
    steady = times[1:] if len(times) > 1 else times
    return {"n": len(times), "first_s": round(times[0], 4), "mean_s": round(statistics.mean(steady), 4),
            "median_s": round(statistics.median(steady), 4), "total_s": round(sum(times), 2),
            "note": "mean/median exclude the first frame (shader compilation)"}


def main():
    a = _args()
    cfg = schema.load_scene(a.config)
    tl = json.loads(Path(a.timeline).read_text())
    out = Path(a.out)
    lay = layout.build_layout(cfg)
    build_scene.build(lay, cfg["lighting"])
    rigs = {}
    for p in cfg["persons"]:
        rigs[p["id"]] = people.build_mannequin(p, cfg["uniform"])
        animate.animate_person(rigs[p["id"]], tl["agents"][str(p["id"])]["frames"], p["height_m"])
    cameras.make_camera(cfg["camera"], f"cam{cfg['camera']['id']}")
    scene = bpy.context.scene
    scene.render.fps = cfg["fps_scene"]
    scene.frame_start, scene.frame_end = 0, schema.n_scene_frames(cfg) - 1
    scene.frame_step = cfg["frame_step"]
    frames = schema.output_scene_frames(cfg)
    if a.max_frames:
        frames = frames[: a.max_frames]
    passes = a.passes.split(",")
    stats = {"blender": bpy.app.version_string, "gpu": None, "frames": len(frames),
             "resolution": [scene.render.resolution_x, scene.render.resolution_y]}
    try:
        import gpu
        stats["gpu"] = gpu.platform.renderer_get()
    except Exception:  # noqa: BLE001 - informational only
        pass
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if "rgb" in passes:
        setup_rgb(scene)
        stats["rgb_eevee"] = _stats(render_frames(scene, frames, out / "rgb"))
    if "mask" in passes or "solo" in passes:
        setup_mask(scene)
    if "mask" in passes:
        stats["mask_workbench"] = _stats(render_frames(scene, frames, out / "mask"))
    if "solo" in passes:
        meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
        solo_times = []
        for pid in rigs:
            for ob in meshes:
                ob.hide_render = ob.get(people.PERSON_TAG) != pid
            solo_times += render_frames(scene, frames, out / "solo" / f"p{pid}")
        for ob in meshes:
            ob.hide_render = False
        stats["solo_workbench"] = _stats(solo_times)
    (out / "render_stats.json").write_text(json.dumps(stats, indent=1))
    print("RENDER_STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
