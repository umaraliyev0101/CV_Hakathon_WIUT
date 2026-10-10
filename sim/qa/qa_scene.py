"""qa_scene: geometry / animation sanity of a rendered scene (measured in Blender on scene.blend,
plus timeline + mask checks in Python). Fails when: person height outside range; car length
outside range; a person's root inside a car footprint (+margin); two people closer than
min_person_dist_m for > min_person_dist_max_s; lowest body vertex off the floor by > feet_floor_tol_m
at sampled frames (some foot is always in contact: no flight phase in walk/idle/talk); an armature
unchanged over frozen_window_s while the action is walk/talk; a missing image texture; an agent
visible (labelled) in no camera for > max_invisible_lifetime_frac of its lifetime."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim.qa.common import REPO, load_qa, report
from sim import labels, layout, schema


def measure(scene_dir, cfg_path, timeline_path, out_json):
    from sim.run_all import find_blender
    cmd = [find_blender(), "-b", str(scene_dir / "scene.blend"), "--python", str(REPO / "sim" / "qa" / "qa_scene_blender.py"),
           "--", "--config", str(cfg_path), "--timeline", str(timeline_path), "--out", str(out_json)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))
    if r.returncode != 0 or not Path(out_json).exists():
        raise RuntimeError(f"blender qa_scene failed: {r.stdout[-1500:]} {r.stderr[-1500:]}")
    return json.loads(Path(out_json).read_text())


def run_scene(scene_dir, qa):
    q = qa["scene"]
    scene_dir = Path(scene_dir)
    tl = json.loads((scene_dir / "timeline.json").read_text())
    cfg_path = REPO / "sim" / "configs" / f"scene_{tl['scene_id'][1:]}.json"
    cfg = schema.load_scene(cfg_path)
    lay = layout.build_layout(cfg)
    fails, info = [], []
    m = measure(scene_dir, cfg_path, scene_dir / "timeline.json", scene_dir / "qa_scene.json")
    fps = cfg["fps_scene"]
    lo, hi = q["person_height_m"]
    for pid, pm in m["persons"].items():
        h = pm["standing_height_m"]
        if h is None or not lo <= h <= hi:
            fails.append(f"person {pid}: standing height {h} m outside {lo}-{hi}")
        zs = [x["zmin"] for x in pm["feet"]]
        bad = [x for x in pm["feet"] if abs(x["zmin"]) > q["feet_floor_tol_m"]]
        info.append(f"person {pid} ({pm['kind']}): height {h:.3f} m (config {schema.person(cfg, int(pid))['height_m']}); "
                    f"lowest vertex z over {len(zs)} sampled frames min/max {min(zs):+.3f}/{max(zs):+.3f} m")
        for x in bad[:5]:
            fails.append(f"person {pid}: lowest vertex at z={x['zmin']:+.3f} m at scene frame {x['f']} ({x['anim']})")
        sig = {int(k): np.asarray(v) for k, v in pm["pose_sig"].items()}
        fr = tl["agents"][pid]["frames"]
        win = int(round(q["frozen_window_s"] * fps))
        for f0 in sorted(sig):
            f1 = f0 + win
            if f1 in sig and all(fr[f]["action"] in ("walk", "talk") for f in range(f0, f1 + 1)) \
                    and len(sig[f0]) and np.abs(sig[f1] - sig[f0]).max() < q["frozen_min_change"]:
                fails.append(f"person {pid}: pose unchanged {f0}->{f1} while {fr[f0]['action']}")
    lo, hi = q["car_length_m"]
    for cid, cm in m["cars"].items():
        L = cm["length_m"]
        if L is None or not lo <= L <= hi:
            fails.append(f"car {cid}: length {L} outside {lo}-{hi}")
    info.append("car lengths: " + ", ".join(f"car_{k}={v['length_m']:.2f} m" for k, v in m["cars"].items()))
    missing = [k for k, v in m["images"].items() if not v["exists"]]
    info.append(f"images: {len(m['images'])} file textures, {len(missing)} missing")
    for k in missing:
        fails.append(f"missing image texture {k}: {m['images'][k]['path']}")
    # timeline geometry
    cars = [c["footprint_m"] for c in lay["cars"]]
    for pid, ag in tl["agents"].items():
        for i, st in enumerate(ag["frames"]):
            if st["rendered"] and layout.dist_to_polygons(st["pos_m"], cars) < q["root_car_margin_m"]:
                fails.append(f"person {pid}: root within {q['root_car_margin_m']} m of a car at scene frame {i}")
                break
    ids = sorted(tl["agents"])
    max_run = int(round(q["min_person_dist_max_s"] * fps))
    closest = math.inf
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            run = 0
            for sa, sb in zip(tl["agents"][ids[a]]["frames"], tl["agents"][ids[b]]["frames"]):
                close = sa["rendered"] and sb["rendered"] and math.dist(sa["pos_m"], sb["pos_m"]) < q["min_person_dist_m"]
                if sa["rendered"] and sb["rendered"]:
                    closest = min(closest, math.dist(sa["pos_m"], sb["pos_m"]))
                run = run + 1 if close else 0
                if run > max_run:
                    fails.append(f"persons {ids[a]},{ids[b]} closer than {q['min_person_dist_m']} m for > {q['min_person_dist_max_s']} s")
                    break
    info.append(f"closest approach between two people: {closest:.2f} m")
    # visible in some camera for >= 1 - max_invisible of lifetime (from the rendered instance masks)
    qa_l = qa["labels"]
    cols = {p["id"]: labels.person_colour(p["id"]) for p in cfg["persons"]}
    out_frames = schema.output_scene_frames(cfg)
    cams = sorted(d for d in scene_dir.glob("cam*") if (d / "mask").exists())
    seen = {p["id"]: np.zeros(len(out_frames), bool) for p in cfg["persons"]}
    for cd in cams:
        for k in range(len(out_frames)):
            mk = labels.read_png_rgb(cd / "mask" / f"{k:04d}.png")
            for pid, b in labels.boxes_from_mask(mk, cols, qa_l["min_mask_px"], qa_l["min_box_h_px"]).items():
                seen[pid][k] = True
    for p in cfg["persons"]:
        alive = np.array([tl["agents"][str(p["id"])]["frames"][f]["rendered"] for f in out_frames])
        frac_inv = float((alive & ~seen[p["id"]]).sum() / max(1, alive.sum()))
        info.append(f"person {p['id']}: invisible in all {len(cams)} rendered camera(s) for {frac_inv:.1%} of lifetime")
        if frac_inv > q["max_invisible_lifetime_frac"]:
            fails.append(f"person {p['id']}: invisible {frac_inv:.1%} of lifetime > {q['max_invisible_lifetime_frac']:.0%}")
    return fails, info


def run(scene_dir, qa):
    return run_scene(scene_dir, qa)


def main(argv=sys.argv):
    fails, info = run_scene(Path(argv[1]), load_qa())
    return report("qa_scene", fails, info)


if __name__ == "__main__":
    sys.exit(main())
