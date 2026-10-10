"""config -> timeline -> Blender render (all cameras x lighting variants) -> labels -> post -> contact sheets.

python sim/run_all.py --config sim/configs/scene_003.json [--cams 1] [--lightings day,glare]
python sim/run_all.py --all [--jobs 2]            # every sim/configs/scene_*.json
Then: python sim/export.py (COCO / MOT / A-format), python sim/share.py (share pack), python sim/verify.py --all

Blender is taken from $BLENDER, then PATH, then ~/opt/blender-*/blender. Every Blender / torch /
onnxruntime subprocess gets CUDA_VISIBLE_DEVICES=1 (GPU 0 belongs to other services); the GPU
that actually ran the render is sampled with `nvidia-smi pmon` and stored as gpu_index.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import behaviour, gt_events, labels, layout, post, projection, schema  # noqa: E402
from sim.qa import contact_sheets  # noqa: E402

DATA = REPO / "data" / "synthetic"
SCENES_ROOT = DATA / "scenes"
CLIPS_ROOT = DATA / "clips"
GPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="1")


def find_blender():
    cands = [os.environ.get("BLENDER"), shutil.which("blender")]
    cands += sorted(glob.glob(os.path.expanduser("~/opt/blender-*/blender")), reverse=True)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise SystemExit("Blender not found: set $BLENDER (needs 4.2+)")


def git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=REPO, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def free_gb(path=REPO):
    return shutil.disk_usage(path).free / 1e9


class GpuWatch(threading.Thread):
    """Samples `nvidia-smi pmon` while a process runs; records GPU indices that show its pid."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.seen, self.stop = str(pid), set(), False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True, timeout=20).stdout
                for line in out.splitlines():
                    f = line.split()
                    if len(f) > 2 and f[1] == self.pid:
                        self.seen.add(int(f[0]))
            except Exception:  # noqa: BLE001 - informational
                pass
            time.sleep(2.0)


def render_scene(cfg_path, cfg, tl_path, sdir, cams, lightings, passes, max_frames):
    blender = find_blender()
    cmd = [blender, "-b", "--factory-startup", "--python", str(REPO / "sim" / "blender" / "render.py"), "--",
           "--config", str(cfg_path), "--timeline", str(tl_path), "--scene-dir", str(sdir),
           "--clips-root", str(CLIPS_ROOT), "--passes", passes]
    if cams:
        cmd += ["--cams", ",".join(map(str, cams))]
    if lightings:
        cmd += ["--lightings", ",".join(lightings)]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    log = sdir / "blender.log"
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=GPU_ENV)
        watch = GpuWatch(proc.pid)
        watch.start()
        rc = proc.wait()
        watch.stop = True
    if rc != 0 or not (sdir / "render_stats.json").exists():
        print(log.read_text()[-3000:])
        raise SystemExit(f"blender render failed for {cfg['scene_id']}")
    st = json.loads((sdir / "render_stats.json").read_text())
    st["gpu_index"] = sorted(watch.seen)
    st["wall_s"] = round(time.time() - t0, 1)
    (sdir / "render_stats.json").write_text(json.dumps(st, indent=1))
    return st


def _link(src, dst):
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def run_scene(cfg_path, cams=None, lightings=None, skip_render=False, max_frames=0, passes="rgb,mask,solo"):
    cfg_path = Path(cfg_path).resolve()
    cfg = schema.load_scene(cfg_path)
    presets = json.loads((REPO / "sim" / "configs" / "presets.json").read_text())
    qa = json.loads((REPO / "sim" / "configs" / "qa.json").read_text())
    sdir = SCENES_ROOT / cfg["scene_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    cams = cams or [c["id"] for c in cfg["cameras"]]
    lightings = lightings or cfg["lightings"]
    t0 = time.time()
    tl = behaviour.build_timeline(cfg)
    tl_path = sdir / "timeline.json"
    schema.dump_json(tl, tl_path)
    print(f"[{cfg['scene_id']}] timeline hash={behaviour.timeline_hash(tl)[:12]}", flush=True)
    if not skip_render:
        if free_gb() < 100:
            raise SystemExit(f"free disk {free_gb():.0f} GB < 100 GB: stop rendering")
        st = render_scene(cfg_path, cfg, tl_path, sdir, cams, lightings, passes, max_frames)
        print(f"[{cfg['scene_id']}] rendered in {st['wall_s']} s on GPU {st['gpu_index']}", flush=True)
    lay = layout.build_layout(cfg)
    ev_rule = {name: gt_events.events_by_rule(tl, cfg, presets[name]) for name in ("short_clip", "dealership")}
    try:
        cfg_rel = str(cfg_path.relative_to(REPO / "sim"))
    except ValueError:
        cfg_rel = str(cfg_path)
    sha = git_sha()
    clip_dirs = []
    for cid in cams:
        cam = projection.Camera.from_params(schema.camera(cfg, cid))
        mask_dir, solo_dir = sdir / f"cam{cid}" / "mask", sdir / f"cam{cid}" / "solo"
        base = labels.build_gt(cfg, tl, cam, lay, mask_dir, solo_dir, ev_rule, sha, cfg_rel, qa["labels"],
                               cid, lightings[0], n_frames=max_frames or None)
        for lt in lightings:
            clip = schema.clip_id(cfg, cid, lt)
            out = CLIPS_ROOT / clip
            out.mkdir(parents=True, exist_ok=True)
            _link(mask_dir, out / "mask")
            _link(solo_dir, out / "solo")
            gt = dict(base, clip_id=clip, lighting=lt)
            labels.write_gt(gt, out / "gt.json")
            mp4 = out / f"{clip}.mp4"
            post.encode(out / "rgb", mp4, schema.out_fps(cfg), f"CAM{cid}", seed=cfg["seed"] * 100 + cid)
            contact_sheets.make_sheet(mp4, out / "gt.json", out / "contact_sheet.png")
            clip_dirs.append(out)
            print(f"[{clip}] gt frames={len(gt['frames'])} -> {mp4.name}", flush=True)
    print(f"[{cfg['scene_id']}] done in {time.time() - t0:.0f} s", flush=True)
    return clip_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cams", default="")
    ap.add_argument("--lightings", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke test only; gates will fail")
    ap.add_argument("--skip-render", action="store_true", help="reuse existing renders")
    a = ap.parse_args()
    cfgs = sorted(glob.glob(str(REPO / "sim" / "configs" / "scene_*.json"))) if a.all else (a.config or [])
    cams = [int(x) for x in a.cams.split(",")] if a.cams else None
    lts = a.lightings.split(",") if a.lightings else None
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(lambda c: run_scene(c, cams, lts, a.skip_render, a.max_frames), cfgs))


if __name__ == "__main__":
    main()
