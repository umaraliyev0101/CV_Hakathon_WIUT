"""config -> timeline -> Blender render -> labels -> post -> contact sheet -> verify.

python sim/run_all.py --config sim/configs/scene_000.json
Blender is taken from $BLENDER, then PATH, then ~/opt/blender-*/blender.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import behaviour, gt_events, labels, layout, post, projection, schema  # noqa: E402
from sim.qa import contact_sheets  # noqa: E402

OUT_ROOT = REPO / "data" / "synthetic" / "quicktest"


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


def clip_dir_for(cfg):
    return OUT_ROOT / schema.clip_id(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sim/configs/scene_000.json")
    ap.add_argument("--max-frames", type=int, default=0, help="smoke test only; gates will fail")
    ap.add_argument("--skip-render", action="store_true", help="reuse existing renders")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    cfg_path = Path(a.config).resolve()
    cfg = schema.load_scene(cfg_path)
    presets = json.loads((REPO / "sim" / "configs" / "presets.json").read_text())
    qa = json.loads((REPO / "sim" / "configs" / "qa.json").read_text())
    out = clip_dir_for(cfg)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tl = behaviour.build_timeline(cfg)
    schema.dump_json(tl, out / "timeline.json")
    print(f"[timeline] {out / 'timeline.json'} hash={behaviour.timeline_hash(tl)[:12]}")

    if not a.skip_render:
        for d in ("rgb", "mask", "solo"):
            shutil.rmtree(out / d, ignore_errors=True)
        blender = find_blender()
        cmd = [blender, "-b", "--factory-startup", "--python", str(REPO / "sim" / "blender" / "render.py"), "--",
               "--config", str(cfg_path), "--timeline", str(out / "timeline.json"), "--out", str(out)]
        if a.max_frames:
            cmd += ["--max-frames", str(a.max_frames)]
        print("[render]", " ".join(cmd))
        log = out / "blender.log"
        with open(log, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not (out / "render_stats.json").exists():
            print(log.read_text()[-3000:])
            raise SystemExit("blender render failed")
        print("[render] stats", (out / "render_stats.json").read_text())

    lay = layout.build_layout(cfg)
    cam = projection.Camera.from_params(cfg["camera"])
    ev_rule = gt_events.events_by_rule(tl, cfg, presets["short_clip"])
    try:
        cfg_rel = str(cfg_path.relative_to(REPO / "sim"))
    except ValueError:
        cfg_rel = str(cfg_path)
    if a.max_frames:
        cfg = dict(cfg, duration_s=a.max_frames * cfg["frame_step"] / cfg["fps_scene"])
    gt = labels.build_gt(cfg, tl, cam, lay, out, ev_rule, git_sha(), cfg_rel, qa["labels"])
    labels.write_gt(gt, out / "gt.json")
    print(f"[labels] {out / 'gt.json'} frames={len(gt['frames'])}")

    mp4 = out / f"{schema.clip_id(cfg)}.mp4"
    n = post.encode(out / "rgb", mp4, schema.out_fps(cfg), f"CAM{cfg['camera']['id']}", seed=cfg["seed"])
    print(f"[post] {mp4} frames={n}")

    sheet = contact_sheets.make_sheet(mp4, out / "gt.json", out / "contact_sheet.png")
    shutil.copy(sheet, REPO / "reports" / "synth_quicktest_contact.png")
    print(f"[contact] {sheet} -> reports/synth_quicktest_contact.png")
    print(f"[run_all] pipeline took {time.time() - t0:.1f} s")

    if not a.no_verify:
        r = subprocess.run([sys.executable, str(REPO / "sim" / "verify.py"), "--clip", str(out)])
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
