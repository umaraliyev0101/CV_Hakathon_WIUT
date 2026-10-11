"""Controlled parallel-render benchmark: the same render job (scene, camera, lighting, frames, passes)
run once alone and then N times concurrently on GPU 1. gain = N * T_alone / T_concurrent.

python sim/bench_parallel.py --config sim/configs/scene_000.json --frames 60 --jobs 3 [--passes rgb]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim import behaviour, schema  # noqa: E402
from sim.run_all import find_blender  # noqa: E402


def job(cfg_path, tl_path, out, frames, passes):
    return [find_blender(), "-b", "--factory-startup", "--python", str(REPO / "sim" / "blender" / "render.py"), "--",
            "--config", str(cfg_path), "--timeline", str(tl_path), "--scene-dir", str(out / "scene"),
            "--clips-root", str(out / "clips"), "--cams", "1", "--lightings", "day",
            "--max-frames", str(frames), "--passes", passes]


def run(cmds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
    t0 = time.time()
    ps = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env) for c in cmds]
    rc = [p.wait() for p in ps]
    if any(rc):
        raise SystemExit(f"render failed {rc}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sim/configs/scene_000.json")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--passes", default="rgb,mask,solo")
    a = ap.parse_args()
    cfg = schema.load_scene(a.config)
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        td = Path(td)
        tl = td / "timeline.json"
        schema.dump_json(behaviour.build_timeline(cfg), tl)
        run([job(a.config, tl, td / "warm", 2, a.passes)])            # warm caches (shaders, files)
        t1 = run([job(a.config, tl, td / "alone", a.frames, a.passes)])
        tn = run([job(a.config, tl, td / f"par{i}", a.frames, a.passes) for i in range(a.jobs)])
    res = {"config": a.config, "frames": a.frames, "passes": a.passes, "jobs": a.jobs,
           "t_alone_s": round(t1, 1), "t_concurrent_s": round(tn, 1), "gain": round(a.jobs * t1 / tn, 3)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
