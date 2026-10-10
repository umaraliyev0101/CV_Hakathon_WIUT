"""THE verification chain: pytest, qa_assets, qa_scene, qa_render, qa_events, qa_detector, qa_video.

python sim/verify.py --all                 # every clip in data/synthetic/share/ (the share pack)
python sim/verify.py --clip <clip_dir> ... # specific clip dirs (e.g. data/synthetic/clips/<clip_id>)
Exits 0 only if every gate passes. qa_scene runs Blender once per distinct scene of the clips.
qa_detector runs RF-DETR (onnxruntime) with CUDA_VISIBLE_DEVICES=1.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 0 belongs to other services; inherited by subprocesses

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.qa import common, qa_assets, qa_detector, qa_events, qa_render, qa_scene, qa_video  # noqa: E402

SHARE = REPO / "data" / "synthetic" / "share"


def _gate(results, name, fn):
    t0 = time.time()
    try:
        fails, info = fn()
        results[name] = common.report(name, fails, info)
    except Exception as e:  # noqa: BLE001 - a crashing gate is a failing gate
        print(f"{name}: FAIL crashed: {type(e).__name__}: {e}")
        results[name] = 1
    print(f"  ({name} took {time.time() - t0:.1f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all clips in data/synthetic/share/")
    ap.add_argument("--clip", nargs="*", default=[])
    a = ap.parse_args()
    clips = [Path(c) for c in a.clip]
    if a.all:
        clips = sorted(d for d in SHARE.iterdir() if (d / "gt.json").exists()) if SHARE.exists() else []
    qa = common.load_qa()
    results = {}
    print(f"=== verify: {len(clips)} clip(s): {[c.name for c in clips]} ===", flush=True)
    if not clips:
        print("verify: FAIL no clips (run sim/run_all.py and sim/share.py first)")
        return 1
    print("=== gate: pytest -q sim/tests ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "sim/tests"], cwd=REPO)
    results["pytest"] = r.returncode
    print("=== gate: qa_assets ===", flush=True)
    _gate(results, "qa_assets", lambda: qa_assets.run(None, qa))
    scenes = sorted({json.loads((c / "gt.json").read_text())["scene_id"] for c in clips})
    for sid in scenes:
        print(f"=== gate: qa_scene ({sid}) ===", flush=True)
        _gate(results, f"qa_scene[{sid}]", lambda sid=sid: qa_scene.run_scene(REPO / "data" / "synthetic" / "scenes" / sid, qa))
    for clip in clips:
        for name, mod in (("qa_render", qa_render), ("qa_events", qa_events), ("qa_video", qa_video)):
            print(f"=== gate: {name} ({clip.name}) ===", flush=True)
            _gate(results, f"{name}[{clip.name}]", lambda mod=mod, clip=clip: mod.run(clip, qa))
    print("=== gate: qa_detector (all clips, per lighting variant) ===", flush=True)
    _gate(results, "qa_detector", lambda: qa_detector.run_clips(clips, qa))
    print("=== summary ===")
    for k, v in results.items():
        print(f"  {k:60s} {'PASS' if v == 0 else 'FAIL'}")
    ok = all(v == 0 for v in results.values())
    print("VERIFY:", "PASS" if ok else "FAIL", f"({sum(v == 0 for v in results.values())}/{len(results)} gates passed)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
