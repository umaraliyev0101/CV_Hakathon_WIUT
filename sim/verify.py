"""THE verification chain for the quick test: pytest, qa_render, qa_events, qa_video.

python sim/verify.py [--clip data/synthetic/quicktest/<clip_id>]   (exit 0 only if every gate passes)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.qa import common, qa_events, qa_render, qa_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=str(common.DEFAULT_CLIP))
    a = ap.parse_args()
    clip = Path(a.clip)
    qa = common.load_qa()
    results = {}
    print("=== gate: pytest -q sim/tests ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "sim/tests"], cwd=REPO)
    results["pytest"] = r.returncode
    for name, mod in (("qa_render", qa_render), ("qa_events", qa_events), ("qa_video", qa_video)):
        print(f"=== gate: {name} ({clip}) ===", flush=True)
        if not (clip / "gt.json").exists():
            print(f"{name}: FAIL missing {clip / 'gt.json'} (run sim/run_all.py first)")
            results[name] = 1
            continue
        try:
            fails, info = mod.run(clip, qa)
            results[name] = common.report(name, fails, info)
        except Exception as e:  # noqa: BLE001 - a crashing gate is a failing gate
            print(f"{name}: FAIL crashed: {type(e).__name__}: {e}")
            results[name] = 1
    print("=== summary ===")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v == 0 else 'FAIL'}")
    ok = all(v == 0 for v in results.values())
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
