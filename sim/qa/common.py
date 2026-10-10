import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CLIP = REPO / "data" / "synthetic" / "quicktest" / "s000_greet_fast_cam1_day"


def load_qa():
    return json.loads((REPO / "sim" / "configs" / "qa.json").read_text())


def clip_arg(argv):
    return Path(argv[1]) if len(argv) > 1 else DEFAULT_CLIP


def report(name, failures, info):
    for line in info:
        print(f"  {name}: {line}")
    for f in failures:
        print(f"  {name}: FAIL {f}")
    print(f"{name}: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
    return 0 if not failures else 1
