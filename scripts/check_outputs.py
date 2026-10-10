"""Validate the files `pulse analyze` writes. Usage: check_outputs.py DIR [DIR ...] [--no-video]
Prints `OK <file>` per file; exits 1 at the first failure with the reason."""
from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from pulse.datatypes import STORE_KEYS_DEFAULTS

REPORT_KEYS = ["banner", "meta", "store", "per_car", "per_zone", "zones", "people", "insights",
               "journey_flow", "charts", "files", "warnings"]
CSV_HEADERS = {
    "visits.csv": "person_id,role,first_t,last_t,already_present,visit_s,zones,stopped_cars,assisted,"
                  "t_first_assist,group".split(","),
    "per_car.csv": "zone_id,name,visitors,stops,stop_rate,hold_rate,median_dwell,total_dwell,"
                   "repeat_visits,deep_interest,unserved,unassisted_share,rank".split(","),
    "occupancy.csv": "t,customers,waiting,customers_per_staff".split(","),
}
FIXED = ["report.json", "stats.json", "insights.json", "zones.json", "roles.json", "people.json",
         "visits.csv", "per_car.csv", "occupancy.csv", "heatmap.png", "occupancy.svg", "waiting.svg",
         "journey_flow.json", "keyframe.jpg"]


class CheckError(Exception):
    pass


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise CheckError(f"not valid JSON ({e})")


def _need(cond, msg):
    if not cond:
        raise CheckError(msg)


def check_json(path: Path, rel: str) -> None:
    d = _json(path)
    if rel == "report.json":
        _need(isinstance(d, dict), "top level must be an object")
        for k in REPORT_KEYS:
            _need(k in d, f"missing key {k!r}")
        _need(isinstance(d["files"], dict), "'files' must be an object")
        for k in ("per_car", "per_zone", "zones", "people", "insights", "warnings"):
            _need(isinstance(d[k], list), f"{k!r} must be a list")
    elif rel == "stats.json":
        _need(isinstance(d, dict) and isinstance(d.get("store"), dict), "missing 'store' object")
        miss = [k for k in STORE_KEYS_DEFAULTS if k not in d["store"]]
        _need(not miss, f"store lacks keys {miss}")
    elif rel == "journey_flow.json":
        _need(isinstance(d, dict) and all(k in d for k in ("nodes", "links", "top_paths")),
              "needs nodes, links, top_paths")
        ids = {n["id"] for n in d["nodes"]}
        _need({"Start", "End"} <= ids, "nodes must include Start and End")
    elif rel == "zones.json":
        _need(isinstance(d, dict) and isinstance(d.get("zones"), list), "missing 'zones' list")
    elif rel == "roles.json":
        _need(isinstance(d, dict) and isinstance(d.get("persons"), list), "missing 'persons' list")
    elif rel == "people.json":
        _need(isinstance(d, dict) and isinstance(d.get("persons"), list), "missing 'persons' list")
    elif rel == "insights.json":
        _need(isinstance(d, list), "must be a list")


def check_csv(path: Path, rel: str) -> None:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    _need(rows and rows[0] == CSV_HEADERS[Path(rel).name], f"bad header {rows[0] if rows else None}")


def check_image(path: Path, rel: str) -> None:
    img = cv2.imread(str(path))
    _need(img is not None and img.size > 0, "image does not decode or is empty")


def check_svg(path: Path, rel: str) -> None:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise CheckError(f"not XML ({e})")
    _need(root.tag.endswith("svg") and "<svg" in path.read_text(), "no <svg> element")


def check_video(path: Path, meta: dict) -> None:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    _need(n >= 1, "video has no decodable frame")
    want = float(meta.get("analysed_duration") or meta.get("duration") or 0) * float(meta.get("sample_fps", 5))
    _need(want <= 0 or abs(n - want) <= 0.1 * want, f"{n} frames, expected about {want:.0f} (+-10%)")


def check_dir(d: Path, video: bool = True, say=print) -> None:
    def run(rel: str, fn) -> None:
        p = d / rel
        try:
            _need(p.is_file(), "missing")
            _need(p.stat().st_size > 0, "empty file")
            fn(p, rel)
        except CheckError as e:
            raise CheckError(f"{p}: {e}")
        say(f"OK {rel}")

    run("report.json", check_json)
    report = json.loads((d / "report.json").read_text())
    names = list(FIXED[1:])
    names += sorted(set(report["files"].values()) - set(FIXED) - {"annotated.mp4"})
    for rel in names:
        suffix = Path(rel).suffix
        fn = {".json": check_json, ".csv": check_csv, ".png": check_image, ".jpg": check_image,
              ".svg": check_svg}[suffix]
        run(rel, fn)
    for rel in report["files"].values():
        _need(rel == "annotated.mp4" and not video or (d / rel).is_file(), f"{d / rel}: listed in files but missing")
    if video:
        run("annotated.mp4", lambda p, rel: check_video(p, report["meta"]))


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    video = "--no-video" not in args
    dirs = [Path(a) for a in args if a != "--no-video"]
    if not dirs:
        print("usage: check_outputs.py DIR [DIR ...] [--no-video]", file=sys.stderr)
        return 2
    for d in dirs:
        try:
            check_dir(d, video)
        except CheckError as e:
            print(f"FAIL {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
