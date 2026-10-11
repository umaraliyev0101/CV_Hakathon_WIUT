"""Entrance-zone coverage check: do GT enter/exit activity foot points fall in an `entrance` zone?

For each `person_enters_scene_through_structure` (first frame) / `person_exits_scene_through_structure`
(last frame) activity, take each actor's GT box at that frame (nearest available box), scale to analysis
size, take the bottom-centre foot point and test it against the clip's entrance zones.

Usage: .venv/bin/python scripts/zone_coverage.py [--group dev|test|all] [--misses]
Tune zones on DEV clips only; test coverage is reported, never tuned."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pulse.eval.meva import load_kpf  # noqa: E402
from pulse.zones import load_zones, point_in_polygon  # noqa: E402

ENTER, EXIT = "person_enters_scene_through_structure", "person_exits_scene_through_structure"
SRC_SIZES = {1080: (1920, 1080), 1072: (1920, 1072)}


def foot_points(clip: dict, width: int = 960):
    """Yield (activity name, act_id, actor, frame, foot_xy normalised to [0,1])."""
    kpf = Path(ROOT / clip["kpf"])
    meva = load_kpf(kpf, clip["clip"])
    import cv2
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    tracks = meva.person_tracks()
    for a in meva.activities:
        if a.name not in (ENTER, EXIT):
            continue
        frame = a.frame_start if a.name == ENTER else a.frame_end
        for actor in a.actors:
            tr = tracks.get(actor)
            if not tr:
                continue
            f = min(tr, key=lambda k: abs(k - frame))
            x1, y1, x2, y2 = tr[f]
            # scale to analysis size (width 960) then normalise: equals normalising by source size
            sx = width / src_w
            ax, ay = (x1 + x2) / 2 * sx, y2 * sx
            yield a.name, a.act_id, actor, f, (ax / width, ay / (src_h * sx))


def coverage(clip: dict):
    zones = [z for z in load_zones(ROOT / "labels" / "zones" / f"{clip['clip']}.json") if z.kind == "entrance"]
    rows = []
    for name, act_id, actor, f, pt in foot_points(clip):
        hit = [z.zone_id for z in zones if point_in_polygon(pt, z.polygon)]
        rows.append((name, act_id, actor, f, pt, hit))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--misses", action="store_true", help="list uncovered activities")
    args = ap.parse_args()
    split = json.loads((ROOT / "labels" / "split.json").read_text())
    groups = ["dev", "test"] if args.group == "all" else [args.group]
    for g in groups:
        tot_hit = tot = 0
        for clip in split[g]:
            rows = coverage(clip)
            hit = sum(1 for r in rows if r[5])
            tot_hit, tot = tot_hit + hit, tot + len(rows)
            pct = 100 * hit / len(rows) if rows else float("nan")
            print(f"{g} {clip['clip']}: {hit}/{len(rows)} = {pct:.1f}%")
            if args.misses:
                for name, act_id, actor, f, pt, h in rows:
                    if not h:
                        print(f"   miss {name.split('_')[1]} act={act_id} actor={actor} frame={f} "
                              f"foot=({pt[0]:.3f},{pt[1]:.3f})")
        if tot:
            print(f"{g} TOTAL: {tot_hit}/{tot} = {100 * tot_hit / tot:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
