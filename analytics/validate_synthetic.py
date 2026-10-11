"""Check box-based floor positions and car dwell against exact synthetic ground truth (pos_m).

Usage from repo root: .venv/bin/python -m analytics.validate_synthetic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analytics.dwell import dwell
from analytics.floor import Camera, person_floor_xy
from analytics.synthetic import car_footprints

CLIPS = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "clips"


def main() -> None:
    errs, vis = [], []
    dw = []  # (gt total s, box total s) per (clip, track, car)
    for gt_path in sorted(CLIPS.glob("*/gt.json")):
        gt = json.loads(gt_path.read_text())
        cam = Camera.from_gt(gt["camera"])
        cars = car_footprints(gt)
        s_gt, s_box = [], []
        for fr in gt["frames"]:
            for p in fr["people"]:
                if not p.get("box_xyxy") or not p.get("present", True):
                    continue
                xy, _ = person_floor_xy(cam, p["box_xyxy"])
                errs.append(np.linalg.norm(xy - p["pos_m"]))
                vis.append(p["visible_frac"])
                s_gt.append((fr["t"], p["id"], np.asarray(p["pos_m"], float)))
                s_box.append((fr["t"], p["id"], xy))
        dt = 1.0 / gt["fps"]
        a = {(d.track_id, d.car): d.total_s for d in dwell(s_gt, cars, dt)}
        b = {(d.track_id, d.car): d.total_s for d in dwell(s_box, cars, dt)}
        dw += [(a.get(k, 0.0), b.get(k, 0.0)) for k in a.keys() | b.keys()]
    errs, vis = np.array(errs), np.array(vis)
    print(f"floor position error, {len(errs)} person-frames over {len(list(CLIPS.glob('*/gt.json')))} clips")
    for lo, hi in [(0.9, 1.01), (0.6, 0.9), (0.0, 0.6)]:
        e = errs[(vis >= lo) & (vis < hi)]
        if len(e):
            print(f"  visible {lo:.1f}-{min(hi, 1):.1f}: n={len(e):6d}  median {np.median(e):.2f} m  p90 {np.percentile(e, 90):.2f} m")
    g, b = np.array(dw).T
    print(f"car dwell, {len(g)} (track, car) pairs: GT total {g.sum():.0f} s, box-based total {b.sum():.0f} s")
    print(f"  per-pair abs error: median {np.median(np.abs(g - b)):.1f} s, p90 {np.percentile(np.abs(g - b), 90):.1f} s;"
          f" pairs only in GT {int(((g > 0) & (b == 0)).sum())}, only in box-based {int(((b > 0) & (g == 0)).sum())}")


if __name__ == "__main__":
    main()
