"""Render car-footprint / distance / dwell overlay videos for synthetic clips (review and demo).

Person positions come from the boxes (analytics.floor.person_floor_xy), exactly as the live system
would compute them, not from the simulator's true positions.

Usage from repo root:
    .venv/bin/python -m analytics.overlay [--out data/synthetic/review_videos_zones] [--clips s000_greet_fast_cam1_day ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from analytics.dwell import NEAR_M, car_distances
from analytics.floor import Camera, person_floor_xy
from analytics.synthetic import car_footprints

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "synthetic" / "clips"
ROLE_COL = {"staff": (255, 140, 0), "customer": (0, 165, 255)}
CAR_COL, NEAR_COL = (180, 180, 180), (60, 220, 60)


def _floor_poly_px(cam: Camera, poly_m) -> np.ndarray | None:
    px = cam.floor_polygon_to_pixel(poly_m)
    return None if px is None else np.clip(px, -10000, 10000).round().astype(np.int32)


def render(args: tuple[Path, Path]) -> str:
    clip_dir, out_dir = args
    gt = json.loads((clip_dir / "gt.json").read_text())
    cam = Camera.from_gt(gt["camera"])
    cars = car_footprints(gt)
    car_px = {k: px for k, v in cars.items() if (px := _floor_poly_px(cam, v)) is not None}
    car_lbl = {k: cam.floor_to_pixel([np.mean(v, axis=0)])[0].astype(int) for k, v in cars.items() if k in car_px}
    roles = {p["id"]: p["role"] for p in gt["persons"]}
    W, H, fps = gt["width"], gt["height"], gt["fps"]
    dt = 1.0 / fps
    dwell_s: dict[tuple[int, str], float] = {}
    cap = cv2.VideoCapture(str(clip_dir / f"{clip_dir.name}.mp4"))
    out = out_dir / f"{clip_dir.name}_zones.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", str(out)], stdin=subprocess.PIPE)
    for fr in gt["frames"]:
        ok, im = cap.read()
        if not ok:
            break
        layer = im.copy()
        near_cars = set()
        people = []
        for p in fr["people"]:
            if not p.get("box_xyxy") or not p.get("present", True):
                continue
            xy, how = person_floor_xy(cam, p["box_xyxy"])
            d = car_distances(xy, cars)
            near = min(d, key=d.get) if d and not np.isnan(min(d.values())) else None
            for c, dist in d.items():
                if dist <= NEAR_M:
                    dwell_s[(p["id"], c)] = dwell_s.get((p["id"], c), 0.0) + dt
                    near_cars.add(c)
            people.append((p, xy, how, near, d.get(near, float("nan")) if near else float("nan")))
        for c, pts in car_px.items():
            cv2.fillPoly(layer, [pts], NEAR_COL if c in near_cars else CAR_COL)
        im = cv2.addWeighted(layer, 0.25, im, 0.75, 0)
        for c, pts in car_px.items():
            cv2.polylines(im, [pts], True, NEAR_COL if c in near_cars else CAR_COL, 2)
            if 0 <= car_lbl[c][0] < W and 0 <= car_lbl[c][1] < H:
                cv2.putText(im, c, tuple(car_lbl[c]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for p, xy, how, near, dist in people:
            col = ROLE_COL.get(roles.get(p["id"]), (200, 200, 200))
            x1, y1, x2, y2 = map(int, p["box_xyxy"])
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            if near is None:
                continue
            foot = cam.floor_to_pixel([xy])[0].astype(int)
            # nearest point on the car footprint, for the distance line
            v = np.asarray(cars[near], float)
            a, b = v, np.roll(v, -1, axis=0)
            ab = b - a
            s = np.clip(np.einsum("ij,ij->i", xy - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
            cand = a + s[:, None] * ab
            q = cand[np.argmin(np.linalg.norm(cand - xy, axis=1))] if dist > 0 else xy
            qp = cam.floor_to_pixel([q])[0].astype(int)
            lc = NEAR_COL if dist <= NEAR_M else (255, 255, 255)
            cv2.circle(im, tuple(foot), 5, col, -1)
            cv2.line(im, tuple(foot), tuple(qp), lc, 2)
            tag = f"#{p['id']} {dist:.1f}m {near}" + (" (head)" if how == "head" else "")
            cv2.putText(im, tag, (x1, max(y1 - 6, 44)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 2)
        cv2.rectangle(im, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(im, f"{clip_dir.name}  t={fr['t']:5.1f}s  near = within {NEAR_M} m of car body",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows = sorted(dwell_s.items(), key=lambda kv: -kv[1])[:6]
        if rows:
            cv2.rectangle(im, (W - 250, 34), (W, 40 + 22 * (len(rows) + 1)), (0, 0, 0), -1)
            cv2.putText(im, "time at car", (W - 240, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for i, ((tid, c), sec) in enumerate(rows):
                cv2.putText(im, f"#{tid} {roles.get(tid, '?')[:4]} {c}: {sec:5.1f}s", (W - 240, 76 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROLE_COL.get(roles.get(tid), (200, 200, 200)), 1)
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    ff.wait()
    return out.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic" / "review_videos_zones")
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clips = sorted(d for d in CLIPS.iterdir() if (d / "gt.json").exists() and (not a.clips or d.name in a.clips))
    with Pool(8) as pool:
        for name in pool.imap_unordered(render, [(c, a.out) for c in clips]):
            print(name, flush=True)


if __name__ == "__main__":
    main()
