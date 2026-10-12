"""Zones: automatic car zones, point-in-polygon, presence hysteresis, zone files."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from pulse.datatypes import Det, Zone
from pulse.geometry import fit_homography

VEHICLES = ("car", "truck", "motorcycle", "bus")


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def auto_zones(car_dets: list[Det], frame_shape: tuple[int, int], apron: float = 0.25,
               foot_frac: float = 0.7, apron_down: float = 0.8, min_area_frac: float = 0.002,
               merge_iou: float = 0.5) -> list[Zone]:
    """One rectangle per car: `apron` x box width on each side, from `foot_frac` x box height above the box bottom
    to `apron_down` x box height below it. People stand beside and in front of a car, so their feet land well outside
    the box. Fitted on the 65 synthetic clips: feet in the right car zone 0.55 -> 0.79 (sides 0.4 reach 0.83 but the
    zones look oversized)."""
    h, w = frame_shape
    cands = [d for d in car_dets if d.cls in VEHICLES
             and (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= min_area_frac * w * h]
    kept: list[Det] = []
    for d in sorted(cands, key=lambda d: -d.score):
        if all(_iou(d.bbox, k.bbox) <= merge_iou for k in kept):
            kept.append(d)
    kept.sort(key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
    zones, moto, cars = [], 0, 0
    for i, d in enumerate(kept, 1):
        x1, y1, x2, y2 = d.bbox
        bw, bh = x2 - x1, y2 - y1
        lx = max(0.0, x1 - apron * bw)
        rx = min(float(w), x2 + apron * bw)
        ty = max(0.0, y2 - foot_frac * bh)
        by = min(float(h), y2 + apron_down * bh)
        if d.cls == "motorcycle":
            moto += 1
            name = f"Motorcycle {moto}"
        else:
            cars += 1
            name = f"Car {cars}"
        poly = [(lx / w, ty / h), (rx / w, ty / h), (rx / w, by / h), (lx / w, by / h)]
        zones.append(Zone(f"car-{i}", name, "car", poly, "auto"))
    return zones


def point_in_polygon(pt, polygon) -> bool:
    x, y = pt
    n = len(polygon)
    eps = 1e-12
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if (abs(cross) <= eps and min(x1, x2) - eps <= x <= max(x1, x2) + eps
                and min(y1, y2) - eps <= y <= max(y1, y2) + eps):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygon_area(polygon) -> float:
    s = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def zone_of(point, zones: list[Zone]) -> str | None:
    best, best_a = None, float("inf")
    for z in zones:
        if point_in_polygon(point, z.polygon):
            a = polygon_area(z.polygon)
            if a < best_a:
                best, best_a = z.zone_id, a
    return best


def presence_intervals(ts: list[float], inside: list[bool], zone_min: float,
                       zone_gap: float) -> list[tuple[float, float]]:
    times = [t for t, i in zip(ts, inside) if i]
    if not times:
        return []
    clusters = [[times[0], times[0]]]
    for t in times[1:]:
        if t - clusters[-1][1] > zone_gap + 1e-6:
            clusters.append([t, t])
        else:
            clusters[-1][1] = t
    return [(a, b) for a, b in clusters if b - a >= zone_min - 1e-6]


class FloorError(ValueError):
    pass


MAX_WORLD_M = 1000.0


def _pts4(v, name: str, lo: float | None, hi: float | None) -> list[tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise FloorError(f"floor.{name} needs exactly 4 points")
    out = []
    for p in v:
        ok = (isinstance(p, (list, tuple)) and len(p) == 2
              and all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)
                      for c in p))
        if not ok:
            raise FloorError(f"floor.{name}: every point must be [x, y] with finite numbers")
        if lo is not None and not all(lo <= c <= hi for c in p):
            raise FloorError(f"floor.{name}: values must be within [{lo:g}, {hi:g}]")
        out.append((float(p[0]), float(p[1])))
    return out


def _non_degenerate(pts, name: str) -> None:
    """No three of the four points may be (nearly) collinear."""
    span = max(max(p[0] for p in pts) - min(p[0] for p in pts),
               max(p[1] for p in pts) - min(p[1] for p in pts))
    for a, b, c in itertools.combinations(pts, 3):
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if span <= 0 or area2 <= 1e-4 * span * span:
            raise FloorError(f"floor.{name}: points are collinear or coincident")


def _convex(pts, name: str) -> None:
    """The four points, in the given order, must form a convex quadrilateral (rejects bow-ties)."""
    cr = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cr.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(c > 0 for c in cr) or all(c < 0 for c in cr)):
        raise FloorError(f"floor.{name}: the 4 points must be given in order around a convex quadrilateral")


def validate_floor(floor) -> dict:
    """Floor calibration dict -> normalised copy {'img': [[u,v]x4], 'world': [[x,y]x4]}; FloorError otherwise."""
    if not isinstance(floor, dict):
        raise FloorError("floor must be an object with 'img' and 'world'")
    img = _pts4(floor.get("img"), "img", 0.0, 1.0)
    world = _pts4(floor.get("world"), "world", -MAX_WORLD_M, MAX_WORLD_M)
    _non_degenerate(img, "img")
    _non_degenerate(world, "world")
    # Both quads convex and in matching point order: a bow-tie in either set gives a wild fit. The
    # windings may differ (image y points down, floor y usually does not); a homography allows either.
    _convex(img, "img")
    _convex(world, "world")
    try:
        h = fit_homography(img, world)
        if not np.all(np.isfinite(h)) or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("singular")
        ws = [(h @ np.array([u, v, 1.0]))[2] for u, v in img]
        if not (all(w > 0 for w in ws) or all(w < 0 for w in ws)):
            raise ValueError("points straddle the horizon")
    except ValueError as e:
        raise FloorError(f"floor calibration could not be fitted ({e})")
    return {"img": [list(p) for p in img], "world": [list(p) for p in world]}


def floor_homography(floor: dict, frame_size: tuple[int, int]) -> np.ndarray:
    """Homography from analysis-frame pixels to floor metres; frame_size is (w, h)."""
    f = validate_floor(floor)
    w, h = frame_size
    pts = [(u * w, v * h) for u, v in f["img"]]
    hm = fit_homography(pts, f["world"])
    cx, cy = np.mean(pts, axis=0)
    if (hm @ np.array([cx, cy, 1.0]))[2] < 0:     # keep w > 0 on the floor side so to_floor can reject w <= 0
        hm = -hm
    return hm


def save_zones(path, zones: list[Zone], frame_size: tuple[int, int] | None = None,
               labeller: str | None = None, floor: dict | None = None) -> None:
    doc: dict = {"version": 1, "frame_size": list(frame_size) if frame_size else None}
    if labeller is not None:
        doc["labeller"] = labeller
    if floor is not None:
        doc["floor"] = validate_floor(floor)
    doc["zones"] = [z.to_dict() for z in zones]
    Path(path).write_text(json.dumps(doc, indent=2))


def load_zones(path) -> list[Zone]:
    doc = json.loads(Path(path).read_text())
    return [Zone.from_dict(d) for d in doc["zones"]]


def load_floor(path) -> dict | None:
    """Optional floor calibration of a zone file (None for files without one)."""
    doc = json.loads(Path(path).read_text())
    return validate_floor(doc["floor"]) if doc.get("floor") is not None else None
