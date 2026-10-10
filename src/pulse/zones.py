"""Zones: automatic car zones, point-in-polygon, presence hysteresis, zone files."""
from __future__ import annotations

import json
from pathlib import Path

from pulse.datatypes import Det, Zone

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
               foot_frac: float = 0.4, min_area_frac: float = 0.002,
               merge_iou: float = 0.5) -> list[Zone]:
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
        by = min(float(h), y2 + apron * bh)
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


def save_zones(path, zones: list[Zone], frame_size: tuple[int, int] | None = None,
               labeller: str | None = None) -> None:
    doc: dict = {"version": 1, "frame_size": list(frame_size) if frame_size else None}
    if labeller is not None:
        doc["labeller"] = labeller
    doc["zones"] = [z.to_dict() for z in zones]
    Path(path).write_text(json.dumps(doc, indent=2))


def load_zones(path) -> list[Zone]:
    doc = json.loads(Path(path).read_text())
    return [Zone.from_dict(d) for d in doc["zones"]]
