"""Room, cars, desk, entrance and zone polygons in floor metres (pure)."""
import math

import numpy as np


def rect(center, length, width, yaw_deg):
    """Corners (CCW) of a rectangle; `length` runs along the yaw direction."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    hl, hw = length / 2.0, width / 2.0
    out = []
    for dx, dy in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
        out.append([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy])
    return out


def point_in_polygon(pt, poly):
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xin > x:
                inside = not inside
    return inside


def _seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_to_polygon(pt, poly):
    """0 inside, else distance to the boundary."""
    if point_in_polygon(pt, poly):
        return 0.0
    return min(_seg_dist(pt, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def dist_to_polygons(pt, polys):
    return min(dist_to_polygon(pt, p) for p in polys)


def build_layout(cfg):
    room = cfg["room"]
    apron = cfg["zone_apron_m"]
    cars = []
    zones = {}
    for c in cfg["cars"]:
        fp = rect(c["center_m"], c["length_m"], c["width_m"], c["yaw_deg"])
        zone = rect(c["center_m"], c["length_m"] + 2 * apron, c["width_m"] + 2 * apron, c["yaw_deg"])
        cars.append(dict(c, footprint_m=fp, zone_m=zone))
        zones[f"car_{c['id']}"] = zone
    d = cfg["desk"]
    desk_fp = rect(d["center_m"], d["length_m"], d["width_m"], d["yaw_deg"])
    zones["desk"] = rect(d["center_m"], d["length_m"] + 2 * apron, d["width_m"] + 2 * apron, d["yaw_deg"])
    e = cfg["entrance"]
    if e["wall"] != "south":
        raise NotImplementedError("quick test supports a south-wall entrance only")
    ex, ew = e["center_x_m"], e["width_m"]
    zones["entrance"] = [[ex - ew / 2, 0.0], [ex + ew / 2, 0.0], [ex + ew / 2, 1.5], [ex - ew / 2, 1.5]]
    x_ = cfg.get("exit")
    if x_:
        if x_["wall"] != "east":
            raise NotImplementedError("test-drive exit supported on the east wall only")
        W = room["width_m"]
        cy, w = x_["center_y_m"], x_["width_m"]
        zones["exit"] = [[W - 1.5, cy - w / 2], [W, cy - w / 2], [W, cy + w / 2], [W - 1.5, cy + w / 2]]
    return {
        "room": room,
        "cars": cars,
        "desk": dict(d, footprint_m=desk_fp),
        "entrance": e,
        "exit": x_,
        "zones": zones,
        "obstacles_m": [c["footprint_m"] for c in cars] + [desk_fp],
    }


def zone_of(pt, zones):
    return [name for name, poly in zones.items() if point_in_polygon(pt, poly)]
