"""Stop precision / recall against hand labels (spec section 6).

A predicted stop is a ZoneVisit with `stopped` set. It matches a ground-truth stop when both are at the
same zone and their time intervals overlap once the GT interval is widened by `tol_s` on each side.
Matching is many-to-many: precision = predicted stops that match any GT stop / predicted stops; recall =
GT stops matched by any predicted stop / GT stops. A ratio with a zero denominator is None.
"""
from __future__ import annotations

import math

from pulse.datatypes import ZoneVisit


def _get(g, key):
    return g[key] if isinstance(g, dict) else getattr(g, key)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval (default 95%) for k successes in n trials; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def usable_gt(gt_stops: list, intervals: list[tuple[float, float]], min_overlap: float) -> list:
    """GT stops whose overlap with the usable (stable, same-view) time is at least `min_overlap` seconds."""
    out = []
    for g in gt_stops:
        a, b = float(_get(g, "t_start")), float(_get(g, "t_end"))
        if sum(max(0.0, min(b, t1) - max(a, t0)) for t0, t1 in intervals) >= min_overlap - 1e-9:
            out.append(g)
    return out


def precision_recall(pred_visits: list[ZoneVisit], gt_stops: list, tol_s: float = 2.0,
                     one_to_one: bool = False) -> dict:
    """one_to_one=False: many-to-many (above). True: each prediction and each GT stop is used at most once,
    pairs chosen greedily by largest overlap (ties: earlier GT start, earlier prediction)."""
    pred = [v for v in pred_visits if v.stopped]
    gts = [(_get(g, "zone_id"), float(_get(g, "t_start")), float(_get(g, "t_end"))) for g in gt_stops]

    def hit(v: ZoneVisit, g) -> bool:
        return v.zone_id == g[0] and v.t_in <= g[2] + tol_s + 1e-9 and v.t_out >= g[1] - tol_s - 1e-9

    if one_to_one:
        pairs = sorted(((-(min(v.t_out, g[2] + tol_s) - max(v.t_in, g[1] - tol_s)), g[1], v.t_in, i, j)
                        for i, v in enumerate(pred) for j, g in enumerate(gts) if hit(v, g)))
        up, ug = set(), set()
        for _, _, _, i, j in pairs:
            if i not in up and j not in ug:
                up.add(i)
                ug.add(j)
        pm, gm = len(up), len(ug)
    else:
        pm = sum(1 for v in pred if any(hit(v, g) for g in gts))
        gm = sum(1 for g in gts if any(hit(v, g) for v in pred))
    return {"n_pred": len(pred), "n_gt": len(gts), "pred_matched": pm, "gt_matched": gm,
            "precision": pm / len(pred) if pred else None,
            "recall": gm / len(gts) if gts else None, "tol_s": tol_s}
