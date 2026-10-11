"""Stop precision / recall against hand labels (spec section 6).

A predicted stop is a ZoneVisit with `stopped` set. It matches a ground-truth stop when both are at the
same zone and their time intervals overlap once the GT interval is widened by `tol_s` on each side.
Matching is many-to-many: precision = predicted stops that match any GT stop / predicted stops; recall =
GT stops matched by any predicted stop / GT stops. A ratio with a zero denominator is None.
"""
from __future__ import annotations

from pulse.datatypes import ZoneVisit


def _get(g, key):
    return g[key] if isinstance(g, dict) else getattr(g, key)


def precision_recall(pred_visits: list[ZoneVisit], gt_stops: list, tol_s: float = 2.0) -> dict:
    pred = [v for v in pred_visits if v.stopped]
    gts = [(_get(g, "zone_id"), float(_get(g, "t_start")), float(_get(g, "t_end"))) for g in gt_stops]

    def hit(v: ZoneVisit, g) -> bool:
        return v.zone_id == g[0] and v.t_in <= g[2] + tol_s + 1e-9 and v.t_out >= g[1] - tol_s - 1e-9

    pm = sum(1 for v in pred if any(hit(v, g) for g in gts))
    gm = sum(1 for g in gts if any(hit(v, g) for v in pred))
    return {"n_pred": len(pred), "n_gt": len(gts), "pred_matched": pm, "gt_matched": gm,
            "precision": pm / len(pred) if pred else None,
            "recall": gm / len(gts) if gts else None, "tol_s": tol_s}
