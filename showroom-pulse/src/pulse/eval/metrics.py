"""Evaluation metrics: tracking (motmetrics), counts, occupancy, person matching, dwell, journeys.

Pure functions over plain data (boxes, Timelines, ZoneVisits); no I/O, no models. Used by run_eval.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

from pulse.datatypes import BBox, Timeline, ZoneVisit
from pulse.journey import journey_accuracy, sequence

MOT_IOU = 0.5          # a prediction matches a GT box when IoU >= 0.5
MATCH_MIN_OVERLAP = 0.3
_MOT_COUNTS = ("num_objects", "num_misses", "num_false_positives", "num_switches",
               "idtp", "idfn", "idfp", "num_matches")


def mot(gt: dict[int, dict[int, BBox]], pred: dict[int, dict[int, BBox]], frames: list[int]) -> dict:
    """CLEAR-MOT / IDF1 with IoU distance (max_iou 0.5). `gt`/`pred` map frame -> {track id: xyxy box}.

    Returns mota, idf1, precision, recall, num_switches plus the raw counts (for pooling with
    `pool_mot`). mota/idf1/precision/recall are None when undefined (no GT / no predictions)."""
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g, p = gt.get(f, {}), pred.get(f, {})
        gids, pids = sorted(g), sorted(p)
        if gids and pids:
            d = _iou_distance([g[i] for i in gids], [p[i] for i in pids])
        else:
            d = np.empty((len(gids), len(pids)))
        acc.update(gids, pids, d, frameid=f)
    if not acc.events.shape[0]:
        return _finish({k: 0 for k in _MOT_COUNTS})
    summ = mm.metrics.create().compute(acc, metrics=list(_MOT_COUNTS), name="acc")
    return _finish({k: int(summ[k].iloc[0]) for k in _MOT_COUNTS})


def _iou_distance(gb: list[BBox], pb: list[BBox]) -> np.ndarray:
    """1 - IoU, NaN (= not matchable) where IoU < MOT_IOU. (motmetrics' own iou_matrix uses
    np.asfarray, which numpy 2 removed.)"""
    d = np.full((len(gb), len(pb)), np.nan)
    for i, a in enumerate(gb):
        for j, b in enumerate(pb):
            v = _iou(a, b)
            if v >= MOT_IOU:
                d[i, j] = 1.0 - v
    return d


def _finish(c: dict) -> dict:
    """Derived rates from the raw counts (the same formulas motmetrics uses)."""
    n = c["num_objects"]
    det = c["num_matches"] + c["num_false_positives"]
    idden = 2 * c["idtp"] + c["idfn"] + c["idfp"]
    out = dict(c)
    out["mota"] = 1.0 - (c["num_misses"] + c["num_false_positives"] + c["num_switches"]) / n if n else None
    out["idf1"] = 2 * c["idtp"] / idden if idden else None
    out["precision"] = c["num_matches"] / det if det else None
    out["recall"] = c["num_matches"] / n if n else None
    return out


def pool_mot(results: list[dict]) -> dict:
    """Pool several `mot` results by summing the raw counts (clip-weighted by GT size)."""
    return _finish({k: sum(r[k] for r in results) for k in _MOT_COUNTS})


def count_error(pred: int, gt: int) -> float | None:
    """|pred - gt| / gt as a percentage; None when gt is 0."""
    return abs(pred - gt) / gt * 100.0 if gt else None


def pooled_count_error(pred: list[int], gt: list[int]) -> float | None:
    """Sum |pred_i - gt_i| / sum gt_i as a percentage over clips (under- and over-counts in different
    clips do not cancel); None when the GT total is 0."""
    total = sum(gt)
    return sum(abs(p - g) for p, g in zip(pred, gt)) / total * 100.0 if total else None


def occupancy_mae(pred: list[int], gt: list[int]) -> float | None:
    """Mean absolute error in persons per second; a shorter series is padded with zeros."""
    n = max(len(pred), len(gt))
    if n == 0:
        return None
    a = list(pred) + [0] * (n - len(pred))
    b = list(gt) + [0] * (n - len(gt))
    return sum(abs(x - y) for x, y in zip(a, b)) / n


def _iou(a: BBox, b: BBox) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_persons(gt_tl: Timeline, pred_tl: Timeline, min_overlap: float = MATCH_MIN_OVERLAP,
                  one_to_one: bool = True) -> dict[int, int]:
    """GT person -> predicted person.

    overlap(g, p) = sum over shared sample times of IoU(box_g, box_p) / n_g: the mean IoU over the GT
    person's own lifetime (a time where we have no box scores 0), so a long predicted person that covers
    a short GT track scores as well as an exact fit. Pairs below `min_overlap` are never matched.

    one_to_one=True: Hungarian assignment on cost 1 - overlap (each predicted person used once).
    one_to_one=False: every GT person takes its best predicted person independently; several GT tracks
    may share one predicted person. MEVA tracks are annotated per activity, so a GT "person" is often a
    short snippet of someone we track for much longer; one-to-one would leave those unmatched."""
    at: dict[float, tuple[list, list]] = defaultdict(lambda: ([], []))
    for pid, ss in gt_tl.items():
        for s in ss:
            at[s.t][0].append((pid, s.bbox))
    for pid, ss in pred_tl.items():
        for s in ss:
            at[s.t][1].append((pid, s.bbox))
    acc: dict[tuple[int, int], float] = defaultdict(float)
    for gs, ps in at.values():
        for gp, gb in gs:
            for pp, pb in ps:
                v = _iou(gb, pb)
                if v > 0:
                    acc[(gp, pp)] += v
    if not acc:
        return {}
    gids = sorted({k[0] for k in acc})
    pids = sorted({k[1] for k in acc})
    gi, pi = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
    score = np.zeros((len(gids), len(pids)))
    for (g, p), v in acc.items():
        score[gi[g], pi[p]] = v / len(gt_tl[g])
    score[score < min_overlap] = 0.0
    if not one_to_one:
        return {g: pids[int(np.argmax(score[i]))] for i, g in enumerate(gids) if score[i].max() >= min_overlap}
    rows, cols = linear_sum_assignment(1.0 - score)
    return {gids[r]: pids[c] for r, c in zip(rows, cols) if score[r, c] >= min_overlap}


def _dwell_by_zone(visits: list[ZoneVisit]) -> dict[str, float]:
    d: dict[str, float] = defaultdict(float)
    for v in visits:
        d[v.zone_id] += v.dwell
    return d


def dwell_errors(gt_j: dict[int, list[ZoneVisit]], pred_j: dict[int, list[ZoneVisit]],
                 match: dict[int, int]) -> list[tuple[float, float]]:
    """(gt_dwell, pred_dwell) seconds per matched person x zone where either is > 0, in
    (GT person id, zone id) order. Dwell is the summed length of the person's visits to the zone."""
    out = []
    for g in sorted(match):
        gd = _dwell_by_zone(gt_j.get(g, []))
        pd = _dwell_by_zone(pred_j.get(match[g], []))
        for z in sorted(set(gd) | set(pd)):
            a, b = gd.get(z, 0.0), pd.get(z, 0.0)
            if a > 0 or b > 0:
                out.append((a, b))
    return out


def dwell_summary(pairs: list[tuple[float, float]]) -> dict:
    """Median absolute dwell error in seconds and in % of the GT dwell, over pairs with GT dwell > 0.
    Pairs with GT dwell 0 and predicted dwell > 0 (a zone the GT person never visited) are counted
    separately as n_pred_only."""
    pos = [(a, b) for a, b in pairs if a > 0]
    return {
        "n_pairs": len(pairs), "n_gt_positive": len(pos), "n_pred_only": len(pairs) - len(pos),
        "median_abs_s": float(median(abs(a - b) for a, b in pos)) if pos else None,
        "median_abs_pct": float(median(abs(a - b) / a * 100.0 for a, b in pos)) if pos else None,
    }


def journey_accuracies(gt_j: dict[int, list[ZoneVisit]], pred_j: dict[int, list[ZoneVisit]],
                       match: dict[int, int]) -> list[float]:
    """1 - normalised edit distance of the zone sequences, per matched person. Pairs where both
    sequences are empty (the person never visited a zone) carry no information and are skipped."""
    out = []
    for g in sorted(match):
        a, b = sequence(pred_j.get(match[g], [])), sequence(gt_j.get(g, []))
        if a or b:
            out.append(journey_accuracy(a, b))
    return out


def match_events(gt_ts: list[float], pred_ts: list[float], window_s: float = 5.0
                 ) -> list[tuple[int, int, float]]:
    """One-to-one matching of GT event times to predicted event times within `window_s`, closest
    pairs first. Returns (gt index, pred index, |dt|) sorted by GT index."""
    cands = sorted((abs(g - p), gi, pj) for gi, g in enumerate(gt_ts) for pj, p in enumerate(pred_ts)
                   if abs(g - p) <= window_s)
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for dt, gi, pj in cands:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        out.append((gi, pj, dt))
    return sorted(out)


def median_timing_error(matches: list[tuple[int, int, float]]) -> float | None:
    return float(median(m[2] for m in matches)) if matches else None
