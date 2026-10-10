"""Identity merge (shared with Showroom Coach): within-clip track fragments -> persons, plus live
re-attach. Never re-identifies across visits; embeddings are only compared inside one clip."""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from pulse.config import Config
from pulse.datatypes import Person, TrackObs
from pulse.geometry import foot_point

OVERLAP_TOL = 0.2


@dataclass
class TrackSummary:
    track_id: int
    t_start: float
    t_end: float
    start_foot: tuple[float, float]   # normalised by frame width/height
    end_foot: tuple[float, float]
    emb: np.ndarray | None
    n_obs: int


def summarize(obs: list[TrackObs], embeddings: dict[int, np.ndarray],
              frame_size: tuple[int, int]) -> dict[int, TrackSummary]:
    w, h = frame_size
    by_track: dict[int, list[tuple[float, tuple[float, float]]]] = {}
    for o in obs:
        u, v = foot_point(o.bbox, o.keypoints)
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        by_track.setdefault(o.track_id, []).append((o.t, (u / w, v / h)))
    out: dict[int, TrackSummary] = {}
    for tid, pts in by_track.items():
        pts.sort(key=lambda p: p[0])
        out[tid] = TrackSummary(tid, pts[0][0], pts[-1][0], pts[0][1], pts[-1][1],
                                embeddings.get(tid), len(pts))
    return out


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _plausible(end_foot, start_foot, gap: float, cfg: Config) -> bool:
    d = math.hypot(start_foot[0] - end_foot[0], start_foot[1] - end_foot[1])
    return d <= cfg.merge_max_speed * max(gap, OVERLAP_TOL) + 0.05


def _overlaps(a: TrackSummary, b: TrackSummary) -> bool:
    return min(a.t_end, b.t_end) - max(a.t_start, b.t_start) > OVERLAP_TOL


def merge(tracks: dict[int, TrackSummary], cfg: Config, max_gap_s: float | None = None) -> list[Person]:
    max_gap = cfg.preset.leave_after if max_gap_s is None else max_gap_s
    cands: list[tuple[float, int, int]] = []
    for a in tracks.values():
        for b in tracks.values():
            if a.track_id == b.track_id or not a.t_start < b.t_start:
                continue
            gap = b.t_start - a.t_end
            if not (-OVERLAP_TOL < gap <= max_gap):
                continue
            sim = _cos(a.emb, b.emb)
            if sim is None or sim < cfg.merge_sim_thr:
                continue
            if not _plausible(a.end_foot, b.start_foot, gap, cfg):
                continue
            cands.append((sim, a.track_id, b.track_id))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))

    cluster_of = {tid: [tid] for tid in tracks}          # tid -> shared member list
    for _, aid, bid in cands:
        A, B = cluster_of[aid], cluster_of[bid]
        if A is B:
            continue
        # chain order: a is A's latest-ending track, b is B's earliest-starting track
        if aid != max(A, key=lambda i: (tracks[i].t_end, i)):
            continue
        if bid != min(B, key=lambda i: (tracks[i].t_start, i)):
            continue
        if any(_overlaps(tracks[x], tracks[y]) for x in A for y in B):
            continue
        A.extend(B)
        for tid in B:
            cluster_of[tid] = A

    clusters, seen = [], set()
    for members in cluster_of.values():
        if id(members) not in seen:
            seen.add(id(members))
            clusters.append(members)
    clusters.sort(key=lambda m: (min(tracks[i].t_start for i in m), min(m)))
    return [Person(n, sorted(m), min(tracks[i].t_start for i in m), max(tracks[i].t_end for i in m))
            for n, m in enumerate(clusters, 1)]


def reattach(lost: list[TrackSummary], new: TrackSummary, now: float, cfg: Config) -> int | None:
    best, best_sim = None, -1.0
    for l in lost:
        gap = now - l.t_end
        if gap > cfg.preset.leave_after:
            continue
        sim = _cos(l.emb, new.emb)
        if sim is None or sim < cfg.merge_sim_thr:
            continue
        if not _plausible(l.end_foot, new.start_foot, gap, cfg):
            continue
        if sim > best_sim:
            best, best_sim = l.track_id, sim
    return best
