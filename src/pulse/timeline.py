"""TrackObs + person assignment -> per-person Samples (spec 4.4): positions, smoothed speed, filters."""
from __future__ import annotations
import math
from collections import defaultdict
from statistics import median

import numpy as np

from pulse.config import Config
from pulse.datatypes import Person, Sample, StableSegment, Timeline, TrackObs
from pulse.geometry import focal_px, foot_point, position_m


def smooth_speed(ts, feet_px, depths_z, f_px: float, window_s: float) -> list[float | None]:
    """Centred moving average of foot pixels over window_s, then |d foot_px|/dt * (Z/f)
    (central difference; one-sided at the ends). None for fewer than 2 samples."""
    n = len(ts)
    if n < 2:
        return [None] * n
    t = np.asarray(ts, dtype=float)
    p = np.asarray(feet_px, dtype=float).reshape(n, 2)
    half = window_s / 2.0
    sm = np.empty_like(p)
    for i in range(n):
        m = np.abs(t - t[i]) <= half + 1e-9
        sm[i] = p[m].mean(axis=0)
    out: list[float | None] = []
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        dt = t[b] - t[a]
        z = depths_z[i]
        if dt <= 0 or z is None or not math.isfinite(z):
            out.append(None)
            continue
        d = float(np.hypot(*(sm[b] - sm[a])))
        out.append(float(d / dt * (z / f_px)))
    return out


def build_timeline(obs: list[TrackObs], track_to_person: dict[int, int], frame_size: tuple[int, int],
                   cfg: Config, floor_h=None) -> Timeline:
    best: dict[tuple[int, int], TrackObs] = {}
    for o in obs:
        pid = track_to_person.get(o.track_id)
        if pid is None:
            continue
        k = (pid, o.t)  # one frame = one timestamp (frame_idx may repeat across test clips)
        if k not in best or o.score > best[k].score:
            best[k] = o
    per: dict[int, list[tuple[TrackObs, tuple[float, float], object]]] = defaultdict(list)
    for (pid, _), o in best.items():
        fp = foot_point(o.bbox, o.keypoints, cfg.kp_thr)
        if not (math.isfinite(fp[0]) and math.isfinite(fp[1])):
            continue
        pe = position_m(o, frame_size, cfg.hfov_deg, cfg.person_height_m, cfg.kp_thr, floor_h)
        per[pid].append((o, fp, pe))
    f = focal_px(frame_size[0], cfg.hfov_deg)
    w, h = frame_size
    tl: Timeline = {}
    for pid, rows in per.items():
        rows.sort(key=lambda r: (r[0].t, r[0].frame_idx))
        # depth for the m-per-pixel scale: own Z, else the person's median valid Z, else None
        zs = [r[2].z if r[2].method != "invalid" and math.isfinite(r[2].z) else None for r in rows]
        valid = [z for z in zs if z is not None]
        fill = median(valid) if valid else None
        speeds = smooth_speed([r[0].t for r in rows], [r[1] for r in rows],
                              [z if z is not None else fill for z in zs], f, cfg.speed_window_s)
        samples = []
        for (o, fp, pe), sp in zip(rows, speeds):
            ok = pe.method != "invalid"
            samples.append(Sample(o.t, pid, (fp[0] / w, fp[1] / h), o.bbox,
                                  (pe.x, pe.z) if ok else None, sp, pe.method))
        tl[pid] = samples
    return dict(sorted(tl.items()))


def filter_stable(timeline: Timeline, segments: list[StableSegment]) -> Timeline:
    segs = sorted(segments, key=lambda s: s.t0)

    def keep(t: float) -> bool:
        for i, s in enumerate(segs):
            last = i == len(segs) - 1
            if s.t0 <= t < s.t1 or (last and t == s.t1):
                return s.stable and s.same_view
        return True

    return {pid: [s for s in ss if keep(s.t)] for pid, ss in timeline.items()}


def already_present(person: Person, leave_after: float) -> bool:
    return person.first_t < leave_after


def at_edge(sample: Sample, frame_size: tuple[int, int], margin: float) -> bool:
    x1, y1, x2, y2 = sample.bbox
    w, h = frame_size
    return (x1 / w < margin or y1 / h < margin
            or 1.0 - x2 / w < margin or 1.0 - y2 / h < margin)
