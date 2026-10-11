"""Journey: per-person zone visits, stop flags, zone sequences, edit distance."""
from __future__ import annotations

from pulse.config import Preset
from pulse.datatypes import Sample, Timeline, Zone, ZoneVisit
from pulse.zones import presence_intervals, zone_of


def _stopped(samples: list[Sample], t_in: float, t_out: float, preset: Preset) -> bool:
    first = last = None
    for s in samples:
        if s.t < t_in - 1e-9 or s.t > t_out + 1e-9:
            continue
        slow = s.speed is not None and s.speed < preset.stop_speed
        if slow and first is not None and s.t - last > preset.zone_gap + 1e-6:
            first = None  # gap breaks the run
        if not slow:
            first = None
            continue
        if first is None:
            first = s.t
        last = s.t
        if last - first >= preset.stop_time - 1e-6:
            return True
    return False


def build(timeline: Timeline, zones: list[Zone], preset: Preset) -> dict[int, list[ZoneVisit]]:
    out: dict[int, list[ZoneVisit]] = {}
    for pid, samples in timeline.items():
        samples = sorted(samples, key=lambda s: s.t)
        ts = [s.t for s in samples]
        labels = [zone_of(s.foot, zones) for s in samples]
        visits: list[ZoneVisit] = []
        for z in zones:
            inside = [lab == z.zone_id for lab in labels]
            for t_in, t_out in presence_intervals(ts, inside, preset.zone_min, preset.zone_gap):
                stop = z.kind == "car" and _stopped(samples, t_in, t_out, preset)
                visits.append(ZoneVisit(pid, z.zone_id, t_in, t_out, stop))
        visits.sort(key=lambda v: v.t_in)
        out[pid] = visits
    return out


def sequence(visits: list[ZoneVisit], zones_by_id: dict[str, Zone] | None = None) -> list[str]:
    seq: list[str] = []
    for v in sorted(visits, key=lambda v: v.t_in):
        if not seq or seq[-1] != v.zone_id:
            seq.append(v.zone_id)
    return seq


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def journey_accuracy(pred: list[str], gt: list[str]) -> float:
    n = max(len(pred), len(gt))
    if n == 0:
        return 1.0
    return 1 - edit_distance(pred, gt) / n
