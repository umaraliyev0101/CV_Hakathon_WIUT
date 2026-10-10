"""Deep interest, staff assistance, unserved-interest episodes and live alerts (pure core)."""
from __future__ import annotations
from dataclasses import dataclass, field

from pulse.config import Preset
from pulse.datatypes import InterestEpisode, Sample, Timeline, Zone, ZoneVisit
from pulse.geometry import distance_m

BRIDGE_S = 1.0
EPS = 1e-6

StaffIndex = dict[float, list[tuple[int, tuple[float, float]]]]


@dataclass
class AssistInfo:
    assisted: bool
    t_first: float | None
    staff_ids: list[int] = field(default_factory=list)


def staff_index(staff_tracks: Timeline) -> StaffIndex:
    idx: StaffIndex = {}
    for sid, samples in staff_tracks.items():
        for s in samples:
            if s.pos_m is not None:
                idx.setdefault(round(s.t, 2), []).append((sid, s.pos_m))
    return idx


def near_staff(sample: Sample, index: StaffIndex, dist: float) -> list[int]:
    if sample.pos_m is None:
        return []
    out = []
    for sid, pos in index.get(round(sample.t, 2), []):
        if distance_m(sample.pos_m, pos) <= dist and sid not in out:
            out.append(sid)
    return out


def _qualifying_run_start(times: list[float], assist_time: float) -> float | None:
    """Start of the earliest bridged run of near-times spanning >= assist_time."""
    times = sorted(times)
    i = 0
    while i < len(times):
        j = i
        while j + 1 < len(times) and times[j + 1] - times[j] <= BRIDGE_S + EPS:
            j += 1
        if times[j] - times[i] >= assist_time - EPS:
            return times[i]
        i = j + 1
    return None


def _near_times(samples: list[Sample], index: StaffIndex, dist: float) -> dict[int, list[float]]:
    per: dict[int, list[float]] = {}
    for s in samples:
        for sid in near_staff(s, index, dist):
            per.setdefault(sid, []).append(s.t)
    return per


def _assist_from(samples: list[Sample], index: StaffIndex, preset: Preset) -> AssistInfo:
    per = _near_times(samples, index, preset.assist_dist)
    starts = {sid: _qualifying_run_start(ts, preset.assist_time) for sid, ts in per.items()}
    starts = {sid: t for sid, t in starts.items() if t is not None}
    return AssistInfo(bool(starts), min(starts.values()) if starts else None, sorted(starts))


def assistance(timeline: Timeline, staff_tracks: Timeline, preset: Preset) -> dict[int, AssistInfo]:
    index = staff_index(staff_tracks)
    return {pid: _assist_from(samples, index, preset)
            for pid, samples in timeline.items() if pid not in staff_tracks}


def _visit_samples(samples: list[Sample], visits: list[ZoneVisit]) -> list[Sample]:
    return [s for s in samples if any(v.t_in <= s.t <= v.t_out for v in visits)]


def episodes(journeys: dict[int, list[ZoneVisit]], staff_tracks: Timeline, preset: Preset,
             timeline: Timeline, zones: list[Zone]) -> list[InterestEpisode]:
    car_ids = {z.zone_id for z in zones if z.kind == "car"}
    index = staff_index(staff_tracks)
    out: list[InterestEpisode] = []
    for pid, visits in journeys.items():
        if pid in staff_tracks:
            continue
        by_zone: dict[str, list[ZoneVisit]] = {}
        for v in visits:
            if v.zone_id in car_ids:
                by_zone.setdefault(v.zone_id, []).append(v)
        for zid, vs in by_zone.items():
            dwell = sum(v.dwell for v in vs)
            if not (dwell >= preset.interest_time or len(vs) >= 2):
                continue
            samples = _visit_samples(timeline.get(pid, []), vs)
            near_ids = sorted({sid for s in samples for sid in near_staff(s, index, preset.assist_dist)})
            info = _assist_from(samples, index, preset)
            out.append(InterestEpisode(
                person_id=pid, zone_id=zid, t_start=min(v.t_in for v in vs),
                t_end=max(v.t_out for v in vs), dwell=dwell, visits=len(vs),
                assisted=info.assisted, unserved=not near_ids, staff_ids=near_ids))
    out.sort(key=lambda e: (e.t_start, e.person_id))
    return out


@dataclass
class LiveAlert:
    person_id: int
    zone_id: str
    dwell: float
    t_in: float
    message: str


def open_alerts(journeys: dict[int, list[ZoneVisit]], staff_tracks: Timeline, preset: Preset,
                timeline: Timeline, zones: list[Zone], now: float) -> list[LiveAlert]:
    cars = {z.zone_id: z for z in zones if z.kind == "car"}
    index = staff_index(staff_tracks)
    out: list[LiveAlert] = []
    for pid, visits in journeys.items():
        if pid in staff_tracks:
            continue
        last: dict[str, ZoneVisit] = {}
        for v in visits:
            if v.zone_id in cars:
                last[v.zone_id] = v  # visits sorted by t_in -> last wins
        for zid, v in last.items():
            if now - v.t_out > preset.zone_gap:
                continue
            dwell = now - v.t_in
            if dwell < preset.alert_after:
                continue
            samples = _visit_samples(timeline.get(pid, []), [v])
            if any(near_staff(s, index, preset.assist_dist) for s in samples):
                continue
            out.append(LiveAlert(pid, zid, dwell, v.t_in,
                                 f"Customer at {cars[zid].name} for {int(dwell)} s, no staff nearby"))
    out.sort(key=lambda a: (a.t_in, a.person_id))
    return out
