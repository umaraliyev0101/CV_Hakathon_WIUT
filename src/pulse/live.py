"""Live mode: one frame at a time through the same perception engine, then the same pure analysis
core (timeline -> journeys -> unserved-interest alerts) over a rolling window.

`t` is supplied by the caller (the server clock, seconds since session start). Everything is
anonymous: person ids are per-session integers, appearance embeddings are colour histograms that
only bridge short tracker gaps, and nothing is written to disk."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from pulse import identity, interest, journey, timeline as timeline_mod
from pulse.config import PRESETS, Config
from pulse.datatypes import Timeline, TrackObs, Zone, ZoneVisit
from pulse.perception.embed import hsv_hist, torso_crop
from pulse.zones import zone_of

TAIL_S = 3.0          # speed smoothing needs a few seconds of context: only this tail is rebuilt
FPS_WINDOW_S = 5.0
EMB_FRAMES = 3        # appearance is averaged over a person's first few frames
CUSTOMER, STAFF = "customer", "staff"


@dataclass
class _Person:
    pid: int
    first_t: float
    last_t: float
    start_foot: tuple[float, float]
    last_foot: tuple[float, float]
    emb_sum: np.ndarray | None = None
    emb_n: int = 0

    @property
    def emb(self) -> np.ndarray | None:
        if self.emb_sum is None:
            return None
        n = float(np.linalg.norm(self.emb_sum))
        return self.emb_sum / n if n > 0 else None


class LiveSession:
    def __init__(self, cfg: Config, engine, zones: list[Zone] | None = None, window_s: float = 300.0):
        self.cfg = cfg
        self.engine = engine
        self.zones: list[Zone] = list(zones or [])
        self.window_s = float(window_s)
        self.roles: dict[int, str] = {}
        self.dropped = 0
        self.timeline: Timeline = {}
        self.journeys: dict[int, list[ZoneVisit]] = {}
        self.frame_size: tuple[int, int] = (0, 0)
        self._persons: dict[int, _Person] = {}
        self._obs: dict[int, list[TrackObs]] = {}           # person -> observations in the window
        self._track_person: dict[int, int] = {}
        self._track_seen: dict[int, float] = {}
        self._next_pid = 1
        self._frame_idx = 0
        self._boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        self._done: deque[float] = deque()
        self._lock = threading.RLock()
        engine.reset()

    # ------------------------------------------------------------------ configuration
    def set_zones(self, zones: list[Zone]) -> None:
        with self._lock:
            self.zones = list(zones)
            self._rebuild_journeys(set(self._persons))

    def set_preset(self, name: str) -> None:
        with self._lock:
            self.cfg = replace(self.cfg, preset=PRESETS[name])
            self._rebuild_journeys(set(self._persons))

    def toggle_role_at(self, x_norm: float, y_norm: float) -> int | None:
        """Flip staff/customer for the person whose current box contains the point (smallest box wins)."""
        with self._lock:
            w, h = self.frame_size
            x, y = x_norm * w, y_norm * h
            hits = [(pid, bb) for pid, bb in self._boxes if bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]]
            if not hits:
                return None
            pid = min(hits, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))[0]
            self.roles[pid] = CUSTOMER if self.roles.get(pid, CUSTOMER) == STAFF else STAFF
            return pid

    def role(self, pid: int) -> str:
        return self.roles.get(pid, CUSTOMER)

    # ------------------------------------------------------------------ identity
    def _assign(self, obs: list[TrackObs], frame: np.ndarray, t: float) -> None:
        size = self.frame_size
        new = [o for o in obs if o.track_id not in self._track_person]
        embs = {}
        for o in new:
            crop = torso_crop(frame, o, self.cfg.kp_thr)
            embs[o.track_id] = hsv_hist(crop) if crop is not None else None
        seen_now = {self._track_person[o.track_id] for o in obs if o.track_id in self._track_person}
        for o in sorted(new, key=lambda o: -o.score):
            summ = identity.summarize([o], {o.track_id: embs[o.track_id]}, size).get(o.track_id)
            if summ is None:                                   # non-finite foot point: ignore
                continue
            lost = [identity.TrackSummary(p.pid, p.first_t, p.last_t, p.start_foot, p.last_foot,
                                          p.emb, 1)
                    for p in self._persons.values()
                    if p.pid not in seen_now and p.last_t < t - 1e-9]
            pid = identity.reattach(lost, summ, t, self.cfg)
            if pid is None:
                pid = self._next_pid
                self._next_pid += 1
                self._persons[pid] = _Person(pid, t, t, summ.start_foot, summ.start_foot)
            self._track_person[o.track_id] = pid
            seen_now.add(pid)

    # ------------------------------------------------------------------ analysis
    def _rebuild_journeys(self, pids) -> None:
        sub = {pid: self.timeline[pid] for pid in pids if pid in self.timeline}
        built = journey.build(sub, self.zones, self.cfg.preset)
        for pid in pids:
            if pid in built:
                self.journeys[pid] = built[pid]

    def _trim(self, t: float) -> None:
        cutoff = t - self.window_s
        for pid in [p for p, ps in self._persons.items() if ps.last_t < cutoff]:
            for d in (self._persons, self.timeline, self._obs, self.journeys):
                d.pop(pid, None)
            for tid in [k for k, v in self._track_person.items() if v == pid]:
                del self._track_person[tid]
                self._track_seen.pop(tid, None)
            self.roles.pop(pid, None)
        for tid in [k for k, seen in self._track_seen.items() if seen < cutoff]:
            self._track_person.pop(tid, None)           # a long-silent track id never comes back
            del self._track_seen[tid]
        for pid, ss in self.timeline.items():
            if ss and ss[0].t < cutoff:
                self.timeline[pid] = [s for s in ss if s.t >= cutoff]
        for pid, os_ in self._obs.items():
            if os_ and os_[0].t < cutoff:
                self._obs[pid] = [o for o in os_ if o.t >= cutoff]
        for pid, vs in self.journeys.items():
            if vs and vs[0].t_out < cutoff:
                self.journeys[pid] = [v for v in vs if v.t_out >= cutoff]

    def step(self, t: float, frame: np.ndarray) -> dict:
        with self._lock:
            t_wall = time.perf_counter()
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            obs = self.engine.process(t, frame, self._frame_idx)
            self._frame_idx += 1
            self._assign(obs, frame, t)

            visible: dict[int, TrackObs] = {}           # best observation per person this frame
            for o in obs:
                pid = self._track_person.get(o.track_id)
                if pid is None:
                    continue
                self._obs.setdefault(pid, []).append(o)
                self._track_seen[o.track_id] = t
                if pid not in visible or o.score > visible[pid].score:
                    visible[pid] = o
            for pid, o in visible.items():
                p = self._persons[pid]
                p.last_t = t
                u, v = o.bbox[0] / 2 + o.bbox[2] / 2, o.bbox[3]
                p.last_foot = (u / w, v / h)
                if p.emb_n < EMB_FRAMES:
                    crop = torso_crop(frame, o, self.cfg.kp_thr)
                    if crop is not None:
                        e = hsv_hist(crop)
                        p.emb_sum = e if p.emb_sum is None else p.emb_sum + e
                        p.emb_n += 1
            self._boxes = [(pid, tuple(o.bbox)) for pid, o in visible.items()]

            # timeline: rebuild only the tail of every visible person
            tails = [o for pid in visible for o in self._obs[pid] if o.t >= t - TAIL_S]
            fresh = timeline_mod.build_timeline(tails, self._track_person, self.frame_size, self.cfg)
            for pid, new in fresh.items():
                if not new:
                    continue
                old = self.timeline.get(pid, [])
                self.timeline[pid] = [s for s in old if s.t < new[0].t] + new

            recent = {pid for pid, p in self._persons.items()
                      if p.last_t >= t - self.cfg.preset.leave_after}
            self._rebuild_journeys(recent)
            self._trim(t)
            return self._overlay(t, visible, t_wall)

    def _alerts(self, t: float) -> list[interest.LiveAlert]:
        preset = self.cfg.preset
        staff = {pid: ss for pid, ss in self.timeline.items() if self.role(pid) == STAFF}
        live = {pid for pid, p in self._persons.items() if p.last_t >= t - preset.zone_gap - 1e-6}
        journeys = {pid: self.journeys.get(pid, []) for pid in live}
        tl = {pid: self.timeline[pid] for pid in live | set(staff) if pid in self.timeline}
        return interest.open_alerts(journeys, staff, preset, tl, self.zones, t)

    def _current_zone(self, pid: int, t: float) -> tuple[str | None, str | None, float]:
        ss = self.timeline.get(pid)
        if not ss or abs(ss[-1].t - t) > 1e-6:
            return None, None, 0.0
        zid = zone_of(ss[-1].foot, self.zones)
        if zid is None:
            return None, None, 0.0
        name = next(z.name for z in self.zones if z.zone_id == zid)
        dwell = 0.0
        for v in self.journeys.get(pid, []):
            if v.zone_id == zid and v.t_out >= t - self.cfg.preset.zone_gap - 1e-6:
                dwell = max(0.0, t - v.t_in)
        return zid, name, dwell

    def _overlay(self, t: float, visible: dict[int, TrackObs], t_wall: float) -> dict:
        alerts = self._alerts(t)
        alert_keys = {(a.person_id, a.zone_id) for a in alerts}
        persons, now_in = [], {z.zone_id: 0 for z in self.zones}
        customers = 0
        for pid in sorted(visible):
            role = self.role(pid)
            zid, zname, dwell = self._current_zone(pid, t)
            if role == CUSTOMER:
                customers += 1
                if zid is not None:
                    now_in[zid] += 1
            persons.append({"person_id": pid, "bbox": [float(v) for v in visible[pid].bbox],
                            "role": role, "zone_id": zid, "zone_name": zname,
                            "dwell": round(dwell, 2), "alert": (pid, zid) in alert_keys})
        total = {z.zone_id: set() for z in self.zones}
        for pid, vs in self.journeys.items():
            if self.role(pid) == STAFF:
                continue
            for v in vs:
                if v.zone_id in total:
                    total[v.zone_id].add(pid)
        zones = [{"zone_id": z.zone_id, "name": z.name, "now": now_in[z.zone_id],
                  "total": len(total[z.zone_id])} for z in self.zones]
        done = time.perf_counter()
        self._done.append(done)
        while self._done and done - self._done[0] > FPS_WINDOW_S:
            self._done.popleft()
        span = self._done[-1] - self._done[0]
        fps = (len(self._done) - 1) / span if len(self._done) >= 2 and span > 0 else 0.0
        return {"type": "overlay", "t": round(t, 3), "frame_size": list(self.frame_size),
                "persons": persons, "zones": zones, "occupancy": customers,
                "alerts": [{"person_id": a.person_id, "zone_id": a.zone_id,
                            "dwell": round(a.dwell, 2), "message": a.message} for a in alerts],
                "fps": round(fps, 2), "dropped": self.dropped,
                "step_ms": round((done - t_wall) * 1000.0, 1)}
