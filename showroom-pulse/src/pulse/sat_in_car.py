"""Sat-in-car detection (spec 4.5, stretch M8). Two rules, both on one car box:

(a) at least `sat_overlap` of the person box lies inside the car box, only torso keypoints (shoulders,
    hips) are confident while knees and ankles are not, and the foot speed stays below `sat_speed` for
    `sat_time_s` or longer;
(b) a track segment ends inside the car box and another track segment starts inside it within
    `sat_reappear_s` (the person disappeared into the car, or somebody came out of it). The last (first)
    second of the ending (starting) segment must be torso-only, and slow at the end, and the end must
    not be at the frame edge: a person walking behind a car or a re-ID fragment of someone standing
    next to it has visible legs and does not qualify.

Known limits: "inside" is an image-plane box overlap, so someone standing in front of or behind a car
can overlap it; both rules were validated on synthetic boxes only. Events with no car zone get
data["zone_id"] = None and are not counted in any per-car number.

Car boxes are the keyframe detections (PerceptionCache.cars): cars in a showroom do not move, so one
box per car stands for the whole clip. Pure functions; no model is run."""
from __future__ import annotations

from bisect import bisect_left

from .config import Config
from .datatypes import BBox, Det, Event, Sample, TrackObs, Zone
from .zones import point_in_polygon

KIND = "sat_in_car"
_VEHICLES = ("car", "truck", "bus")          # motorcycles cannot be sat "in"
_TORSO = (5, 6, 11, 12)                      # COCO shoulders and hips
_LEGS = (13, 14, 15, 16)                     # COCO knees and ankles


def _inside(person: BBox, car: BBox) -> float:
    iw = min(person[2], car[2]) - max(person[0], car[0])
    ih = min(person[3], car[3]) - max(person[1], car[1])
    area = (person[2] - person[0]) * (person[3] - person[1])
    return iw * ih / area if iw > 0 and ih > 0 and area > 0 else 0.0


def _torso_only(kp, thr: float) -> bool:
    if not kp or len(kp) < 17:
        return False
    return sum(kp[i][2] >= thr for i in _TORSO) >= 2 and all(kp[i][2] < thr for i in _LEGS)


def _speed_at(samples: list[Sample], ts: list[float], t: float, tol: float) -> float | None:
    if not samples:
        return None
    i = bisect_left(ts, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(samples)), key=lambda j: abs(ts[j] - t))
    return samples[best].speed if abs(ts[best] - t) <= tol else None


def _at_edge(b: BBox, frame_size, margin: float, bottom_only: bool = False) -> bool:
    if not frame_size:
        return False
    w, h = frame_size
    mx, my = margin * w, margin * h
    if b[3] >= h - my:
        return True
    return not bottom_only and (b[0] <= mx or b[1] <= my or b[2] >= w - mx)


def _rule_a(pid: int, obs: list[TrackObs], samples: list[Sample], car_i: int, car: BBox,
            cfg: Config, frame_size=None) -> list[Event]:
    ts = [s.t for s in samples]
    gap = cfg.sat_gap_s
    runs: list[list[float]] = []
    for o in obs:
        sp = _speed_at(samples, ts, o.t, 0.5 / cfg.sample_fps)
        ok = (_inside(o.bbox, car) >= cfg.sat_overlap and _torso_only(o.keypoints, cfg.kp_thr)
              and car[1] <= o.bbox[1] <= car[3]              # a seated person's head is inside the car outline
              and not _at_edge(o.bbox, frame_size, cfg.edge_margin, bottom_only=True)  # legs cut by the frame
              and sp is not None and sp < cfg.sat_speed)
        if not ok:
            continue
        if runs and o.t - runs[-1][-1] <= gap:
            runs[-1].append(o.t)
        else:
            runs.append([o.t])
    return [Event(KIND, r[0], r[-1], [pid], {"rule": "a", "car": car_i, "car_bbox": list(car)})
            for r in runs if r[-1] - r[0] >= cfg.sat_time_s]


def _segments(obs: list[TrackObs], gap: float) -> list[list[TrackObs]]:
    segs: list[list[TrackObs]] = []
    for o in obs:
        if segs and o.t - segs[-1][-1].t <= gap:
            segs[-1].append(o)
        else:
            segs.append([o])
    return segs


def _hidden_legs(tail: list[TrackObs], samples: list[Sample], cfg: Config, slow: bool) -> bool:
    """Every observation of `tail` torso-only (and, if `slow`, with a known speed below sat_speed)."""
    if not tail or tail[-1].t - tail[0].t < 1.0 - 1e-6:
        return False
    ts = [x.t for x in samples]
    for o in tail:
        if not _torso_only(o.keypoints, cfg.kp_thr):
            return False
        if slow:
            sp = _speed_at(samples, ts, o.t, 0.5 / cfg.sample_fps)
            if sp is None or sp >= cfg.sat_speed:
                return False
    return True


def _rule_b(segs: list[tuple[int, list[TrackObs]]], samples: dict[int, list[Sample]], car_i: int, car: BBox,
            cfg: Config, frame_size=None) -> list[Event]:
    inside = lambda o: _inside(o.bbox, car) >= cfg.sat_overlap  # noqa: E731
    ends = [(pid, s[-1]) for pid, s in segs
            if inside(s[-1]) and not _at_edge(s[-1].bbox, frame_size, cfg.edge_margin)
            and _hidden_legs([o for o in s if o.t >= s[-1].t - 1.0], samples.get(pid, []), cfg, True)]
    starts = [(pid, s[0]) for pid, s in segs
              if inside(s[0]) and _hidden_legs([o for o in s if o.t <= s[0].t + 1.0], [], cfg, False)]
    out = []
    for pid, last in ends:
        later = [(o.t, q) for q, o in starts if 0 < o.t - last.t <= cfg.sat_reappear_s]
        if later:
            t_back, _ = min(later)
            out.append(Event(KIND, last.t, t_back, [pid],
                              {"rule": "b", "car": car_i, "car_bbox": list(car)}))
    return out


def detect(timeline: dict[int, list[Sample]], obs_by_person: dict[int, list[TrackObs]],
           car_dets: list[Det], cfg: Config, frame_size: tuple[int, int] | None = None) -> list[Event]:
    cars = [(i, d.bbox) for i, d in enumerate(car_dets) if d.cls in _VEHICLES]
    events: list[Event] = []
    if not cars:
        return events
    obs = {pid: sorted(o, key=lambda x: x.t) for pid, o in obs_by_person.items()}
    sam = {pid: sorted(s, key=lambda x: x.t) for pid, s in timeline.items()}
    segs = [(pid, s) for pid in sorted(obs) for s in _segments(obs[pid], cfg.sat_gap_s)]
    for i, box in cars:
        for pid in sorted(obs):
            events += _rule_a(pid, obs[pid], sam.get(pid, []), i, box, cfg, frame_size)
        events += _rule_b(segs, sam, i, box, cfg, frame_size)
    events.sort(key=lambda e: (e.t_start, e.person_ids[0], e.data["car"], e.data["rule"]))
    return events


def assign_zones(events: list[Event], car_dets: list[Det], zones: list[Zone],
                 frame_size: tuple[int, int]) -> list[Event]:
    """Set data['zone_id'] to the car zone holding the car box's bottom centre (else its centre)."""
    w, h = frame_size
    cz = [z for z in zones if z.kind == "car"]
    for e in events:
        b = car_dets[e.data["car"]].bbox
        pts = [((b[0] + b[2]) / 2 / w, b[3] / h), ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h)]
        e.data["zone_id"] = next((z.zone_id for p in pts for z in cz
                                  if point_in_polygon(p, z.polygon)), None)
    return events
