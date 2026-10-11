"""Per-person, per-car dwell time from floor positions over time.

A person is "at" a car while their floor point is within `near_m` of the car footprint. Short
dropouts (< `gap_s`, e.g. a missed detection or a brief step back) do not split a visit, and visits
shorter than `min_visit_s` (walking past) are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analytics.floor import distance_to_polygon

NEAR_M = 1.5
GAP_S = 1.0
MIN_VISIT_S = 2.0


@dataclass
class Visit:
    start_t: float
    end_t: float
    min_dist_m: float

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t


@dataclass
class CarDwell:
    track_id: int
    car: str
    visits: list[Visit] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(v.duration_s for v in self.visits)


def car_distances(xy, cars: dict[str, list]) -> dict[str, float]:
    """Distance (m) from one floor point to every car footprint. NaN point -> NaN distances."""
    if xy is None or np.any(np.isnan(xy)):
        return {c: float("nan") for c in cars}
    return {c: distance_to_polygon(xy, poly) for c, poly in cars.items()}


def dwell(samples: list[tuple[float, int, np.ndarray]], cars: dict[str, list], dt: float,
          near_m: float = NEAR_M, gap_s: float = GAP_S, min_visit_s: float = MIN_VISIT_S) -> list[CarDwell]:
    """samples: (t, track_id, floor_xy) for every frame a person is seen; dt: frame period (s).

    Returns one CarDwell per (track, car) with at least one visit.
    """
    by_track: dict[int, list[tuple[float, np.ndarray]]] = {}
    for t, tid, xy in samples:
        by_track.setdefault(tid, []).append((t, xy))
    out = []
    for tid, seq in sorted(by_track.items()):
        seq.sort(key=lambda s: s[0])
        for car, poly in cars.items():
            near = [(t, d) for t, xy in seq
                    if not np.any(np.isnan(xy)) and (d := distance_to_polygon(xy, poly)) <= near_m]
            visits: list[Visit] = []
            for t, d in near:
                if visits and t - visits[-1].end_t <= gap_s + dt / 2:
                    visits[-1].end_t = t + dt
                    visits[-1].min_dist_m = min(visits[-1].min_dist_m, d)
                else:
                    visits.append(Visit(t, t + dt, d))
            visits = [v for v in visits if v.duration_s >= min_visit_s]
            if visits:
                out.append(CarDwell(tid, car, visits))
    return out
