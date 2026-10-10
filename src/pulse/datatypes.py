"""Core dataclasses. Det/TrackObs/Person/RoleSuggestion/Event are shared with Showroom Coach
(spec §4.2) — keep names and fields identical. All serialise to JSON via to_dict/from_dict."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from typing import Any

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2, analysis-frame pixels


class JsonMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Det(JsonMixin):
    bbox: BBox
    score: float
    cls: str  # COCO class name: 'person', 'car', 'truck', 'motorcycle', 'bus'


@dataclass
class TrackObs(JsonMixin):
    t: float
    frame_idx: int                      # index of the sampled frame
    track_id: int
    bbox: BBox
    score: float
    keypoints: list[list[float]] | None = None  # 17 x [x, y, score], COCO order, pixels


@dataclass
class Person(JsonMixin):
    person_id: int
    track_ids: list[int]
    first_t: float
    last_t: float


@dataclass
class RoleSuggestion(JsonMixin):
    person_id: int
    role: str            # 'staff' | 'customer' | 'unsure'
    confidence: float    # 0..1
    reasons: list[str]
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Event(JsonMixin):
    kind: str
    t_start: float
    t_end: float | None
    person_ids: list[int]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Zone(JsonMixin):
    zone_id: str
    name: str
    kind: str                               # 'car' | 'desk' | 'entrance' | 'lounge' | 'other'
    polygon: list[tuple[float, float]]      # normalised [0,1] image coords
    source: str = "user"                    # 'auto' | 'user'


@dataclass
class Sample(JsonMixin):
    t: float
    person_id: int
    foot: tuple[float, float]               # normalised [0,1] image coords
    bbox: BBox
    pos_m: tuple[float, float] | None       # (X, Z) metres
    speed: float | None                     # smoothed m/s
    dist_method: str = "box"                # 'full' | 'torso' | 'box' | 'floor'


Timeline = dict[int, list[Sample]]


@dataclass
class ZoneVisit(JsonMixin):
    person_id: int
    zone_id: str
    t_in: float
    t_out: float
    stopped: bool = False

    @property
    def dwell(self) -> float:
        return self.t_out - self.t_in


@dataclass
class InterestEpisode(JsonMixin):
    person_id: int
    zone_id: str
    t_start: float
    t_end: float
    dwell: float
    visits: int
    assisted: bool
    unserved: bool
    staff_ids: list[int] = field(default_factory=list)


@dataclass
class Insight(JsonMixin):
    rule: str
    title: str
    evidence: dict[str, Any]
    action: str
    cars: list[str]
    window: tuple[float, float] | None
    confidence: str                         # 'insufficient' | 'low' | 'medium' | 'high'
    sample: int
    moments: list[dict[str, Any]]
    demo_only: bool = False


@dataclass
class StableSegment(JsonMixin):
    t0: float
    t1: float
    stable: bool
    same_view: bool = True                  # zones drawn on the keyframe apply here


STORE_KEYS_DEFAULTS: dict[str, Any] = {
    "visitors": 0, "staff": 0, "unsure": 0, "already_present": 0, "entries": 0, "exits": 0,
    "groups": 0, "n_solo": 0, "n_group_members": 0, "median_visit_s": None, "browse_rate": None,
    "assisted_share": None, "assisted_customers": 0, "median_time_to_assist": None,
    "unserved_episodes": 0, "deep_interest_visitors": 0, "reached_desk": 0,
    "deep_interest_reached_desk": 0, "has_desk": False, "peak_occupancy": 0, "peak_t": None,
    "funnel": {}, "group_assisted_share": None, "solo_assisted_share": None,
    "duration_s": 0.0, "analysed_share": 1.0, "distance_approx": False,
}


@dataclass
class Stats(JsonMixin):
    store: dict[str, Any]
    per_car: list[dict[str, Any]]
    per_zone: list[dict[str, Any]]
    occupancy: list[list[float]]            # [[t, customers present]] every 1 s
    waiting: list[list[float]]              # [[t, customers present & not yet assisted]]
    crowding: list[list[float]]             # [[t, customers per staff]] (-1 when no staff present)
    time_buckets: list[dict[str, Any]]
    heatmap: dict[str, Any]                 # {"w": 48, "h": 27, "counts": [[int]*48]*27}
    paths: list[dict[str, Any]]             # [{"person_id", "points": [[x, y], ...]}] normalised
    visits: list[dict[str, Any]]            # one anonymous record per person
    groups: list[list[int]]
    warnings: list[str]

    @classmethod
    def empty(cls) -> "Stats":
        return cls(dict(STORE_KEYS_DEFAULTS), [], [], [], [], [], [],
                   {"w": 48, "h": 27, "counts": [[0] * 48 for _ in range(27)]}, [], [], [], [])
