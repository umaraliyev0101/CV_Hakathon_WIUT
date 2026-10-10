"""Core data types shared by every stage (perception, roles, events, metrics, export)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in processed-frame pixels


@dataclass
class Det:
    bbox: BBox
    score: float
    cls: int = 0  # COCO class id; 0 = person, 2 = car


@dataclass
class TrackObs:
    track_id: int
    frame_idx: int  # sampled-frame index
    t: float  # seconds from clip start
    bbox: BBox
    score: float
    kpts: list[list[float]] | None = None  # 17 x [x, y, conf] (COCO order)


@dataclass
class Person:
    pid: int
    track_ids: list[int]
    t_first: float
    t_last: float
    n_obs: int
    thumb: str | None = None  # relative file name of the blurred thumbnail


@dataclass
class RoleSuggestion:
    pid: int
    role: str  # "staff" | "customer" | "unsure"
    confidence: float  # P(staff) in [0, 1]
    reasons: list[str]
    features: dict[str, float]


@dataclass
class Event:
    # enter | leave | greet | engagement_start | engagement | wait_alert | walkout | party | handoff
    type: str
    t: float
    customer: int | None = None
    staff: int | None = None
    t_end: float | None = None
    data: dict = field(default_factory=dict)


@dataclass
class StaffCard:
    staff: int
    name: str | None
    greets: int
    median_time_to_greet: float | None
    customers_engaged: int
    engaged_total_s: float
    engaged_median_s: float | None
    missed_while_free: int
    walkouts_while_free: int
    sold: int | None = None
    conversion: float | None = None


@dataclass
class StoreCard:
    customers: int
    customers_entry_known: int
    greeted: int
    greet_rate: float | None
    median_time_to_greet: float | None
    walkouts: int
    unattended_wait_s: float
    wait_alerts: int
    engagements: int


@dataclass
class Scorecard:
    staff: list[StaffCard]
    store: StoreCard


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses, numpy scalars/arrays, tuples and non-string dict keys to plain JSON types.

    Non-finite floats (NaN, +/-inf) become None, because they are not valid JSON and
    `JSON.parse` in the browser rejects them.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, dict):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _key(k: Any) -> str:
    if isinstance(k, str):
        return k
    k = to_jsonable(k)
    return k if isinstance(k, str) else str(k)
