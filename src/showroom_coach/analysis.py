"""The offline analysis pass: one perception result in, coaching results out.

1. persons  - drop flicker tracks and merge the track fragments of one person by appearance (``build_persons``)
2. geometry - floor position in metres, distance source, facing and frame-edge flag per observation
3. roles    - contacts, initiative, presence and shared outfit per person -> staff suggestion with reasons
4. events   - every sampled frame's person states run through the event engine (greet, engagement, walkout, ...)
5. metrics  - scorecards, per-person timelines and warnings; ``rescore`` repeats only 4-5 when roles or preset change
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .config import CameraConfig, RoleWeights, Thresholds, get_preset
from .events import PersonState, run_events
from .geometry import Camera, at_frame_edge, facing_vector, position_m
from .identity import TrackSummary, merge_tracks
from .metrics import scorecard as build_scorecard
from .roles import interaction_features, outfit_peers, presence, suggest_roles
from .types import Event, Person, RoleSuggestion, Scorecard, to_jsonable

if TYPE_CHECKING:
    from .perception.pipeline import PerceptionResult

log = logging.getLogger(__name__)

__all__ = ["Analysis", "analyze", "build_persons", "compute_geometry", "person_timeline", "rescore"]

ROLES = ("staff", "customer")
SOURCE_CODES = {"full": 0, "torso": 1, "box": 2, "floor": 3}  # obs_src values; "box" is the approximate one
APPROX_WARN_SHARE = 0.5  # warn when more than this share of the kept observations used the box height
_EPS = 1e-6  # time tolerance: sampled times such as 20.2 are not exact in binary floating point

W_NOBODY = "nobody detected — the camera may be too far or too dark"
W_ONE_PERSON = "fewer than 2 people — store metrics only"
W_NO_STAFF = "nobody marked as staff — mark staff to get scorecards"


@dataclass(eq=False)
class Analysis:
    """Everything the results page needs, plus the per-observation arrays that make ``rescore`` cheap."""

    preset: str
    thresholds: Thresholds
    duration: float
    persons: list[Person]  # sorted by pid; pids are 1..N by first appearance
    suggestions: dict[int, RoleSuggestion]
    roles: dict[int, str]  # final: user override, else suggestion ("unsure" -> "customer")
    overrides: dict[int, str]  # what the user set
    names: dict[int, str]
    events: list[Event]
    scorecard: Scorecard
    timelines: dict[int, dict]  # pid -> {"present": [[t0, t1], ...], "waiting": [...], "engaged": [...], "greeted": t}
    warnings: list[str]
    approx_share: float  # share of kept observations whose distance came from the box height
    # per-observation arrays aligned with PerceptionResult.obs_*:
    obs_pid: np.ndarray  # (M,) person id, 0 = dropped (flicker track, or a duplicate in the same frame)
    obs_xz: np.ndarray  # (M, 2) floor position in metres
    obs_src: np.ndarray  # (M,) position source code (SOURCE_CODES)
    obs_facing: np.ndarray  # (M, 2) unit heading on the floor, nan when unknown
    obs_edge: np.ndarray  # (M,) box touches a frame edge
    features: dict[int, dict]  # role features per person

    def to_dict(self) -> dict:
        """The web API's results object (api-contract.md) without the URL fields the web layer adds."""
        return to_jsonable({
            "preset": self.preset, "duration": self.duration, "thresholds": self.thresholds,
            "persons": self.persons, "suggestions": self.suggestions, "roles": self.roles, "names": self.names,
            "events": self.events, "scorecard": self.scorecard, "timelines": self.timelines,
            "warnings": self.warnings,
        })


# --------------------------------------------------------------------------------------------- 1. persons

def build_persons(res: PerceptionResult, merge_thr: float = 0.30, min_track_obs: int = 3,
                  min_track_s: float = 1.0) -> tuple[list[Person], np.ndarray]:
    """Persons sorted by pid, and ``obs_pid`` giving each observation's person (0 = not used).

    Tracks with fewer than ``min_track_obs`` observations or spanning less than ``min_track_s`` seconds are
    flicker and are dropped. The rest are merged by appearance (``identity.merge_tracks``); a track without an
    embedding can't be matched, so it stays a person of its own.
    """
    obs_t = res.frame_times[res.obs_frame]
    summaries = []
    for tid in res.track_ids():
        t = obs_t[res.obs_idx(tid)]
        if len(t) >= min_track_obs and t.max() - t.min() >= min_track_s - _EPS:
            summaries.append(TrackSummary(tid, float(t.min()), float(t.max()), res.track_emb.get(tid), len(t)))
    track_pid = merge_tracks(summaries, thr=merge_thr)

    obs_pid = np.array([track_pid.get(int(tid), 0) for tid in res.obs_track], np.int64)
    obs_pid = _one_per_frame(obs_pid, res.obs_frame, res.obs_score)
    by_pid = _indices_by_pid(obs_pid)
    persons = []
    for pid in sorted(by_pid):
        tracks = [s for s in summaries if track_pid[s.track_id] == pid]
        longest = max(tracks, key=lambda s: (s.t1 - s.t0, s.n_obs, -s.track_id))
        t = obs_t[by_pid[pid]]
        persons.append(Person(pid=pid, track_ids=sorted(s.track_id for s in tracks), t_first=float(t.min()),
                              t_last=float(t.max()), n_obs=len(t), thumb=f"{longest.track_id}.jpg"))
    return persons, obs_pid


def _one_per_frame(obs_pid: np.ndarray, obs_frame: np.ndarray, obs_score: np.ndarray) -> np.ndarray:
    """Merged tracks may both see one frame: keep that person's highest-score observation, drop (pid 0) the rest."""
    out = obs_pid.copy()
    kept = np.flatnonzero(out > 0)
    order = kept[np.lexsort((-obs_score[kept], out[kept], obs_frame[kept]))]  # by frame, then pid, best first
    same = (obs_frame[order][1:] == obs_frame[order][:-1]) & (out[order][1:] == out[order][:-1])
    out[order[1:][same]] = 0
    return out


def _indices_by_pid(obs_pid: np.ndarray) -> dict[int, np.ndarray]:
    """pid -> indices of its observations, in observation order (pid 0 left out)."""
    order = np.argsort(obs_pid, kind="stable")
    pids, starts = np.unique(obs_pid[order], return_index=True)
    groups = np.split(order, starts[1:])
    return {int(p): g for p, g in zip(pids, groups) if p > 0}


# -------------------------------------------------------------------------------------------- 2. geometry

def compute_geometry(res: PerceptionResult, camera: CameraConfig = CameraConfig()
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(obs_xz, obs_src, obs_facing, obs_edge)`` for every observation (see ``Analysis``)."""
    cam = Camera.from_config(res.video.width, res.video.height, camera)
    m = len(res.obs_track)
    xz = np.zeros((m, 2))
    src = np.zeros(m, np.int8)
    facing = np.full((m, 2), np.nan)
    edge = np.zeros(m, bool)
    for i, (bbox, kpts) in enumerate(zip(res.obs_bbox, res.obs_kpts)):
        pos = position_m(bbox, kpts, cam)
        xz[i], src[i] = pos.xz, SOURCE_CODES[pos.source]
        heading = facing_vector(bbox, kpts, cam)
        if heading is not None:
            facing[i] = heading
        edge[i] = at_frame_edge(bbox, cam.width, cam.height)
    return xz, src, facing, edge


def _approx_share(obs_pid: np.ndarray, obs_src: np.ndarray) -> float:
    kept = obs_pid > 0
    return float(np.mean(obs_src[kept] == SOURCE_CODES["box"])) if kept.any() else 0.0


# ----------------------------------------------------------------------------------------------- 3. roles

def _role_features(res: PerceptionResult, persons: list[Person], obs_pid: np.ndarray, obs_xz: np.ndarray,
                   weights: RoleWeights) -> dict[int, dict]:
    """outfit_peers, contacts, presence, initiated (and the contact partners) per person."""
    obs_t = res.frame_times[res.obs_frame]
    traj = {pid: np.column_stack([obs_t[idx], obs_xz[idx]]) for pid, idx in _indices_by_pid(obs_pid).items()}
    talk = interaction_features(traj, weights.contact_dist, weights.contact_min_s)
    seen = presence(traj, res.frame_times)
    outfit = outfit_peers(_person_outfits(res, persons), weights.outfit_sim_thr)
    return {p.pid: {"outfit_peers": outfit.get(p.pid, 0), "contacts": talk[p.pid]["contacts"],
                    "presence": seen[p.pid], "initiated": talk[p.pid]["initiated"],
                    "partners": talk[p.pid]["partners"]} for p in persons}


def _person_outfits(res: PerceptionResult, persons: list[Person]) -> dict[int, np.ndarray]:
    """Mean outfit descriptor of each person's tracks; zeros (matches nobody) when no track has one."""
    vecs = {p.pid: [np.asarray(res.track_outfit[t], np.float64).ravel() for t in p.track_ids if t in res.track_outfit]
            for p in persons}
    dim = next((v[0].size for v in vecs.values() if v), 0)
    if not dim:
        return {}
    return {pid: np.mean(v, axis=0) if v else np.zeros(dim) for pid, v in vecs.items()}


def _final_roles(persons: list[Person], suggestions: dict[int, RoleSuggestion],
                 overrides: dict[int, str]) -> dict[int, str]:
    """The user's override, else the suggestion, with "unsure" counted as a customer."""
    return {p.pid: overrides.get(p.pid) or ("staff" if suggestions[p.pid].role == "staff" else "customer")
            for p in persons}


def _as_pid(key) -> int | None:
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _merge_roles(old: dict[int, str], new: dict | None, pids: set[int]) -> dict[int, str]:
    """``old`` updated with ``new``; unknown pids and roles other than staff/customer are ignored."""
    out = dict(old)
    for key, role in (new or {}).items():
        pid = _as_pid(key)
        if pid in pids and role in ROLES:
            out[pid] = role
        else:
            log.warning("ignoring role %r for person %r", role, key)
    return dict(sorted(out.items()))


def _merge_names(old: dict[int, str], new: dict | None, pids: set[int]) -> dict[int, str]:
    """``old`` updated with ``new``; a blank name removes it, unknown pids are ignored."""
    out = dict(old)
    for key, name in (new or {}).items():
        pid = _as_pid(key)
        if pid not in pids:
            log.warning("ignoring name for unknown person %r", key)
            continue
        name = "" if name is None else str(name).strip()
        if name:
            out[pid] = name
        else:
            out.pop(pid, None)
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------------------------- 4. events

def person_timeline(res: PerceptionResult, obs_pid: np.ndarray, obs_xz: np.ndarray, obs_facing: np.ndarray,
                    obs_edge: np.ndarray, roles: dict[int, str]) -> list[tuple[float, list[PersonState]]]:
    """``(t, states)`` for every sampled frame, empty frames included, so the engine sees time pass."""
    kept = np.flatnonzero(obs_pid > 0)
    kept = kept[np.argsort(res.obs_frame[kept], kind="stable")]
    bounds = np.searchsorted(res.obs_frame[kept], np.arange(len(res.frame_times) + 1))
    pid, xz, facing, edge = obs_pid.tolist(), obs_xz.tolist(), obs_facing.tolist(), obs_edge.tolist()

    def state(i: int) -> PersonState:
        heading = None if math.isnan(facing[i][0]) else tuple(facing[i])
        return PersonState(pid[i], tuple(xz[i]), roles.get(pid[i], "customer"), heading, bool(edge[i]))

    return [(float(t), [state(i) for i in kept[bounds[f]:bounds[f + 1]].tolist()])
            for f, t in enumerate(res.frame_times)]


# --------------------------------------------------------------------------------------------- 5. metrics

def _timelines(res: PerceptionResult, obs_pid: np.ndarray, persons: list[Person], events: list[Event],
               th: Thresholds, t_end: float) -> dict[int, dict]:
    obs_t = res.frame_times[res.obs_frame]
    by_pid = _indices_by_pid(obs_pid)
    waits = _waits(events, t_end)
    out = {}
    for p in persons:
        engaged = [[e.t, t_end if e.t_end is None else e.t_end] for e in events
                   if e.type == "engagement" and p.pid in (e.customer, e.staff)]
        greets = [e.t for e in events if e.type == "greet" and e.customer == p.pid]
        out[p.pid] = {"present": _present(np.sort(obs_t[by_pid[p.pid]]), th.leave_after),
                      "waiting": waits.get(p.pid, []), "engaged": _union(engaged),
                      "greeted": min(greets) if greets else None}
    return out


def _present(times: np.ndarray, max_gap: float) -> list[list[float]]:
    """Intervals of observation times, split wherever the person was unseen for more than ``max_gap``."""
    if not len(times):
        return []
    cut = np.flatnonzero(np.diff(times) > max_gap + _EPS) + 1
    starts, ends = np.r_[0, cut], np.r_[cut - 1, len(times) - 1]
    return [[float(times[s]), float(times[e])] for s, e in zip(starts, ends)]


def _waits(events: list[Event], t_end: float) -> dict[int, list[list[float]]]:
    """[enter, first greet or leave or clip end] of each visit with a known entry, per customer."""
    out: dict[int, list[list[float]]] = {}
    waiting: dict[int, list[float]] = {}  # customer -> the open interval of their current visit
    for e in sorted(events, key=lambda e: (e.t, e.type != "enter")):  # an enter comes first at equal times
        if e.customer is None:
            continue
        if e.type == "enter":
            waiting.pop(e.customer, None)
            if e.data.get("entry_known"):
                waiting[e.customer] = [e.t, t_end]
                out.setdefault(e.customer, []).append(waiting[e.customer])
        elif e.type in ("greet", "leave") and e.customer in waiting:
            waiting.pop(e.customer)[1] = e.t
    return out


def _union(intervals: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + _EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _warnings(res: PerceptionResult, persons: list[Person], roles: dict[int, str], approx_share: float) -> list[str]:
    out = []
    if not persons:
        out.append(W_NOBODY)
    elif len(persons) < 2:
        out.append(W_ONE_PERSON)
    if persons and "staff" not in roles.values():
        out.append(W_NO_STAFF)
    if approx_share > APPROX_WARN_SHARE:
        out.append(f"distance is approximate for {round(100 * approx_share)}% of samples")
    if res.video.trimmed:
        out.append(f"video trimmed to the first {res.video.max_seconds:g} s")
    return out


# --------------------------------------------------------------------------------------------- entry points

def analyze(res: PerceptionResult, preset: str = "short", roles: dict[int, str] | None = None,
            names: dict[int, str] | None = None, camera: CameraConfig = CameraConfig(),
            weights: RoleWeights = RoleWeights()) -> Analysis:
    """The whole offline pass: steps 1-3 here, then ``rescore`` for events, metrics, timelines and warnings."""
    th = get_preset(preset)
    duration = float(res.video.duration)
    persons, obs_pid = build_persons(res)
    obs_xz, obs_src, obs_facing, obs_edge = compute_geometry(res, camera)
    features = _role_features(res, persons, obs_pid, obs_xz, weights)
    suggestions = {s.pid: s for s in suggest_roles(features, duration, weights)}
    unscored = Analysis(
        preset=preset, thresholds=th, duration=duration, persons=persons, suggestions=suggestions, roles={},
        overrides={}, names={}, events=[], scorecard=build_scorecard([], [], [], duration), timelines={},
        warnings=[], approx_share=_approx_share(obs_pid, obs_src), obs_pid=obs_pid, obs_xz=obs_xz,
        obs_src=obs_src, obs_facing=obs_facing, obs_edge=obs_edge, features=features)
    return rescore(unscored, res, roles=roles, names=names)


def rescore(prev: Analysis, res: PerceptionResult, roles: dict[int, str] | None = None,
            names: dict[int, str] | None = None, preset: str | None = None) -> Analysis:
    """New roles, names or preset: re-run events and metrics only, reusing persons, geometry and suggestions.

    ``roles`` and ``names`` update the earlier ones (people left out keep theirs). ``prev`` is not modified.
    """
    preset = prev.preset if preset is None else preset
    th = get_preset(preset)
    pids = {p.pid for p in prev.persons}
    overrides = _merge_roles(prev.overrides, roles, pids)
    names = _merge_names(prev.names, names, pids)
    final = _final_roles(prev.persons, prev.suggestions, overrides)

    timeline = person_timeline(res, prev.obs_pid, prev.obs_xz, prev.obs_facing, prev.obs_edge, final)
    events = run_events(timeline, th, t_end=prev.duration, t0=0.0)
    staff = [pid for pid, role in final.items() if role == "staff"]
    customers = [pid for pid, role in final.items() if role != "staff"]
    return replace(
        prev, preset=preset, thresholds=th, roles=final, overrides=overrides, names=names, events=events,
        scorecard=build_scorecard(events, staff, customers, prev.duration, names),
        timelines=_timelines(res, prev.obs_pid, prev.persons, events, th, prev.duration),
        warnings=_warnings(res, prev.persons, final, prev.approx_share))
