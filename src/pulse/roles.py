"""Staff vs customer role features, suggestions with plain reasons, roles.json import/export.

Shared by name with Showroom Coach (spec 4.2). Pure functions, no I/O except roles.json."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from pulse.config import Config, get_config
from pulse.datatypes import Person, RoleSuggestion, Timeline
from pulse.geometry import distance_m

VALID_ROLES = ("staff", "customer")
_BRIDGE_S = 1.0
_INITIATOR_WINDOW_S = 3.0
_MIN_REASON_NORM = 0.25  # controller ruling R3


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pos_index(samples) -> dict[float, tuple[float, float]]:
    return {round(s.t, 2): s.pos_m for s in samples if s.pos_m is not None}


def _contact_runs(pa: dict, pb: dict, cfg: Config) -> list[tuple[float, float]]:
    """Qualifying (t_start, t_end) runs where both are within contact_dist_m."""
    near = sorted(t for t in pa.keys() & pb.keys()
                  if distance_m(pa[t], pb[t]) <= cfg.contact_dist_m)
    runs: list[list[float]] = []
    for t in near:
        if runs and t - runs[-1][1] <= _BRIDGE_S + 1e-6:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [(a, b) for a, b in runs if b - a >= cfg.contact_time_s - 1e-6]


def _approach(pa: dict, pb: dict, t0: float, t1: float) -> float:
    """How much A's own movement reduced its distance to B's (earlier) position over [t0, t1]."""
    ts = sorted(t for t in pa.keys() & pb.keys() if t0 - 1e-6 <= t <= t1 + 1e-6)
    if len(ts) < 2:
        return 0.0
    then, now = ts[0], ts[-1]
    return distance_m(pa[then], pb[then]) - distance_m(pa[now], pb[then])


def interactions(persons: list[Person], timeline: Timeline, embeddings: dict[int, np.ndarray],
                 duration: float, cfg: Config) -> dict[int, dict[str, float]]:
    ids = [p.person_id for p in persons]
    pos = {i: _pos_index(timeline.get(i, [])) for i in ids}
    out: dict[int, dict[str, float]] = {}
    contact_cnt = {i: 0 for i in ids}
    init_cnt = {i: 0 for i in ids}
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            runs = _contact_runs(pos[a], pos[b], cfg)
            if not runs:
                continue
            contact_cnt[a] += 1
            contact_cnt[b] += 1
            for t_start, _ in runs[:1]:
                t0 = t_start - _INITIATOR_WINDOW_S
                ra = _approach(pos[a], pos[b], t0, t_start)
                rb = _approach(pos[b], pos[a], t0, t_start)
                if ra > rb + 1e-9:
                    init_cnt[a] += 1
                elif rb > ra + 1e-9:
                    init_cnt[b] += 1
    for i in ids:
        peers = 0
        if i in embeddings:
            peers = sum(1 for j in ids if j != i and j in embeddings
                        and _cos(embeddings[i], embeddings[j]) >= cfg.outfit_sim_thr)
        if duration < cfg.presence_min_clip_s:
            presence = None
        else:
            presence = min(1.0, len(timeline.get(i, [])) / (duration * cfg.sample_fps))
        out[i] = {"outfit_peers": float(peers), "contacts": float(contact_cnt[i]),
                  "initiator": init_cnt[i] / contact_cnt[i] if contact_cnt[i] else 0.0,
                  "presence": presence}
    return out


_NORM = {"outfit_peers": lambda v: min(v, 3) / 3, "contacts": lambda v: min(v, 4) / 4,
         "presence": lambda v: v, "initiator": lambda v: v}


def _reason(name: str, v: float) -> str:
    if name == "outfit_peers":
        return f"same outfit as {int(v)} others"
    if name == "contacts":
        return f"close to {int(v)} people for 5 s or more"
    if name == "presence":
        return f"on screen {round(v * 100)}% of the clip"
    return f"approached others first in {round(v * 100)}% of contacts"


def suggest(persons: list[Person], interactions: dict[int, dict[str, float]],
            cfg: Config | None = None) -> list[RoleSuggestion]:
    cfg = cfg or get_config("dealership")
    w = cfg.role_weights
    result = []
    for p in persons:
        feats = interactions.get(p.person_id, {})
        active = {k: feats[k] for k in _NORM if feats.get(k) is not None}
        wsum = sum(w[k] for k in active)
        contrib = {k: w[k] * _NORM[k](v) for k, v in active.items()}
        score = sum(contrib.values()) / wsum if wsum > 0 else 0.5
        if score >= cfg.role_staff_thr:
            role, conf = "staff", score
        elif score <= cfg.role_customer_thr:
            role, conf = "customer", 1.0 - score
        else:
            role, conf = "unsure", 0.5
        cand = [k for k in active if _NORM[k](active[k]) >= _MIN_REASON_NORM and contrib[k] > 0]
        cand.sort(key=lambda k: contrib[k], reverse=True)
        reasons = [_reason(k, active[k]) for k in cand[:3]]
        result.append(RoleSuggestion(p.person_id, role, float(conf), reasons,
                                     {k: float(v) for k, v in active.items()}))
    return result


def load_roles_json(path) -> dict[int, str]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and isinstance(data.get("persons"), list):
        pairs = [(e["person_id"], e["role"]) for e in data["persons"]]
    elif isinstance(data, dict):
        pairs = list(data.items())
    else:
        raise ValueError("roles.json must be an object")
    out: dict[int, str] = {}
    for pid, role in pairs:
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role {role!r} for person {pid}")
        out[int(pid)] = role
    return out


def final_roles(suggestions: list[RoleSuggestion], imported: dict[int, str] | None,
                overrides: dict[int, str] | None) -> dict[int, str]:
    roles = {s.person_id: s.role for s in suggestions}
    roles.update(imported or {})
    roles.update(overrides or {})
    return roles


def export_roles_json(roles: dict[int, str], names: dict[int, str] | None = None) -> dict:
    persons = []
    for pid in sorted(roles):
        e = {"person_id": pid, "role": roles[pid]}
        if names and pid in names:
            e["name"] = names[pid]
        persons.append(e)
    return {"persons": persons}
