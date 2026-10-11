"""Transparent staff-versus-customer suggestion (spec section 4.4).

Pure numpy. Four hand-set features feed one logistic score, and the suggestion carries the plain-language
reasons behind it. The user's correction always wins elsewhere; this module only suggests.

Inputs are trajectories ``pid -> time-sorted [(t, x, z)]`` in metres on the floor plane. All trajectories come
from the same sampled frames, so a timestamp shared by several people is the same float; timestamps within
``_TOL`` seconds are still treated as one frame. People may be missing in some frames (live mode).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np

from .config import RoleWeights
from .types import RoleSuggestion

_TOL = 1e-6  # timestamps closer than this are the same sampled frame
_EPS = 1e-9  # float slack on distance / duration thresholds
_PRESENCE_MIN_CLIP_S = 60.0  # presence says nothing about a short clip
_REASON_MIN_CONTRIB = 0.05  # a staff reason must push the score up by at least this much
_MAX_REASONS = 3
_FEATURE_ORDER = ("outfit_peers", "contacts", "presence", "initiated")  # also the tie-break order


# --------------------------------------------------------------------------------------------------
# Interaction features
# --------------------------------------------------------------------------------------------------

def _samples(samples) -> np.ndarray:
    """(n, 3) float array [t, x, z], time-sorted, rows with non-finite values dropped."""
    a = np.asarray(samples, dtype=float).reshape(-1, 3)
    if len(a):
        a = a[np.isfinite(a).all(axis=1)]
    if len(a) > 1 and np.any(np.diff(a[:, 0]) < 0):
        a = a[np.argsort(a[:, 0], kind="stable")]
    return a


def _approach_m(arr: np.ndarray, i: int, other_xz: np.ndarray, window: float) -> float:
    """Own displacement over [t_i - window, t_i] projected on the unit vector toward ``other_xz``."""
    ts = arr[:, 0]
    j = int(np.searchsorted(ts, ts[i] - window - _TOL, side="left"))
    to_other = other_xz - arr[i, 1:]
    dist = float(np.hypot(to_other[0], to_other[1]))
    if dist < _EPS:
        return 0.0
    return float((arr[i, 1:] - arr[j, 1:]) @ to_other / dist)


def interaction_features(traj: dict[int, list[tuple[float, float, float]]], contact_dist: float = 2.0,
                         contact_min: float = 5.0, approach_window: float = 3.0,
                         min_approach_m: float = 0.5) -> dict[int, dict]:
    """Per person: ``{"contacts": int, "initiated": int, "partners": [pid, ...]}``.

    Two people are close in a frame when their distance is at most ``contact_dist``. A pair becomes a
    *contact* once its cumulative close time reaches ``contact_min`` seconds, each close frame counting the
    median sampling step of the shared frame grid (so gaps in someone's track do not inflate it).

    Who initiated: at the pair's first close frame ``tc`` each person's displacement toward the other (over
    ``[tc - approach_window, tc]``, projected on the direction to the other's position at ``tc``) is
    compared. The larger one initiated if it leads by at least ``min_approach_m`` and is a real approach;
    a mutual approach (both walk up, similar distance) gives the initiative to nobody.

    Cost is O(total samples x people in view): one pass over the shared frames, no per-pair rescans.
    """
    arrays = {pid: _samples(s) for pid, s in traj.items()}
    out: dict[int, dict] = {pid: {"contacts": 0, "initiated": 0, "partners": []} for pid in arrays}
    pids = [pid for pid, a in arrays.items() if len(a)]
    if len(pids) < 2:
        return out

    # One flat, time-ordered table of every sample: (time, x, z, person index, index in that person's track).
    t = np.concatenate([arrays[p][:, 0] for p in pids])
    xz = np.concatenate([arrays[p][:, 1:] for p in pids])
    who = np.concatenate([np.full(len(arrays[p]), i) for i, p in enumerate(pids)])
    loc = np.concatenate([np.arange(len(arrays[p])) for p in pids])
    order = np.argsort(t, kind="stable")
    t, xz, who, loc = t[order], xz[order], who[order], loc[order]

    cut = np.flatnonzero(np.diff(t) > _TOL) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [len(t)]))
    frame_t = t[starts]
    step = float(np.median(np.diff(frame_t))) if len(frame_t) > 1 else 0.0

    close_n: dict[tuple[int, int], int] = {}
    first_close: dict[tuple[int, int], tuple[int, int]] = {}  # pair -> (local index of a, of b) at first close
    for s, e in zip(starts, stops):
        if e - s < 2:
            continue
        w, p, li = who[s:e], xz[s:e], loc[s:e]
        uniq, first = np.unique(w, return_index=True)
        if len(uniq) < len(w):  # the same person twice in one frame (duplicate sample): keep the first
            w, p, li = w[first], p[first], li[first]
            if len(w) < 2:
                continue
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        ii, jj = np.nonzero(np.triu(d <= contact_dist, 1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b = int(w[i]), int(w[j])
            if a > b:
                a, b, i, j = b, a, j, i
            key = (a, b)
            if key not in close_n:
                close_n[key] = 0
                first_close[key] = (int(li[i]), int(li[j]))
            close_n[key] += 1

    for (a, b), n in close_n.items():
        if n * step + _EPS < contact_min:
            continue
        pa, pb = pids[a], pids[b]
        out[pa]["partners"].append(pb)
        out[pb]["partners"].append(pa)
        ia, ib = first_close[(a, b)]
        xa, xb = arrays[pa], arrays[pb]
        move_a = _approach_m(xa, ia, xb[ib, 1:], approach_window)
        move_b = _approach_m(xb, ib, xa[ia, 1:], approach_window)
        if move_a - move_b >= min_approach_m - _EPS and move_a > 0:
            out[pa]["initiated"] += 1
        elif move_b - move_a >= min_approach_m - _EPS and move_b > 0:
            out[pb]["initiated"] += 1

    for v in out.values():
        v["partners"].sort()
        v["contacts"] = len(v["partners"])
    return out


def presence(traj: dict[int, list[tuple[float, float, float]]], frame_times: Sequence[float]) -> dict[int, float]:
    """Share of ``frame_times`` at which each person has a sample (matched within 1e-6 s)."""
    ft = np.asarray(frame_times, dtype=float).ravel()
    ft = ft[np.isfinite(ft)]
    out: dict[int, float] = {}
    for pid, samples in traj.items():
        ts = np.sort(_samples(samples)[:, 0])
        if not len(ft) or not len(ts):
            out[pid] = 0.0
            continue
        idx = np.searchsorted(ts, ft - _TOL, side="left")
        have = idx < len(ts)
        hit = np.zeros(len(ft), dtype=bool)
        hit[have] = ts[idx[have]] <= ft[have] + _TOL
        out[pid] = float(hit.mean())
    return out


def outfit_peers(descr: dict[int, np.ndarray], thr: float) -> dict[int, int]:
    """For each person, how many *other* people have a torso descriptor with cosine similarity >= ``thr``.

    A zero or non-finite descriptor matches nobody.
    """
    pids = list(descr)
    if not pids:
        return {}
    m = np.stack([np.asarray(descr[p], dtype=float).ravel() for p in pids])
    norms = np.linalg.norm(m, axis=1)
    ok = np.isfinite(norms) & (norms > 1e-12)
    unit = np.zeros_like(m)
    unit[ok] = m[ok] / norms[ok, None]
    match = (unit @ unit.T >= thr - _EPS) & ok[:, None] & ok[None, :]
    np.fill_diagonal(match, False)
    counts = match.sum(axis=1)
    return {p: int(c) for p, c in zip(pids, counts)}


# --------------------------------------------------------------------------------------------------
# Score and reasons
# --------------------------------------------------------------------------------------------------

def _number(value, default: float | None) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _reason(feature: str, value: float) -> str:
    if feature == "outfit_peers":
        n = int(round(value))
        return "outfit unlike anyone else" if n <= 0 else f"same outfit as {n} {_plural(n, 'other', 'others')}"
    if feature == "contacts":
        n = int(round(value))
        return f"talked to {n} {_plural(n, 'person', 'people')}"
    if feature == "presence":
        return f"on screen {math.floor(100.0 * value + 0.5)}% of the clip"
    n = int(round(value))
    return "never approached anyone first" if n <= 0 else f"approached {n} {_plural(n, 'person', 'people')} first"


def _scalar_features(feats: dict) -> dict[str, float]:
    """Flat copy of the numeric (finite) entries, as plain Python ints and floats."""
    flat: dict[str, float] = {}
    for k, v in feats.items():
        if isinstance(v, (bool, np.bool_)):
            flat[str(k)] = int(v)
        elif isinstance(v, numbers.Integral):
            flat[str(k)] = int(v)
        elif isinstance(v, numbers.Real) and math.isfinite(float(v)):
            flat[str(k)] = float(v)
    return flat


def suggest_roles(features: dict[int, dict], clip_len: float, weights: RoleWeights = RoleWeights()) -> list[RoleSuggestion]:
    """Staff / customer / unsure suggestion per person with up to three plain-language reasons.

    ``features[pid]`` keys: ``outfit_peers``, ``contacts``, ``presence``, ``initiated`` (others are ignored
    for the score but copied into ``RoleSuggestion.features`` when numeric). A missing count counts as 0;
    a missing presence is neutral (0.5) and is never given as a reason.

    z = bias + sum(w * f) with f_outfit = min(n,3)/3, f_contacts = min(n,4)/4,
    f_presence = (p - 0.5) * 2 (only when ``clip_len`` >= 60 s, else 0), f_init = min(n,3)/3;
    P(staff) = 1 / (1 + e^-z). Staff if P >= staff_thr, customer if P <= customer_thr, else unsure.
    Results are ordered by pid.
    """
    use_presence = _number(clip_len, 0.0) >= _PRESENCE_MIN_CLIP_S
    result: list[RoleSuggestion] = []
    for pid, feats in sorted(features.items(), key=lambda kv: kv[0]):
        n_outfit = max(0.0, _number(feats.get("outfit_peers"), 0.0))
        n_contacts = max(0.0, _number(feats.get("contacts"), 0.0))
        n_init = max(0.0, _number(feats.get("initiated"), 0.0))
        p_raw = _number(feats.get("presence"), None)
        p = 0.5 if p_raw is None else min(1.0, max(0.0, p_raw))

        contrib = {
            "outfit_peers": weights.outfit_peers * min(n_outfit, 3.0) / 3.0,
            "contacts": weights.contacts * min(n_contacts, 4.0) / 4.0,
            "presence": weights.presence * ((p - 0.5) * 2.0 if use_presence else 0.0),
            "initiated": weights.initiator * min(n_init, 3.0) / 3.0,
        }
        shown = {"outfit_peers": n_outfit, "contacts": n_contacts, "presence": p, "initiated": n_init}
        conf = _sigmoid(weights.bias + sum(contrib.values()))
        role = "staff" if conf >= weights.staff_thr else "customer" if conf <= weights.customer_thr else "unsure"

        usable = [k for k in _FEATURE_ORDER if k != "presence" or (use_presence and p_raw is not None)]
        if role == "staff":
            ranked = sorted((k for k in usable if contrib[k] > _REASON_MIN_CONTRIB), key=lambda k: -contrib[k])
        elif role == "customer":
            ranked = sorted(usable, key=lambda k: contrib[k])
        else:
            ranked = sorted(usable, key=lambda k: -abs(contrib[k]))
        if not ranked:  # staff on the strength of the bias alone: still say what mattered most
            ranked = sorted(usable, key=lambda k: -abs(contrib[k]))
        reasons = [_reason(k, shown[k]) for k in ranked[:_MAX_REASONS]]

        result.append(RoleSuggestion(pid=int(pid), role=role, confidence=conf, reasons=reasons,
                                     features=_scalar_features(feats)))
    return result
