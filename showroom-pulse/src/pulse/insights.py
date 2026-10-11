"""Rule-based recommendations R1-R7 (spec 4.7): pure functions over Stats and episodes."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .config import Config
from .datatypes import InterestEpisode, Insight, Stats

MAX_MOMENTS = 5
BUCKET_S = 60.0
EN_DASH = "–"


def confidence(n: int, min_sample: int) -> str:
    if n < min_sample:
        return "insufficient"
    if n < 2 * min_sample:
        return "low"
    if n < 4 * min_sample:
        return "medium"
    return "high"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _int(v: Any) -> int:
    f = _num(v)
    return int(f) if f is not None else 0


def _mmss(t: float) -> str:
    s = int(round(t))
    return f"{s // 60:02d}:{s % 60:02d}"


def _fmt_window(w: tuple[float, float]) -> str:
    return f"{_mmss(w[0])}{EN_DASH}{_mmss(w[1])}"


def _moments(eps: list[InterestEpisode]) -> list[dict[str, Any]]:
    return [{"t": e.t_start, "person_id": e.person_id, "zone_id": e.zone_id}
            for e in sorted(eps, key=lambda e: e.t_start)[:MAX_MOMENTS]]


def _make(cfg: Config, rule: str, title: str, evidence: dict, action: str, cars: list[str],
          window, sample: int, moments=()) -> Insight:
    return Insight(rule=rule, title=title, evidence=evidence, action=action, cars=cars,
                   window=window, confidence=confidence(sample, cfg.preset.min_sample),
                   sample=sample, moments=list(moments), demo_only=cfg.preset.demo_only)


def _car_eps(episodes: list[InterestEpisode]) -> dict[str, list[InterestEpisode]]:
    by: dict[str, list[InterestEpisode]] = defaultdict(list)
    for e in episodes:
        by[e.zone_id].append(e)
    return by


def _names(stats: Stats) -> dict[str, str]:
    return {c["zone_id"]: (c.get("name") or c["zone_id"]) for c in stats.per_car if "zone_id" in c}


def _r1(stats, episodes, cfg, names):
    out = []
    for zid, eps in _car_eps(episodes).items():
        unserved = [e for e in eps if e.unserved]
        u, m = len(unserved), len(eps)
        if not (u >= 3 or (m > 0 and u / m >= 0.4)):
            continue
        buckets: dict[int, int] = defaultdict(int)
        for e in unserved:
            buckets[int(math.floor(e.t_start / BUCKET_S))] += 1
        best = min(b for b, n in buckets.items() if n == max(buckets.values()))
        window = (best * BUCKET_S, (best + 1) * BUCKET_S)
        name = names.get(zid, zid)
        out.append((name, _make(
            cfg, "R1", f"Unserved hotspot at {name}", {"unserved": u, "deep_interest": m,
                                                      "share": round(u / m, 3) if m else None},
            f"Station a salesperson near {name} during {_fmt_window(window)}; "
            f"{u} of {m} interested customers got no contact.",
            [zid], window, m, _moments(unserved))))
    return out


def _rates(stats, key):
    return [(c, _num(c.get(key))) for c in stats.per_car if _num(c.get(key)) is not None]


def _r2(stats, episodes, cfg, names):
    stop = {c["zone_id"]: _num(c.get("stop_rate")) for c in stats.per_car}
    dwell = {c["zone_id"]: _num(c.get("median_dwell")) for c in stats.per_car}
    with_dwell = [z for z in stop if dwell[z] is not None and stop[z] is not None]
    if len(with_dwell) < 4:
        return []
    q3 = float(np.percentile([stop[z] for z in with_dwell], 75))
    q1 = float(np.percentile([dwell[z] for z in with_dwell], 25))
    by = _car_eps(episodes)
    sample = _int(stats.store.get("visitors"))
    out = []
    for z in with_dwell:
        if stop[z] > 0 and stop[z] >= q3 and dwell[z] <= q1:
            name = names.get(z, z)
            out.append((name, _make(
                cfg, "R2", f"Glance-and-go at {name}",
                {"stop_rate": stop[z], "median_dwell": dwell[z], "stop_rate_q3": q3, "dwell_q1": q1},
                f"{name} draws people in but doesn't hold them. Check the price board or info display.",
                [z], None, sample, _moments(by.get(z, [])))))
    return out


def _r3(stats, episodes, cfg, names):
    rates = [_num(c.get("stop_rate")) for c in stats.per_car]
    rates = [r for r in rates if r is not None]
    if not rates:
        return []
    med = float(np.median(rates))
    if med <= 0.30:
        return []
    sample = _int(stats.store.get("visitors"))
    out = []
    for c in stats.per_car:
        r = _num(c.get("stop_rate"))
        if r is not None and r < 0.10:
            name = names.get(c["zone_id"], c["zone_id"])
            out.append((name, _make(
                cfg, "R3", f"Dead zone at {name}", {"stop_rate": r, "store_median_stop_rate": med},
                f"{name} is rarely visited. Consider moving it, lighting or signage.",
                [c["zone_id"]], None, sample)))
    return out


def _r4(stats, episodes, cfg, names):
    t = _num(stats.store.get("median_time_to_assist"))
    if t is None or t <= cfg.preset.alert_after * 2:
        return []
    return [("", _make(
        cfg, "R4", "Long unattended waits",
        {"median_time_to_assist": t, "threshold": cfg.preset.alert_after * 2},
        f"Customers wait {t:.0f} s on average before contact. Review floor staffing. "
        f"Showroom Coach has per-salesperson detail.",
        [], None, _int(stats.store.get("assisted_customers"))))]


def _r5(stats, episodes, cfg, names):
    best: tuple[float, float, float] | None = None   # (length, t_first, t_last)
    run: list[tuple[float, float]] = []

    def close():
        nonlocal best
        if run:
            ln = run[-1][0] - run[0][0]
            if ln >= cfg.crowding_time_s and (best is None or ln > best[0]):
                best = (ln, run[0][0], run[-1][0], max(r for _, r in run))
    for p in stats.crowding:
        t, r = _num(p[0]), _num(p[1])
        if t is not None and r is not None and r > cfg.crowding_ratio:
            run.append((t, r))
        else:
            close()
            run = []
    close()
    if best is None:
        return []
    _, t0, t1, k = best
    return [("", _make(
        cfg, "R5", "Crowding at peak", {"max_ratio": k, "duration_s": t1 - t0},
        f"Peak at {_fmt_window((t0, t1))}: {k:.1f} customers per staff member.",
        [], (t0, t1), _int(stats.store.get("visitors"))))]


def _r6(stats, episodes, cfg, names):
    s = stats.store
    deep = _int(s.get("deep_interest_visitors"))
    if not s.get("has_desk") or deep <= 0:
        return []
    reached = _int(s.get("deep_interest_reached_desk"))
    missed = deep - reached
    if missed / deep < 0.5:
        return []
    return [("", _make(
        cfg, "R6", "Desk drop-off", {"deep_interest": deep, "reached_desk": reached,
                                      "share_missed": round(missed / deep, 3)},
        "Interest isn't turning into desk conversations. Review the follow-up process.",
        [], None, deep))]


def _r7(stats, episodes, cfg, names):
    s = stats.store
    g, so = _num(s.get("group_assisted_share")), _num(s.get("solo_assisted_share"))
    if g is None or so is None or so - g < 0.20 - 1e-9:
        return []
    return [("", _make(
        cfg, "R7", "Groups get less attention",
        {"group_assisted_share": g, "solo_assisted_share": so, "gap": round(so - g, 3)},
        "Groups get less attention. Consider a family-friendly approach.",
        [], None, min(_int(s.get("groups")), _int(s.get("n_solo")))))]


def recommend(stats: Stats, episodes: list[InterestEpisode], cfg: Config) -> list[Insight]:
    names = _names(stats)
    found: list[tuple[str, str, Insight]] = []
    for rule in (_r1, _r2, _r3, _r4, _r5, _r6, _r7):
        for name, ins in rule(stats, episodes, cfg, names):
            found.append((ins.rule, name, ins))
    found.sort(key=lambda x: (x[0], x[1]))
    return [i for _, _, i in found]
