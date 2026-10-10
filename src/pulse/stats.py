"""Timeline + journeys + episodes -> Stats (spec 4.5, 4.8): store KPIs, per-car table, occupancy,
funnel, groups, heatmap, paths, per-person visit records."""
from __future__ import annotations
import math
from collections import defaultdict
from statistics import median

from pulse.config import Config
from pulse.datatypes import (InterestEpisode, Person, Sample, STORE_KEYS_DEFAULTS, Stats, Timeline,
                             Zone, ZoneVisit)
from pulse.geometry import distance_m
from pulse.interest import AssistInfo
from pulse.journey import sequence
from pulse.timeline import already_present, at_edge
from pulse.zones import zone_of


def _share(n: int, d: int) -> float | None:
    return n / d if d else None


def _med(xs: list[float]) -> float | None:
    return float(median(xs)) if xs else None


def _find(parent: dict[int, int], a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _same_group(a: list[Sample], b: list[Sample], cfg: Config) -> bool:
    if abs(a[0].t - b[0].t) > cfg.group_arrival_s:
        return False
    t0 = max(a[0].t, b[0].t)
    pb = {s.t: s for s in b if s.pos_m is not None}
    n = close = 0
    for s in a:
        o = pb.get(s.t)
        if s.pos_m is None or o is None or not t0 <= s.t <= t0 + cfg.group_window_s:
            continue
        n += 1
        close += distance_m(s.pos_m, o.pos_m) <= cfg.group_dist_m
    return n >= 3 and close / n > 0.5


def _groups(ids: list[int], tl: Timeline, cfg: Config) -> list[list[int]]:
    ids = [i for i in ids if tl.get(i)]
    parent = {i: i for i in ids}
    for x, a in enumerate(ids):
        for b in ids[x + 1:]:
            if _same_group(tl[a], tl[b], cfg):
                parent[_find(parent, a)] = _find(parent, b)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in ids:
        comps[_find(parent, i)].append(i)
    return sorted(sorted(c) for c in comps.values() if len(c) >= 2)


def _heatmap(samples: list[Sample], w: int, h: int) -> dict:
    counts = [[0] * w for _ in range(h)]
    for s in samples:
        x, y = s.foot
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        counts[min(max(int(y * h), 0), h - 1)][min(max(int(x * w), 0), w - 1)] += 1
    return {"w": w, "h": h, "counts": counts}


def _paths(tl: Timeline, ids: list[int]) -> list[dict]:
    out = []
    for pid in sorted(ids, key=lambda i: -len(tl[i]))[:10]:
        pts = [list(s.foot) for s in tl[pid]]
        if len(pts) > 50:
            pts = [pts[round(k * (len(pts) - 1) / 49)] for k in range(50)]
        out.append({"person_id": pid, "points": pts})
    return out


def _zone_rows(zones: list[Zone], visits: list[ZoneVisit]) -> dict[str, dict]:
    by_zone: dict[str, list[ZoneVisit]] = defaultdict(list)
    for v in visits:
        by_zone[v.zone_id].append(v)
    rows = {}
    for z in zones:
        vs = by_zone[z.zone_id]
        rows[z.zone_id] = {"zone_id": z.zone_id, "name": z.name, "kind": z.kind,
                           "visitors": len({v.person_id for v in vs}),
                           "median_dwell": _med([v.dwell for v in vs]),
                           "total_dwell": float(sum(v.dwell for v in vs))}
    return rows


def _per_car(zones, visits, episodes, n_cust, rows) -> list[dict]:
    cars = []
    for z in zones:
        if z.kind != "car":
            continue
        vs = [v for v in visits if v.zone_id == z.zone_id]
        per_p: dict[int, int] = defaultdict(int)
        for v in vs:
            per_p[v.person_id] += 1
        eps = [e for e in episodes if e.zone_id == z.zone_id]
        stops = len({v.person_id for v in vs if v.stopped})
        deep = len({e.person_id for e in eps})
        unserved = len({e.person_id for e in eps if e.unserved})
        r = rows[z.zone_id]
        cars.append({**{k: r[k] for k in ("zone_id", "name", "visitors")}, "stops": stops,
                     "stop_rate": stops / n_cust if n_cust else 0.0,
                     "hold_rate": _share(stops, r["visitors"]),
                     "median_dwell": r["median_dwell"], "total_dwell": r["total_dwell"],
                     "repeat_visits": sum(1 for n in per_p.values() if n >= 2),
                     "deep_interest": deep, "unserved": unserved,
                     "unassisted_share": _share(unserved, deep)})
    for rank, c in enumerate(sorted(cars, key=lambda c: (-c["total_dwell"], -c["stops"])), 1):
        c["rank"] = rank
    return sorted(cars, key=lambda c: c["rank"])


def compute(persons: list[Person], journeys: dict[int, list[ZoneVisit]], roles: dict[int, str],
            cfg: Config, *, timeline: Timeline, episodes: list[InterestEpisode],
            assist: dict[int, AssistInfo], zones: list[Zone], duration: float,
            frame_size: tuple[int, int], analysed_share: float = 1.0,
            extra_warnings: list[str] = ()) -> Stats:
    pre = cfg.preset
    role = lambda p: roles.get(p.person_id, "customer")  # noqa: E731
    cust = [p for p in persons if role(p) != "staff"]
    staff = [p for p in persons if role(p) == "staff"]
    cids = {p.person_id for p in cust}
    tl = {pid: sorted(ss, key=lambda s: s.t) for pid, ss in timeline.items()}
    n = len(cust)
    fresh = [p for p in cust if not already_present(p, pre.leave_after)]
    cv = [v for p in cust for v in journeys.get(p.person_id, [])]
    ceps = [e for e in episodes if e.person_id in cids]

    entrance = [z for z in zones if z.kind == "entrance"]
    ent_ids = {z.zone_id for z in entrance}

    def through_entrance(s: Sample) -> bool:
        if entrance:
            return zone_of(s.foot, entrance) in ent_ids
        return at_edge(s, frame_size, cfg.edge_margin)

    entries = sum(1 for p in fresh if tl.get(p.person_id) and through_entrance(tl[p.person_id][0]))
    exits = sum(1 for p in cust if tl.get(p.person_id) and p.last_t <= duration - pre.leave_after
                and through_entrance(tl[p.person_id][-1]))

    groups = _groups(sorted(cids), tl, cfg)
    gmember = {i: k for k, g in enumerate(groups) for i in g}
    asst = lambda pid: assist.get(pid)  # noqa: E731
    is_asst = lambda pid: bool(asst(pid) and asst(pid).assisted)  # noqa: E731
    assisted_ids = {pid for pid in cids if is_asst(pid)}
    browsers = {v.person_id for v in cv if v.stopped}
    deep_ids = {e.person_id for e in ceps}
    desk_ids = {z.zone_id for z in zones if z.kind == "desk"}
    desk_reach = {v.person_id for v in cv if v.zone_id in desk_ids and v.dwell >= pre.desk_time}
    tta = [asst(p.person_id).t_first - p.first_t for p in fresh
           if is_asst(p.person_id) and asst(p.person_id).t_first is not None]
    solo = [i for i in cids if i not in gmember]

    # occupancy / waiting / crowding, sampled at t = 0, 1, 2, ...
    occ, wait, crowd = [], [], []
    for k in range(int(math.floor(duration)) + 1):
        t = float(k)
        here = [p for p in cust if p.first_t <= t <= p.last_t]
        w = sum(1 for p in here if not is_asst(p.person_id)
                or (asst(p.person_id).t_first is not None and t < asst(p.person_id).t_first))
        ns = sum(1 for p in staff if p.first_t <= t <= p.last_t)
        occ.append([t, len(here)])
        wait.append([t, w])
        crowd.append([t, len(here) / ns if ns else -1])
    peak = max(occ, key=lambda r: r[1]) if occ and n else None  # max() keeps the first maximum

    # time buckets
    size = 60.0 if duration < 3600 else 3600.0
    buckets = []
    for k in range(max(1, math.ceil(duration / size))):
        t0, t1 = k * size, (k + 1) * size
        occs = [r[1] for r in occ if t0 <= r[0] < t1]
        buckets.append({
            "t0": t0, "t1": t1, "arrivals": sum(1 for p in fresh if t0 <= p.first_t < t1),
            "mean_occupancy": sum(occs) / len(occs) if occs else 0.0,
            "unserved": sum(1 for e in ceps if e.unserved and t0 <= e.t_start < t1),
            "stops": sum(1 for v in cv if v.stopped and t0 <= v.t_in < t1)})

    funnel = {"entered": 1.0 if n else 0.0}
    for key, cnt in (("browsed", len(browsers)), ("deep_interest", len(deep_ids)),
                     ("assisted", len(assisted_ids)), ("reached_desk", len(desk_reach))):
        funnel[key] = cnt / n if n else 0.0

    store = dict(STORE_KEYS_DEFAULTS)
    store.update({
        "visitors": n, "staff": len(staff), "unsure": sum(1 for p in cust if role(p) == "unsure"),
        "already_present": n - len(fresh), "entries": entries, "exits": exits,
        "groups": len(groups), "n_group_members": len(gmember), "n_solo": len(solo),
        "median_visit_s": _med([p.last_t - p.first_t for p in fresh]),
        "browse_rate": _share(len(browsers), n), "assisted_share": _share(len(assisted_ids), n),
        "assisted_customers": len(assisted_ids), "median_time_to_assist": _med(tta),
        "unserved_episodes": sum(1 for e in ceps if e.unserved),
        "deep_interest_visitors": len(deep_ids), "has_desk": bool(desk_ids),
        "reached_desk": len(desk_reach), "deep_interest_reached_desk": len(deep_ids & desk_reach),
        "peak_occupancy": int(peak[1]) if peak else 0, "peak_t": peak[0] if peak else None,
        "funnel": funnel,
        "group_assisted_share": _share(sum(1 for i in gmember if is_asst(i)), len(gmember)),
        "solo_assisted_share": _share(sum(1 for i in solo if is_asst(i)), len(solo)),
        "duration_s": float(duration), "analysed_share": float(analysed_share)})

    all_samples = [s for ss in tl.values() for s in ss]
    approx = bool(all_samples) and sum(s.dist_method == "box" for s in all_samples) / len(all_samples) > 0.5
    store["distance_approx"] = approx

    rows = _zone_rows(zones, cv)
    cust_tl_ids = [i for i in tl if i in cids and tl[i]]
    warnings = []
    if not persons:
        warnings.append("Nobody detected — camera too far or too dark?")
    if not zones:
        warnings.append("No zones: draw zones to get per-car statistics")
    if approx:
        warnings.append("Distance estimate approximate (box-height fallback for >50% of samples)")
    if analysed_share < 0.99:
        warnings.append(f"Only {int(analysed_share * 100)}% of the clip analysed (camera moving or cut)")
    warnings.extend(extra_warnings)

    visits = []
    for p in cust:
        vs = journeys.get(p.person_id, [])
        a = asst(p.person_id)
        visits.append({
            "person_id": p.person_id, "role": role(p), "first_t": p.first_t, "last_t": p.last_t,
            "already_present": already_present(p, pre.leave_after), "visit_s": p.last_t - p.first_t,
            "zones": sequence(vs),
            "stopped_cars": sorted({v.zone_id for v in vs if v.stopped}),
            "assisted": bool(a and a.assisted), "t_first_assist": a.t_first if a else None,
            "group": gmember.get(p.person_id)})

    return Stats(store=store, per_car=_per_car(zones, cv, ceps, n, rows),
                 per_zone=list(rows.values()), occupancy=occ, waiting=wait, crowding=crowd,
                 time_buckets=buckets,
                 heatmap=_heatmap([s for i in cust_tl_ids for s in tl[i]], cfg.heatmap_w, cfg.heatmap_h),
                 paths=_paths(tl, cust_tl_ids), visits=visits, groups=groups, warnings=warnings)
