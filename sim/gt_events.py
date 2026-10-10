"""A's event rules applied to true 3D states (pure). Thresholds come from configs/presets.json.

A's spec (salespeople_prompts/2026-10-11-showroom-coach-design.md) is not in the repo, so the
rules below are the documented fallback (FLAG in presets.json "_source"). Event times are the
ONSET of the qualifying run, not the moment it is confirmed.

greet       staff within greet_dist_m of a customer, staff facing the customer within
            greet_facing_deg, sustained >= greet_min_s. First qualifying run per customer.
handoff     a later qualifying greet-run by a different staff member than the one currently
            engaged with the customer (from_staff -> to_staff).
engagement  staff-customer distance <= engage_dist_m (no facing test), gaps <= engage_gap_s
            merged, total span >= engage_min_s. Reports [t, t_end].
party       two customers both present and within party_dist_m for >= party_min_s, the run
            starting within party_window_s of the later one's entry. Greeting / handing off /
            engaging one member applies to every member (earliest onset wins).
walkout     a customer who was present >= walkout_min_present_s, was never greeted (after party
            propagation) and left the room; t = first frame no longer present.
wait_alert  a customer still present and not greeted wait_alert_s after entering;
            t = entry + wait_alert_s. Rule-only (no scripted counterpart by design).
"""
import math


def _ang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _runs(flags):
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            yield start, i
            start = None


def greet_condition(s, c, preset):
    if not (s["present"] and c["present"]):
        return False
    d = math.dist(s["pos_m"], c["pos_m"])
    if d > preset["greet_dist_m"]:
        return False
    bearing = math.degrees(math.atan2(c["pos_m"][1] - s["pos_m"][1], c["pos_m"][0] - s["pos_m"][0]))
    return _ang(s["yaw_deg"], bearing) <= preset["greet_facing_deg"]


def _near(a, b, dist):
    return a["present"] and b["present"] and math.dist(a["pos_m"], b["pos_m"]) <= dist


def _merge(runs, gap):
    out = []
    for a, b in runs:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def parties_by_rule(agents, fps, preset):
    custs = sorted(int(k) for k, a in agents.items() if a["role"] == "customer")
    first = {c: next((i for i, s in enumerate(agents[str(c)]["frames"]) if s["present"]), None) for c in custs}
    min_f = int(math.ceil(preset["party_min_s"] * fps - 1e-9))
    win = preset["party_window_s"] * fps
    pairs = []
    for i, a in enumerate(custs):
        for b in custs[i + 1:]:
            if first[a] is None or first[b] is None:
                continue
            flags = [_near(x, y, preset["party_dist_m"])
                     for x, y in zip(agents[str(a)]["frames"], agents[str(b)]["frames"])]
            for s, e in _runs(flags):
                if e - s >= min_f and s - max(first[a], first[b]) <= win:
                    pairs.append((a, b, s))
                    break
    # connected components
    groups = []
    for a, b, s in pairs:
        hit = [g for g in groups if a in g["m"] or b in g["m"]]
        g = {"m": {a, b}, "s": s}
        for h in hit:
            g["m"] |= h["m"]
            g["s"] = min(g["s"], h["s"])
            groups.remove(h)
        groups.append(g)
    return [(sorted(g["m"]), g["s"]) for g in groups]


def events_by_rule(timeline, cfg, preset):
    fps = timeline["fps"]
    agents = timeline["agents"]
    t = lambda f: round(f / fps, 6)  # noqa: E731
    staff = sorted(int(k) for k, a in agents.items() if a["role"] == "staff")
    custs = sorted(int(k) for k, a in agents.items() if a["role"] == "customer")
    F = {k: agents[str(k)]["frames"] for k in staff + custs}
    g_min = int(math.ceil(preset["greet_min_s"] * fps - 1e-9))
    e_min = int(math.ceil(preset["engage_min_s"] * fps - 1e-9))
    e_gap = int(round(preset["engage_gap_s"] * fps))

    # per customer: greet runs, engagements
    greet_seq, engage = {}, {}
    for c in custs:
        runs = []
        for s in staff:
            flags = [greet_condition(a, b, preset) for a, b in zip(F[s], F[c])]
            for a, b in _runs(flags):
                if b - a >= g_min:
                    runs.append((a, s))
            near = [_near(a, b, preset["engage_dist_m"]) for a, b in zip(F[s], F[c])]
            for a, b in _merge(list(_runs(near)), e_gap):
                if b - a >= e_min:
                    engage.setdefault(c, []).append((a, b, s))
        greet_seq[c] = sorted(runs)

    parties = parties_by_rule(agents, fps, preset)
    party_of = {c: members for members, _ in parties for c in members}

    # greet/handoff sequence per party (or single customer): merge members' runs by onset
    greets, handoffs = {}, {}
    done = set()
    for c in custs:
        members = party_of.get(c, [c])
        key = tuple(members)
        if key in done:
            continue
        done.add(key)
        seq = sorted(r for m in members for r in greet_seq[m])
        cur = None
        for onset, s in seq:
            if cur is None:
                for m in members:
                    greets[m] = (onset, s)
                cur = s
            elif s != cur:
                for m in members:
                    handoffs.setdefault(m, []).append((onset, cur, s))
                cur = s

    events = []
    for c, (f0, s) in greets.items():
        events.append({"type": "greet", "staff": s, "customer": c, "t": t(f0)})
    for c, lst in handoffs.items():
        for f0, a, b in lst:
            events.append({"type": "handoff", "customer": c, "from_staff": a, "to_staff": b, "t": t(f0)})
    # engagements, propagated within parties (earliest-onset member interval per staff)
    done = set()
    for c in custs:
        members = party_of.get(c, [c])
        key = tuple(members)
        if key in done:
            continue
        done.add(key)
        for s in staff:
            ivs = sorted(iv for m in members for iv in engage.get(m, []) if iv[2] == s)
            merged = _merge([(a, b) for a, b, _ in ivs], e_gap)
            for a, b in merged:
                for m in members:
                    events.append({"type": "engagement", "staff": s, "customer": m, "t": t(a), "t_end": t(b)})
    for members, s in parties:
        events.append({"type": "party", "customers": members, "t": t(s)})
    for c in custs:
        pres = [s["present"] for s in F[c]]
        if not any(pres):
            continue
        first = pres.index(True)
        last = len(pres) - 1 - pres[::-1].index(True)
        if c not in greets and last + 1 < len(pres) and (last + 1 - first) >= preset["walkout_min_present_s"] * fps:
            events.append({"type": "walkout", "customer": c, "t": t(last + 1)})
        wf = first + int(round(preset["wait_alert_s"] * fps))
        greeted_by = greets[c][0] if c in greets else None
        if wf < len(pres) and pres[wf] and (greeted_by is None or greeted_by > wf):
            events.append({"type": "wait_alert", "customer": c, "t": t(wf)})
    return sorted(events, key=lambda e: (e["t"], e["type"], str(e.get("customer", e.get("customers")))))


def _same_subject(e, r):
    if e["type"] == "party":
        return sorted(e["customers"]) == sorted(r["customers"])
    if e.get("customer") != r.get("customer"):
        return False
    if e["type"] in ("greet", "engagement"):
        return e.get("staff") == r.get("staff")
    if e["type"] == "handoff":
        return (e.get("from_staff"), e.get("to_staff")) == (r.get("from_staff"), r.get("to_staff"))
    return True


def match_events(scripted, by_rule, tol_s, types=("greet",)):
    """Returns (unmatched_scripted, unmatched_rule) lists, greedy nearest-in-time per type and subject."""
    un_s, used = [], set()
    for e in scripted:
        if e["type"] not in types:
            continue
        cands = [(abs(r["t"] - e["t"]), i) for i, r in enumerate(by_rule)
                 if i not in used and r["type"] == e["type"] and _same_subject(e, r)
                 and abs(r["t"] - e["t"]) <= tol_s]
        if cands:
            used.add(min(cands)[1])
        else:
            un_s.append(e)
    un_r = [r for i, r in enumerate(by_rule) if r["type"] in types and i not in used]
    return un_s, un_r
