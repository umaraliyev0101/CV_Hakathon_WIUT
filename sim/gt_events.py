"""A's event rules applied to true 3D states (pure).

greet: staff within greet_dist_m of a customer, the staff member facing the customer within
greet_facing_deg, sustained >= greet_min_s. Event time = start of the first qualifying run
(per customer, first staff member to qualify). Thresholds come from configs/presets.json.
"""
import math


def _ang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _runs(flags):
    start = None
    for i, f in enumerate(flags + [False]):
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


def events_by_rule(timeline, cfg, preset):
    fps = timeline["fps"]
    min_frames = int(math.ceil(preset["greet_min_s"] * fps - 1e-9))
    agents = timeline["agents"]
    staff = [k for k, a in agents.items() if a["role"] == "staff"]
    custs = [k for k, a in agents.items() if a["role"] == "customer"]
    events = []
    for ck in custs:
        best = None
        for sk in staff:
            flags = [greet_condition(s, c, preset)
                     for s, c in zip(agents[sk]["frames"], agents[ck]["frames"])]
            for a, b in _runs(flags):
                if b - a >= min_frames:
                    if best is None or a < best[0]:
                        best = (a, sk)
                    break
        if best is not None:
            events.append({"type": "greet", "staff": int(best[1]), "customer": int(ck),
                           "t": round(best[0] / fps, 6)})
    return sorted(events, key=lambda e: e["t"])


def match_events(scripted, by_rule, tol_s, types=("greet",)):
    """Returns (unmatched_scripted, unmatched_rule) lists, greedy nearest-in-time per type."""
    un_s, used = [], set()
    for e in scripted:
        if e["type"] not in types:
            continue
        cands = [(abs(r["t"] - e["t"]), i) for i, r in enumerate(by_rule)
                 if i not in used and r["type"] == e["type"]
                 and r.get("customer") == e.get("customer") and abs(r["t"] - e["t"]) <= tol_s]
        if cands:
            used.add(min(cands)[1])
        else:
            un_s.append(e)
    un_r = [r for i, r in enumerate(by_rule) if r["type"] in types and i not in used]
    return un_s, un_r
