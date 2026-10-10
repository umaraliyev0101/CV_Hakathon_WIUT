import copy

from sim import gt_events


def test_greet_fast_produces_one_greet(cfg, timeline, presets):
    ev = gt_events.events_by_rule(timeline, cfg, presets["short_clip"])
    greets = [e for e in ev if e["type"] == "greet"]
    assert len(greets) == 1
    scripted = [e for e in timeline["events_scripted"] if e["type"] == "greet"][0]
    assert (greets[0]["staff"], greets[0]["customer"]) == (scripted["staff"], scripted["customer"])
    assert abs(greets[0]["t"] - scripted["t"]) <= 1.0


def test_no_greet_when_far(cfg, timeline, presets):
    p = dict(presets["short_clip"], greet_dist_m=0.5, engage_dist_m=0.5)
    assert [e for e in gt_events.events_by_rule(timeline, cfg, p) if e["type"] == "greet"] == []


def test_no_greet_when_too_short(cfg, timeline, presets):
    p = dict(presets["short_clip"], greet_min_s=cfg["duration_s"] + 1)
    assert [e for e in gt_events.events_by_rule(timeline, cfg, p) if e["type"] == "greet"] == []


def _synthetic(frames_by_agent, roles, fps=10):
    return {"fps": fps, "agents": {str(k): {"role": roles[k], "frames": v} for k, v in frames_by_agent.items()}}


def _st(x, y, yaw, present=True):
    return {"pos_m": [x, y], "yaw_deg": yaw, "present": present, "rendered": present}


def test_walkout_rule_on_synthetic_track(presets):
    p = presets["short_clip"]
    fps = 10
    staff = [_st(0, 0, 0)] * 300
    cust = [_st(10, 10, 0)] * 200 + [_st(10, -1, 0, present=False)] * 100
    ev = gt_events.events_by_rule(_synthetic({1: staff, 2: cust}, {1: "staff", 2: "customer"}, fps), {}, p)
    w = [e for e in ev if e["type"] == "walkout"]
    assert w == [{"type": "walkout", "customer": 2, "t": 20.0}]
    assert not any(e["type"] == "wait_alert" for e in ev)      # gone (20 s) before the 30 s wait alert


def test_handoff_and_party_propagation(presets):
    p = presets["short_clip"]
    fps = 10
    n = 400
    c1 = [_st(5, 5, 0)] * n
    c2 = [_st(5.8, 5, 0)] * n           # party member 0.8 m away for the whole clip
    s1 = [_st(0, 0, 0)] * 50 + [_st(3.8, 5, 0)] * 150 + [_st(0, 0, 180)] * 200      # faces +x toward c1
    s2 = [_st(9, 9, 0)] * 180 + [_st(5, 6.2, -90)] * 220                             # faces -y toward c1
    tl = _synthetic({1: s1, 2: s2, 3: c1, 4: c2}, {1: "staff", 2: "staff", 3: "customer", 4: "customer"}, fps)
    ev = gt_events.events_by_rule(tl, {}, p)
    greets = sorted((e["customer"], e["staff"], e["t"]) for e in ev if e["type"] == "greet")
    assert greets == [(3, 1, 5.0), (4, 1, 5.0)]
    hand = sorted((e["customer"], e["from_staff"], e["to_staff"], e["t"]) for e in ev if e["type"] == "handoff")
    assert hand == [(3, 1, 2, 18.0), (4, 1, 2, 18.0)]
    assert [e["customers"] for e in ev if e["type"] == "party"] == [[3, 4]]
    assert not any(e["type"] == "walkout" for e in ev)


def test_match_events_requires_same_subject():
    s = [{"type": "greet", "staff": 1, "customer": 2, "t": 5.0}]
    r = [{"type": "greet", "staff": 3, "customer": 2, "t": 5.0}]
    un_s, un_r = gt_events.match_events(s, r, 1.0, ("greet",))
    assert un_s == s and un_r == r
    r2 = copy.deepcopy(s)
    r2[0]["t"] = 5.9
    assert gt_events.match_events(s, r2, 1.0, ("greet",)) == ([], [])
