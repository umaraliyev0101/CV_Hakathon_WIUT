from sim import gt_events


def test_greet_fast_produces_one_greet(cfg, timeline, presets):
    ev = gt_events.events_by_rule(timeline, cfg, presets["short_clip"])
    greets = [e for e in ev if e["type"] == "greet"]
    assert len(greets) == 1
    g = greets[0]
    sc = cfg["script"]
    assert (g["staff"], g["customer"]) == (sc["staff"], sc["customer"])
    scripted = timeline["events_scripted"][0]
    assert abs(g["t"] - scripted["t"]) <= 1.0


def test_no_greet_when_far(cfg, timeline, presets):
    p = dict(presets["short_clip"], greet_dist_m=0.5)
    assert gt_events.events_by_rule(timeline, cfg, p) == []


def test_no_greet_when_too_short(cfg, timeline, presets):
    p = dict(presets["short_clip"], greet_min_s=cfg["duration_s"] + 1)
    assert gt_events.events_by_rule(timeline, cfg, p) == []
