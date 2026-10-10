"""Every generated scene: deterministic timeline, valid motion, intended scripted events, and the
rule engine (A's fallback presets) reproduces them within the qa tolerance."""
import math

import pytest

from sim import behaviour, gt_events, layout, schema
from sim.tests.conftest import scene_and_timeline

TYPES = ("greet", "handoff", "walkout", "party", "engagement")


def _ang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_deterministic(scene):
    cfg, tl = scene
    assert behaviour.timeline_hash(behaviour.build_timeline(cfg)) == behaviour.timeline_hash(tl)


def test_frame_counts(scene):
    cfg, tl = scene
    for ag in tl["agents"].values():
        assert len(ag["frames"]) == schema.n_scene_frames(cfg)


def test_speed_and_yaw_rate(scene):
    cfg, tl = scene
    fps = cfg["fps_scene"]
    rate = cfg["navigation"]["turn_rate_dps"]
    for pid, ag in tl["agents"].items():
        hi = schema.person(cfg, int(pid))["speed_mps"][1]
        fr = ag["frames"]
        for a, b in zip(fr[:-1], fr[1:]):
            if a["rendered"] and b["rendered"]:
                assert math.dist(a["pos_m"], b["pos_m"]) * fps <= hi + 1e-6
                assert _ang(a["yaw_deg"], b["yaw_deg"]) <= rate / fps + 1e-6


def test_never_inside_cars_or_each_other(scene):
    cfg, tl = scene
    lay = layout.build_layout(cfg)
    cars = [c["footprint_m"] for c in lay["cars"]]
    ags = list(tl["agents"].values())
    for ag in ags:
        for st in ag["frames"]:
            if st["rendered"]:
                assert layout.dist_to_polygons(st["pos_m"], cars) >= cfg["navigation"]["body_radius_m"] - 1e-6
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            for a, b in zip(ags[i]["frames"], ags[j]["frames"]):
                if a["rendered"] and b["rendered"]:
                    assert math.dist(a["pos_m"], b["pos_m"]) >= 0.5


def test_talk_choreography(scene):
    """During every staff talk: 0.9-1.3 m from the customer, facing them within 20 deg; talks last
    >= greet_min_s + 1 s (both presets' greet minimum is <= 5 s)."""
    cfg, tl = scene
    fps = cfg["fps_scene"]
    for e in tl["events_scripted"]:
        if e["type"] != "engagement":
            continue
        s = tl["agents"][str(e["staff"])]["frames"]
        c = tl["agents"][str(e["customer"])]["frames"]
        f0, f1 = int(round(e["t"] * fps)), int(round(e["t_end"] * fps))
        assert (f1 - f0) / fps >= 5.0 + 1.0
        lead = [p for p in cfg.get("parties", []) if e["customer"] in p]
        if lead and lead[0][0] != e["customer"]:
            continue    # party member: the staff faces the party leader
        for i in range(f0, min(f1, len(s))):
            if s[i]["action"] != "talk":
                continue
            d = math.dist(s[i]["pos_m"], c[i]["pos_m"])
            assert 0.9 <= d <= 1.3, f"frame {i}: distance {d}"
            bearing = math.degrees(math.atan2(c[i]["pos_m"][1] - s[i]["pos_m"][1], c[i]["pos_m"][0] - s[i]["pos_m"][0]))
            assert _ang(s[i]["yaw_deg"], bearing) <= 20.0


def test_intended_events(scene):
    cfg, tl = scene
    ev = tl["events_scripted"]
    kinds = [e["type"] for e in ev]
    g = [e["t"] for e in ev if e["type"] == "greet"]
    sc = cfg["scenario"]
    if sc == "greet_fast":
        assert len(g) == 1 and 5.0 <= g[0] <= 15.0
    elif sc == "greet_slow":
        assert len(g) == 1 and 40.0 <= g[0] <= 70.0
    elif sc == "walkout":
        w = [e["t"] for e in ev if e["type"] == "walkout"]
        assert not g and len(w) == 1 and 30.0 <= w[0] <= 60.0
    elif sc == "party":
        assert len(g) == 2 and kinds.count("party") == 1
    elif sc == "handoff":
        assert len(g) == 1 and kinds.count("handoff") == 1
    elif sc == "dealership_long":
        assert any(e["type"] == "engagement" and e["t_end"] - e["t"] > 120 for e in ev)


def test_rules_reproduce_scripted_events(scene, presets):
    cfg, tl = scene
    for pr in cfg["eval_presets"]:
        rule = gt_events.events_by_rule(tl, cfg, presets[pr])
        un_s, un_r = gt_events.match_events(tl["events_scripted"], rule, 1.0, TYPES)
        un_r = [r for r in un_r if r["type"] in ("greet", "walkout", "handoff")]
        assert not un_s, f"[{pr}] unmatched scripted {un_s}"
        assert not un_r, f"[{pr}] unscripted rule events {un_r}"


def test_greet_offset_by_choreography(scene, presets):
    """Arrival-defined scripted greet vs rule onset: |dt| <= 0.8 s (margin under the 1.0 s gate)."""
    cfg, tl = scene
    rule = gt_events.events_by_rule(tl, cfg, presets["short_clip"])
    for e in tl["events_scripted"]:
        if e["type"] in ("greet", "handoff"):
            r = [x for x in rule if x["type"] == e["type"] and x.get("customer") == e["customer"]]
            assert r and min(abs(x["t"] - e["t"]) for x in r) <= 0.8
