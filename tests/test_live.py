"""Live session engine (`LiveSession`) and the WebSocket protocol handler (`serve_live`).

Session tests feed frames straight in, with explicit times, using the fake perception stack. The WebSocket tests mount
`serve_live` on a tiny local FastAPI app, the way the real app does with `/ws/live`.
"""
import asyncio
import dataclasses
import json
import re
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from showroom_coach.live import LiveSession
from showroom_coach.live_ws import serve_live
from showroom_coach.perception.stack import fake_stack
from showroom_coach.sources import iter_frames

BLUE = (255, 0, 0)  # BGR
RED = (0, 0, 255)
GREEN = (0, 200, 0)
W, H = 640, 360
BOX_W, BOX_H = 60, 160


def _scene(*people):
    """Grey 640x360 frame with one coloured 60x160 block per person given as (x1, colour); the fake detector finds
    each block as a person. Blocks stand 5.5 m from the camera, so 100 px is about 1.2 m sideways."""
    img = np.full((H, W, 3), 128, np.uint8)
    for x1, colour in people:
        img[100:100 + BOX_H, x1:x1 + BOX_W] = colour
    return img


def _jpeg(frame):
    return cv2.imencode(".jpg", frame)[1].tobytes()


def _run(session, script, step=0.2):
    """Feed `script(t) -> frame` for t = 0, step, ... up to `script.until`; returns every overlay."""
    return [session.process(script(t), t) for t in _times(script.until, step)]


def _times(until, step=0.2):
    return [round(i * step, 6) for i in range(int(round(until / step)) + 1)]


def _person(overlay, pid):
    return next(p for p in overlay["people"] if p["pid"] == pid)


# --------------------------------------------------------------------------- session: tracking and roles


def test_session_tracks_and_toggles(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    out = None
    for _, t, frame in iter_frames(meet_video, sample_fps=5):
        out = s.process(frame, t)
    assert out["people"] and all(p["role"] in ("staff", "customer") for p in out["people"])
    pid = out["people"][0]["pid"]
    assert s.toggle(pid) == "staff"
    frame = next(iter_frames(meet_video))[2]
    again = s.process(frame, out["t"] + 0.2)
    assert [p for p in again["people"] if p["pid"] == pid][0]["override"] is True


def test_session_emits_events_with_staff(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    frames = list(iter_frames(meet_video, sample_fps=5))
    first = s.process(frames[0][2], frames[0][1])
    s.toggle(first["people"][0]["pid"])           # the blue stander is the only person at t=0
    events = []
    for _, t, f in frames[1:]:
        events += s.process(f, t)["events"]
    assert any(e["type"] == "greet" for e in events)


def test_overlay_has_the_contract_fields(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    frames = list(iter_frames(meet_video, sample_fps=5))
    outs = [s.process(f, t) for _, t, f in frames[:60]]     # up to 11.8 s: both people are in view
    out = outs[-1]
    assert out["type"] == "overlay"
    assert (out["w"], out["h"]) == (frames[0][2].shape[1], frames[0][2].shape[0])
    assert out["t"] == pytest.approx(11.8) and out["fps"] > 0
    assert set(out["store"]) == {"customers", "greeted", "walkouts"}
    assert len(out["people"]) == 2 and [p["pid"] for p in out["people"]] == sorted(p["pid"] for p in out["people"])
    for p in out["people"]:
        assert set(p) == {"pid", "bbox", "face", "role", "suggested", "confidence", "override", "state", "wait_s",
                          "label"}
        x1, y1, x2, y2 = p["bbox"]
        fx1, fy1, fx2, fy2 = p["face"]
        assert x1 < x2 and y1 < y2 and fx1 < fx2 and fy1 < fy2
        assert fy1 <= y1 + 0.2 * (y2 - y1) and fy2 <= y1 + 0.5 * (y2 - y1)     # a face sits near the top of the box
        assert p["suggested"] == "unsure" and p["override"] is False
    json.dumps(out, allow_nan=False)                         # plain JSON types only


def test_no_people_gives_an_empty_overlay():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    out = s.process(_scene(), 0.0)
    assert out["people"] == [] and out["pairs"] == [] and out["events"] == [] and out["alerts"] == []
    assert out["store"] == {"customers": 0, "greeted": 0, "walkouts": 0}


def test_labels_and_names():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    script = lambda t: _scene((50, BLUE), (500, RED))
    script.until = 1.0
    out = _run(s, script)[-1]
    staff, customer = out["people"][0]["pid"], out["people"][1]["pid"]
    assert out["people"][0]["label"] == f"C{staff}"
    s.toggle(staff)
    out = s.process(script(1.2), 1.2)
    assert _person(out, staff)["label"] == f"S{staff}"
    s.set_name(staff, "Aziz")
    out = s.process(script(1.4), 1.4)
    assert _person(out, staff)["label"] == f"S{staff} Aziz"
    assert _person(out, customer)["label"].startswith(f"C{customer}")
    s.set_name(staff, "  ")                                  # a blank name removes it
    assert _person(s.process(script(1.6), 1.6), staff)["label"] == f"S{staff}"


def test_toggle_flips_back_and_unknown_people_are_rejected():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    out = s.process(_scene((50, BLUE)), 0.0)
    pid = out["people"][0]["pid"]
    assert s.toggle(pid) == "staff"
    assert s.toggle(pid) == "customer"
    with pytest.raises(KeyError):
        s.toggle(999)
    with pytest.raises(KeyError):
        s.set_name(999, "Nobody")


# --------------------------------------------------------------------------- session: waiting, alerts, events


def _walkin(staff_x=None, customers=((500, 4.0, None),), until=20.0):
    """Script: an optional staff blue block from t=0; customers as (x1, appears, leaves-or-None) red blocks."""
    def script(t):
        people = [(staff_x, BLUE)] if staff_x is not None else []
        people += [(x, RED) for x, a, b in customers if t >= a and (b is None or t < b)]
        return _scene(*people)
    script.until = until
    return script


def test_waiting_customer_has_a_timer_and_a_long_wait_alert():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)         # wait_alert = 10 s
    outs = _run(s, _walkin(until=17.0))
    # the customer appears at 4.0, the tracker returns the box from 4.2, and the engine counts them once confirmed
    # (3 observations over >= 1 s), which is at 5.2: that is their "enter"
    at12 = next(o for o in outs if o["t"] == pytest.approx(12.0))
    p = _person(at12, 1)
    assert p["state"] == "waiting" and p["wait_s"] == pytest.approx(12.0 - 5.2, abs=0.01)
    assert p["label"] == "C1 ⏱ 0:06"
    assert at12["alerts"] == []
    alert_frames = [o for o in outs if o["alerts"]]
    assert alert_frames and alert_frames[0]["t"] == pytest.approx(15.4)
    assert alert_frames[0]["alerts"] == ["0:15 long wait: customer C1 (no staff free)"]
    wait_alerts = [e for o in outs for e in o["events"] if e["type"] == "wait_alert"]
    assert len(wait_alerts) == 1 and wait_alerts[0]["customer"] == 1
    assert outs[-1]["store"]["customers"] == 1 and outs[-1]["store"]["greeted"] == 0


def test_alert_names_the_free_staff_and_the_newest_comes_first():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _walkin(staff_x=290, customers=((40, 4.0, None), (540, 6.0, None)), until=18.0)
    first = s.process(script(0.0), 0.0)
    staff = first["people"][0]["pid"]
    s.toggle(staff)
    s.set_name(staff, "Aziz")
    outs = [first] + [s.process(script(t), t) for t in _times(18.0)[1:]]
    alerts = outs[-1]["alerts"]
    assert len(alerts) == 2
    assert re.fullmatch(r"0:17 long wait: customer C3 \(free: Aziz\)", alerts[0])      # confirmed at 7.2, alert at 17.4
    assert re.fullmatch(r"0:15 long wait: customer C2 \(free: Aziz\)", alerts[1])      # confirmed at 5.2, alert at 15.4
    assert _person(outs[-1], staff)["state"] == "free"


def test_walkout_raises_an_alert():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)         # leave_after = 3 s
    script = _walkin(staff_x=290, customers=((40, 4.0, 8.0),), until=14.0)
    first = s.process(script(0.0), 0.0)
    s.toggle(first["people"][0]["pid"])
    outs = [first] + [s.process(script(t), t) for t in _times(14.0)[1:]]
    walkouts = [e for o in outs for e in o["events"] if e["type"] == "walkout"]
    assert len(walkouts) == 1 and walkouts[0]["customer"] == 2
    assert any("walk-out: customer C2" in a and "(free: S1)" in a for a in outs[-1]["alerts"])
    assert outs[-1]["store"] == {"customers": 1, "greeted": 0, "walkouts": 1}


def test_alerts_keep_only_the_last_ten():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    # twelve different customers come and go without being greeted (14 s apart, longer than the 10 s re-attach
    # window, so each is a new person): twelve walk-outs, ten alerts kept
    visits = [(40 + (i % 2) * 400, 4.0 + 14.0 * i, 8.0 + 14.0 * i) for i in range(12)]
    script = _walkin(customers=tuple(visits), until=4.0 + 14.0 * 12)
    outs = _run(s, script, step=0.5)
    assert sum(e["type"] == "walkout" for o in outs for e in o["events"]) == 12
    alerts = outs[-1]["alerts"]
    assert len(alerts) == 10
    assert "customer C12" in alerts[0] and "customer C3" in alerts[-1]       # newest first, oldest two dropped


def test_open_engagement_counts_in_the_summary(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    frames = [f for f in iter_frames(meet_video, sample_fps=5)][:81]          # up to t = 16.0, mid-engagement
    first = s.process(frames[0][2], frames[0][1])
    s.toggle(first["people"][0]["pid"])
    for _, t, f in frames[1:]:
        s.process(f, t)
    summary = s.summary()
    assert summary["type"] == "summary"
    kinds = [e["type"] for e in summary["events"]]
    assert "enter" in kinds and "greet" in kinds and "engagement_start" in kinds
    engagements = [e for e in summary["events"] if e["type"] == "engagement"]
    assert len(engagements) == 1 and engagements[0]["t_end"] is None
    card = summary["scorecard"]
    assert card["store"]["customers"] == 1 and card["store"]["greeted"] == 1
    assert card["staff"][0]["customers_engaged"] == 1 and card["staff"][0]["engaged_total_s"] > 5
    json.dumps(summary, allow_nan=False)


def test_summary_of_an_empty_session():
    summary = LiveSession(fake_stack()).summary()
    assert summary["type"] == "summary" and summary["events"] == []
    assert summary["scorecard"]["staff"] == [] and summary["scorecard"]["store"]["customers"] == 0


# --------------------------------------------------------------------------- session: confirmation of people


BLIP_COLOURS = (GREEN, BLUE, (0, 255, 255), (255, 255, 0))    # four different looks, so four different phantom people


def _with_flickers(real=((500, 4.0, None),), flickers=((100, 7.0), (200, 10.0), (300, 13.0), (100, 16.0)), until=20.0,
                   colours=BLIP_COLOURS):
    """Script: real red customers as (x1, appears, leaves-or-None) plus 3-frame blips as (x1, starts)."""
    def script(t):
        people = [(x, RED) for x, a, b in real if t >= a - 1e-9 and (b is None or t < b)]
        people += [(x, colours[i % len(colours)]) for i, (x, a) in enumerate(flickers) if a - 1e-9 <= t <= a + 0.4 + 1e-9]
        return _scene(*people)
    script.until = until
    return script


def test_flickers_are_not_counted_as_customers_or_walkouts():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    outs = _run(s, _with_flickers())
    assert outs[-1]["store"] == {"customers": 1, "greeted": 0, "walkouts": 0}
    kinds = [e["type"] for o in outs for e in o["events"]]
    assert kinds.count("enter") == 1 and "walkout" not in kinds and "leave" not in kinds
    assert [e["customer"] for o in outs for e in o["events"] if e["type"] == "wait_alert"] == [1]
    summary = s.summary()
    assert summary["scorecard"]["store"]["customers"] == 1 and summary["scorecard"]["store"]["walkouts"] == 0
    assert not any(a for o in outs for a in o["alerts"] if "walk-out" in a)
    # the blips are still shown (tentative: no state) so the client can draw and blur them
    tentative = [p for o in outs for p in o["people"] if p["state"] is None and p["pid"] != 1]
    assert tentative


def test_a_person_is_tentative_until_seen_three_times_over_a_second():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = lambda t: _scene((300, RED))
    script.until = 1.2
    outs = {o["t"]: o for o in _run(s, script)}
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        p = outs[t]["people"][0]
        assert p["state"] is None and p["wait_s"] is None and p["label"] == "C1"
        assert outs[t]["store"]["customers"] == 0
    assert outs[1.0]["people"][0]["state"] == "waiting"
    assert outs[1.0]["store"]["customers"] == 1
    assert outs[1.0]["events"] and outs[1.0]["events"][0]["type"] == "enter"


def test_toggling_a_tentative_person_applies_once_confirmed():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    first = s.process(_scene((300, RED)), 0.0)
    assert first["people"][0]["state"] is None
    assert s.toggle(first["people"][0]["pid"]) == "staff"
    for t in (0.2, 0.4, 0.6, 0.8):
        out = s.process(_scene((300, RED)), t)
        assert out["people"][0]["state"] is None and out["people"][0]["role"] == "staff"
    out = s.process(_scene((300, RED)), 1.0)
    p = out["people"][0]
    assert p["state"] == "free" and p["role"] == "staff" and p["override"] is True and p["label"] == "S1"
    assert out["store"]["customers"] == 0                         # they never were a customer


def test_confirmation_is_per_track_so_repeated_blips_of_one_look_stay_unconfirmed():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    # the same green blob blinks six times, 3 s apart (longer than the tracker keeps a lost track, so each blink is a
    # new 3-frame track) and each re-attaches to the same person
    script = _with_flickers(real=(), flickers=tuple((100, 3.0 * i) for i in range(6)), until=16.0, colours=(GREEN,))
    outs = _run(s, script)
    assert len({p["pid"] for o in outs for p in o["people"]}) == 1          # one person, re-attached each time
    assert all(p["state"] is None for o in outs for p in o["people"])
    assert outs[-1]["store"]["customers"] == 0 and s.summary()["events"] == []


# --------------------------------------------------------------------------- session: face rects for everyone


def _detected(frame):
    return len([d for d in fake_stack().detect(frame) if d.cls == 0])


def test_every_detected_person_is_in_people_or_faces():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _with_flickers(real=((500, 0.0, None), (60, 4.0, None)))
    for t in _times(script.until):
        frame = script(t)
        out = s.process(frame, t)
        assert len(out["people"]) + len(out["faces"]) == _detected(frame), t
    json.dumps(out, allow_nan=False)


def test_first_sighting_has_a_face_rect_but_no_person_yet():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    out = s.process(_scene(), 0.0)
    assert out["people"] == [] and out["faces"] == []
    out = s.process(_scene((300, RED)), 0.2)                      # the tracker confirms a new box on its second sighting
    assert out["people"] == [] and len(out["faces"]) == 1
    x1, y1, x2, y2 = out["faces"][0]
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
    assert (x1, y1, x2) == (300, 100, 360) and y2 == 100 + round(0.2 * BOX_H)     # the top 20 % of the box, full width
    out = s.process(_scene((300, RED)), 0.4)
    assert len(out["people"]) == 1 and out["faces"] == []        # now tracked: its rect is people[0]["face"]


# --------------------------------------------------------------------------- session: alerts follow roles


def test_alerts_and_counters_follow_a_role_change():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _walkin(customers=((300, 4.0, None),), until=17.0)
    outs = _run(s, script)
    assert outs[-1]["alerts"] == ["0:15 long wait: customer C1 (no staff free)"]
    assert s.summary()["scorecard"]["store"]["wait_alerts"] == 1
    assert s.toggle(1) == "staff"                                  # it was a salesperson all along
    out = s.process(script(17.2), 17.2)
    assert out["alerts"] == [] and out["store"] == {"customers": 0, "greeted": 0, "walkouts": 0}
    summary = s.summary()
    assert summary["scorecard"]["store"]["wait_alerts"] == 0 and summary["scorecard"]["store"]["customers"] == 0
    assert not any(e["customer"] == 1 for e in summary["events"])
    assert s.toggle(1) == "customer"                               # changed their mind: everything is back
    back = s.process(script(17.4), 17.4)
    assert back["alerts"] == ["0:15 long wait: customer C1 (no staff free)"]
    assert s.summary()["scorecard"]["store"]["wait_alerts"] == 1


def test_walkout_alert_shows_current_names_and_drops_when_the_leaver_becomes_staff():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _walkin(staff_x=290, customers=((40, 4.0, 8.0),), until=14.0)
    first = s.process(script(0.0), 0.0)
    s.toggle(first["people"][0]["pid"])
    outs = [first] + [s.process(script(t), t) for t in _times(14.0)[1:]]
    assert any("(free: S1)" in a for a in outs[-1]["alerts"])
    s.set_name(1, "Aziz")
    out = s.process(script(14.2), 14.2)
    assert any("walk-out: customer C2 (free: Aziz)" in a for a in out["alerts"])
    s.toggle(2)
    out = s.process(script(14.4), 14.4)
    assert out["alerts"] == [] and out["store"]["walkouts"] == 0
    assert s.summary()["scorecard"]["store"]["walkouts"] == 0


# --------------------------------------------------------------------------- session: presets


def test_set_preset_keeps_identities_and_roles():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _walkin(staff_x=290, customers=((40, 4.0, None),), until=16.0)
    first = s.process(script(0.0), 0.0)
    staff = first["people"][0]["pid"]
    s.toggle(staff)
    outs = [first] + [s.process(script(t), t) for t in _times(16.0)[1:]]
    customer = outs[-1]["people"][1]["pid"]
    assert _person(outs[-1], customer)["state"] == "waiting" and outs[-1]["alerts"]      # 10 s limit already passed
    s.set_preset("dealership")
    assert s.preset == "dealership"
    after = s.process(script(16.2), 16.2)
    assert _person(after, staff)["role"] == "staff" and _person(after, staff)["override"] is True
    assert {p["pid"] for p in after["people"]} == {staff, customer}


def test_set_preset_rescores_the_session_so_far_under_the_new_thresholds():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    script = _walkin(customers=((300, 4.0, None),), until=16.0)
    outs = _run(s, script)
    assert outs[-1]["alerts"] and any(e["type"] == "wait_alert" for o in outs for e in o["events"])
    assert s.summary()["scorecard"]["store"]["wait_alerts"] == 1
    s.set_preset("dealership")                              # wait alert after 60 s, "already there" if seen < 10 s in
    summary = s.summary()
    assert summary["scorecard"]["store"]["wait_alerts"] == 0
    assert [e["type"] for e in summary["events"]].count("enter") == 1          # one visit, not counted twice
    after = s.process(script(16.2), 16.2)
    assert after["alerts"] == [] and after["store"]["customers"] == 1
    assert after["events"] == []                            # nothing replayed is announced again
    assert after["people"][0]["wait_s"] is None             # seen at 4.2 s: before the 10 s that make an entry certain


def test_set_preset_rejects_unknown_names_and_changes_nothing():
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    s.process(_scene((50, RED)), 0.0)
    with pytest.raises(ValueError):
        s.set_preset("nonsense")
    assert s.preset == "short"
    assert s.process(_scene((50, RED)), 0.2)["people"]
    with pytest.raises(ValueError):
        LiveSession(fake_stack(), preset="nonsense")


# --------------------------------------------------------------------------- session: auto-suggest


def test_everyone_is_an_unsure_customer_before_suggest_after():
    s = LiveSession(fake_stack(), suggest_after=5.0, suggest_every=1.0)
    script = lambda t: _scene((50, RED))
    script.until = 4.0
    out = _run(s, script)[-1]
    p = out["people"][0]
    assert (p["role"], p["suggested"], p["confidence"], p["override"]) == ("customer", "unsure", 0.5, False)


def test_suggestions_start_at_suggest_after_and_refresh_every_suggest_every():
    s = LiveSession(fake_stack(), suggest_after=5.0, suggest_every=2.0)
    script = lambda t: _scene((50, RED))
    script.until = 9.0
    outs = _run(s, script)
    by_t = {o["t"]: _person(o, 1) for o in outs}
    assert by_t[4.8]["suggested"] == "unsure"
    assert by_t[5.0]["suggested"] == "customer" and by_t[5.0]["confidence"] == pytest.approx(0.18, abs=0.01)
    assert by_t[9.0]["suggested"] == "customer"


def test_matching_outfits_that_stay_together_are_suggested_as_staff():
    s = LiveSession(fake_stack(), suggest_after=8.0, suggest_every=2.0)
    four = [(250 + 70 * i, BLUE) for i in range(4)]          # four in the same outfit, standing together
    script = lambda t: _scene(*four, (20, RED))                # and a lone customer in red, far to the left
    script.until = 14.0
    outs = _run(s, script)
    last = outs[-1]
    blues = [p for p in last["people"] if p["bbox"][0] > 200]
    red = [p for p in last["people"] if p["bbox"][0] < 100]
    assert len(blues) == 4 and len(red) == 1
    assert all(p["suggested"] == "staff" and p["role"] == "staff" and p["override"] is False for p in blues)
    assert red[0]["suggested"] == "customer" and red[0]["role"] == "customer"
    assert blues[0]["label"].startswith("S")
    assert outs[0]["people"][0]["suggested"] == "unsure"


def test_a_user_override_beats_the_suggestion():
    s = LiveSession(fake_stack(), suggest_after=8.0, suggest_every=2.0)
    four = [(250 + 70 * i, BLUE) for i in range(4)]
    script = lambda t: _scene(*four)
    script.until = 10.0
    out = _run(s, script)[-1]
    pid = out["people"][0]["pid"]
    assert out["people"][0]["role"] == "staff"
    assert s.toggle(pid) == "customer"
    for t in _times(16.0)[51:]:                                # several more suggestion rounds (t = 10.2 ... 16.0)
        out = s.process(script(t), t)
    p = _person(out, pid)
    assert p["role"] == "customer" and p["override"] is True and p["suggested"] == "staff"


def test_trajectories_are_kept_for_five_minutes_only():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    frame = _scene((50, RED))
    for i in range(400):
        s.process(frame, float(i))                              # one frame per second for 400 s
    (person,) = s._people.values()
    assert len(person.traj) <= 301 and person.traj[0][0] >= 399 - 300
    assert len(s._frame_times) <= 301


# --------------------------------------------------------------------------- session: identity and cost


def test_fps_is_the_processing_rate_not_the_arrival_rate():
    base = fake_stack()

    def slow_detect(frame):
        time.sleep(0.05)
        return base.detect(frame)

    s = LiveSession(dataclasses.replace(base, detect=slow_detect), suggest_after=1e9)
    frame = _scene((50, RED))
    outs = [s.process(frame, 0.2 * i) for i in range(8)]        # frames arrive instantly, one every 0.2 s of "video"
    assert 5.0 <= outs[-1]["fps"] <= 20.0


def test_new_tracks_are_embedded_at_once_and_refreshes_are_budgeted():
    base = fake_stack()
    calls = []

    def embed(crops):
        calls.append(len(crops))
        return base.embed(crops)

    s = LiveSession(dataclasses.replace(base, embed=embed), suggest_after=1e9, embed_every=5)
    frame = _scene((50, RED), (300, BLUE))
    s.process(frame, 0.0)
    assert calls == [2]                                           # both new tracks, in one call, on the first frame
    for i in range(1, 31):
        s.process(frame, 0.2 * i)
    assert max(calls[1:]) == 1                                    # at most one refresh crop per frame
    assert 6 <= len(calls) - 1 <= 14                              # each of 2 people refreshed about every 5 frames


def test_a_stack_without_an_embedder_still_tracks_and_reattaches():
    s = LiveSession(dataclasses.replace(fake_stack(), embed=None), suggest_after=1e9)
    out = s.process(_scene((50, RED)), 0.0)
    assert len(out["people"]) == 1


def test_a_person_who_returns_after_a_gap_keeps_their_id():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    present = _scene((300, GREEN))
    empty = _scene()
    first = None
    for i in range(10):
        first = s.process(present, 0.2 * i)
    pid = first["people"][0]["pid"]
    for i in range(10, 30):                                        # 4 s out of view: the tracker forgets the track
        assert s.process(empty, 0.2 * i)["people"] == []
    back = None
    for i in range(30, 36):
        back = s.process(present, 0.2 * i)
    assert [p["pid"] for p in back["people"]] == [pid]
    assert back["people"][0]["state"] is not None                  # re-attached to a confirmed person: not tentative again


def test_a_dissimilar_newcomer_gets_a_new_id():
    s = LiveSession(fake_stack(), suggest_after=1e9)
    for i in range(10):
        out = s.process(_scene((300, GREEN)), 0.2 * i)
    old = out["people"][0]["pid"]
    for i in range(10, 30):
        s.process(_scene(), 0.2 * i)
    for i in range(30, 36):
        out = s.process(_scene((300, RED)), 0.2 * i)
    assert [p["pid"] for p in out["people"]] == [old + 1]


# --------------------------------------------------------------------------- WebSocket


def _make_app(stack_factory=fake_stack, finished=None, sessions=None):
    """A local app mounting the handler the way the real app mounts it at /ws/live."""
    app = FastAPI()
    stack = stack_factory()

    def make(preset):
        session = LiveSession(stack, preset=preset)
        if sessions is not None:
            sessions.append(session)
        return session

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        await serve_live(websocket, make)
        if finished is not None:
            finished.append(True)

    return app


def test_ws_file_fed(meet_video):
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live?preset=short") as ws:
        n = 0
        for _, t, frame in iter_frames(meet_video, sample_fps=5):
            ws.send_text(json.dumps({"type": "time", "t": t}))
            ws.send_bytes(_jpeg(frame))
            msg = ws.receive_json()
            n += 1
            assert msg["type"] == "overlay" and "people" in msg and "fps" in msg
            assert msg["t"] == pytest.approx(t)
        assert n == 100
        ws.send_bytes(b"garbage")
        assert ws.receive_json() == {"type": "error", "error": "bad frame"}
        ws.send_bytes(b"")
        assert ws.receive_json() == {"type": "error", "error": "bad frame"}
        ws.send_text(json.dumps({"type": "summary"}))
        summary = ws.receive_json()
        assert summary["type"] == "summary" and summary["events"] and "scorecard" in summary


def test_ws_frames_wider_than_960_px_are_resized_before_processing():
    big = cv2.resize(_scene((50, RED)), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_bytes(_jpeg(big))
        out = ws.receive_json()
    assert (out["w"], out["h"]) == (960, 540) and len(out["people"]) == 1
    assert out["people"][0]["bbox"][2] <= 960 and out["people"][0]["bbox"][3] <= 540


def test_ws_disconnect_and_reconnect(meet_video):
    finished = []
    frame = next(iter_frames(meet_video))[2]
    with TestClient(_make_app(finished=finished)) as c:
        with c.websocket_connect("/ws/live") as ws:
            ws.send_bytes(_jpeg(frame))
            ws.receive_json()
        with c.websocket_connect("/ws/live") as ws:          # new session after a drop
            ws.send_bytes(_jpeg(frame))
            assert "people" in ws.receive_json()
    assert finished == [True, True]                           # both handlers returned cleanly, no exception


def test_ws_each_connection_has_its_own_session(meet_video):
    sessions = []
    frame = next(iter_frames(meet_video))[2]
    with TestClient(_make_app(sessions=sessions)) as c:
        with c.websocket_connect("/ws/live") as a, c.websocket_connect("/ws/live?preset=dealership") as b:
            a.send_bytes(_jpeg(frame)); a.receive_json()
            b.send_bytes(_jpeg(frame)); b.receive_json()
            b.send_bytes(_jpeg(frame)); second = b.receive_json()
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert sorted(s.preset for s in sessions) == ["dealership", "short"]
    assert second["t"] >= 0


def test_ws_controls_toggle_name_preset_and_time():
    frame = _jpeg(_scene((50, BLUE), (500, RED)))
    sessions = []
    with TestClient(_make_app(sessions=sessions)) as c, c.websocket_connect("/ws/live?preset=short") as ws:
        ws.send_text(json.dumps({"type": "time", "t": 1.0}))
        ws.send_bytes(frame)
        first = ws.receive_json()
        assert first["t"] == pytest.approx(1.0)
        pid = first["people"][0]["pid"]
        assert first["people"][0]["role"] == "customer"
        ws.send_text(json.dumps({"type": "toggle", "pid": pid}))
        ws.send_text(json.dumps({"type": "name", "pid": pid, "name": "Aziz"}))
        ws.send_text(json.dumps({"type": "preset", "name": "dealership"}))
        ws.send_text(json.dumps({"type": "time", "t": 1.2}))
        ws.send_bytes(frame)
        second = ws.receive_json()                            # exactly one reply per frame; controls sent none
        assert second["t"] == pytest.approx(1.2)
        p = _person(second, pid)
        assert p["role"] == "staff" and p["override"] is True and p["label"] == f"S{pid} Aziz"
        assert sessions[0].preset == "dealership"
        ws.send_bytes(frame)                                   # the override applied to one frame only: the session's
        ws.send_bytes(frame)                                   # own clock takes over from 1.2 and moves on
        assert ws.receive_json()["t"] >= 1.2
        assert ws.receive_json()["t"] > 1.2


def test_ws_bad_controls_get_an_error_and_the_session_continues():
    frame = _jpeg(_scene((50, RED)))
    bad = ['not json', '[]', '{"type": "dance"}', '{"type": "toggle"}', '{"type": "toggle", "pid": "x"}',
           '{"type": "toggle", "pid": 999}', '{"type": "name", "pid": 1}', '{"type": "preset", "name": "nonsense"}',
           '{"type": "time", "t": "soon"}', '{"type": "time", "t": -3}']
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_bytes(frame)
        assert ws.receive_json()["people"]
        for text in bad:
            ws.send_text(text)
            reply = ws.receive_json()
            assert reply["type"] == "error" and reply["error"], text
        ws.send_bytes(frame)
        assert ws.receive_json()["type"] == "overlay"


def test_ws_unknown_preset_is_refused_with_an_error():
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live?preset=bogus") as ws:
        reply = ws.receive_json()
        assert reply["type"] == "error" and "preset" in reply["error"]


def test_ws_a_failing_frame_does_not_end_the_session():
    base = fake_stack()
    state = {"n": 0}

    def flaky(frame):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("model hiccup")
        return base.detect(frame)

    frame = _jpeg(_scene((50, RED)))
    app = _make_app(lambda: dataclasses.replace(base, detect=flaky))
    with TestClient(app) as c, c.websocket_connect("/ws/live") as ws:
        replies = []
        for _ in range(3):
            ws.send_bytes(frame)
            replies.append(ws.receive_json())
    assert [r["type"] for r in replies] == ["overlay", "error", "overlay"]


def test_ws_frames_sent_back_to_back_are_answered_in_order():
    frame = _jpeg(_scene((50, RED)))
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live") as ws:
        for t in (0.0, 0.2, 0.4):
            ws.send_text(json.dumps({"type": "time", "t": t}))
            ws.send_bytes(frame)
        assert [ws.receive_json()["t"] for _ in range(3)] == pytest.approx([0.0, 0.2, 0.4])


class _FakeSocket:
    """The slice of starlette's WebSocket that `serve_live` uses, driven by a list of incoming messages."""

    def __init__(self, incoming, query=None):
        self.query_params = query or {}
        self._incoming = list(incoming)
        self.sent = []
        self.accepted = False
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        await asyncio.sleep(0)
        return self._incoming.pop(0)

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=None):
        self.closed = code


def test_ws_inference_runs_off_the_event_loop():
    class SlowSession:
        def process(self, frame, t=None):
            time.sleep(0.4)
            return {"type": "overlay", "t": 0.0}

    sock = _FakeSocket([{"type": "websocket.receive", "bytes": _jpeg(_scene((50, RED)))},
                        {"type": "websocket.disconnect", "code": 1001}])
    ticks = []

    async def main():
        async def heartbeat():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await serve_live(sock, lambda preset: SlowSession())
        beat.cancel()

    asyncio.run(main())
    assert sock.accepted and sock.sent == [{"type": "overlay", "t": 0.0}]
    assert len(ticks) >= 8                                         # the loop kept running during the 0.4 s of inference


def test_ws_oversized_frames_are_refused_without_decoding(monkeypatch):
    from showroom_coach import live_ws

    decoded = []
    real_decode = live_ws.decode_jpeg

    def spy(data, *args, **kwargs):
        decoded.append(len(data))
        return real_decode(data, *args, **kwargs)

    monkeypatch.setattr(live_ws, "decode_jpeg", spy)
    frame = _jpeg(_scene((50, RED)))
    with TestClient(_make_app()) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"\xff" * (live_ws.MAX_FRAME_BYTES + 1))
        assert ws.receive_json() == {"type": "error", "error": "bad frame"}
        assert decoded == []                                       # refused on size alone
        ws.send_bytes(frame)
        assert ws.receive_json()["type"] == "overlay"              # the session carries on
    assert decoded == [len(frame)] and live_ws.MAX_FRAME_BYTES == 2 * 1024 * 1024


def test_ws_a_factory_failure_other_than_an_unknown_preset_is_not_reported_as_one():
    app = FastAPI()

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        def broken(preset):
            raise ValueError("malformed detector spec")
        await serve_live(websocket, broken)

    with TestClient(app) as c, c.websocket_connect("/ws/live?preset=short") as ws:
        reply = ws.receive_json()
        assert reply["type"] == "error" and "preset" not in reply["error"]


def test_ws_slow_controls_run_off_the_event_loop():
    class SlowSession:
        def summary(self):
            time.sleep(0.4)
            return {"type": "summary"}

        def set_preset(self, name):
            time.sleep(0.4)

    sock = _FakeSocket([{"type": "websocket.receive", "text": json.dumps({"type": "summary"})},
                        {"type": "websocket.receive", "text": json.dumps({"type": "preset", "name": "dealership"})},
                        {"type": "websocket.receive", "text": json.dumps({"type": "summary"})},
                        {"type": "websocket.disconnect", "code": 1001}])
    ticks = []

    async def main():
        async def heartbeat():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await serve_live(sock, lambda preset: SlowSession())
        beat.cancel()

    asyncio.run(main())
    assert sock.sent == [{"type": "summary"}, {"type": "summary"}]            # in order; the preset control sent nothing
    assert len(ticks) >= 20                                                    # ~1.2 s of slow work: the loop kept ticking


# --------------------------------------------------------------------------- real models (slow)


def _real_stack():
    from showroom_coach import config, models as M

    for name in (config.LIVE_DETECTOR.partition("@")[0], "rtmpose_s", "dinov2_small"):
        if not (M.models_dir() / name / M.MODELS[name].filename).is_file():
            pytest.skip(f"model {name} not downloaded")
    from showroom_coach.perception.stack import default_stack

    return default_stack(config.LIVE_DETECTOR)


@pytest.mark.slow
@pytest.mark.parametrize("clip", ["meva_a_60s.mp4", "meva_b_60s.mp4"])
def test_real_stack_live_speed(clip, capsys):
    """Measures, and checks only that the result is sane: the spec target (>= 4 fps with 1-4 people) is reported by
    this test's output (run with -s), not asserted, because it depends on the machine and on how crowded the clip is."""
    from showroom_coach.config import repo_root

    path = repo_root() / "data" / "samples" / clip
    if not path.is_file():
        pytest.skip(f"{clip} missing")
    stack = _real_stack()
    s = LiveSession(stack, preset="short")
    outs = []
    t_wall = time.perf_counter()
    for i, (_, t, frame) in enumerate(iter_frames(path, sample_fps=5)):
        if i >= 100:
            break
        outs.append(s.process(frame, t))
    wall = time.perf_counter() - t_wall
    people = [len(o["people"]) for o in outs]
    with capsys.disabled():
        print(f"\n[live speed] {clip}: {len(outs)} frames in {wall:.1f} s = {len(outs) / wall:.2f} fps wall "
              f"(EMA {outs[-1]['fps']}); people per frame mean {np.mean(people):.1f} max {max(people)}")
    assert len(outs) == 100 and max(people) >= 1
    assert len(outs) / wall > 0.5
