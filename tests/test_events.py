import pytest
from showroom_coach.config import PRESETS
from showroom_coach.events import EventEngine, PersonState, run_events
from tests.synth import script, when

SHORT, DEAL = PRESETS["short"], PRESETS["dealership"]


def of(events, kind):
    return [e for e in events if e.type == kind]


def approach(t_enter, t_move, xs=6.0, xe=1.0, z=5.0, t_out=40.0):
    """Customer enters at t_enter at x=xs, walks to xe during [t_move, t_move+1], stays until t_out."""
    def pos(t):
        if t < t_enter or t > t_out:
            return None
        if t < t_move:
            return (xs, z)
        if t < t_move + 1:
            return (xs + (xe - xs) * (t - t_move), z)
        return (xe, z)
    return pos


STAFF = {"role": "staff", "pos": when(0, 40, (0.0, 5.0))}


def test_greet_time_to_greet_short():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(5, 8)}}), SHORT)
    (g,) = of(ev, "greet")
    assert (g.staff, g.customer) == (1, 2)
    assert g.t == pytest.approx(9.0, abs=0.21)
    assert g.data["time_to_greet"] == pytest.approx(4.0, abs=0.21)
    (en,) = of(ev, "enter")
    assert en.data["entry_known"] is True


def test_engagement_short_closed_at_finish():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(5, 8)}}), SHORT, t_end=40.0)
    (e,) = of(ev, "engagement")
    assert e.t == pytest.approx(8.8, abs=0.21) and e.t_end == pytest.approx(40.0, abs=0.21)
    assert len(of(ev, "engagement_start")) == 1


def test_no_greet_when_both_face_away():
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0)), "facing": lambda t: (-1.0, 0.0)},
        2: {"pos": lambda t: (1.0, 5.0) if 5 <= t <= 10 else ((6.0, 5.0) if 10 < t <= 40 else None),
            "facing": lambda t: (1.0, 0.0)},
    }
    ev = run_events(script(40, 0.2, people), SHORT)
    assert of(ev, "greet") == []


def test_greet_when_one_faces_the_other():
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0)), "facing": lambda t: (1.0, 0.0)},
        2: {"pos": lambda t: (1.0, 5.0) if 5 <= t <= 10 else ((6.0, 5.0) if 10 < t <= 40 else None)},
    }
    assert len(of(run_events(script(40, 0.2, people), SHORT), "greet")) == 1


def _gap_customer(gap_end):
    def pos(t):
        if t < 5 or t > 40:
            return None
        close = (10 <= t <= 20) or (gap_end <= t <= 32)
        return (1.0, 5.0) if close else (5.0, 5.0)
    return pos


def test_short_gap_does_not_split_engagement():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": _gap_customer(21.0)}}), SHORT, t_end=40)
    (e,) = of(ev, "engagement")
    assert e.t == pytest.approx(10.0, abs=0.21) and e.t_end == pytest.approx(32.0, abs=0.21)


def test_long_gap_splits_engagement():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": _gap_customer(23.0)}}), SHORT, t_end=40)
    es = of(ev, "engagement")
    assert [(round(e.t), round(e.t_end)) for e in es] == [(10, 20), (23, 32)]


@pytest.mark.parametrize("close_until,expected", [(120.0, 0), (150.0, 1)])
def test_dealership_engagement_needs_120s(close_until, expected):
    people = {
        1: {"role": "staff", "pos": when(0, 300, (0.0, 5.0))},
        2: {"pos": lambda t: None if t < 15 else ((1.0, 5.0) if 20 <= t <= close_until else (6.0, 5.0))},
    }
    ev = run_events(script(300, 1.0, people), DEAL, t_end=300)
    assert len(of(ev, "engagement")) == expected
    assert len(of(ev, "greet")) == 1


def test_wait_alert_lists_free_staff():
    people = {1: STAFF, 2: {"pos": when(5, 40, (6.0, 5.0))}}
    (w,) = of(run_events(script(40, 0.2, people), SHORT), "wait_alert")
    assert w.customer == 2 and w.t == pytest.approx(15.2, abs=0.21)
    assert w.data["free_staff"] == [1] and w.data["wait_s"] > 10


def test_wait_alert_staff_busy_with_other_customer():
    people = {1: STAFF, 3: {"pos": when(0, 40, (0.5, 5.0))}, 2: {"pos": when(5, 40, (6.0, 5.0))}}
    w = [e for e in run_events(script(40, 0.2, people), SHORT) if e.type == "wait_alert" and e.customer == 2]
    assert len(w) == 1 and w[0].data["free_staff"] == []


def test_walkout_after_leave_after():
    people = {1: STAFF, 2: {"pos": when(5, 12, (6.0, 5.0))}}
    ev = run_events(script(40, 0.2, people), SHORT)
    (wo,) = of(ev, "walkout")
    assert wo.t == pytest.approx(12.0) and wo.data["free_staff"] == [1]
    (lv,) = [e for e in of(ev, "leave") if e.customer == 2]
    assert lv.data["detected_at"] == pytest.approx(15.0, abs=0.21)


def test_walkout_at_frame_edge_is_fast():
    people = {1: STAFF, 2: {"pos": when(5, 12, (6.0, 5.0)), "edge": lambda t: t >= 11.5}}
    ev = run_events(script(40, 0.2, people), SHORT)
    (lv,) = [e for e in of(ev, "leave") if e.customer == 2]
    assert lv.data["detected_at"] <= 13.01


def test_already_present_customer_excluded_from_wait_and_walkout():
    people = {1: STAFF, 2: {"pos": when(0, 20, (6.0, 5.0))}}
    ev = run_events(script(40, 0.2, people), SHORT)
    assert of(ev, "wait_alert") == [] and of(ev, "walkout") == []
    (en,) = of(ev, "enter")
    assert en.data["entry_known"] is False


def test_already_present_greet_has_no_time_to_greet():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(0, 8)}}), SHORT)
    (g,) = of(ev, "greet")
    assert g.data["time_to_greet"] is None


def test_party_greet_propagates():
    def c2(t):
        if t < 5:
            return None
        return (6.0, 5.0) if t < 20 else (1.0, 5.0)
    people = {
        1: {"role": "staff", "pos": when(0, 60, (0.0, 5.0))},
        2: {"pos": c2},
        3: {"pos": when(7, 60, (6.0, 6.0))},
    }
    ev = run_events(script(60, 0.2, people), SHORT)
    (p,) = of(ev, "party")
    assert p.data["members"] == [2, 3]
    greets = {g.customer: g for g in of(ev, "greet")}
    assert set(greets) == {2, 3}
    assert greets[3].data["via"] == "party" and greets[3].staff == 1


def test_handoff_between_staff():
    def c(t):
        if t < 5 or t > 40:
            return None
        return (1.0, 5.0) if t <= 20 else (9.0, 5.0)
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0))},
        4: {"role": "staff", "pos": when(0, 40, (10.0, 5.0))},
        2: {"pos": c},
    }
    ev = run_events(script(40, 0.2, people), SHORT, t_end=40)
    assert sorted((e.staff, e.customer) for e in of(ev, "engagement")) == [(1, 2), (4, 2)]
    (h,) = of(ev, "handoff")
    assert h.data == {"from_staff": 1, "to_staff": 4} and h.t == pytest.approx(20.2, abs=0.21)


def test_unsure_is_treated_as_customer():
    ev = run_events(script(10, 0.2, {9: {"role": "unsure", "pos": when(4, 10, (3.0, 5.0))}}), SHORT)
    assert [e.customer for e in of(ev, "enter")] == [9]


def test_step_and_status_for_live_overlay():
    eng = EventEngine(SHORT)
    for t, states in script(16, 0.2, {1: STAFF, 2: {"pos": when(5, 40, (6.0, 5.0))}}):
        eng.step(t, states)
    st = eng.status(16.0)
    assert st[2]["state"] == "waiting" and st[2]["wait_s"] == pytest.approx(11.0, abs=0.21)
    assert st[1] == {"role": "staff", "state": "free"}


def test_role_flip_mid_stream_drops_customer_state():
    eng = EventEngine(SHORT)
    eng.step(4.0, [PersonState(5, (0, 5), "customer")])
    eng.step(4.2, [PersonState(5, (0, 5), "staff")])
    assert eng.status(4.2)[5]["role"] == "staff"
