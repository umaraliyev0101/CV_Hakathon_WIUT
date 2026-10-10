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


# ---------------------------------------------------------------------------------------------- fix round 1

def _run_engine(timeline, th=SHORT):
    """Step an engine over a timeline; return (engine, events, {t: active_pairs()}, {t: status(t)})."""
    eng, ev, pairs, status = EventEngine(th), [], {}, {}
    for t, states in timeline:
        ev += eng.step(t, states)
        pairs[t], status[t] = eng.active_pairs(), eng.status(t)
    return eng, ev, pairs, status


def test_no_wait_alert_after_customer_has_gone():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": when(5, 14, (6.0, 5.0))}}), SHORT)
    assert of(ev, "wait_alert") == []
    (wo,) = of(ev, "walkout")
    assert wo.t == pytest.approx(14.0)


def _approach_at(t_near, t_far=None):
    """Customer from t=5 at (6, 5), beside staff (0, 5) during [t_near, t_far], back at (6, 5) afterwards."""
    def pos(t):
        if t < 5:
            return None
        near = t >= t_near and (t_far is None or t <= t_far)
        return (1.0, 5.0) if near else (6.0, 5.0)
    return pos


def test_open_greet_streak_that_confirms_suppresses_wait_alert():
    ev = run_events(script(30, 0.2, {1: STAFF, 2: {"pos": _approach_at(14.0)}}), SHORT)
    (g,) = of(ev, "greet")
    assert g.t == pytest.approx(14.0) and g.data["confirmed_at"] == pytest.approx(16.0)
    assert of(ev, "wait_alert") == []


def test_open_greet_streak_that_fails_delays_wait_alert():
    ev = run_events(script(30, 0.2, {1: STAFF, 2: {"pos": _approach_at(14.0, 15.0)}}), SHORT)
    assert of(ev, "greet") == []
    (w,) = of(ev, "wait_alert")
    assert w.t == pytest.approx(16.6, abs=0.21) and w.data["free_staff"] == [1]


def test_status_browsing_between_engagements():
    tl = script(20, 0.2, {1: STAFF, 2: {"pos": _approach_at(5.0, 15.0)}})
    _, _, _, status = _run_engine(tl)
    assert status[6.0][2]["state"] == "waiting"
    assert status[9.0][2]["state"] == "greeted" and status[9.0][2]["greeted_by"] == 1
    assert status[14.0][2]["state"] == "engaged"
    assert status[18.0][2]["state"] == "browsing"


def test_active_pairs_close_then_engaged():
    _, _, pairs, _ = _run_engine(script(20, 0.2, {1: STAFF, 2: {"pos": when(5, 20, (1.0, 5.0))}}))
    assert pairs[4.8] == []
    assert pairs[12.8] == [(1, 2, "close")]
    assert pairs[13.0] == [(1, 2, "engaged")]


def test_handoff_once_when_second_staff_engages_twice():
    def c(t):
        if t < 5:
            return None
        if t <= 15:
            return (1.0, 5.0)
        return (9.0, 5.0) if t <= 25 or t >= 28 else (5.0, 5.0)
    people = {1: STAFF, 4: {"role": "staff", "pos": when(0, 40, (10.0, 5.0))}, 2: {"pos": c}}
    ev = run_events(script(40, 0.2, people), SHORT, t_end=40)
    assert [e.staff for e in of(ev, "engagement")] == [1, 4, 4]
    (h,) = of(ev, "handoff")
    assert h.data == {"from_staff": 1, "to_staff": 4}


def test_party_greet_not_before_mate_entry():
    people = {1: STAFF, 2: {"pos": when(5, 30, (1.0, 5.0))}, 3: {"pos": when(7, 30, (2.2, 5.6))}}
    ev = run_events(script(30, 0.2, people), SHORT)
    greets = {g.customer: g for g in of(ev, "greet")}
    assert greets[2].t == pytest.approx(5.0)
    assert greets[3].data["via"] == "party" and greets[3].t == pytest.approx(7.0)
    assert greets[3].data["time_to_greet"] == pytest.approx(0.0)


def test_party_greet_not_after_mate_last_seen():
    people = {1: STAFF,
              2: {"pos": lambda t: None if t < 5 else ((6.0, 5.0) if t < 20.6 else (1.0, 5.0))},
              3: {"pos": when(6, 19.8, (6.0, 6.0))}}
    ev = run_events(script(30, 0.2, people), SHORT)
    greets = {g.customer: g for g in of(ev, "greet")}
    assert greets[2].t == pytest.approx(20.6)
    (lv,) = [e for e in of(ev, "leave") if e.customer == 3]
    assert greets[3].t == pytest.approx(19.8) and greets[3].t <= lv.t and lv.data["greeted"] is True
    assert of(ev, "walkout") == []


def test_party_resolved_when_member_leaves():
    people = {1: STAFF, 2: {"pos": when(5, 30, (1.0, 5.0))}, 3: {"pos": when(6, 10, (1.0, 6.2))}}
    ev = run_events(script(30, 0.2, people), SHORT)
    (p,) = of(ev, "party")
    assert p.data["members"] == [2, 3] and p.data["confirmed_at"] == pytest.approx(13.0)
    (lv,) = [e for e in of(ev, "leave") if e.customer == 3]
    assert lv.data["greeted"] is True and of(ev, "walkout") == []


def test_party_resolved_at_finish():
    people = {1: STAFF, 2: {"pos": when(30, 36, (6.0, 5.0))}, 3: {"pos": when(31, 36, (6.0, 6.0))}}
    (p,) = of(run_events(script(36, 0.2, people), SHORT), "party")
    assert p.data["members"] == [2, 3] and p.data["confirmed_at"] == pytest.approx(36.0)


def test_three_member_party_one_event_per_growth():
    people = {1: STAFF, 2: {"pos": when(5, 30, (6.0, 5.0))}, 3: {"pos": when(6, 30, (6.0, 6.0))},
              4: {"pos": when(7, 30, (6.5, 5.5))}}
    ev = run_events(script(30, 0.2, people), SHORT)
    assert [p.data["members"] for p in of(ev, "party")] == [[2, 3], [2, 3, 4]]


def test_party_member_who_left_is_not_greeted_later():
    people = {1: STAFF,
              2: {"pos": lambda t: None if t < 5 else ((6.0, 5.0) if t < 25 else (1.0, 5.0))},
              3: {"pos": when(6, 18, (6.0, 6.0))}}
    ev = run_events(script(35, 0.2, people), SHORT)
    assert [p.data["members"] for p in of(ev, "party")] == [[2, 3]]
    assert [g.customer for g in of(ev, "greet")] == [2]
    assert [w.customer for w in of(ev, "walkout")] == [3]


def test_party_needs_three_shared_samples():
    people = {1: STAFF, 2: {"pos": when(5, 30, (6.0, 5.0))}, 3: {"pos": when(6, 6.2, (6.0, 6.0))}}
    assert of(run_events(script(30, 0.2, people), SHORT), "party") == []


def test_reentry_starts_new_visit_and_can_walk_out_again():
    def pos(t):
        return (6.0, 5.0) if 5 <= t <= 8 or 14 <= t <= 18 else None
    ev = run_events(script(25, 0.2, {1: STAFF, 2: {"pos": pos}}), SHORT)
    assert [e.t for e in of(ev, "enter")] == pytest.approx([5.0, 14.0])
    assert [e.t for e in of(ev, "walkout")] == pytest.approx([8.0, 18.0])


def test_reappearing_after_stall_ends_old_visit():
    eng = EventEngine(SHORT)
    staff, cust = PersonState(1, (0.0, 5.0), "staff"), PersonState(2, (6.0, 5.0))
    eng.step(5.0, [staff, cust])
    eng.step(5.2, [staff, cust])
    ev = eng.step(10.0, [staff, cust])  # no step for 4.8 s, longer than leave_after (3 s)
    assert [e.type for e in ev] == ["leave", "walkout", "enter"]
    assert ev[0].t == pytest.approx(5.2) and ev[0].data["detected_at"] == pytest.approx(10.0)
    assert ev[2].t == pytest.approx(10.0)


def test_role_flip_closes_engagement_and_drops_visit():
    eng, ev, _, _ = _run_engine(script(20, 0.2, {1: STAFF, 5: {"pos": when(4, 20, (1.0, 5.0))}}))
    ev += eng.step(20.2, [PersonState(1, (0.0, 5.0), "staff"), PersonState(5, (1.0, 5.0), "staff")])
    (e,) = of(ev, "engagement")
    assert (e.staff, e.customer) == (1, 5) and (e.t, e.t_end) == pytest.approx((4.0, 20.0))
    assert of(ev, "leave") == [] and of(ev, "walkout") == []
    assert eng.active_pairs() == []
    assert eng.status(20.2) == {1: {"role": "staff", "state": "free"}, 5: {"role": "staff", "state": "free"}}
    assert eng.finish(20.2) == []  # nothing is still open for pid 5 as a customer
    assert [e.type for e in eng.step(20.4, [PersonState(5, (1.0, 5.0), "customer")])] == ["enter"]


def test_role_flip_staff_to_customer_closes_their_engagements():
    eng, ev, _, _ = _run_engine(script(20, 0.2, {1: STAFF, 2: {"pos": when(4, 20, (1.0, 5.0))}}))
    ev += eng.step(20.2, [PersonState(1, (0.0, 5.0), "customer"), PersonState(2, (1.0, 5.0))])
    (e,) = of(ev, "engagement")
    assert (e.staff, e.customer) == (1, 2) and e.t_end == pytest.approx(20.0)
    assert [x.customer for x in ev if x.type == "enter"] == [2, 1]
    assert eng.active_pairs() == []


def test_dealership_greet():
    people = {1: {"role": "staff", "pos": when(0, 100, (0.0, 5.0))},
              2: {"pos": lambda t: None if t < 30 else ((6.0, 5.0) if t < 50 else (1.0, 5.0))}}
    (g,) = of(run_events(script(100, 1.0, people), DEAL), "greet")
    assert g.t == pytest.approx(50.0) and g.data["confirmed_at"] == pytest.approx(55.0)
    assert g.data["time_to_greet"] == pytest.approx(20.0)


def test_dealership_wait_alert_and_walkout():
    people = {1: {"role": "staff", "pos": when(0, 130, (0.0, 5.0))}, 2: {"pos": when(20, 100, (6.0, 5.0))}}
    ev = run_events(script(130, 1.0, people), DEAL)
    (w,) = of(ev, "wait_alert")
    assert w.t == pytest.approx(81.0) and w.data["wait_s"] == pytest.approx(61.0) and w.data["free_staff"] == [1]
    (wo,) = of(ev, "walkout")
    (lv,) = of(ev, "leave")
    assert wo.t == pytest.approx(100.0) and lv.data["detected_at"] == pytest.approx(110.0)
