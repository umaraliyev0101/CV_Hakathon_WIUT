import pytest
from pulse.config import PRESETS
from pulse import interest as I, journey as J
from tests.conftest import stay

S = PRESETS["short"]
POS_A = (0.0, 4.0)

def cust_at_a(pid, t0, secs):
    return stay(pid, 0.2, 0.7, t0, secs, speed=0.05, pos=POS_A)

def staff_at(pid, t0, secs, pos):
    return stay(pid, 0.95, 0.3, t0, secs, pos=pos)

def run(tl, staff, zones):
    journeys = J.build(tl, zones, S)
    return I.episodes(journeys, staff, S, tl, zones)

def test_deep_by_cumulative_dwell_unserved(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 12, (5.0, 4.0))}
    ep = run(tl, staff, zones4)
    assert len(ep) == 1 and ep[0].zone_id == "car-a" and ep[0].dwell == pytest.approx(12)
    assert ep[0].unserved and not ep[0].assisted

def test_deep_by_two_visits(zones4):
    tl = {1: cust_at_a(1, 0, 2) + stay(1, 0.45, 0.7, 2.2, 3) + cust_at_a(1, 5.4, 2)}
    ep = run(tl, {}, zones4)
    assert len(ep) == 1 and ep[0].visits == 2

def test_short_single_visit_not_deep(zones4):
    assert run({1: cust_at_a(1, 0, 5)}, {}, zones4) == []

def test_assisted_when_staff_close_long_enough(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 4, (5.0, 4.0)) + staff_at(9, 4.2, 3, (1.0, 4.0)) + staff_at(9, 7.4, 4.6, (5.0, 4.0))}
    ep = run(tl, staff, zones4)[0]
    assert ep.assisted and not ep.unserved and ep.staff_ids == [9]

def test_brief_staff_contact_not_unserved_not_assisted(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 4, (5.0, 4.0)) + staff_at(9, 4.2, 1, (1.0, 4.0)) + staff_at(9, 5.4, 6.6, (5.0, 4.0))}
    ep = run(tl, staff, zones4)[0]
    assert not ep.unserved and not ep.assisted

def test_desk_dwell_is_not_interest(zones4):
    assert run({1: stay(1, 0.45, 0.15, 0, 20, pos=POS_A)}, {}, zones4) == []

def test_assistance_store_level(zones4):
    tl = {1: cust_at_a(1, 0, 12), 2: cust_at_a(2, 0, 12)}
    staff = {9: staff_at(9, 3, 4, (0.5, 4.0))}
    info = I.assistance({k: v for k, v in tl.items()}, staff, S)
    assert info[1].assisted and info[1].t_first == pytest.approx(3.0)

def test_live_alert_after_threshold(zones4):
    tl = {1: cust_at_a(1, 0, 9)}
    j = J.build(tl, zones4, S)
    alerts = I.open_alerts(j, {}, S, tl, zones4, now=9.0)
    assert len(alerts) == 1 and "Car A" in alerts[0].message and alerts[0].dwell == pytest.approx(9.0)
    tl7 = {1: cust_at_a(1, 0, 7)}
    assert I.open_alerts(J.build(tl7, zones4, S), {}, S, tl7, zones4, now=7.0) == []

def test_live_alert_suppressed_by_staff(zones4):
    tl = {1: cust_at_a(1, 0, 9)}
    staff = {9: staff_at(9, 3, 0.4, (0.5, 4.0))}
    assert I.open_alerts(J.build(tl, zones4, S), staff, S, tl, zones4, now=9.0) == []
