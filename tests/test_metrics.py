import pytest
from showroom_coach.metrics import scorecard, unattended_wait
from showroom_coach.types import Event

EV = [
    Event("enter", 5.0, customer=2, data={"entry_known": True}),
    Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": 4.0, "via": "direct"}),
    Event("engagement", 9.0, customer=2, staff=1, t_end=40.0),
    Event("enter", 10.0, customer=3, data={"entry_known": True}),
    Event("wait_alert", 20.0, customer=3, data={"wait_s": 10.0, "free_staff": [4]}),
    Event("walkout", 25.0, customer=3, data={"free_staff": [4]}),
    Event("leave", 25.0, customer=3, data={"greeted": False, "entry_known": True}),
    Event("enter", 12.0, customer=5, data={"entry_known": True}),
    Event("greet", 18.0, customer=5, staff=4, data={"time_to_greet": 6.0, "via": "direct"}),
    Event("engagement", 18.0, customer=5, staff=4, t_end=30.0),
]


def test_staff_cards():
    sc = scorecard(EV, staff_ids=[4, 1], customer_ids=[2, 3, 5], t_end=60.0, names={1: "Aziz"})
    a, b = sc.staff
    assert (a.staff, a.name, a.greets, a.median_time_to_greet) == (1, "Aziz", 1, 4.0)
    assert (a.customers_engaged, a.engaged_total_s, a.engaged_median_s) == (1, 31.0, 31.0)
    assert (a.missed_while_free, a.walkouts_while_free) == (0, 0)
    assert (b.staff, b.greets, b.median_time_to_greet, b.engaged_total_s) == (4, 1, 6.0, 12.0)
    assert (b.missed_while_free, b.walkouts_while_free) == (1, 1)


def test_store_card():
    s = scorecard(EV, [1, 4], [2, 3, 5], 60.0).store
    assert (s.customers, s.greeted, s.walkouts, s.wait_alerts, s.engagements) == (3, 2, 1, 1, 2)
    assert s.greet_rate == pytest.approx(2 / 3)
    assert s.median_time_to_greet == pytest.approx(5.0)
    assert s.unattended_wait_s == pytest.approx(4 + 15 + 6)


def test_unattended_wait_open_visit_runs_to_end():
    ev = [Event("enter", 50.0, customer=7, data={"entry_known": True})]
    assert unattended_wait(ev, t_end=60.0) == pytest.approx(10.0)


def test_unknown_entry_not_counted_in_wait():
    ev = [Event("enter", 0.0, customer=7, data={"entry_known": False})]
    assert unattended_wait(ev, t_end=60.0) == 0.0


def test_empty_events():
    sc = scorecard([], [], [], 0.0)
    assert sc.staff == [] and sc.store.customers == 0
    assert sc.store.greet_rate is None and sc.store.median_time_to_greet is None


def test_staff_without_events_gets_zero_card():
    sc = scorecard([], [8], [], 10.0)
    assert sc.staff[0].greets == 0 and sc.staff[0].median_time_to_greet is None


def test_reentry_greet_and_leave_match_latest_enter():
    ev = [
        Event("enter", 5.0, customer=9, data={"entry_known": True}),
        Event("leave", 20.0, customer=9, data={"greeted": False, "entry_known": True}),
        Event("enter", 30.0, customer=9, data={"entry_known": True}),
        Event("greet", 35.0, customer=9, staff=1, data={"time_to_greet": 5.0, "via": "direct"}),
    ]
    s = scorecard(ev, [1], [9], 60.0).store
    assert (s.customers, s.customers_entry_known, s.greeted) == (2, 2, 1)
    assert s.greet_rate == pytest.approx(0.5)
    assert s.median_time_to_greet == pytest.approx(5.0)
    assert s.unattended_wait_s == pytest.approx(15 + 5)  # visit 1 to its leave, visit 2 to its greet


def test_events_out_of_order_and_ties_still_match():
    ev = [
        Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": 4.0, "via": "direct"}),
        Event("leave", 9.0, customer=2, data={"greeted": True, "entry_known": True}),
        Event("enter", 5.0, customer=2, data={"entry_known": True}),
    ]
    s = scorecard(ev, [1], [2], 60.0).store
    assert (s.customers, s.greeted) == (1, 1)
    assert s.unattended_wait_s == pytest.approx(4.0)


def test_greet_without_time_counts_as_greeted_but_not_in_median():
    ev = [
        Event("enter", 5.0, customer=2, data={"entry_known": False}),
        Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": None, "via": "approach"}),
    ]
    sc = scorecard(ev, [1], [2], 60.0)
    assert sc.store.greeted == 1 and sc.store.customers_entry_known == 0
    assert sc.store.median_time_to_greet is None
    assert sc.store.unattended_wait_s == 0.0
    assert sc.staff[0].greets == 1 and sc.staff[0].median_time_to_greet is None


def test_staff_ids_sorted_deduped_and_unknown_staff_ignored():
    ev = [Event("greet", 9.0, customer=2, staff=99, data={"time_to_greet": 1.0, "via": "direct"})]
    sc = scorecard(ev, iter([7, 3, 7]), [], 60.0)
    assert [c.staff for c in sc.staff] == [3, 7]
    assert all(c.greets == 0 for c in sc.staff)


def test_visit_of_person_marked_staff_is_not_counted():
    # Person 8 was seen as a customer (enter emitted, known entry) and later marked staff:
    # final roles win, so the orphan enter must not become a phantom visit.
    ev = EV + [Event("enter", 20.0, customer=8, data={"entry_known": True})]
    sc = scorecard(ev, staff_ids=[1, 4, 8], customer_ids=[2, 3, 5], t_end=60.0)
    s = sc.store
    assert (s.customers, s.customers_entry_known, s.greeted) == (3, 3, 2)
    assert s.greet_rate == pytest.approx(2 / 3)
    assert s.unattended_wait_s == pytest.approx(4 + 15 + 6)  # no 40 s tail for person 8
    assert [c.staff for c in sc.staff] == [1, 4, 8]
    # the same events with 8 treated as a customer do count it
    s2 = scorecard(ev, [1, 4], [2, 3, 5, 8], 60.0).store
    assert (s2.customers, s2.greeted) == (4, 2)
    assert s2.unattended_wait_s == pytest.approx(4 + 15 + 6 + 40)


def test_unattended_wait_can_exclude_staff_customers():
    ev = [
        Event("enter", 10.0, customer=7, data={"entry_known": True}),
        Event("enter", 20.0, customer=8, data={"entry_known": True}),
    ]
    assert unattended_wait(ev, t_end=60.0) == pytest.approx(50 + 40)
    assert unattended_wait(ev, t_end=60.0, exclude_customers=[8]) == pytest.approx(50.0)


def test_greet_at_the_same_instant_as_enter_matches_that_enter():
    # A party mate greeted on arrival: greet.t == enter.t, listed before the enter in the input.
    ev = [
        Event("greet", 5.0, customer=2, staff=1, data={"time_to_greet": 0.0, "via": "party"}),
        Event("enter", 5.0, customer=2, data={"entry_known": True}),
    ]
    s = scorecard(ev, [1], [2], 60.0).store
    assert (s.customers, s.greeted) == (1, 1)
    assert s.median_time_to_greet == 0.0
    assert s.unattended_wait_s == 0.0


def test_unattended_wait_returns_float_on_empty_input():
    out = unattended_wait([], t_end=10.0)
    assert out == 0.0 and isinstance(out, float)
