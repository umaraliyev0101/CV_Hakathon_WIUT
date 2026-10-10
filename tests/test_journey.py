import pytest
from pulse.config import PRESETS
from pulse import journey as J
from tests.conftest import stay, track

S = PRESETS["short"]

def test_walk_a_then_b(zones4):
    tl = {1: stay(1, 0.2, 0.7, 0, 3) + track(1, [(0.45, 0.7)] * 10, t0=3.2) + stay(1, 0.7, 0.7, 5.2, 4)}
    v = J.build(tl, zones4, S)[1]
    assert [x.zone_id for x in v] == ["car-a", "car-b"]
    assert v[0].t_in == pytest.approx(0.0) and v[0].t_out == pytest.approx(3.0)
    assert v[1].dwell == pytest.approx(4.0)

def test_short_exit_merged(zones4):
    tl = {1: stay(1, 0.2, 0.7, 0, 2) + track(1, [(0.45, 0.7)] * 3, t0=2.2) + stay(1, 0.2, 0.7, 2.8, 2)}
    v = J.build(tl, zones4, S)[1]
    assert len(v) == 1 and v[0].t_out == pytest.approx(4.8)

def test_flicker_below_zone_min(zones4):
    tl = {1: stay(1, 0.45, 0.7, 0, 2) + stay(1, 0.7, 0.7, 2.2, 0.4) + stay(1, 0.45, 0.7, 2.8, 2)}
    assert J.build(tl, zones4, S)[1] == []

def test_stop_flag_requires_slow_and_long(zones4):
    slow = {1: stay(1, 0.2, 0.7, 0, 4, speed=0.05)}
    fast = {2: stay(2, 0.2, 0.7, 0, 4, speed=1.0)}
    brief = {3: stay(3, 0.2, 0.7, 0, 2, speed=0.05)}
    assert J.build(slow, zones4, S)[1][0].stopped
    assert not J.build(fast, zones4, S)[2][0].stopped
    assert not J.build(brief, zones4, S)[3][0].stopped

def test_desk_visit_never_stopped(zones4):
    v = J.build({1: stay(1, 0.45, 0.15, 0, 5, speed=0.0)}, zones4, S)[1]
    assert v[0].zone_id == "desk" and not v[0].stopped

def test_sequence_and_accuracy():
    from pulse.datatypes import ZoneVisit
    vs = [ZoneVisit(1, "a", 0, 1), ZoneVisit(1, "b", 2, 3), ZoneVisit(1, "b", 5, 6), ZoneVisit(1, "a", 8, 9)]
    assert J.sequence(vs) == ["a", "b", "a"]
    assert J.edit_distance(["a", "b", "c"], ["a", "b", "d"]) == 1
    assert J.journey_accuracy(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(2 / 3)
    assert J.journey_accuracy([], []) == 1.0
    assert J.journey_accuracy(["a"], []) == 0.0

def test_no_zones_gives_empty_visits():
    assert J.build({1: stay(1, 0.2, 0.7, 0, 3)}, [], S) == {1: []}
