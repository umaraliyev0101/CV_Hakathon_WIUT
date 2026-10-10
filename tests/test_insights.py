import pytest
from pulse.config import get_config
from pulse.datatypes import Stats, InterestEpisode
from pulse.insights import recommend, confidence

CFG = get_config("short")       # min_sample 2, alert_after 8

def car(zid, stop_rate=0.3, median_dwell=10.0, deep=0, unserved=0, stops=3):
    return {"zone_id": zid, "name": zid.upper(), "visitors": 10, "stops": stops, "stop_rate": stop_rate,
            "hold_rate": 0.5, "median_dwell": median_dwell, "total_dwell": 100.0, "repeat_visits": 0,
            "deep_interest": deep, "unserved": unserved, "unassisted_share": None, "rank": 1}

def stats(per_car=(), crowding=(), **store):
    s = Stats.empty()
    s.store.update({"visitors": 10, "assisted_customers": 4}, **store)
    s.per_car = list(per_car)
    s.crowding = [list(x) for x in crowding]
    return s

def eps(zid, n_unserved, n_served):
    return ([InterestEpisode(i, zid, 10.0 * i, 10.0 * i + 12, 12, 1, False, True) for i in range(n_unserved)]
            + [InterestEpisode(100 + i, zid, 5.0, 17.0, 12, 1, True, False, [9]) for i in range(n_served)])

def rules(ins): return [i.rule for i in ins]

def test_r1_fires_on_three_unserved():
    ins = recommend(stats([car("a", deep=3, unserved=3)]), eps("a", 3, 0), CFG)
    r1 = [i for i in ins if i.rule == "R1"][0]
    assert "3 of 3" in r1.action and r1.cars == ["a"] and r1.moments

def test_r1_quiet_when_mostly_served():
    assert "R1" not in rules(recommend(stats([car("a", deep=5, unserved=1)]), eps("a", 1, 4), CFG))

def test_r2_glance_and_go():
    cs = [car("a", 0.8, 2.0), car("b", 0.3, 20), car("c", 0.35, 25), car("d", 0.4, 30)]
    ins = [i for i in recommend(stats(cs), [], CFG) if i.rule == "R2"]
    assert [i.cars for i in ins] == [["a"]]

def test_r2_quiet_when_dwell_high():
    cs = [car("a", 0.8, 40.0), car("b", 0.3, 20), car("c", 0.35, 25), car("d", 0.4, 30)]
    assert "R2" not in rules(recommend(stats(cs), [], CFG))

def test_r3_dead_zone():
    cs = [car("a", 0.05), car("b", 0.4), car("c", 0.5)]
    assert [i.cars for i in recommend(stats(cs), [], CFG) if i.rule == "R3"] == [["a"]]

def test_r3_quiet_when_store_median_low():
    cs = [car("a", 0.05), car("b", 0.2), car("c", 0.25)]
    assert "R3" not in rules(recommend(stats(cs), [], CFG))

def test_r4_long_waits():
    assert "R4" in rules(recommend(stats(median_time_to_assist=20.0), [], CFG))
    assert "R4" not in rules(recommend(stats(median_time_to_assist=10.0), [], CFG))

def test_r5_crowding_needs_two_minutes():
    long = [(t, 4.0) for t in range(0, 150)]
    short = [(t, 4.0) for t in range(0, 60)] + [(t, 1.0) for t in range(60, 150)]
    assert "R5" in rules(recommend(stats(crowding=long), [], CFG))
    assert "R5" not in rules(recommend(stats(crowding=short), [], CFG))

def test_r6_desk_drop_off():
    fire = stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=2)
    quiet = stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=5)
    nodesk = stats(has_desk=False, deep_interest_visitors=6, deep_interest_reached_desk=0)
    assert "R6" in rules(recommend(fire, [], CFG))
    assert "R6" not in rules(recommend(quiet, [], CFG))
    assert "R6" not in rules(recommend(nodesk, [], CFG))

def test_r7_groups_get_less_attention():
    fire = stats(groups=3, n_solo=5, group_assisted_share=0.2, solo_assisted_share=0.6)
    quiet = stats(groups=3, n_solo=5, group_assisted_share=0.5, solo_assisted_share=0.6)
    assert "R7" in rules(recommend(fire, [], CFG))
    assert "R7" not in rules(recommend(quiet, [], CFG))

def test_confidence_levels_and_demo_flag():
    assert [confidence(n, 5) for n in (2, 5, 12, 25)] == ["insufficient", "low", "medium", "high"]
    ins = recommend(stats(median_time_to_assist=20.0, assisted_customers=1), [], CFG)
    r4 = [i for i in ins if i.rule == "R4"][0]
    assert r4.confidence == "insufficient" and r4.demo_only

# extra robustness tests (beyond the brief)
def test_missing_values_do_not_crash():
    cs = [car("a", None, None), car("b", None, 5.0), car("c", 0.4, None), car("d", 0.2, 3.0)]
    recommend(stats(cs, median_time_to_assist=None, has_desk=True, deep_interest_visitors=None,
                    group_assisted_share=None, solo_assisted_share=0.5, groups=None, n_solo=None), [], CFG)

def test_r1_episode_only_car_and_window_format():
    ins = recommend(stats([]), eps("zz", 3, 0), CFG)
    r1 = [i for i in ins if i.rule == "R1"][0]
    assert r1.cars == ["zz"] and "ZZ" not in r1.action and "zz" in r1.action
    assert "00:00–01:00" in r1.action and r1.window == (0.0, 60.0)

def test_order_by_rule_then_name():
    cs = [car("b", 0.05), car("a", 0.05), car("c", 0.5), car("d", 0.6), car("e", 0.7)]
    r3 = [i for i in recommend(stats(cs), [], CFG) if i.rule == "R3"]
    assert [i.cars for i in r3] == [["a"], ["b"]]
