from pulse.config import get_config
from pulse.insights import recommend, confidence
from pulse.eval.scenarios import car, stats, eps, build

CFG = get_config("short")       # min_sample 2, alert_after 8

def rules(ins): return [i.rule for i in ins]

def test_r1_fires_on_three_unserved():
    ins = recommend(*build("r1_fire"), CFG)
    r1 = [i for i in ins if i.rule == "R1"][0]
    assert "3 of 3" in r1.action and r1.cars == ["a"] and r1.moments

def test_r1_quiet_when_mostly_served():
    assert "R1" not in rules(recommend(*build("r1_quiet"), CFG))

def test_r2_glance_and_go():
    ins = [i for i in recommend(*build("r2_fire"), CFG) if i.rule == "R2"]
    assert [i.cars for i in ins] == [["a"]]

def test_r2_quiet_when_dwell_high():
    assert "R2" not in rules(recommend(*build("r2_quiet"), CFG))

def test_r3_dead_zone():
    assert [i.cars for i in recommend(*build("r3_fire"), CFG) if i.rule == "R3"] == [["a"]]

def test_r3_quiet_when_store_median_low():
    assert "R3" not in rules(recommend(*build("r3_quiet"), CFG))

def test_r4_long_waits():
    assert "R4" in rules(recommend(*build("r4_fire"), CFG))
    assert "R4" not in rules(recommend(*build("r4_quiet"), CFG))

def test_r5_crowding_needs_two_minutes():
    assert "R5" in rules(recommend(*build("r5_fire"), CFG))
    assert "R5" not in rules(recommend(*build("r5_quiet"), CFG))

def test_r6_desk_drop_off():
    assert "R6" in rules(recommend(*build("r6_fire"), CFG))
    assert "R6" not in rules(recommend(*build("r6_quiet"), CFG))
    assert "R6" not in rules(recommend(*build("r6_nodesk"), CFG))

def test_r7_groups_get_less_attention():
    assert "R7" in rules(recommend(*build("r7_fire"), CFG))
    assert "R7" not in rules(recommend(*build("r7_quiet"), CFG))

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
