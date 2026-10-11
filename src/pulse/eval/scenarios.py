"""Insight-rule scenarios (R1-R7): the exact inputs the unit tests in tests/test_insights.py use.

Shared by the tests and by `pulse eval`, so the EVAL.md insights section runs the same inputs through
`insights.recommend` instead of quoting them. Each scenario either fires its rule or stays quiet."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pulse.datatypes import InterestEpisode, Stats


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


def _glance(a=(0.8, 2.0)):
    return [car("a", *a), car("b", 0.3, 20), car("c", 0.35, 25), car("d", 0.4, 30)]


def _long_crowd():
    return [(t, 4.0) for t in range(0, 150)]


def _short_crowd():
    return [(t, 4.0) for t in range(0, 60)] + [(t, 1.0) for t in range(60, 150)]


@dataclass(frozen=True)
class Scenario:
    key: str
    rule: str
    fires: bool                    # whether the rule is expected to fire
    describe: str
    build: Callable[[], tuple[Stats, list[InterestEpisode]]]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("r1_fire", "R1", True, "3 deep-interest episodes at car A, none assisted",
             lambda: (stats([car("a", deep=3, unserved=3)]), eps("a", 3, 0))),
    Scenario("r1_quiet", "R1", False, "5 deep-interest episodes at car A, 4 assisted",
             lambda: (stats([car("a", deep=5, unserved=1)]), eps("a", 1, 4))),
    Scenario("r2_fire", "R2", True, "car A: stop rate 0.8 but median dwell 2 s (others 0.3-0.4, 20-30 s)",
             lambda: (stats(_glance()), [])),
    Scenario("r2_quiet", "R2", False, "car A: stop rate 0.8 and median dwell 40 s",
             lambda: (stats(_glance((0.8, 40.0))), [])),
    Scenario("r3_fire", "R3", True, "car A: stop rate 0.05 while the others have 0.4 and 0.5",
             lambda: (stats([car("a", 0.05), car("b", 0.4), car("c", 0.5)]), [])),
    Scenario("r3_quiet", "R3", False, "car A: stop rate 0.05 but the store median is only 0.2",
             lambda: (stats([car("a", 0.05), car("b", 0.2), car("c", 0.25)]), [])),
    Scenario("r4_fire", "R4", True, "median time to assist 20 s",
             lambda: (stats(median_time_to_assist=20.0), [])),
    Scenario("r4_quiet", "R4", False, "median time to assist 10 s",
             lambda: (stats(median_time_to_assist=10.0), [])),
    Scenario("r5_fire", "R5", True, "4 customers per staff member for 150 s",
             lambda: (stats(crowding=_long_crowd()), [])),
    Scenario("r5_quiet", "R5", False, "4 customers per staff member for only 60 s",
             lambda: (stats(crowding=_short_crowd()), [])),
    Scenario("r6_fire", "R6", True, "desk exists; 2 of 6 deep-interest visitors reached it",
             lambda: (stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=2), [])),
    Scenario("r6_quiet", "R6", False, "desk exists; 5 of 6 deep-interest visitors reached it",
             lambda: (stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=5), [])),
    Scenario("r6_nodesk", "R6", False, "no desk zone drawn",
             lambda: (stats(has_desk=False, deep_interest_visitors=6, deep_interest_reached_desk=0), [])),
    Scenario("r7_fire", "R7", True, "groups assisted 20% of the time, solo visitors 60%",
             lambda: (stats(groups=3, n_solo=5, group_assisted_share=0.2, solo_assisted_share=0.6), [])),
    Scenario("r7_quiet", "R7", False, "groups assisted 50%, solo visitors 60%",
             lambda: (stats(groups=3, n_solo=5, group_assisted_share=0.5, solo_assisted_share=0.6), [])),
)

BY_KEY: dict[str, Scenario] = {s.key: s for s in SCENARIOS}


def build(key: str) -> tuple[Stats, list[InterestEpisode]]:
    return BY_KEY[key].build()
