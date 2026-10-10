import pytest
from pulse.config import get_config
from pulse.datatypes import Person
from pulse import stats as ST, journey as J, interest as I
from tests.conftest import stay, track

CFG = get_config("short")

def scenario(zones):
    tl = {
        # already present, browses car A 12 s (deep, unserved), no assist
        1: stay(1, 0.2, 0.7, 0, 12, speed=0.05, pos=(0.0, 4.0)),
        # enters via entrance at t=5, stops at car B 5 s, walks to desk 6 s
        2: stay(2, 0.95, 0.5, 5, 1, pos=(6.0, 4.0)) + stay(2, 0.7, 0.7, 6.2, 5, speed=0.05, pos=(3.0, 4.0))
           + stay(2, 0.45, 0.15, 11.4, 6, pos=(1.0, 8.0)),
        # group with 2 (arrives within 5 s, stays 1 m away)
        3: stay(3, 0.95, 0.55, 6, 1, pos=(6.0, 5.0)) + stay(3, 0.72, 0.72, 7.2, 4, speed=0.05, pos=(3.5, 4.5)),
        # staff near desk the whole time
        9: stay(9, 0.47, 0.15, 0, 20, pos=(1.5, 8.0)),
    }
    persons = [Person(pid, [pid], s[0].t, s[-1].t) for pid, s in tl.items()]
    roles = {1: "customer", 2: "customer", 3: "unsure", 9: "staff"}
    cust = {k: v for k, v in tl.items() if roles[k] != "staff"}
    staff = {9: tl[9]}
    journeys = J.build(tl, zones, CFG.preset)
    eps = I.episodes(journeys, staff, CFG.preset, tl, zones)
    assist = I.assistance(cust, staff, CFG.preset)
    return ST.compute(persons, journeys, roles, CFG, timeline=tl, episodes=eps, assist=assist,
                      zones=zones, duration=20.0, frame_size=(960, 540))

def test_store_kpis(zones4):
    s = scenario(zones4).store
    assert s["visitors"] == 3 and s["staff"] == 1 and s["unsure"] == 1
    assert s["already_present"] == 1 and s["entries"] == 2
    assert s["unserved_episodes"] == 1 and s["deep_interest_visitors"] == 1
    assert s["has_desk"] and s["reached_desk"] == 1
    assert s["assisted_customers"] == 1          # customer 2 at the desk next to staff
    assert s["browse_rate"] == pytest.approx(3 / 3)
    assert s["groups"] == 1 and s["n_group_members"] == 2 and s["n_solo"] == 1
    assert s["median_visit_s"] == pytest.approx((12.4 + 5.0) / 2, abs=0.3)

def test_per_car_table(zones4):
    cars = {c["zone_id"]: c for c in scenario(zones4).per_car}
    assert set(cars) == {"car-a", "car-b"}
    assert cars["car-b"]["visitors"] == 2 and cars["car-b"]["stops"] == 2
    assert cars["car-a"]["unserved"] == 1 and cars["car-a"]["unassisted_share"] == 1.0
    assert cars["car-a"]["stop_rate"] == pytest.approx(1 / 3)
    assert sorted(c["rank"] for c in cars.values()) == [1, 2]

def test_occupancy_and_peak(zones4):
    st = scenario(zones4)
    occ = dict((t, n) for t, n in st.occupancy)
    assert occ[0] == 1 and occ[8] == 3 and st.store["peak_occupancy"] == 3
    assert all(r == -1 or r >= 0 for _, r in st.crowding)

def test_funnel_shares(zones4):
    f = scenario(zones4).store["funnel"]
    assert f["entered"] == 1.0 and f["deep_interest"] == pytest.approx(1 / 3)
    assert f["reached_desk"] == pytest.approx(1 / 3)

def test_heatmap_counts_customer_samples_only(zones4):
    st = scenario(zones4)
    total = sum(sum(r) for r in st.heatmap["counts"])
    n_customer_samples = 61 + (6 + 26 + 31) + (6 + 21)
    assert total == n_customer_samples

def test_empty_inputs_no_crash():
    st = ST.compute([], {}, {}, CFG, timeline={}, episodes=[], assist={}, zones=[], duration=10.0,
                    frame_size=(960, 540))
    assert st.store["visitors"] == 0 and st.per_car == []
    assert any("Nobody detected" in w for w in st.warnings)
    assert any("No zones" in w for w in st.warnings)

def test_entries_by_frame_edge_without_entrance_zone(zones4):
    no_ent = [z for z in zones4 if z.kind != "entrance"]
    tl = {5: track(5, [(0.0, 0.6)] + [(0.1, 0.6)] * 30, t0=6.0)}
    tl[5][0].bbox = (0, 200, 30, 320)                   # touching the left edge
    persons = [Person(5, [5], 6.0, tl[5][-1].t)]
    j = J.build(tl, no_ent, CFG.preset)
    st = ST.compute(persons, j, {5: "customer"}, CFG, timeline=tl, episodes=[], assist={},
                    zones=no_ent, duration=20.0, frame_size=(960, 540))
    assert st.store["entries"] == 1
