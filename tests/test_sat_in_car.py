"""Sat-in-car (spec 4.5, stretch M8) on synthetic boxes."""
from pulse import sat_in_car as S
from pulse.config import get_config
from pulse.datatypes import Det, Sample, TrackObs

CFG = get_config("dealership")
CAR = Det((100, 200, 500, 400), 0.9, "car")
FAR_CAR = Det((700, 200, 900, 400), 0.9, "car")
INSIDE = (250, 220, 330, 380)       # person box fully inside the car box
BESIDE = (520, 220, 600, 380)       # next to the car, no overlap
TORSO = [5, 6, 11, 12]


def kps(torso=0.9, legs=0.05):
    return [[0.0, 0.0, torso if i in TORSO else (legs if i >= 13 else 0.5)] for i in range(17)]


def person(pid, t0, seconds, box, k=None, speed=0.0, dt=0.2):
    n = int(round(seconds / dt)) + 1
    obs = [TrackObs(round(t0 + i * dt, 3), i, pid, box, 0.9, k) for i in range(n)]
    sam = [Sample(o.t, pid, (0.5, 0.7), box, None, speed, "full") for o in obs]
    return obs, sam


def run(*parts, cars=(CAR,)):
    obs = {}
    tl = {}
    for o, s in parts:
        obs.setdefault(o[0].track_id, []).extend(o)
        tl.setdefault(o[0].track_id, []).extend(s)
    return S.detect(tl, obs, list(cars), CFG)


def test_rule_a_torso_only_still_inside_car_fires():
    ev = run(person(1, 0, 6, INSIDE, kps()))
    assert len(ev) == 1
    e = ev[0]
    assert e.kind == "sat_in_car" and e.person_ids == [1]
    assert e.data["rule"] == "a" and e.data["car"] == 0
    assert e.t_start == 0.0 and e.t_end >= 5.0


def test_rule_a_needs_five_seconds():
    assert run(person(1, 0, 4, INSIDE, kps())) == []


def test_rule_a_needs_60_percent_inside():
    half = (450, 220, 570, 380)         # 50/120 of the width inside
    assert run(person(1, 0, 8, half, kps())) == []


def test_rule_a_moving_person_does_not_fire():
    assert run(person(1, 0, 8, INSIDE, kps(), speed=0.5)) == []


def test_standing_next_to_car_with_ankles_visible_does_not_fire():
    # in front of the car (overlapping it) but legs fully visible: standing, not sitting
    assert run(person(1, 0, 8, INSIDE, kps(legs=0.9))) == []
    assert run(person(1, 0, 8, BESIDE, kps(legs=0.9))) == []
    assert run(person(1, 0, 8, BESIDE, kps())) == []     # torso only but not over the car


def test_walking_past_car_does_not_fire():
    obs, sam = [], []
    for i in range(30):
        x = 100 + i * 12
        box = (x, 220, x + 80, 380)
        obs.append(TrackObs(i * 0.2, i, 1, box, 0.9, kps()))
        sam.append(Sample(i * 0.2, 1, (0.5, 0.7), box, None, 1.2, "full"))
    assert run((obs, sam)) == []


def test_no_keypoints_does_not_fire():
    assert run(person(1, 0, 8, INSIDE, None)) == []


def test_rule_b_disappear_and_reappear_in_car_zone():
    a = person(1, 0, 3, INSIDE, kps(legs=0.9), speed=1.0)       # walks to car, track ends inside it
    b = person(2, 20, 3, INSIDE, kps(legs=0.9), speed=1.0)      # a track starts inside the car later
    ev = run(a, b)
    assert len(ev) == 1
    e = ev[0]
    assert e.kind == "sat_in_car" and e.data["rule"] == "b" and e.person_ids == [1]
    assert e.t_start == a[0][-1].t and e.t_end == b[0][0].t and e.data["car"] == 0


def test_rule_b_same_person_gap_counts():
    a = person(1, 0, 3, INSIDE, kps(legs=0.9), speed=1.0)
    b = person(1, 15, 3, INSIDE, kps(legs=0.9), speed=1.0)
    ev = run(a, b)
    assert [e.data["rule"] for e in ev] == ["b"] and ev[0].person_ids == [1]


def test_rule_b_reappear_too_late_or_elsewhere_does_not_fire():
    a = person(1, 0, 3, INSIDE, kps(legs=0.9), speed=1.0)
    assert run(a, person(2, 90, 3, INSIDE, kps(legs=0.9), speed=1.0)) == []      # > 60 s
    assert run(a, person(2, 20, 3, BESIDE, kps(legs=0.9), speed=1.0)) == []      # not at the car
    assert run(a, person(2, 20, 3, (710, 220, 790, 380), kps(legs=0.9), speed=1.0)) == []  # another car (not in cars)


def test_rule_b_track_ending_beside_car_does_not_fire():
    a = person(1, 0, 3, BESIDE, kps(legs=0.9), speed=1.0)
    assert run(a, person(2, 20, 3, INSIDE, kps(legs=0.9), speed=1.0)) == []


def test_non_vehicle_dets_ignored_and_empty_inputs():
    assert S.detect({}, {}, [], CFG) == []
    assert run(person(1, 0, 8, INSIDE, kps()), cars=(Det(INSIDE, 0.9, "person"),)) == []


def test_assign_zones_maps_event_to_zone():
    from pulse.zones import auto_zones
    zs = auto_zones([CAR, FAR_CAR], (540, 960))
    ev = run(person(1, 0, 6, INSIDE, kps()))
    S.assign_zones(ev, [CAR], zs, (960, 540))
    assert ev[0].data["zone_id"] == "car-1"


def test_per_car_sat_in_car_column():
    from pulse import stats as ST
    from pulse.datatypes import Event, Person, Zone
    z = Zone("car-1", "Car 1", "car", [(0.1, 0.4), (0.5, 0.4), (0.5, 0.9), (0.1, 0.9)], "auto")
    z2 = Zone("car-2", "Car 2", "car", [(0.6, 0.4), (0.9, 0.4), (0.9, 0.9), (0.6, 0.9)], "auto")
    evs = [Event("sat_in_car", 0, 6, [1], {"rule": "a", "zone_id": "car-1"}),
           Event("sat_in_car", 9, 15, [1], {"rule": "a", "zone_id": "car-1"}),
           Event("sat_in_car", 0, 6, [2], {"rule": "b", "zone_id": "car-1"})]
    st = ST.compute([Person(1, [1], 0, 1)], {}, {}, CFG, timeline={}, episodes=[], assist={},
                    zones=[z, z2], duration=10.0, frame_size=(960, 540), sat_events=evs)
    cars = {c["zone_id"]: c for c in st.per_car}
    assert cars["car-1"]["sat_in_car"] == 2 and cars["car-2"]["sat_in_car"] == 0
