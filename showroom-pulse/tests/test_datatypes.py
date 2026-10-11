import json
from pulse.datatypes import (Det, TrackObs, Person, RoleSuggestion, Event, Zone, Sample,
                             ZoneVisit, InterestEpisode, Insight, StableSegment, Stats)

def roundtrip(obj):
    d = json.loads(json.dumps(obj.to_dict()))
    return type(obj).from_dict(d)

def test_roundtrip_all_types():
    objs = [
        Det((1, 2, 3, 4), 0.9, "person"),
        TrackObs(0.2, 1, 7, (1, 2, 3, 4), 0.8, [[1.0, 2.0, 0.9]] * 17),
        Person(1, [7, 9], 0.0, 12.5),
        RoleSuggestion(1, "staff", 0.8, ["same outfit as 2 others"], {"contacts": 3.0}),
        Event("unserved_interest", 1.0, 5.0, [1], {"zone_id": "car-1"}),
        Zone("car-1", "Car 1", "car", [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)], "auto"),
        Sample(0.2, 1, (0.5, 0.9), (10, 20, 30, 40), (0.1, 3.0), 0.2, "full"),
        ZoneVisit(1, "car-1", 1.0, 5.0, True),
        InterestEpisode(1, "car-1", 1.0, 13.0, 12.0, 1, False, True, []),
        Insight("R1", "t", {"n": 1}, "a", ["car-1"], (0.0, 60.0), "low", 2, [{"t": 1.0}], True),
        StableSegment(0.0, 10.0, True, True),
    ]
    for o in objs:
        assert roundtrip(o).to_dict() == json.loads(json.dumps(o.to_dict()))

def test_zone_visit_dwell():
    assert ZoneVisit(1, "a", 2.0, 5.5).dwell == 3.5

def test_stats_empty_factory():
    s = Stats.empty()
    assert s.store["visitors"] == 0 and s.per_car == [] and s.occupancy == []
