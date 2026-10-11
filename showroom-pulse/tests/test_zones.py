import pytest
from pulse.datatypes import Det, Zone
from pulse import zones as Z

FS = (540, 960)  # (h, w)

def n(x, y): return (x / 960, y / 540)

def test_auto_zones_two_cars_left_to_right():
    dets = [Det((500, 220, 800, 420), 0.8, "car"), Det((100, 200, 400, 400), 0.9, "car"),
            Det((0, 0, 10, 10), 0.9, "person")]
    zs = Z.auto_zones(dets, FS)
    assert [z.name for z in zs] == ["Car 1", "Car 2"]
    assert all(z.kind == "car" and z.source == "auto" for z in zs)
    assert Z.zone_of(n(250, 430), zs) == zs[0].zone_id     # just below the car: feet of a viewer
    assert Z.zone_of(n(250, 205), zs) is None               # roof is not floor
    assert Z.zone_of(n(90, 380), zs) == zs[0].zone_id       # beside the car (apron)
    for z in zs:
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in z.polygon)

def test_auto_zones_merges_duplicates_and_drops_tiny():
    dets = [Det((100, 200, 400, 400), 0.9, "car"), Det((110, 205, 405, 400), 0.6, "car"),
            Det((900, 10, 905, 14), 0.9, "car")]
    zs = Z.auto_zones(dets, FS)
    assert len(zs) == 1

def test_motorcycle_named():
    zs = Z.auto_zones([Det((100, 300, 200, 400), 0.9, "motorcycle")], FS)
    assert zs[0].name == "Motorcycle 1" and zs[0].kind == "car"

def test_point_in_polygon_concave_and_boundary():
    L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert Z.point_in_polygon((0.5, 1.5), L)
    assert not Z.point_in_polygon((1.5, 1.5), L)
    assert Z.point_in_polygon((2, 0.5), L)

def test_zone_of_prefers_smallest():
    big = Zone("big", "Floor", "other", [(0, 0), (1, 0), (1, 1), (0, 1)])
    small = Zone("s", "Desk", "desk", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    assert Z.zone_of((0.5, 0.5), [big, small]) == "s"
    assert Z.zone_of((0.1, 0.1), [big, small]) == "big"
    assert Z.zone_of((0.5, 0.5), []) is None

def test_hysteresis_bridges_short_exit():
    ts = [round(i * 0.2, 3) for i in range(50)]
    inside = [False] * 5 + [True] * 20 + [False] * 3 + [True] * 12 + [False] * 10
    assert Z.presence_intervals(ts, inside, zone_min=1.0, zone_gap=1.0) == [pytest.approx((1.0, 7.8))]

def test_hysteresis_ignores_blip_and_splits_long_gap():
    ts = [round(i * 0.2, 3) for i in range(60)]
    blip = [False] * 10 + [True] * 3 + [False] * 47              # 0.4 s inside
    assert Z.presence_intervals(ts, blip, 1.0, 1.0) == []
    two = [True] * 10 + [False] * 10 + [True] * 10 + [False] * 30  # gap 2.2 s > 1 s
    assert Z.presence_intervals(ts, two, 1.0, 1.0) == [pytest.approx((0.0, 1.8)), pytest.approx((4.0, 5.8))]

def test_hysteresis_missing_observations_count_as_gap():
    ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]   # unseen 1.0 → 4.0
    assert len(Z.presence_intervals(ts, [True] * 12, 1.0, 2.0)) == 2

def test_save_load_roundtrip(tmp_path):
    zs = [Zone("desk", "Desk", "desk", [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], "user")]
    p = tmp_path / "z.json"
    Z.save_zones(p, zs, (960, 540))
    assert [z.to_dict() for z in Z.load_zones(p)] == [
        {**zs[0].to_dict(), "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}]
