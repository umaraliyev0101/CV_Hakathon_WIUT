import pytest
from pulse.config import get_config
from pulse.datatypes import TrackObs, StableSegment, Sample
from pulse import timeline as T

CFG = get_config("short")

def walker(tid, x0, vx_px, n, y=400, t0=0.0, h=200):
    return [TrackObs(round(t0 + i * 0.2, 3), i, tid, (x0 + vx_px * i - 30, y - h, x0 + vx_px * i + 30, y), 0.9, None)
            for i in range(n)]

def test_build_timeline_maps_tracks_to_persons_and_normalises():
    obs = walker(1, 100, 0, 5) + walker(2, 500, 0, 5, t0=1.0)
    tl = T.build_timeline(obs, {1: 10, 2: 10}, (960, 540), CFG)
    assert list(tl) == [10]
    s = tl[10]
    assert len(s) == 10 and s[0].foot == pytest.approx((100 / 960, 400 / 540))
    assert all(a.t < b.t for a, b in zip(s, s[1:]))

def test_speed_still_vs_walking():
    still = T.build_timeline(walker(1, 300, 0, 20), {1: 1}, (960, 540), CFG)[1]
    assert max(x.speed for x in still) < 0.05
    # 200 px tall person at 1.7 m → 0.0085 m/px; 20 px per 0.2 s = 100 px/s ≈ 0.85 m/s
    walk = T.build_timeline(walker(2, 100, 20, 20), {2: 2}, (960, 540), CFG)[2]
    mid = [x.speed for x in walk[5:15]]
    assert all(0.6 < v < 1.1 for v in mid)

def test_filter_stable_drops_moving_segments():
    tl = {1: [Sample(t, 1, (0.5, 0.5), (0, 0, 1, 1), None, 0.0) for t in (0.0, 1.0, 2.0, 3.0)]}
    segs = [StableSegment(0.0, 1.5, True, True), StableSegment(1.5, 2.5, False, True),
            StableSegment(2.5, 4.0, True, False)]
    assert [s.t for s in T.filter_stable(tl, segs)[1]] == [0.0, 1.0]

def test_already_present_and_edge():
    from pulse.datatypes import Person
    assert T.already_present(Person(1, [1], 2.0, 9.0), 3.0)
    assert not T.already_present(Person(1, [1], 3.5, 9.0), 3.0)
    s = Sample(0, 1, (0.01, 0.5), (2, 100, 40, 300), None, 0.0)
    assert T.at_edge(s, (960, 540), 0.02)
    s2 = Sample(0, 1, (0.5, 0.5), (400, 100, 440, 300), None, 0.0)
    assert not T.at_edge(s2, (960, 540), 0.02)

# --- extra coverage for controller rulings ---
def test_same_frame_two_tracks_keep_higher_score():
    a = TrackObs(0.0, 0, 1, (70, 200, 130, 400), 0.5, None)
    b = TrackObs(0.0, 0, 2, (470, 200, 530, 400), 0.9, None)
    s = T.build_timeline([a, b], {1: 7, 2: 7}, (960, 540), CFG)[7]
    assert len(s) == 1 and s[0].foot[0] == pytest.approx(500 / 960)

def test_invalid_position_stored_as_none():
    bad = TrackObs(0.0, 0, 1, (70, 200, 130, 200), 0.9, None)  # zero-height box
    s = T.build_timeline([bad], {1: 1}, (960, 540), CFG)[1]
    assert s[0].pos_m is None and s[0].dist_method == "invalid" and s[0].speed is None

def test_non_finite_foot_dropped_and_unmapped_ignored():
    nan = float("nan")
    bad = TrackObs(0.0, 0, 1, (nan, 200, nan, nan), 0.9, None)
    ok = TrackObs(0.2, 1, 3, (70, 200, 130, 400), 0.9, None)
    tl = T.build_timeline([bad, ok], {1: 1}, (960, 540), CFG)
    assert tl == {}

def test_smooth_speed_basic_and_short():
    assert T.smooth_speed([0.0], [(0, 0)], [5.0], 700.0, 1.0) == [None]
    ts = [i * 0.2 for i in range(10)]
    v = T.smooth_speed(ts, [(100 + 20 * i, 400) for i in range(10)], [5.0] * 10, 500.0, 0.4)
    assert v[5] == pytest.approx(100 * 5.0 / 500.0, rel=0.05)

def test_filter_stable_last_segment_includes_t1_and_outside_kept():
    tl = {1: [Sample(t, 1, (0.5, 0.5), (0, 0, 1, 1), None, 0.0) for t in (0.0, 1.0, 5.0)], 2: [
        Sample(1.0, 2, (0.5, 0.5), (0, 0, 1, 1), None, 0.0)]}
    segs = [StableSegment(0.0, 1.0, True), StableSegment(1.0, 2.0, False)]
    out = T.filter_stable(tl, segs)
    assert [s.t for s in out[1]] == [0.0, 5.0] and out[2] == []
    segs2 = [StableSegment(0.0, 1.0, False)]
    assert [s.t for s in T.filter_stable(tl, segs2)[1]] == [5.0]

def test_smooth_speed_does_not_difference_across_gap():
    ts = [0, .2, .4, 5.4, 5.6, 5.8]
    feet = [(100, 400)] * 3 + [(400, 400)] * 3
    v = T.smooth_speed(ts, feet, [5.0] * 6, 500.0, 1.0)
    assert all(x == pytest.approx(0.0) for x in v)  # cross-gap average would be ~0.58 m/s

def test_smooth_speed_lone_sample_run_is_none():
    v = T.smooth_speed([0, .2, .4, 9.0], [(0, 0), (0, 0), (0, 0), (50, 0)], [5.0] * 4, 500.0, 1.0)
    assert v[3] is None and v[0] == pytest.approx(0.0)

def test_smooth_speed_constant_walk_ends_unbiased():
    ts = [i * 0.2 for i in range(20)]
    v = T.smooth_speed(ts, [(100 + 20 * i, 400) for i in range(20)], [5.0] * 20, 500.0, 1.0)
    true = 100 * 5.0 / 500.0
    assert v[0] == pytest.approx(true, rel=0.2) and v[-1] == pytest.approx(true, rel=0.2)
