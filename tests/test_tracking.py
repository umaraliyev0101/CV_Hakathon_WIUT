import warnings

from showroom_coach.perception.tracking import Tracker
from showroom_coach.types import Det


def test_two_movers_keep_ids():
    tr = Tracker(fps=5.0)
    ids = []
    for k in range(20):
        dets = [Det((100 + 5 * k, 100, 160 + 5 * k, 300), 0.9), Det((600, 100, 660, 300), 0.85),
                Det((10, 10, 40, 40), 0.9, cls=2)]           # a car is ignored
        out = tr.update(dets)
        assert len(out) == 2
        ids.append(tuple(sorted(i for i, _ in out)))
    assert len(set(ids[2:])) == 1


def test_short_occlusion_keeps_id():
    tr = Tracker(fps=5.0, lost_seconds=2.0)
    first = tr.update([Det((100, 100, 160, 300), 0.9)])[0][0]
    for _ in range(5):
        tr.update([])
    again = tr.update([Det((104, 100, 164, 300), 0.9)])
    assert again and again[0][0] == first


def test_returns_the_input_detections_with_positive_int_ids():
    tr = Tracker(fps=5.0)
    for k in range(6):
        dets = [Det((100 + 7 * k, 100, 160 + 7 * k, 300), 0.9), Det((600, 100 + 3 * k, 660, 300 + 3 * k), 0.8)]
        out = tr.update(dets)
        assert len(out) == 2
        for tid, det in out:
            assert type(tid) is int and tid > 0
            assert any(det is d for d in dets)          # the observed box, not a Kalman prediction
        assert {id(d) for _, d in out} == {id(d) for d in dets}


def test_empty_and_non_person_frames():
    tr = Tracker(fps=5.0)
    assert tr.update([]) == []
    assert tr.update([Det((10, 10, 40, 40), 0.9, cls=2), Det((60, 60, 90, 90), 0.9, cls=56)]) == []
    for _ in range(2):          # ByteTrack confirms a track born after the first frame on its second sighting
        out = tr.update([Det((100, 100, 160, 300), 0.9)])
    assert len(out) == 1 and out[0][0] >= 1


def test_long_gap_starts_a_new_track():
    tr = Tracker(fps=5.0, lost_seconds=1.0)               # 1 s at 5 fps = 5 frames of memory
    first = tr.update([Det((100, 100, 160, 300), 0.9)])[0][0]
    for _ in range(9):
        tr.update([])
    for _ in range(2):
        again = tr.update([Det((100, 100, 160, 300), 0.9)])
    assert again and again[0][0] != first


def test_no_library_warnings_leak():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr = Tracker(fps=5.0)
        tr.update([Det((100, 100, 160, 300), 0.9)])
        tr.update([])
