import pytest
from pulse.config import get_config
from pulse.datatypes import Det
from pulse.tracking import Tracker


def _dets(i, skip_b=False):
    out = [Det((100 + 5 * i, 100, 160 + 5 * i, 260), 0.9, "person")]
    if not skip_b:
        out.append(Det((400 - 5 * i, 120, 460 - 5 * i, 280), 0.9, "person"))
    return out


def test_two_moving_boxes_keep_stable_ids():
    tr = Tracker(5.0, get_config("short"))
    ids = []
    for i in range(20):
        obs = tr.update(_dets(i), i * 0.2, i)
        ids.append({o.track_id for o in obs})
        assert all(o.keypoints is None and o.frame_idx == i for o in obs)
    stable = ids[-1]
    assert len(stable) == 2
    assert all(s == stable for s in ids[2:])


def test_short_gap_keeps_id():
    tr = Tracker(5.0, get_config("short"))
    seen = {}
    for i in range(20):
        obs = tr.update(_dets(i, skip_b=8 <= i < 11), i * 0.2, i)
        for o in obs:
            seen.setdefault(round(o.bbox[1]), set()).add(o.track_id)
    assert all(len(v) == 1 for v in seen.values()) and len(seen) == 2


def test_reset():
    tr = Tracker(5.0, get_config("short"))
    for i in range(5):
        tr.update(_dets(i), i * 0.2, i)
    tr.reset()
    obs = tr.update(_dets(0), 0.0, 0)
    assert len(obs) == 2


def test_gap_longer_than_buffer_gives_new_id():
    tr = Tracker(5.0, get_config("short"))  # buffer = 10 sampled frames
    first = None
    for i in range(30):
        obs = tr.update(_dets(i, skip_b=6 <= i < 20), i * 0.2, i)
        if i == 0:
            first = [o.track_id for o in obs if round(o.bbox[1]) == 120][0]
    last = [o.track_id for o in obs if round(o.bbox[1]) == 120][0]
    assert last != first


def test_obs_carry_detection_bbox_and_score():
    tr = Tracker(5.0, get_config("short"))
    for i in range(3):
        dets = [Det((100 + 5 * i, 100, 160 + 5 * i, 260), 0.87, "person")]
        obs = tr.update(dets, i * 0.2, i)
    assert len(obs) == 1
    assert obs[0].bbox == pytest.approx(dets[0].bbox, abs=1e-3)
    assert obs[0].score == pytest.approx(0.87, abs=1e-6)
    assert (obs[0].t, obs[0].frame_idx) == (pytest.approx(0.4), 2)
