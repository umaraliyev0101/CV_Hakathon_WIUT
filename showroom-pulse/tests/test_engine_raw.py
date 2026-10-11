from types import SimpleNamespace

import numpy as np

from pulse.config import get_config
from pulse.datatypes import Det
from pulse.engine import PerceptionEngine


class _Det:
    def detect(self, frame, classes=("person",), score_thr=None):
        thr = 0.3 if score_thr is None else score_thr
        raw = [Det((10, 10, 50, 100), 0.9, "person"), Det((200, 10, 240, 100), 0.25, "person"),
               Det((100, 10, 140, 100), 0.1, "person")]
        return [d for d in raw if d.score >= thr]


class _Tracker:
    def __init__(self):
        self.seen = []

    def update(self, dets, t, i):
        self.seen.append(list(dets))
        return []


def _engine():
    e = object.__new__(PerceptionEngine)
    e.cfg = get_config("short")
    e.detector, e.tracker = _Det(), _Tracker()
    from collections import deque
    e._ms = {k: deque(maxlen=5) for k in ("det", "track", "pose")}
    e.pose_est = SimpleNamespace()
    return e


def test_tracker_only_gets_dets_above_det_score_thr_and_raw_dets_are_kept():
    e = _engine()
    e.process(0.0, np.zeros((120, 260, 3), np.uint8), 0)
    assert [d.score for d in e.tracker.seen[0]] == [0.9]
    assert sorted(d.score for d in e.last_person_dets) == [0.25, 0.9]
