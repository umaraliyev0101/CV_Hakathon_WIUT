"""Perception engine: detect persons -> track -> pose on tracked boxes. Works per frame (file or live)."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .config import Config
from .datatypes import Det, TrackObs
from .perception.detect import VEHICLE_CLASSES, Detector
from .perception.pose import PoseEstimator
from .tracking import Tracker

_WINDOW = 50


class PerceptionEngine:
    def __init__(self, cfg: Config, providers=("CPUExecutionProvider",)):
        self.cfg = cfg
        self.providers = tuple(providers)
        self.detector = Detector(cfg.detector, cfg, providers=providers)
        self.pose_est = PoseEstimator(cfg.pose_model, providers=providers)
        self.tracker = Tracker(cfg.sample_fps, cfg)
        self.last_person_dets: list[Det] = []   # raw person boxes of the last frame (privacy: blur these too)
        self._ms = {k: deque(maxlen=_WINDOW) for k in ("det", "track", "pose")}

    @property
    def timings(self) -> dict[str, float]:
        """Rolling mean ms per stage."""
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in self._ms.items()}

    def reset(self) -> None:
        """Forget tracker state and timings (call before each new video)."""
        self.tracker.reset()
        self.last_person_dets = []
        for d in self._ms.values():
            d.clear()

    def _time(self, key: str, t0: float) -> None:
        self._ms[key].append((time.perf_counter() - t0) * 1000.0)

    def process(self, t: float, frame: np.ndarray, frame_idx: int) -> list[TrackObs]:
        t0 = time.perf_counter()
        low = min(0.2, self.cfg.det_score_thr)
        raw = self.detector.detect(frame, ("person",), score_thr=low)
        self.last_person_dets = raw
        dets = [d for d in raw if d.score > self.cfg.det_score_thr]    # tracker input unchanged
        self._time("det", t0)
        t0 = time.perf_counter()
        obs = self.tracker.update(dets, t, frame_idx)
        self._time("track", t0)
        if obs:
            t0 = time.perf_counter()
            kps = self.pose_est.pose(frame, [o.bbox for o in obs])
            assert len(kps) == len(obs), "pose returned a different number of skeletons than boxes"
            for o, kp in zip(obs, kps):
                o.keypoints = [[float(v) for v in row] for row in kp]
            self._time("pose", t0)
        return obs

    def detect_vehicles(self, frame: np.ndarray) -> list[Det]:
        return self.detector.detect(frame, tuple(VEHICLE_CLASSES))
