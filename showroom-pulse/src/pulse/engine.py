"""Perception engine: detect persons -> track -> pose on tracked boxes. Works per frame (file or live)."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .config import Config
from .datatypes import Det, TrackObs
from .perception import finetuned
from .perception.detect import VEHICLE_CLASSES, Detector
from .perception.device import default_providers
from .perception.pose import PoseEstimator
from .tracking import Tracker

_WINDOW = 50


class PerceptionEngine:
    def __init__(self, cfg: Config, providers=None):
        self.cfg = cfg
        providers = default_providers() if providers is None else providers   # ORT_DEVICE=cuda -> GPU
        self.providers = tuple(providers)
        self.detector = Detector(cfg.detector, cfg, providers=providers)     # vehicles (and persons by default)
        # PULSE_RFDETR_FT: our fine-tuned person-only RF-DETR handles persons; cars stay on cfg.detector
        ft = finetuned.path_from_env()
        self._person_detector = Detector.finetuned(cfg, ft, providers) if ft is not None else None
        self.pose_est = PoseEstimator(cfg.pose_model, providers=providers)
        self.tracker = Tracker(cfg.sample_fps, cfg)
        self.last_person_dets: list[Det] = []   # raw person boxes of the last frame (privacy: blur these too)
        self._ms = {k: deque(maxlen=_WINDOW) for k in ("det", "track", "pose")}

    @property
    def person_detector(self) -> Detector:
        """The model used for persons: the fine-tuned one when configured, else `detector`."""
        return getattr(self, "_person_detector", None) or self.detector

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
        raw = self.person_detector.detect(frame, ("person",), score_thr=low)
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

    @property
    def vehicle_detector(self) -> Detector:
        """The model used for cars (cfg.vehicle_detector), built on first use; `detector` when they are the same."""
        if self.cfg.vehicle_detector == self.cfg.detector:
            return self.detector
        if getattr(self, "_vehicle_detector", None) is None:
            self._vehicle_detector = Detector(self.cfg.vehicle_detector, self.cfg, providers=self.providers)
        return self._vehicle_detector

    def detect_vehicles(self, frame: np.ndarray) -> list[Det]:
        return self.vehicle_detector.detect(frame, tuple(VEHICLE_CLASSES))
