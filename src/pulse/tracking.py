"""ByteTrack wrapper (supervision, MIT). Shared unit with Showroom Coach."""
from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from .config import Config
from .datatypes import Det, TrackObs


class Tracker:
    def __init__(self, sample_fps: float, cfg: Config):
        self.sample_fps = sample_fps
        self.cfg = cfg
        self._make()

    def _make(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._bt = sv.ByteTrack(
                track_activation_threshold=self.cfg.track_activation_thr,
                lost_track_buffer=int(self.cfg.track_buffer_s * self.sample_fps),
                minimum_matching_threshold=self.cfg.track_match_thr,
                # supervision scales the buffer by frame_rate/30 (max_time_lost = frame_rate/30*buffer);
                # frame_rate=30 makes lost_track_buffer an exact count of *sampled* frames (2 s).
                frame_rate=30,
            )

    def reset(self) -> None:
        self._make()

    def update(self, dets: list[Det], t: float, frame_idx: int) -> list[TrackObs]:
        if dets:
            xyxy = np.array([d.bbox for d in dets], dtype=np.float32)
            conf = np.array([d.score for d in dets], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), np.float32)
            conf = np.zeros((0,), np.float32)
        sd = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(dets), dtype=int))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._bt.update_with_detections(sd)
        res = []
        for i in range(len(out)):
            b = tuple(float(v) for v in out.xyxy[i])
            res.append(TrackObs(t=t, frame_idx=frame_idx, track_id=int(out.tracker_id[i]), bbox=b,
                                score=float(out.confidence[i]) if out.confidence is not None else 0.0))
        return res
