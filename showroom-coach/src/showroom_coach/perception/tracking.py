"""Person tracking: a thin wrapper over supervision's ByteTrack (MIT) that speaks `Det`.

ByteTrack associates boxes frame to frame with a Kalman motion model and keeps lost tracks alive for a
while, so a person who is briefly hidden (or missed by the detector for a few sampled frames) comes back with
the same id. Longer gaps and look-alikes are handled later by `identity` (appearance-based merging).
"""

from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from ..types import Det

PERSON = 0  # COCO class id


class Tracker:
    """Frame-by-frame person tracker.

    `fps` is the rate at which `update` is called (the sampled frame rate, not the video's), `lost_seconds` how
    long a track is kept without a matching box, `activation` the confidence a box needs to join a track and
    `match_thr` ByteTrack's matching threshold on the IoU cost. Track ids are positive ints, unique per tracker.

    ByteTrack starts a brand-new track only for a box scoring at least `activation + 0.1`; lower-scoring boxes
    (down to 0.1) can only extend existing tracks. The default `activation=0.25` therefore puts that floor at 0.35,
    the same as `Detector.score_thr`, so every box the detector reports can start a track (with the usual 0.30 a
    person steadily detected at 0.37 never got one: 0 of 10 frames).

    Two ByteTrack behaviours to know about:
    - A track born after the very first frame is confirmed only when it is matched again on the next one, so such
      a person is first returned on their second sighting (0.2 s at 5 fps); a box seen once and never again
      produces no track at all, which filters one-frame false detections.
    - supervision hard-codes the IoU needed for that confirming match at 0.7 (not configurable). A person who
      moves so far sideways between two sampled frames that their boxes overlap less than that is never
      confirmed: at 5 fps a fast sideways walker (about 1.8 m/s and up) may never get a track (measured: 0 of 25
      frames at 1.8 m/s against 24 of 25 at 1.4 m/s). Sample faster for such scenes.
    """

    def __init__(self, fps: float = 5.0, lost_seconds: float = 2.0, activation: float = 0.25,
                 match_thr: float = 0.8):
        with warnings.catch_warnings():
            # supervision >= 0.28 announces ByteTrack's move to the `trackers` package; the class still works
            warnings.filterwarnings("ignore", message=r".*ByteTrack.*deprecated", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r".*ByteTrack.*deprecated", category=DeprecationWarning)
            self._bt = sv.ByteTrack(
                track_activation_threshold=activation,
                lost_track_buffer=int(round(lost_seconds * 30)),   # frames at 30 fps; ByteTrack rescales by fps
                minimum_matching_threshold=match_thr,
                frame_rate=fps,
                minimum_consecutive_frames=1,
            )

    def update(self, dets: list[Det]) -> list[tuple[int, Det]]:
        """Feed one frame's detections (non-person classes are ignored) and get back `(track_id, det)` for every
        person box that belongs to a live track. `det` is the very object that was passed in, not a prediction."""
        people = [d for d in dets if d.cls == PERSON]
        n = len(people)
        detections = sv.Detections(
            xyxy=np.asarray([d.bbox for d in people], dtype=np.float32).reshape(n, 4),
            confidence=np.asarray([d.score for d in people], dtype=np.float32),
            class_id=np.zeros(n, dtype=int),
            data={"det_index": np.arange(n)},
        )
        tracked = self._bt.update_with_detections(detections)
        if len(tracked) == 0 or tracked.tracker_id is None:
            return []
        out = []
        for tid, idx in zip(tracked.tracker_id, tracked.data["det_index"]):
            if int(tid) > 0:
                out.append((int(tid), people[int(idx)]))
        return out
