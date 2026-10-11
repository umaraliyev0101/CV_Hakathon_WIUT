"""RTMPose top-down pose estimation (vendored rtmlib, onnxruntime-only)."""
from __future__ import annotations

import numpy as np

from pulse.datatypes import BBox
from pulse.perception.models import ensure_model
from pulse.perception.vendor_rtmlib.rtmpose import RTMPose


class PoseEstimator:
    def __init__(self, name: str, providers=("CPUExecutionProvider",), models_dir="models"):
        self.name = name
        self.model = RTMPose(str(ensure_model(name, models_dir)), model_input_size=(192, 256),
                             providers=providers)

    def pose(self, frame_bgr: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        """Returns [N, 17, 3] (x, y, score) in frame pixels."""
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        kps, scores = self.model(frame_bgr, [list(b) for b in boxes])
        return np.concatenate([kps, scores[..., None]], -1).astype(np.float32)
