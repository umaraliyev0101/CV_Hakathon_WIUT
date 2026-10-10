"""Top-down 17-keypoint (COCO order) pose estimation with RTMPose."""

from __future__ import annotations

import numpy as np

from ..models import ensure_model
from . import rtm_onnx as R

NUM_KEYPOINTS = 17
MIN_BOX_HEIGHT = 40.0  # px; keypoints on a smaller person are noise and cost ~11 ms each


class PoseEstimator:
    """RTMPose-S on person boxes. `__call__(frame_bgr, boxes (N,4), min_height=40) -> (N,17,3)`
    with columns x, y (frame pixels) and confidence in [0, 1]. Boxes shorter than `min_height`
    pixels (and degenerate boxes) are not run through the model and get zeros (confidence 0)."""

    def __init__(self, name: str = "rtmpose_s", threads: int | None = None):
        self.name = name
        self.session = R.make_session(ensure_model(name), threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_wh = (int(inp.shape[3]), int(inp.shape[2]))  # model input is NCHW
        self._out_names = [o.name for o in self.session.get_outputs()]  # simcc_x, simcc_y

    def __call__(self, frame_bgr: np.ndarray, boxes: np.ndarray,
                 min_height: float = MIN_BOX_HEIGHT) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = np.zeros((len(boxes), NUM_KEYPOINTS, 3), np.float32)
        for i, box in enumerate(boxes):
            if not (box[2] - box[0] >= 1 and box[3] - box[1] >= max(1.0, min_height)):
                continue
            blob, center, scale = R.rtmpose_preprocess(frame_bgr, box, self.input_wh)
            sx, sy = self.session.run(self._out_names, {self.input_name: blob})
            kpts, conf = R.rtmpose_decode(sx, sy, center, scale, self.input_wh)
            out[i, :, :2] = kpts
            out[i, :, 2] = conf
        return out
