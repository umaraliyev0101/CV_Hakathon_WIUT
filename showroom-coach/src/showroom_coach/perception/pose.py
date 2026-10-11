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
    pixels (and degenerate boxes) are not run through the model and get zeros (confidence 0).

    All eligible boxes of a call go through the model in ONE `session.run` (the exported model has a dynamic batch
    axis); each box is preprocessed and decoded on its own exactly as before, so batching changes the cost, not the
    keypoints. A model exported with a fixed batch size is fed that many boxes at a time instead (the last group is
    padded with a repeat and the padding rows are dropped).
    """

    def __init__(self, name: str = "rtmpose_s", threads: int | None = None):
        self.name = name
        self.session = R.make_session(ensure_model(name), threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_wh = (int(inp.shape[3]), int(inp.shape[2]))  # model input is NCHW
        batch = inp.shape[0]
        self._fixed_batch = batch if isinstance(batch, int) and batch > 0 else None
        self._out_names = [o.name for o in self.session.get_outputs()]  # simcc_x, simcc_y

    def __call__(self, frame_bgr: np.ndarray, boxes: np.ndarray,
                 min_height: float = MIN_BOX_HEIGHT) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = np.zeros((len(boxes), NUM_KEYPOINTS, 3), np.float32)
        todo = [i for i, box in enumerate(boxes) if box[2] - box[0] >= 1 and box[3] - box[1] >= max(1.0, min_height)]
        if not todo:
            return out
        prepared = [R.rtmpose_preprocess(frame_bgr, boxes[i], self.input_wh) for i in todo]
        step = self._fixed_batch or len(todo)
        for s in range(0, len(todo), step):
            group = prepared[s:s + step]
            blobs = [p[0] for p in group]
            if self._fixed_batch:
                blobs += [blobs[-1]] * (step - len(blobs))
            sx, sy = self.session.run(self._out_names, {self.input_name: np.concatenate(blobs)})
            for n, (_, center, scale) in enumerate(group):
                kpts, conf = R.rtmpose_decode(sx[n:n + 1], sy[n:n + 1], center, scale, self.input_wh)
                out[todo[s + n], :, :2] = kpts
                out[todo[s + n], :, 2] = conf
        return out
