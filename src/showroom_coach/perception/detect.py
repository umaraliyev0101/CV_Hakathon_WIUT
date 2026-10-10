"""Person/object detector over the vendored ONNX wrappers (YOLOX family or RF-DETR)."""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_DETECTOR
from ..models import ensure_model
from ..types import Det
from . import rtm_onnx as R


class Detector:
    """Runs one detector model on BGR frames.

    `__call__` returns every COCO class above `score_thr` (cls is the contiguous COCO-80 index,
    0 = person); `persons` keeps class 0 only. Boxes are clipped to the frame.
    """

    def __init__(self, name: str = DEFAULT_DETECTOR, score_thr: float = 0.35, nms_thr: float = 0.45,
                 threads: int | None = None):
        self.name = name
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.kind = "rfdetr" if name.startswith("rfdetr") else "yolox"
        self.session = R.make_session(ensure_model(name), threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self._out_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, frame_bgr: np.ndarray) -> list[Det]:
        h, w = frame_bgr.shape[:2]
        if self.kind == "yolox":
            blob, ratio = R.yolox_preprocess(frame_bgr, self.input_hw)
            (out,) = self.session.run(self._out_names, {self.input_name: blob})
            boxes, scores, cls = R.yolox_postprocess(out, self.input_hw, ratio, self.score_thr, self.nms_thr)
        else:
            blob = R.rfdetr_preprocess(frame_bgr, self.input_hw)
            raw_boxes, logits = self.session.run(self._out_names, {self.input_name: blob})
            boxes, scores, cls = R.rfdetr_postprocess(raw_boxes, logits, (h, w), self.score_thr)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [Det(tuple(float(v) for v in b), float(s), int(c)) for b, s, c in zip(boxes, scores, cls)]

    def persons(self, frame_bgr: np.ndarray) -> list[Det]:
        return [d for d in self(frame_bgr) if d.cls == 0]
