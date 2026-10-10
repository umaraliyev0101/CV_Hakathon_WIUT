"""Object detector wrapper: YOLOX (letterbox + grid decode + class-aware NMS, scores kept) and
RF-DETR (resize + sigmoid top-k, scores kept). One ort.InferenceSession per Detector."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from pulse.config import Config
from pulse.datatypes import Det
from pulse.perception.models import REGISTRY, ensure_model
from pulse.perception.vendor_rtmlib.det_post import multiclass_nms

VEHICLE_CLASSES = ("car", "truck", "motorcycle", "bus")

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]
# COCO category ids (1..90) of the 80 classes, in COCO80 order. RF-DETR heads are indexed by
# these ids (ids with no category are unused).
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
assert len(COCO_IDS) == 80


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Detector:
    def __init__(self, name: str, cfg: Config, providers=("CPUExecutionProvider",),
                 models_dir="models"):
        path = Path(ensure_model(name, models_dir))
        # the fallback chain may have substituted another model: trust the folder name
        self.name = path.parent.name if path.parent.name in REGISTRY else name
        self.kind = REGISTRY[self.name].kind
        self.cfg = cfg
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 1
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=list(providers))
        inp = self.session.get_inputs()[0]
        shp = inp.shape
        size = REGISTRY[self.name].input_size
        self.in_hw = (int(shp[2]), int(shp[3])) if all(isinstance(v, int) for v in shp[2:]) \
            else (size, size)
        self.input_name = inp.name
        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std = np.array([0.229, 0.224, 0.225], np.float32)

    # ---- public ----
    def detect(self, frame_bgr: np.ndarray, classes=("person",)) -> list[Det]:
        if self.kind == "yolox":
            return self._yolox(frame_bgr, classes)
        if self.kind == "rfdetr":
            return self._rfdetr(frame_bgr, classes)
        raise ValueError(f"{self.name} is not a detector")

    # ---- YOLOX ----
    def _yolox(self, img, classes):
        h_in, w_in = self.in_hw
        ratio = min(h_in / img.shape[0], w_in / img.shape[1])
        padded = np.full((h_in, w_in, 3), 114, np.uint8)
        nh, nw = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        out = self.session.run(None, {self.input_name: x})[0]
        strides = [8, 16, 32]
        grids, strs = [], []
        for s in strides:
            hs, ws = h_in // s, w_in // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            g = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(g)
            strs.append(np.full((1, g.shape[1], 1), s))
        grids, strs = np.concatenate(grids, 1), np.concatenate(strs, 1)
        out = out.copy()
        out[..., :2] = (out[..., :2] + grids) * strs
        out[..., 2:4] = np.exp(out[..., 2:4]) * strs
        p = out[0]
        b = p[:, :4]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
        want = [COCO80.index(c) for c in classes]
        scores = p[:, 4:5] * p[:, 5:][:, want]
        dets, _ = multiclass_nms(xyxy, scores, nms_thr=self.cfg.det_nms_thr,
                                 score_thr=self.cfg.det_score_thr)
        if dets is None:
            return []
        return [self._mk(d[:4], d[4], classes[int(d[5])], img.shape) for d in dets]

    # ---- RF-DETR ----
    def _rfdetr(self, img, classes):
        h_in, w_in = self.in_hw
        r = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - self._mean) / self._std
        x = np.ascontiguousarray(r.transpose(2, 0, 1)[None], dtype=np.float32)
        names = [o.name for o in self.session.get_outputs()]
        outs = dict(zip(names, self.session.run(names, {self.input_name: x})))
        boxes = outs.get("dets", outs.get("boxes")) if ("dets" in outs or "boxes" in outs) \
            else list(outs.values())[0]
        logits = outs.get("labels", outs.get("logits")) if ("labels" in outs or "logits" in outs) \
            else list(outs.values())[1]
        probs = _sigmoid(logits[0])                       # [Q, C]
        cid = {COCO_IDS[COCO80.index(c)]: c for c in classes}
        H, W = img.shape[:2]
        res = []
        k = min(300, probs.size)
        flat = probs.ravel()
        top = np.argpartition(flat, -k)[-k:]
        for idx in top:
            q, c = divmod(int(idx), probs.shape[1])
            s = float(flat[idx])
            if s <= self.cfg.det_score_thr or c not in cid:
                continue
            cx, cy, w, h = boxes[0][q]
            xyxy = np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            res.append(self._mk(xyxy, s, cid[c], img.shape))
        return sorted(res, key=lambda d: -d.score)

    @staticmethod
    def _mk(xyxy, score, cls, shape) -> Det:
        h, w = shape[:2]
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        return Det((min(max(x1, 0.0), w), min(max(y1, 0.0), h),
                    min(max(x2, 0.0), w), min(max(y2, 0.0), h)), float(score), cls)
