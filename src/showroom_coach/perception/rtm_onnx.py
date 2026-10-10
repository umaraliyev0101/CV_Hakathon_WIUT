# Adapted from rtmlib 0.0.16 (Apache-2.0, https://github.com/Tau-J/rtmlib); YOLOX postprocess from Megvii YOLOX (Apache-2.0).
"""ONNX pre/post-processing for YOLOX, RTMPose and RF-DETR.

Only the pieces this project needs are ported, as plain functions over numpy arrays, so that
rtmlib itself (whose dependencies clash with opencv-python) is not a dependency. Nothing here
touches the filesystem except `make_session`, which loads one .onnx file.

Conventions: images are BGR uint8 (OpenCV); boxes are xyxy in source-image pixels.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

YOLOX_PAD_VALUE = 114
YOLOX_STRIDES = (8, 16, 32)
RTMPOSE_PADDING = 1.25
RTMPOSE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB order
RTMPOSE_STD = np.array([58.395, 57.12, 57.375], np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # RGB order
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    """CPU-only onnxruntime session; `threads` intra-op threads (default 4)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or 4
    opts.log_severity_level = 3  # errors only: the exports carry harmless unused-initializer notes
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- NMS


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Single-class greedy NMS (YOLOX/rtmlib "+1 pixel" convention). Returns kept indices."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.asarray(scores).argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[rest] - inter)
        order = rest[ovr <= thr]
    return np.array(keep, dtype=np.int64)


def _multiclass_nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float):
    """Class-aware NMS. boxes (N,4), scores (N,C) -> (boxes (K,4), scores (K,), cls (K,))."""
    out_b, out_s, out_c = [], [], []
    for c in range(scores.shape[1]):
        sel = np.nonzero(scores[:, c] > score_thr)[0]
        if sel.size == 0:
            continue
        keep = sel[nms(boxes[sel], scores[sel, c], nms_thr)]
        out_b.append(boxes[keep])
        out_s.append(scores[keep, c])
        out_c.append(np.full(keep.size, c, dtype=np.int64))
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    return (np.concatenate(out_b).astype(np.float32), np.concatenate(out_s).astype(np.float32),
            np.concatenate(out_c))


# --------------------------------------------------------------------------- YOLOX


def yolox_preprocess(img_bgr: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Letterbox (top-left, pad 114), BGR float32 0-255, NCHW. Returns (blob, ratio)."""
    ih, iw = int(input_hw[0]), int(input_hw[1])
    h, w = img_bgr.shape[:2]
    padded = np.full((ih, iw, 3), YOLOX_PAD_VALUE, np.uint8)
    ratio = min(ih / h, iw / w)
    nh, nw = min(ih, max(1, round(h * ratio))), min(iw, max(1, round(w * ratio)))
    if (nh, nw) == (h, w):
        padded[:nh, :nw] = img_bgr
    else:
        padded[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, float(ratio)


@lru_cache(maxsize=8)
def _yolox_grid(input_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in YOLOX_STRIDES:
        gh, gw = input_hw[0] // s, input_hw[1] // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((gh * gw, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def yolox_postprocess(out: np.ndarray, input_hw: tuple[int, int], ratio: float,
                      score_thr: float, nms_thr: float):
    """Decode a Megvii YOLOX ONNX output [1,N,5+C] (sigmoid already applied; box terms are
    raw dx, dy, log w, log h) to (boxes (K,4) xyxy in source pixels, scores (K,), cls (K,))."""
    pred = np.asarray(out, dtype=np.float32)
    if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[-1] <= 5:
        raise ValueError(f"unsupported YOLOX output shape {pred.shape}; expected [1, N, 5+classes]")
    pred = pred[0]
    grid, stride = _yolox_grid((int(input_hw[0]), int(input_hw[1])))
    if pred.shape[0] != grid.shape[0]:
        raise ValueError(f"YOLOX output has {pred.shape[0]} anchors; {tuple(input_hw)} needs {grid.shape[0]}")
    scores = pred[:, 4:5] * pred[:, 5:]  # objectness * class probability
    cand = np.nonzero(scores.max(axis=1) > score_thr)[0]  # decode only anchors that can survive
    if cand.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    p, g, s = pred[cand], grid[cand], stride[cand]
    cxcy = (p[:, :2] + g) * s
    wh = np.exp(p[:, 2:4]) * s
    xyxy = np.concatenate([cxcy - wh / 2.0, cxcy + wh / 2.0], axis=1) / ratio
    return _multiclass_nms(xyxy, scores[cand], nms_thr, score_thr)


# --------------------------------------------------------------------------- RTMPose


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return b + np.array([-d[1], d[0]])


def rtmpose_preprocess(img_bgr: np.ndarray, bbox_xyxy, input_wh: tuple[int, int] = (192, 256)):
    """Crop the (padded) person box to the model's aspect ratio and warp it to `input_wh`.

    Returns (blob float32 [1,3,H,W] RGB-normalised, center (2,), scale (2,)); center and scale
    describe the crop in source pixels and are what `rtmpose_decode` needs.
    """
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    w, h = int(input_wh[0]), int(input_wh[1])
    center = (bbox[:2] + bbox[2:]) * 0.5
    scale = (bbox[2:] - bbox[:2]) * RTMPOSE_PADDING
    aspect = w / h  # crop is widened or heightened to the model's aspect ratio
    scale = (np.array([scale[0], scale[0] / aspect]) if scale[0] > scale[1] * aspect
             else np.array([scale[1] * aspect, scale[1]]))

    src = np.zeros((3, 2), np.float32)  # centre, a point above it, a third point (right angle)
    src[0] = center
    src[1] = center + np.array([0.0, scale[0] * -0.5])
    src[2] = _third_point(src[0], src[1])
    dst = np.zeros((3, 2), np.float32)
    dst[0] = [w * 0.5, h * 0.5]
    dst[1] = dst[0] + np.array([0.0, w * -0.5])
    dst[2] = _third_point(dst[0], dst[1])
    warp = cv2.getAffineTransform(src, dst)

    crop = cv2.warpAffine(img_bgr, warp, (w, h), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = ((rgb - RTMPOSE_MEAN) / RTMPOSE_STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, dtype=np.float32), center, scale


def rtmpose_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, center, scale,
                   input_wh: tuple[int, int], split_ratio: float = 2.0):
    """SimCC argmax decode for one person. Inputs [1,K,Wx] / [1,K,Wy] (or [K,W*]).

    Returns (kpts (K,2) in source pixels, scores (K,) in [0,1]). The score is the mean of the
    x and y peak responses (as rtmlib does), clipped to [0,1].
    """
    sx, sy = np.asarray(simcc_x), np.asarray(simcc_y)
    if sx.ndim == 3:
        if sx.shape[0] != 1 or sy.shape[0] != 1:
            raise ValueError("rtmpose_decode takes one person at a time (batch of 1)")
        sx, sy = sx[0], sy[0]
    locs = np.stack([sx.argmax(axis=1), sy.argmax(axis=1)], axis=-1).astype(np.float32) / split_ratio
    peak = 0.5 * (sx.max(axis=1) + sy.max(axis=1))
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    kpts = locs / np.asarray(input_wh, dtype=np.float32) * scale + center - scale / 2.0
    return kpts.astype(np.float32), np.clip(peak, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- RF-DETR

# The export uses COCO-91 category ids (0 = background, 1 = person, ...). Detector.cls is the
# contiguous COCO-80 index (0 = person, 2 = car) so both detector families agree.
_COCO91_UNUSED = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)
_COCO91_TO_80 = np.full(91, -1, dtype=np.int64)
_COCO91_TO_80[[i for i in range(1, 91) if i not in _COCO91_UNUSED]] = np.arange(80)


def rfdetr_preprocess(img_bgr: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    """Plain resize to the model size (no letterbox), RGB, ImageNet normalisation, NCHW."""
    resized = cv2.resize(img_bgr, (int(input_hw[1]), int(input_hw[0])), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, dtype=np.float32)


def rfdetr_postprocess(boxes: np.ndarray, logits: np.ndarray, orig_hw: tuple[int, int],
                       score_thr: float, num_select: int = 300):
    """Decode RF-DETR outputs: boxes [1,Q,4] normalised cx,cy,w,h and logits [1,Q,C].

    Sigmoid scores; the background column (index 0) is never selected; the top `num_select`
    (query, class) pairs above `score_thr` are returned as
    (boxes (K,4) xyxy in source pixels, scores (K,), cls (K,) COCO-80 index).
    """
    logits = np.asarray(logits, dtype=np.float32)
    nclass = logits.shape[-1]
    probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1, nclass)))
    probs[:, 0] = -1.0
    flat = probs.ravel()
    k = min(num_select, flat.size)
    top = np.argpartition(flat, -k)[-k:]
    top = top[np.argsort(-flat[top])]
    query, label = top // nclass, top % nclass
    cls = _COCO91_TO_80[np.minimum(label, 90)]
    keep = (flat[top] > score_thr) & (label > 0) & (cls >= 0)

    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[query[keep]]
    wh = np.clip(b[:, 2:4], 0.0, None)
    xyxy = np.concatenate([b[:, :2] - 0.5 * wh, b[:, :2] + 0.5 * wh], axis=1)
    h, w = orig_hw
    xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
    return xyxy.astype(np.float32), flat[top][keep].astype(np.float32), cls[keep]
