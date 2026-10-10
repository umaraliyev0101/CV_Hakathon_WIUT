"""The perception stack: one bundle of detector, pose estimator, embedder and tracker factory.

`default_stack` builds the real thing (cached per detector spec, with the spec §8 fallbacks when weights cannot
be fetched); `fake_stack` is a model-free test double that finds saturated colour blobs, used by unit tests,
the app's test mode and CI. Both expose the same callables, so the pipeline and the app never care which one
they were given.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..config import DEFAULT_DETECTOR
from ..models import ModelUnavailable
from ..types import Det
from .detect import Detector, parse_detector_spec
from .embed import Embedder, hsv_hist, usable_crop
from .pose import PoseEstimator
from .tracking import Tracker

log = logging.getLogger(__name__)

# Spec §8: when a detector's weights cannot be fetched (ensure_model already retried 3 times), try the alternative
# listed here. Keys and values are detector specs (`<model>[@<tiles>]`); chains are followed until one loads.
DETECTOR_FALLBACKS: dict[str, str] = {
    "yolox_nano@2": "yolox_tiny@2",
    "yolox_tiny@2": "yolox_nano",
    "yolox_nano": "yolox_tiny",
    "yolox_tiny": "yolox_nano",
    "yolox_s": "yolox_tiny",
    "yolox_s@2": "yolox_tiny@2",
    "rfdetr_nano": "yolox_nano@2",
}

# The int8 DINOv2 export quantises activations dynamically with one scale per input tensor, so a crop's embedding
# shifts slightly (cosine >= 0.98 against its batch-of-one value) with whatever else is in the batch, and on this
# CPU a batch of 8 is no faster per crop than one at a time (57 vs 64 ms). Production therefore embeds one crop
# per model run: every crop gets the same embedding however it was grouped (offline, live and cached runs agree).
EMBED_BATCH = 1


@dataclass
class PerceptionStack:
    """Everything the pipeline needs from the models. `name` describes the exact combination that is running
    (e.g. "yolox_nano@2+rtmpose_s+dinov2_small") and is part of the perception cache key, so it is stable across runs
    and changes when a fallback kicks in. `pose` / `embed` are None when unavailable: geometry then falls back to
    box height and identity to HSV histograms only."""

    name: str
    detect: Callable[[np.ndarray], list[Det]]                      # all classes (COCO-80 index, 0 = person)
    pose: Callable[[np.ndarray, np.ndarray], np.ndarray] | None    # (frame, boxes (N,4)) -> (N, 17, 3)
    embed: Callable[[list[np.ndarray]], np.ndarray] | None         # crops -> (N, D), rows L2-normalised

    def new_tracker(self, fps: float) -> Tracker:
        """A fresh tracker for one video (or live session) sampled at `fps`."""
        return Tracker(fps=fps)


# --------------------------------------------------------------------------- the real stack

_STACKS: dict[str, PerceptionStack] = {}
_LOCK = threading.Lock()


def _canonical(spec: str) -> str:
    """'yolox_nano@1' -> 'yolox_nano'; raises ValueError for a bad spec."""
    name, tiles = parse_detector_spec(spec)
    return name if tiles == 1 else f"{name}@{tiles}"


def _next_fallback(spec: str) -> str | None:
    if spec in DETECTOR_FALLBACKS:
        return DETECTOR_FALLBACKS[spec]
    base = spec.partition("@")[0]
    return DETECTOR_FALLBACKS.get(base) if base != spec else None


def _load_detector(spec: str) -> Detector:
    """The detector for `spec`, or the first one along its fallback chain whose weights can be loaded."""
    tried: list[str] = []
    last: ModelUnavailable | None = None
    cur: str | None = spec
    while cur is not None and cur not in tried:
        try:
            return Detector(cur)
        except ModelUnavailable as exc:
            tried.append(cur)
            last = exc
            nxt = _next_fallback(cur)
            if nxt is not None and nxt not in tried:
                log.warning("detector %s unavailable (%s); falling back to %s", cur, exc, nxt)
            cur = nxt
    raise ModelUnavailable(f"no detector could be loaded; tried {', '.join(tried)}: {last}") from last


def _optional(factory: Callable[[], object], what: str, consequence: str):
    try:
        return factory()
    except ModelUnavailable as exc:
        log.warning("%s unavailable (%s); running without it: %s", what, exc, consequence)
        return None


def default_stack(detector: str = DEFAULT_DETECTOR) -> PerceptionStack:
    """The real perception stack (detector + RTMPose-S + DINOv2-small), built once per detector spec.

    If the detector's weights cannot be fetched the stack uses the alternative from `DETECTOR_FALLBACKS`
    (logged as a warning naming both); if pose or embedder weights are missing the stack runs without them.
    Raises ModelUnavailable only when no detector along the chain loads, and ValueError for a malformed spec.
    The result is cached under the requested spec, fallback included, so a missing model is retried once per
    process rather than once per job.
    """
    key = _canonical(detector)
    with _LOCK:
        cached = _STACKS.get(key)
        if cached is not None:
            return cached
        det = _load_detector(key)
        pose = _optional(PoseEstimator, "pose model", "distance falls back to box height")
        emb = _optional(lambda: Embedder(batch=EMBED_BATCH), "appearance embedder",
                        "identity uses colour histograms only")
        name = "+".join([det.spec] + [m.name for m in (pose, emb) if m is not None])
        stack = PerceptionStack(name, det, pose, emb)
        _STACKS[key] = stack
        return stack


# --------------------------------------------------------------------------- the fake stack

FAKE_SAT_MIN = 120      # HSV saturation above this counts as "person" pixels
FAKE_VAL_MIN = 60
FAKE_MIN_AREA = 400     # px
FAKE_SCORE = 0.9
FAKE_KPT_CONF = 0.9
FAKE_MIN_BOX_HEIGHT = 40.0   # like PoseEstimator: smaller boxes get no keypoints

# Standing person facing the camera, laid out in the box with the proportions of tests/synth.py STAND:
# (y as a fraction of the box height from the top, x offset from the box centre in box widths). The person's left
# side is on the image right; the right side mirrors it. COCO order: nose, eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles.
_FAKE_LAYOUT = np.array([
    (0.07, 0.0),
    (0.055, 0.06), (0.055, -0.06),
    (0.065, 0.15), (0.065, -0.15),
    (0.18, 0.40), (0.18, -0.40),
    (0.37, 0.45), (0.37, -0.45),
    (0.52, 0.45), (0.52, -0.45),
    (0.48, 0.20), (0.48, -0.20),
    (0.715, 0.20), (0.715, -0.20),
    (0.96, 0.20), (0.96, -0.20),
], np.float64)


def _fake_detect(frame: np.ndarray) -> list[Det]:
    """One person per connected blob of strongly saturated pixels (area > 400 px)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > FAKE_SAT_MIN) & (hsv[..., 2] > FAKE_VAL_MIN)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [Det((float(x), float(y), float(x + w), float(y + h)), FAKE_SCORE, 0)
            for x, y, w, h, area in stats[1:n] if area > FAKE_MIN_AREA]


def _fake_pose(frame: np.ndarray, boxes: np.ndarray, min_height: float = FAKE_MIN_BOX_HEIGHT) -> np.ndarray:
    """Synthetic standing keypoints spread over each box (conf 0.9); boxes under `min_height` px get zeros."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((len(boxes), 17, 3), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        w, h = x2 - x1, y2 - y1
        if not (w >= 1 and h >= max(1.0, min_height)):
            continue
        out[i, :, 0] = (x1 + x2) / 2.0 + _FAKE_LAYOUT[:, 1] * w
        out[i, :, 1] = y1 + _FAKE_LAYOUT[:, 0] * h
        out[i, :, 2] = FAKE_KPT_CONF
    return out


def _fake_embed(crops: list[np.ndarray]) -> np.ndarray:
    """L2-normalised [unit mean colour (BGR), HSV histogram] per crop; zeros for an unusable crop."""
    out = np.zeros((len(crops), 3 + 128), np.float32)
    for i, c in enumerate(crops):
        if not usable_crop(c):
            continue
        mean = c.reshape(-1, 3).astype(np.float64).mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-12)
        v = np.concatenate([mean, hsv_hist(c).astype(np.float64)])
        out[i] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out


def fake_stack() -> PerceptionStack:
    """Model-free stand-in: persons are saturated colour blobs, pose is a synthetic standing skeleton, the embedding
    is mean colour plus HSV histogram. Deterministic and fast; used by tests and the app's test mode."""
    return PerceptionStack("fake", _fake_detect, _fake_pose, _fake_embed)
