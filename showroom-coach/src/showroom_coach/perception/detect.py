"""Person/object detector over the vendored ONNX wrappers (YOLOX family or RF-DETR)."""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_DETECTOR
from ..models import FINETUNED_DETECTOR, ModelUnavailable, ensure_model, finetuned_detector_path, short_sha256
from ..types import Det
from . import rtm_onnx as R

SUPPORTED_TILES = (1, 2)


def parse_detector_spec(spec: str) -> tuple[str, int]:
    """'yolox_tiny@2' -> ('yolox_tiny', 2); a bare model name means one tile ('yolox_s' -> ('yolox_s', 1))."""
    name, sep, rest = spec.partition("@")
    if not name or (sep and (not rest.isascii() or not rest.isdigit())):
        raise ValueError(f"bad detector spec {spec!r}; expected '<model>' or '<model>@<tiles>'")
    tiles = int(rest) if sep else 1
    if tiles not in SUPPORTED_TILES:
        raise ValueError(f"bad detector spec {spec!r}: tiles must be one of {SUPPORTED_TILES}")
    return name, tiles


class Detector:
    """Runs one detector model on BGR frames.

    `name` is a model name or a spec such as 'yolox_tiny@2' (see `parse_detector_spec`);
    `tiles=2` runs the model on two overlapping crops of the frame (long axis split) and merges
    the results, which helps small, distant people at the cost of a second inference.

    `__call__` returns every COCO class above `score_thr` (cls is the contiguous COCO-80 index,
    0 = person); `persons` keeps class 0 only. Boxes are clipped to the frame.
    """

    def __init__(self, name: str = DEFAULT_DETECTOR, score_thr: float = 0.35, nms_thr: float = 0.45,
                 tiles: int = 1, threads: int | None = None):
        base, spec_tiles = parse_detector_spec(name)
        if tiles not in SUPPORTED_TILES:
            raise ValueError(f"tiles must be one of {SUPPORTED_TILES}, got {tiles}")
        if "@" in name and tiles not in (1, spec_tiles):
            raise ValueError(f"conflicting tiles: spec {name!r} but tiles={tiles}")
        self.name = base
        self.tiles = spec_tiles if "@" in name else tiles
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.kind = "rfdetr" if base.startswith("rfdetr") else "yolox"
        if base == FINETUNED_DETECTOR:
            # our fine-tuned weights (SHOWROOM_RFDETR_FT); the sha prefix in `spec` makes the perception
            # cache key change whenever the weights do. `spec` is a label only: nothing parses it back.
            path = finetuned_detector_path()
            label = f"{base}-{short_sha256(path)}"
        else:
            path, label = ensure_model(base), base
        self.spec = label if self.tiles == 1 else f"{label}@{self.tiles}"
        try:
            self.session = R.make_session(path, threads)
        except Exception as exc:  # a corrupt or foreign file: let the stack fall back like a missing one
            if base != FINETUNED_DETECTOR:
                raise
            raise ModelUnavailable(f"{base}: cannot load {path}: {exc}") from exc
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self._out_names = [o.name for o in self.session.get_outputs()]
        if self.kind == "rfdetr":
            self._out_names = R.rfdetr_output_names(self.session)

    def _infer(self, img: np.ndarray):
        """One model run on `img`: (boxes (K,4), scores (K,), cls (K,)) in `img` pixel coordinates."""
        h, w = img.shape[:2]
        if self.kind == "yolox":
            blob, ratio = R.yolox_preprocess(img, self.input_hw)
            (out,) = self.session.run(self._out_names, {self.input_name: blob})
            return R.yolox_postprocess(out, self.input_hw, ratio, self.score_thr, self.nms_thr)
        blob = R.rfdetr_preprocess(img, self.input_hw)
        raw_boxes, logits = self.session.run(self._out_names, {self.input_name: blob})
        class_map = R.FINETUNED_PERSON_MAP if self.name == FINETUNED_DETECTOR else None
        return R.rfdetr_postprocess(raw_boxes, logits, (h, w), self.score_thr, class_map=class_map)

    def __call__(self, frame_bgr: np.ndarray) -> list[Det]:
        h, w = frame_bgr.shape[:2]
        windows = R.tile_windows(w, h, self.tiles)
        if len(windows) == 1:
            boxes, scores, cls = self._infer(frame_bgr)
        else:
            parts = []
            for k, (x0, y0, x1, y1) in enumerate(windows):
                b, s, c = self._infer(np.ascontiguousarray(frame_bgr[y0:y1, x0:x1]))
                parts.append((b + np.array([x0, y0, x0, y0], np.float32), s, c, np.full(len(s), k)))
            boxes, scores, cls = R.merge_tiles(*(np.concatenate(p) for p in zip(*parts)), windows, self.nms_thr)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [Det(tuple(float(v) for v in b), float(s), int(c)) for b, s, c in zip(boxes, scores, cls)]

    def persons(self, frame_bgr: np.ndarray) -> list[Det]:
        return [d for d in self(frame_bgr) if d.cls == 0]
