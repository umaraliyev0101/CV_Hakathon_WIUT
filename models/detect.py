"""P0 baseline person detector: COCO-pretrained RF-DETR Nano exported to ONNX.

Usage from repo root:
    from models.detect import detect, Det, Detector
    dets = detect(frame_bgr, conf=0.5)

CLI:
    python models/detect.py IMAGE_OR_VIDEO [--frame N] [--out annotated.jpg] [--conf 0.5]

GPU policy for this machine: GPU 0 belongs to other people. CUDA is only used when
CUDA_VISIBLE_DEVICES is set explicitly (e.g. CUDA_VISIBLE_DEVICES=1); otherwise we
fall back to CPU. See models/README.md for the tensor spec.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "showroom_detector.onnx"

# RF-DETR COCO checkpoints emit 91 logits indexed by the *raw COCO category id*
# (1..90, sparse; index 0 unused). Person is category id 1. Verified empirically
# against rfdetr.RFDETRNano.predict() (class_id == 1 -> "person") by
# models/export_onnx.py --validate.
PERSON_CLASS_ID = 1

# ImageNet normalisation, identical to rfdetr.detr.RFDETR.means / stds.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# rfdetr PostProcess default: top-300 over the flattened (query, class) scores.
NUM_SELECT = 300


# FLAG: reconcile with teammate A's Det type
@dataclass
class Det:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the input frame
    score: float
    cls: str = "person"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _cuda_allowed() -> bool:
    """CUDA only when the caller pinned a GPU via CUDA_VISIBLE_DEVICES (never implicit GPU 0)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    return cvd is not None and cvd.strip() not in ("", "-1")


def _preload_cuda_libs() -> None:
    # onnxruntime-gpu can load the CUDA/cuDNN libs shipped as nvidia-* pip wheels (pulled in by torch).
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # pragma: no cover - best effort
            pass


class Detector:
    """onnxruntime wrapper around the exported RF-DETR Nano COCO model, person class only."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        device: str = "auto",
        threads: int | None = None,
        conf: float = 0.5,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path) if model_path else MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found; regenerate with `python models/export_onnx.py`"
            )
        self.conf = conf

        available = ort.get_available_providers()
        want_cuda = device in ("cuda", "auto")
        providers: list = ["CPUExecutionProvider"]
        if want_cuda:
            if "CUDAExecutionProvider" in available and _cuda_allowed():
                _preload_cuda_libs()
                # device_id 0 == first *visible* device (e.g. physical GPU 1 with CUDA_VISIBLE_DEVICES=1)
                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            elif device == "cuda":
                warnings.warn(
                    "device='cuda' requested but CUDA EP unavailable or CUDA_VISIBLE_DEVICES unset; using CPU",
                    stacklevel=2,
                )

        so = ort.SessionOptions()
        if threads is not None and threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.in_h, self.in_w = (int(d) for d in inp.shape)
        outs = {o.name: o for o in self.session.get_outputs()}
        self.boxes_name = "dets" if "dets" in outs else self.session.get_outputs()[0].name
        self.logits_name = "labels" if "labels" in outs else self.session.get_outputs()[1].name

    # ---- pipeline stages -------------------------------------------------
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HxWx3 -> float32 1x3xHxW, RGB, bilinear resize to model res, ImageNet-normalised."""
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        img = (img - _MEAN) / _STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def postprocess(
        self, boxes: np.ndarray, logits: np.ndarray, orig_hw: tuple[int, int], conf: float
    ) -> list[Det]:
        """Mirror rfdetr PostProcess: sigmoid, top-k over (query, class), cxcywh->xyxy, scale; keep person."""
        h, w = orig_hw
        boxes, logits = boxes[0], logits[0]  # (Q,4), (Q,C)
        num_classes = logits.shape[1]
        prob = _sigmoid(logits.astype(np.float32)).reshape(-1)
        k = min(NUM_SELECT, prob.size)
        top = np.argpartition(-prob, k - 1)[:k]
        top = top[np.argsort(-prob[top])]
        scores = prob[top]
        labels = top % num_classes
        queries = top // num_classes
        keep = (labels == PERSON_CLASS_ID) & (scores > conf)
        if not np.any(keep):
            return []
        b = boxes[queries[keep]]
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        xyxy *= np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        return [
            Det(bbox=tuple(float(v) for v in bb), score=float(s), cls="person")
            for bb, s in zip(xyxy, scores[keep])
        ]

    def __call__(self, frame_bgr: np.ndarray, conf: float | None = None) -> list[Det]:
        return self.detect(frame_bgr, conf)

    def detect(self, frame_bgr: np.ndarray, conf: float | None = None) -> list[Det]:
        conf = self.conf if conf is None else conf
        x = self.preprocess(frame_bgr)
        boxes, logits = self.session.run([self.boxes_name, self.logits_name], {self.input_name: x})
        return self.postprocess(boxes, logits, frame_bgr.shape[:2], conf)


_DEFAULT: Detector | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_detector() -> Detector:
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = Detector(device="auto")
    return _DEFAULT


def detect(frame_bgr: np.ndarray, conf: float = 0.5) -> list[Det]:
    """Detect people in a BGR frame with a lazily-created default Detector."""
    return get_default_detector().detect(frame_bgr, conf)


def draw(frame_bgr: np.ndarray, dets: list[Det]) -> np.ndarray:
    out = frame_bgr.copy()
    t = max(2, round(min(out.shape[:2]) / 400))
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), t)
        label = f"{d.cls} {d.score:.2f}"
        fs = 0.5 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = max(y1, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t)
    return out


def read_frame(path: str, frame: int = 0) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path} as image or video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame} from {path}")
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RF-DETR Nano (COCO) person detector, ONNX Runtime")
    ap.add_argument("source", help="image or video path")
    ap.add_argument("--frame", type=int, default=0, help="frame index for videos")
    ap.add_argument("--out", help="write annotated image here")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    img = read_frame(args.source, args.frame)
    det = Detector(device=args.device, threads=args.threads)
    dets = det.detect(img, args.conf)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        print(f"{d.cls} score={d.score:.2f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    print(f"# {len(dets)} detection(s), {img.shape[1]}x{img.shape[0]}, provider={det.device}", file=sys.stderr)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, draw(img, dets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
