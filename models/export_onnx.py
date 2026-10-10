"""Regenerate models/showroom_detector.onnx from COCO-pretrained RF-DETR Nano and validate it.

    CUDA_VISIBLE_DEVICES=1 python models/export_onnx.py              # export + validate
    CUDA_VISIBLE_DEVICES=1 python models/export_onnx.py --skip-export # validate only

Needs `pip install rfdetr onnx onnxsim onnxruntime-gpu` (torch comes with rfdetr).
Weights are fetched by rfdetr into ~/.roboflow/models/rf-detr-nano.pth
(https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth, Apache-2.0).

Validation compares the ONNX pipeline in models/detect.py against rfdetr's own
RFDETRNano.predict() on real frames: matches person boxes by IoU and reports the
max pixel / score difference, and checks that predict() labels our class id as "person".
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.detect import MODEL_PATH, PERSON_CLASS_ID, Detector, read_frame  # noqa: E402

DEFAULT_VAL = [
    ("data/real_open/pexels_7154214_1920_1080_25fps.mp4", 100),
    ("data/real_open/pexels_7154217_1920_1080_25fps.mp4", 150),
    ("data/real_open/pexels_7154241_1066_1920_25fps.mp4", 200),
    ("data/real_open/pexels_36512656_2160_3840_50fps.mp4", 240),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out: Path, opset: int, simplify: bool) -> None:
    from rfdetr import RFDETRNano

    model = RFDETRNano()
    with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp:
        p = Path(model.export(output_dir=tmp, opset_version=opset, batch_size=1, dynamic_batch=False, verbose=False))
        if simplify:
            import onnx
            from onnxsim import simplify as onnx_simplify

            m = onnx.load(str(p))
            m2, ok = onnx_simplify(m)
            if ok:
                onnx.save(m2, str(p))
            else:
                print("onnxsim check failed; keeping unsimplified graph")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) sha256={sha256(out)}")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def validate(samples: list[tuple[str, int]], conf: float) -> bool:
    import cv2
    from rfdetr import RFDETRNano

    ref = RFDETRNano()
    det = Detector(device="cpu")
    all_ok = True
    for path, frame in samples:
        if not (REPO / path).exists():
            print(f"skip {path} (missing)")
            continue
        bgr = read_frame(str(REPO / path), frame)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = ref.predict(rgb, threshold=conf, include_source_image=False)
        names = list(r.data.get("class_name", [])) if hasattr(r, "data") else []
        person_mask = r.class_id == PERSON_CLASS_ID
        for cid, nm in zip(r.class_id, names or [None] * len(r.class_id)):
            if nm is not None and (cid == PERSON_CLASS_ID) != (nm == "person"):
                print(f"  CLASS MAP MISMATCH: class_id={cid} name={nm}")
                all_ok = False
        t_boxes, t_scores = r.xyxy[person_mask], r.confidence[person_mask]
        o = det.detect(bgr, conf)
        o_boxes = np.array([d.bbox for d in o]).reshape(-1, 4)
        o_scores = np.array([d.score for d in o])
        max_px, max_ds, unmatched = 0.0, 0.0, 0
        for tb, ts in zip(t_boxes, t_scores):
            if len(o_boxes) == 0:
                unmatched += 1
                continue
            ious = [iou(tb, ob) for ob in o_boxes]
            j = int(np.argmax(ious))
            if ious[j] < 0.5:
                unmatched += 1
                continue
            max_px = max(max_px, float(np.abs(tb - o_boxes[j]).max()))
            max_ds = max(max_ds, abs(float(ts) - float(o_scores[j])))
        cls_seen = sorted(set(zip(r.class_id.tolist(), names))) if names else sorted(set(r.class_id.tolist()))
        ok = unmatched == 0 and max_px <= 5.0 and abs(len(o) - len(t_boxes)) <= 1
        all_ok &= ok
        print(
            f"{Path(path).name}@{frame} {bgr.shape[1]}x{bgr.shape[0]}: torch persons={len(t_boxes)} "
            f"onnx persons={len(o)} unmatched={unmatched} max|dbox|={max_px:.2f}px max|dscore|={max_ds:.4f} "
            f"torch classes={cls_seen} -> {'OK' if ok else 'CHECK'}"
        )
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # this machine: never GPU 0
    if not args.skip_export:
        export(args.out, args.opset, not args.no_simplify)
    if not args.skip_validate:
        ok = validate(DEFAULT_VAL, args.conf)
        print("VALIDATION", "PASS" if ok else "CHECK")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
