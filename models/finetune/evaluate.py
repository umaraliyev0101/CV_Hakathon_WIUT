"""Compare COCO RF-DETR Nano vs the fine-tuned Nano on what the showroom use case needs.

1. Real test (data/real_labels/coco/test, reviewed Pexels): AP50, AP50:95, recall / FP per image at conf.
2. Synthetic test (CCTV angle): AP50, recall on people < 60 % visible (behind cars).
3. Hard negatives: detections on boxes removed in review (headrests, reflections) in the real test split.
4. Use case: floor-position error (analytics.floor.person_floor_xy) from detector boxes vs simulator pos_m.

Usage from repo root:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/evaluate.py --ckpt data/finetune_runs/nano_mix/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analytics.floor import Camera, person_floor_xy  # noqa: E402

REAL = ROOT / "data" / "real_labels"
SYN = ROOT / "data" / "synthetic"


class YoloAdapter:
    """Ultralytics YOLO with the rfdetr `predict(image, threshold)` -> .xyxy/.confidence/.class_id shape."""

    def __init__(self, weights: str):
        from ultralytics import YOLO
        self.m = YOLO(weights)

    def predict(self, img, threshold: float):
        from types import SimpleNamespace
        r = self.m.predict(img, conf=threshold, imgsz=640, verbose=False, device=0)[0].boxes
        return SimpleNamespace(xyxy=r.xyxy.cpu().numpy(), confidence=r.conf.cpu().numpy(),
                               class_id=r.cls.cpu().numpy().astype(int))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def predict(model, path: Path, person_ids: set[int]):
    det = model.predict(Image.open(path).convert("RGB"), threshold=0.05)
    keep = [i for i, c in enumerate(det.class_id) if int(c) in person_ids]
    return [(det.xyxy[i].tolist(), float(det.confidence[i])) for i in keep]


def coco_ap(gt_coco: dict, preds: dict[int, list]) -> tuple[float, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = {"images": gt_coco["images"], "categories": [{"id": 1, "name": "person"}],
                     "annotations": [dict(a, category_id=1) for a in gt_coco["annotations"]]}
        g.createIndex()
        dets = [{"image_id": iid, "category_id": 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
                for iid, ps in preds.items() for b, s in ps]
        if not dets:
            return 0.0, 0.0
        e = COCOeval(g, g.loadRes(dets), "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[1]), float(e.stats[0])


def match(gts, ps, conf):
    """Greedy IoU>=0.5 matching at conf. Returns (matched gt idx -> pred box, n_fp)."""
    ps = sorted([p for p in ps if p[1] >= conf], key=lambda p: -p[1])
    used, m = set(), {}
    for b, _ in ps:
        best, bi = 0.5, None
        for gi, g in enumerate(gts):
            if gi not in used and (v := iou(b, g)) >= best:
                best, bi = v, gi
        if bi is not None:
            used.add(bi)
            m[bi] = b
    return m, len(ps) - len(m)


def evaluate(model, person_ids, conf, syn_step):
    out = {}
    # 1 + 3. real test
    test = json.loads((REAL / "coco" / "test" / "_annotations.coco.json").read_text())
    preds = {im["id"]: predict(model, REAL / "coco" / "test" / im["file_name"], person_ids) for im in test["images"]}
    out["real_AP50"], out["real_AP50_95"] = coco_ap(test, preds)
    tp = fp = n = 0
    for im in test["images"]:
        gts = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
               for a in test["annotations"] if a["image_id"] == im["id"]]
        m, f = match(gts, preds[im["id"]], conf)
        tp += len(m); fp += f; n += len(gts)
    out["real_recall"], out["real_fp_per_img"] = tp / n, fp / len(test["images"])
    pre = {r["file"]: r for r in json.loads((REAL / "prelabels.json").read_text())}
    review = json.loads((REAL / "review.json").read_text())
    names = {im["file_name"]: im["id"] for im in test["images"]}
    hn = hn_hit = 0
    for f, idxs in review.get("delete", {}).items():
        if f not in names:
            continue
        for i in idxs:
            b = pre[f]["boxes"][i]["bbox_xyxy"]
            # skip deleted duplicates of a kept person: only count boxes not overlapping any kept GT
            kept = [a["bbox"] for a in test["annotations"] if a["image_id"] == names[f]]
            if any(iou(b, [k[0], k[1], k[0] + k[2], k[1] + k[3]]) > 0.3 for k in kept):
                continue
            hn += 1
            hn_hit += any(iou(b, p) >= 0.5 for p, s in preds[names[f]] if s >= conf)
    out["hard_neg_regions"], out["hard_neg_detected"] = hn, hn_hit
    # 2 + 4. synthetic test
    st = json.loads((SYN / "coco" / "test" / "_annotations.coco.json").read_text())
    imgs = st["images"][::syn_step]
    ids = {i["id"] for i in imgs}
    sub = {"images": imgs, "annotations": [a for a in st["annotations"] if a["image_id"] in ids]}
    gts_by = {}
    for a in sub["annotations"]:
        gts_by.setdefault(a["image_id"], []).append(a)
    spreds = {im["id"]: predict(model, SYN / "coco" / "test" / im["file_name"], person_ids) for im in imgs}
    out["syn_AP50"], out["syn_AP50_95"] = coco_ap(sub, spreds)
    gt_cache, occ_n, occ_tp, errs, all_tp, all_n, sfp = {}, 0, 0, [], 0, 0, 0
    for im in imgs:
        anns = gts_by.get(im["id"], [])
        gts = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        m, f = match(gts, spreds[im["id"]], conf)
        all_tp += len(m); all_n += len(gts); sfp += f
        if im["clip_id"] not in gt_cache:
            gt_cache[im["clip_id"]] = json.loads((SYN / "clips" / im["clip_id"] / "gt.json").read_text())
        g = gt_cache[im["clip_id"]]
        cam = Camera.from_gt(g["camera"])
        fr = g["frames"][im["frame"]]  # COCO `frame` is the index into gt.json frames
        pos = {p["id"]: p["pos_m"] for p in fr["people"]}
        for gi, a in enumerate(anns):
            vis = a["attributes"]["visible_frac"]
            if vis < 0.6:
                occ_n += 1; occ_tp += gi in m
            if gi in m and a["attributes"]["track_id"] in pos:
                xy, _ = person_floor_xy(cam, m[gi])
                errs.append(float(np.linalg.norm(xy - pos[a["attributes"]["track_id"]])))
    out["syn_recall"], out["syn_fp_per_img"] = all_tp / all_n, sfp / len(imgs)
    out["syn_occluded_recall"], out["syn_occluded_n"] = (occ_tp / occ_n if occ_n else float("nan")), occ_n
    e = np.array(errs)
    out["floor_err_median_m"], out["floor_err_p90_m"] = float(np.median(e)), float(np.percentile(e, 90))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="fine-tuned RF-DETR Nano checkpoint")
    ap.add_argument("--yolo-base", help="COCO YOLO weights (benchmark only)")
    ap.add_argument("--yolo-ft", help="fine-tuned YOLO weights (benchmark only)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--syn-step", type=int, default=3)
    ap.add_argument("--out", default="data/finetune_runs/eval.json")
    a = ap.parse_args()
    res = {"conf": a.conf}
    if a.ckpt:
        from rfdetr import RFDETRNano
        res["coco_nano"] = evaluate(RFDETRNano(), {1}, a.conf, a.syn_step)
        print("coco_nano", json.dumps(res["coco_nano"]), flush=True)
        res["finetuned_nano"] = evaluate(RFDETRNano(pretrain_weights=a.ckpt), set(range(0, 100)), a.conf, a.syn_step)
        print("finetuned_nano", json.dumps(res["finetuned_nano"]), flush=True)
    if a.yolo_base:
        res["coco_yolo11n"] = evaluate(YoloAdapter(a.yolo_base), {0}, a.conf, a.syn_step)
        print("coco_yolo11n", json.dumps(res["coco_yolo11n"]), flush=True)
    if a.yolo_ft:
        res["finetuned_yolo11n"] = evaluate(YoloAdapter(a.yolo_ft), {0}, a.conf, a.syn_step)
        print("finetuned_yolo11n", json.dumps(res["finetuned_yolo11n"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
