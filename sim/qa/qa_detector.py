"""qa_detector: off-the-shelf person detector (models/detect.py, RF-DETR Nano COCO, ONNX) recall at
IoU >= iou on GT people with box height >= min_box_h_px and visible_frac >= min_visible_frac,
per lighting variant, over the final (CCTV-degraded) MP4 frames of the given clips.

Also reports (not a gate) unmatched detections: 'reflection' when the box centre lies on a car
zone in the image or right under a GT person (floor reflection), else 'other'.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # GPU 0 belongs to other services

from sim.qa.common import REPO, load_qa, report  # noqa: E402
from sim import labels, layout  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(gts, dets, thr):
    """Greedy by IoU. Returns set of matched gt indices and matched det indices."""
    pairs = sorted(((iou(g, d), i, j) for i, g in enumerate(gts) for j, d in enumerate(dets)), reverse=True)
    mg, md = set(), set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i in mg or j in md:
            continue
        mg.add(i)
        md.add(j)
    return mg, md


def _classify_unmatched(box, gt_boxes, car_polys):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for g in gt_boxes:
        if g[0] <= cx <= g[2] and g[3] - 0.1 * (g[3] - g[1]) <= box[1] <= g[3] + 0.5 * (g[3] - g[1]):
            return "reflection"
    for poly in car_polys:
        if len(poly) >= 3 and layout.point_in_polygon((cx, cy), poly):
            return "reflection"
    return "other"


def run_clips(clips, qa, detector=None):
    import cv2
    from models.detect import Detector
    q = qa["detector"]
    det = detector or Detector(device="auto", conf=q["conf"])
    stats = defaultdict(lambda: {"gt": 0, "tp": 0, "unmatched": defaultdict(int), "clips": 0})
    info = [f"detector providers={det.providers} conf={q['conf']} iou={q['iou']}"]
    for clip in clips:
        gt = labels.read_gt(clip / "gt.json")
        lt = gt["lighting"]
        s = stats[lt]
        s["clips"] += 1
        car_polys = [z["image_px"] for n, z in gt["zones"].items() if n.startswith("car_")]
        cap = cv2.VideoCapture(str(clip / f"{gt['clip_id']}.mp4"))
        c_gt = c_tp = 0
        for fr in gt["frames"]:
            ok, img = cap.read()
            if not ok:
                break
            dets = [d.bbox for d in det(img)] if callable(det) else [d.bbox for d in det.detect(img)]
            all_gt = [p["box_xyxy"] for p in fr["people"]]
            mg, md = match(all_gt, dets, q["iou"])
            for i, p in enumerate(fr["people"]):
                x1, y1, x2, y2 = p["box_xyxy"]
                if (y2 - y1) >= q["min_box_h_px"] and p["visible_frac"] >= q["min_visible_frac"]:
                    c_gt += 1
                    c_tp += i in mg
            for j, d in enumerate(dets):
                if j not in md and all(iou(g, d) < 0.1 for g in all_gt):
                    s["unmatched"][_classify_unmatched(d, all_gt, car_polys)] += 1
        cap.release()
        s["gt"] += c_gt
        s["tp"] += c_tp
        info.append(f"{gt['clip_id']}: recall {c_tp}/{c_gt} = {c_tp / max(1, c_gt):.3f}")
    fails = []
    for lt, s in sorted(stats.items()):
        r = s["tp"] / max(1, s["gt"])
        info.append(f"[{lt}] clips={s['clips']} recall@IoU{q['iou']} = {s['tp']}/{s['gt']} = {r:.3f} "
                    f"(gate >= {q['recall_min']}); unmatched detections (report only): {dict(s['unmatched'])}")
        if s["gt"] == 0:
            fails.append(f"[{lt}] no eligible GT people")
        elif r < q["recall_min"]:
            fails.append(f"[{lt}] recall {r:.3f} < {q['recall_min']}")
    return fails, info


def run(clip_dir, qa):
    return run_clips([Path(clip_dir)], qa)


def main(argv=sys.argv):
    clips = [Path(a) for a in argv[1:]]
    fails, info = run_clips(clips, load_qa())
    return report("qa_detector", fails, info)


if __name__ == "__main__":
    sys.exit(main())
