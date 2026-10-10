"""Detector benchmark on a MEVA clip (decides `config.DEFAULT_DETECTOR`).

Protocol: N frames sampled evenly across the clip, each resized to width 960 (ground-truth
boxes scaled by the same factor); ground truth = boxes of `person` actors at that frame index,
ignoring boxes under 20 px high after scaling; one score threshold for every detector; greedy
IoU >= 0.5 matching; time = median ms for the detector call only, after 3 untimed warm-up frames.
MEVA only boxes annotated actors, so unannotated bystanders count as false positives and the
reported precision is a lower bound.

The MEVA geom/types files are untrusted input: they are read line by line and parsed with
simple regular expressions only (never a YAML loader).
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import MAX_WIDTH, repo_root

CANDIDATES = ("yolox_nano", "yolox_tiny", "yolox_s", "rfdetr_nano")
SCORE_THR = 0.3
IOU_THR = 0.5
MIN_GT_HEIGHT = 20.0  # px, after resizing to MAX_WIDTH
HEIGHT_BUCKETS = ((MIN_GT_HEIGHT, 40.0, "20-40px"), (40.0, 80.0, "40-80px"), (80.0, float("inf"), "80px+"))
WARMUP_FRAMES = 3
RECALL_MARGIN = 0.05  # "within 5 points of the best recall"
PRECISION_NOTE = ("MEVA only boxes annotated actors, so precision is a lower bound: "
                  "unannotated bystanders count as false positives.")

_NUM = r"-?\d+(?:\.\d+)?"
_TS0 = re.compile(r"'ts0':\s*(\d+)")
_ID1 = re.compile(r"'id1':\s*(\d+)")
_G0 = re.compile(rf"'g0':\s*'({_NUM}) ({_NUM}) ({_NUM}) ({_NUM})'")
_CSET = re.compile(r"'cset3':\s*\{([^}]*)\}")
_CSET_ITEM = re.compile(rf"'([A-Za-z_]+)':\s*({_NUM})")


def _load_person_boxes(geom: Path, types: Path, frames: set[int]) -> dict[int, list[tuple[float, float, float, float]]]:
    """frame index -> source-pixel boxes of `person` actors, for the wanted frames only."""
    person_ids: set[int] = set()
    with open(types, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_id, m_set = _ID1.search(line), _CSET.search(line)
            if not m_id or not m_set:
                continue
            items = [(float(v), k) for k, v in _CSET_ITEM.findall(m_set.group(1))]
            if items and max(items)[1] == "person":  # the actor's most likely type
                person_ids.add(int(m_id.group(1)))
    found: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    with open(geom, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_ts = _TS0.search(line)
            if not m_ts or int(m_ts.group(1)) not in frames:
                continue
            m_id, m_g = _ID1.search(line), _G0.search(line)
            if not m_id or not m_g or int(m_id.group(1)) not in person_ids:
                continue
            x1, y1, x2, y2 = (float(v) for v in m_g.groups())
            found[(int(m_ts.group(1)), int(m_id.group(1)))] = (x1, y1, x2, y2)
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    for (ts, _), box in found.items():
        out.setdefault(ts, []).append(box)
    return out


def sample_indices(n_total: int, n: int) -> list[int]:
    """n frame indices spread evenly over n_total frames (mid-points of n equal slices)."""
    if n_total <= n:
        return list(range(max(0, n_total)))
    return [int((i + 0.5) * n_total / n) for i in range(n)]


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_flags(gt, det_boxes, det_scores, iou_thr: float = IOU_THR) -> list[bool]:
    """Detections by descending score each take the best still-unmatched GT box with
    IoU >= iou_thr. Returns, per GT box, whether it was matched."""
    taken: set[int] = set()
    for d in sorted(range(len(det_boxes)), key=lambda i: -det_scores[i]):
        best, best_iou = -1, iou_thr
        for g, box in enumerate(gt):
            if g in taken:
                continue
            v = _iou(det_boxes[d], box)
            if v >= best_iou:
                best, best_iou = g, v
        if best >= 0:
            taken.add(best)
    return [g in taken for g in range(len(gt))]


def greedy_match(gt, det_boxes, det_scores, iou_thr: float = IOU_THR) -> tuple[int, int, int]:
    """Returns (true positives, false positives, false negatives)."""
    tp = sum(greedy_flags(gt, det_boxes, det_scores, iou_thr))
    return tp, len(det_boxes) - tp, len(gt) - tp


def height_bucket(h: float) -> str:
    for lo, hi, label in HEIGHT_BUCKETS:
        if lo <= h < hi:
            return label
    return HEIGHT_BUCKETS[-1][2]


def choose_detector(rows: list[dict], margin: float = RECALL_MARGIN) -> str:
    """The fastest model whose recall is within `margin` of the best recall."""
    best = max(r["recall"] for r in rows)
    eligible = [r for r in rows if r["recall"] >= best - margin - 1e-9]
    return min(eligible, key=lambda r: r["ms_median"])["model"]


def _read_frames(video: Path, indices: list[int]) -> tuple[list[np.ndarray], tuple[float, float], tuple[int, int]]:
    """Decode the wanted frames in one sequential pass (exact, unlike seeking in H.264) and
    resize each to width MAX_WIDTH. Returns (frames, (sx, sy) scale factors, (src_w, src_h))."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = min(MAX_WIDTH, src_w)
        out_h = round(src_h * out_w / src_w)
        wanted = set(indices)
        frames: list[np.ndarray] = []
        pos = 0
        last = max(indices)
        while pos <= last:
            if pos in wanted:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA))
            elif not cap.grab():
                break
            pos += 1
    finally:
        cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f"decoded {len(frames)} of {len(indices)} sampled frames from {video}")
    return frames, (out_w / src_w, out_h / src_h), (src_w, src_h)


def bench_detectors(video: Path, geom: Path, types: Path, frames: int = 60,
                    models: tuple[str, ...] = CANDIDATES, score_thr: float = SCORE_THR,
                    out: Path | None = None) -> dict:
    from .detect import Detector

    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if n_total <= 0:
        raise RuntimeError(f"cannot read the frame count of {video}")
    indices = sample_indices(n_total, frames)
    imgs, (sx, sy), (src_w, src_h) = _read_frames(video, indices)
    gt_src = _load_person_boxes(geom, types, set(indices))

    gt_per_frame: list[list[tuple[float, float, float, float]]] = []
    n_small = 0
    for idx in indices:
        scaled = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in gt_src.get(idx, [])]
        keep = [b for b in scaled if b[3] - b[1] >= MIN_GT_HEIGHT]
        n_small += len(scaled) - len(keep)
        gt_per_frame.append(keep)
    n_gt = sum(len(g) for g in gt_per_frame)
    if n_gt == 0:
        raise RuntimeError("no ground-truth person boxes at the sampled frames; wrong geom/types files?")

    rows = []
    for name in models:
        det = Detector(name, score_thr=score_thr)
        for img in imgs[:WARMUP_FRAMES]:
            det.persons(img)
        times, tp, fp, fn, n_det = [], 0, 0, 0, 0
        hit = {label: [0, 0] for _, _, label in HEIGHT_BUCKETS}  # bucket -> [matched, total]
        for img, gt in zip(imgs, gt_per_frame):
            t0 = time.perf_counter()
            people = det.persons(img)
            times.append((time.perf_counter() - t0) * 1000.0)
            flags = greedy_flags(gt, [p.bbox for p in people], [p.score for p in people])
            for box, matched in zip(gt, flags):
                cell = hit[height_bucket(box[3] - box[1])]
                cell[0] += matched
                cell[1] += 1
            a = sum(flags)
            tp, fp, fn, n_det = tp + a, fp + len(people) - a, fn + len(gt) - a, n_det + len(people)
        rows.append({
            "model": name,
            "ms_median": round(statistics.median(times), 2),
            "ms_mean": round(statistics.fmean(times), 2),
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "n_det": n_det,
            "recall_by_gt_height": {k: {"n": v[1], "recall": round(v[0] / v[1], 4) if v[1] else None}
                                    for k, v in hit.items()},
        })
    chosen = choose_detector(rows)
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "clip": Path(video).name,
        "source_size": [src_w, src_h],
        "processed_width": MAX_WIDTH,
        "frames": len(indices),
        "frame_indices": indices,
        "gt_boxes": n_gt,
        "gt_boxes_ignored_small": n_small,
        "score_thr": score_thr,
        "iou_thr": IOU_THR,
        "min_gt_height_px": MIN_GT_HEIGHT,
        "warmup_frames": WARMUP_FRAMES,
        "rule": f"fastest model whose recall is within {RECALL_MARGIN * 100:g} points of the best",
        "rows": rows,
        "chosen": chosen,
        "precision_note": PRECISION_NOTE,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "threads_per_model": 4},
    }
    out = Path(out) if out else repo_root() / "runs" / "bench" / "detectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    result["out"] = str(out)
    return result


def format_table(result: dict) -> str:
    lines = [f"clip {result['clip']}  frames {result['frames']}  GT person boxes {result['gt_boxes']} "
             f"(+{result['gt_boxes_ignored_small']} ignored, < {result['min_gt_height_px']:g} px high)  "
             f"score_thr {result['score_thr']}  IoU {result['iou_thr']}",
             f"{'model':<13}{'ms/frame':>10}{'recall':>9}{'precision':>11}{'tp':>7}{'fp':>7}{'fn':>7}"]
    for r in result["rows"]:
        lines.append(f"{r['model']:<13}{r['ms_median']:>10.1f}{r['recall'] * 100:>8.1f}%{r['precision'] * 100:>10.1f}%"
                     f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}")
    lines.append(f"recall by GT person height (px at width {result['processed_width']}):")
    for r in result["rows"]:
        parts = [f"{k} {v['recall'] * 100:.1f}% (n={v['n']})" if v["recall"] is not None else f"{k} n/a"
                 for k, v in r["recall_by_gt_height"].items()]
        lines.append(f"  {r['model']:<13}" + "   ".join(parts))
    lines.append(f"rule: {result['rule']}")
    lines.append(f"chosen: {result['chosen']}")
    lines.append(f"note: {result['precision_note']}")
    return "\n".join(lines)
