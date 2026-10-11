"""Pre-label real Pexels showroom clips with RF-DETR Large and export COCO for fine-tuning.

Stage 1 (prelabel): sample frames at --fps from every clip in data/real_open/, run COCO RF-DETR
Large, keep `person` boxes with score >= --threshold, write data/real_labels/prelabels.json and
overlay JPEGs for visual review.

Stage 2 (export): apply data/real_labels/review.json (box deletions, additions, dropped frames)
and write data/real_labels/coco/{train,valid,test}/_annotations.coco.json plus the images, in the
same layout as data/synthetic/coco/.

The split is by clip, never by frame (neighbouring frames are near-duplicates). The test split is
pre-labelled + model-reviewed only; it must be human-verified before it is reported as
human-labelled (prompts/teammate_c_full_run_goal.md).

Labelling policy: every visible real person gets a box, including people seen through glass, people
cut off at the frame edge and blurred background staff. Reflections (car paint, windows) are not
people and get no box.

Usage from repo root:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/label_real.py prelabel
    .venv/bin/python models/label_real.py export
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "real_open"
OUT = ROOT / "data" / "real_labels"
PERSON_CLASS_ID = 1  # raw COCO category id, see models/detect.py

TEST_CLIPS = {"7154214", "7154217", "7154241", "36512656", "7154228", "7154230", "7154216",
              "7154233", "7154238"}
VALID_CLIPS = {"7154212", "7154225", "7154236"}


def split_of(clip_id: str) -> str:
    if clip_id in TEST_CLIPS:
        return "test"
    if clip_id in VALID_CLIPS:
        return "valid"
    return "train"


def prelabel(fps: float, threshold: float) -> None:
    from PIL import Image
    from rfdetr import RFDETRLarge

    model = RFDETRLarge()
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "viz").mkdir(parents=True, exist_ok=True)
    records = []
    for clip in sorted(CLIPS.glob("pexels_*.mp4")):
        clip_id = clip.stem.split("_")[1]
        cap = cv2.VideoCapture(str(clip))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = src_fps / fps
        idx = step / 2
        while idx < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                break
            name = f"pexels_{clip_id}_f{int(idx):05d}.jpg"
            cv2.imwrite(str(OUT / "frames" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            det = model.predict(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                                threshold=threshold)
            boxes = [{"bbox_xyxy": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                     for b, s, c in zip(det.xyxy, det.confidence, det.class_id)
                     if int(c) == PERSON_CLASS_ID]
            records.append({"file": name, "clip_id": clip_id, "split": split_of(clip_id),
                            "frame": int(idx), "width": bgr.shape[1], "height": bgr.shape[0],
                            "boxes": boxes})
            draw(bgr, boxes, OUT / "viz" / name)
            idx += step
        print(clip.name, sum(r["clip_id"] == clip_id for r in records), "frames")
    (OUT / "prelabels.json").write_text(json.dumps(records, indent=1))
    print(len(records), "frames,", sum(len(r["boxes"]) for r in records), "boxes")


def draw(bgr, boxes, path: Path) -> None:
    im = bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b["bbox_xyxy"])
        col = (0, 200, 255) if b.get("src") == "review" else (0, 255, 0)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
        cv2.putText(im, f"#{i} {b.get('score', '')}", (x1 + 4, y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    scale = 1152 / max(im.shape[:2])
    cv2.imwrite(str(path), cv2.resize(im, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])


def apply_review(records: list[dict], review: dict) -> list[dict]:
    """review.json: {"drop_frames": [file], "delete": {file: [box idx]}, "add": {file: [[x1,y1,x2,y2]]},
    "set": {file: {idx: [x1,y1,x2,y2]}}}. Indices refer to prelabels.json box order."""
    out = []
    for r in records:
        if r["file"] in review.get("drop_frames", []):
            continue
        dele = set(review.get("delete", {}).get(r["file"], []))
        sets = review.get("set", {}).get(r["file"], {})
        boxes = []
        for i, b in enumerate(r["boxes"]):
            if i in dele:
                continue
            b = dict(b, src="model")
            if str(i) in sets:
                b = {"bbox_xyxy": sets[str(i)], "score": b["score"], "src": "review"}
            boxes.append(b)
        boxes += [{"bbox_xyxy": xyxy, "score": None, "src": "review"}
                  for xyxy in review.get("add", {}).get(r["file"], [])]
        out.append(dict(r, boxes=boxes))
    return out


def export() -> None:
    records = json.loads((OUT / "prelabels.json").read_text())
    review_path = OUT / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    records = apply_review(records, review)
    stats = {}
    for split in ("train", "valid", "test"):
        d = OUT / "coco" / split
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        images, anns = [], []
        for r in (r for r in records if r["split"] == split):
            shutil.copy2(OUT / "frames" / r["file"], d / r["file"])
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": r["file"], "width": r["width"],
                           "height": r["height"], "clip_id": r["clip_id"], "frame": r["frame"]})
            for b in r["boxes"]:
                x1, y1, x2, y2 = b["bbox_xyxy"]
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(r["width"]), x2), min(float(r["height"]), y2)
                w, h = round(x2 - x1, 1), round(y2 - y1, 1)
                anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1,
                             "bbox": [round(x1, 1), round(y1, 1), w, h], "area": round(w * h, 1),
                             "iscrowd": 0, "attributes": {"source": b["src"], "score": b["score"]}})
        coco = {"info": {"description": "real Pexels showroom clips; RF-DETR Large pre-labels, "
                                         "model-reviewed; test split pending human verification",
                         "version": "1"},
                "licenses": [], "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
                "images": images, "annotations": anns}
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
        stats[split] = {"clips": len({i["clip_id"] for i in images}), "images": len(images),
                        "boxes": len(anns)}
    (OUT / "export_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prelabel", "export"])
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.3)
    a = ap.parse_args()
    prelabel(a.fps, a.threshold) if a.stage == "prelabel" else export()


if __name__ == "__main__":
    main()
