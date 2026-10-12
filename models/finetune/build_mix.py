"""Build data/finetune/ (symlinks): real train x3 + every 4th synthetic frame; real test untouched.

Usage from repo root: .venv/bin/python -I models/finetune/build_mix.py
"""

import json
import os
import shutil
from pathlib import Path

R, S, O = Path("data/real_labels/coco"), Path("data/synthetic/coco"), Path("data/finetune")


def add(out, split, src_dir, coco, keep, prefix, reps):
    for r in range(reps):
        idmap = {}
        for im in coco["images"]:
            if not keep(im):
                continue
            nid = len(out["images"]) + 1
            idmap[im["id"]] = nid
            fn = f"{prefix}{r}_{im['file_name']}"
            os.symlink((src_dir / im["file_name"]).resolve(), O / split / fn)
            out["images"].append({"id": nid, "file_name": fn, "width": im["width"], "height": im["height"]})
        for a in coco["annotations"]:
            if a["image_id"] in idmap:
                out["annotations"].append({"id": len(out["annotations"]) + 1, "image_id": idmap[a["image_id"]],
                                           "category_id": 1, "bbox": a["bbox"], "area": a["area"], "iscrowd": 0})


if __name__ == "__main__":
    if O.exists():
        shutil.rmtree(O)
    plan = {"train": [(R / "train", lambda i: True, "real", 3), (S / "train", lambda i: i["id"] % 4 == 0, "syn", 1)],
            "valid": [(R / "valid", lambda i: True, "real", 1), (S / "valid", lambda i: i["id"] % 4 == 0, "syn", 1)],
            "test": [(R / "test", lambda i: True, "real", 1)]}
    for split, srcs in plan.items():
        (O / split).mkdir(parents=True)
        out = {"info": {"description": "fine-tune mix: real Pexels (reviewed) + synthetic CCTV"}, "licenses": [],
               "categories": [{"id": 1, "name": "person", "supercategory": "person"}], "images": [], "annotations": []}
        for d, keep, pre, reps in srcs:
            add(out, split, d, json.loads((d / "_annotations.coco.json").read_text()), keep, pre, reps)
        (O / split / "_annotations.coco.json").write_text(json.dumps(out))
        print(split, len(out["images"]), "images", len(out["annotations"]), "boxes")
