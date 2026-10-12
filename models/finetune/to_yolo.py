"""Convert data/finetune (COCO, symlinked images) to Ultralytics YOLO layout in data/finetune_yolo/."""

import json
import os
import shutil
from pathlib import Path

SRC, DST = Path("data/finetune"), Path("data/finetune_yolo")

if __name__ == "__main__":
    if DST.exists():
        shutil.rmtree(DST)
    for split, ysplit in (("train", "train"), ("valid", "val")):
        (DST / "images" / ysplit).mkdir(parents=True)
        (DST / "labels" / ysplit).mkdir(parents=True)
        c = json.loads((SRC / split / "_annotations.coco.json").read_text())
        by = {}
        for a in c["annotations"]:
            by.setdefault(a["image_id"], []).append(a["bbox"])
        for im in c["images"]:
            os.symlink((SRC / split / im["file_name"]).resolve(), DST / "images" / ysplit / im["file_name"])
            W, H = im["width"], im["height"]
            lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in by.get(im["id"], [])]
            (DST / "labels" / ysplit / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        print(split, len(c["images"]))
    (DST / "data.yaml").write_text(f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
