"""Exports from per-clip gt.json (pure: numpy/cv2/json).

python sim/export.py   ->  data/synthetic/coco/{train,valid,test}/_annotations.coco.json + JPEGs (q95)
                           data/synthetic/mot/<clip_id>/gt/gt.txt + seqinfo.ini
                           data/synthetic/labels/<clip_id>.json   (A-format adapter, see a_format())

COCO frames are decoded from the final CCTV-look MP4 (what a deployed detector sees), sampled
every `sample_every_n_frames` output frames (2 fps at 10 fps), split by scene (configs/splits.json).
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import labels  # noqa: E402

DATA = REPO / "data" / "synthetic"
A_FORMAT_VERSION = "adapter-0.1"


def a_format(gt):
    """A-format labels. FLAG: A's spec section 6 schema was not available; this is a documented adapter
    from gt.json: persons with roles, greet times, engagement intervals, walk-outs, parties, handoffs
    (ground truth = scripted events) plus rule-only wait alerts. Reconcile field names with A."""
    ev = gt["events_scripted"]
    preset = gt.get("eval_presets", ["short_clip"])[0]
    rule = gt["events_by_rule"][preset]
    first_seen, last_seen = {}, {}
    for fr in gt["frames"]:
        for p in fr["people"]:
            first_seen.setdefault(p["id"], fr["t"])
            last_seen[p["id"]] = fr["t"]
    return {
        "format": A_FORMAT_VERSION, "_flag": "adapter from gt.json; A's spec (salespeople_prompts/...design.md section 6) missing",
        "clip_id": gt["clip_id"], "fps": gt["fps"], "duration_s": gt["duration_s"], "preset": preset,
        "camera": {"hfov_deg": gt["camera"]["hfov_deg"], "height_m": gt["camera"]["height_m"],
                   "pitch_deg": gt["camera"]["pitch_deg"]},
        "persons": [{"track_id": p["id"], "role": p["role"], "height_m": p["height_m"],
                     "enter_t": p["enter_t"], "exit_t": p["exit_t"],
                     "first_seen_t": first_seen.get(p["id"]), "last_seen_t": last_seen.get(p["id"]),
                     "hard_negative": p["hard_negative"]} for p in gt["persons"]],
        "greets": [{"staff": e["staff"], "customer": e["customer"], "t": e["t"]} for e in ev if e["type"] == "greet"],
        "engagements": [{"staff": e["staff"], "customer": e["customer"], "t_start": e["t"], "t_end": e["t_end"]}
                        for e in ev if e["type"] == "engagement"],
        "walkouts": [{"customer": e["customer"], "t": e["t"]} for e in ev if e["type"] == "walkout"],
        "parties": [{"customers": e["customers"], "t": e["t"]} for e in ev if e["type"] == "party"],
        "handoffs": [{"customer": e["customer"], "from_staff": e["from_staff"], "to_staff": e["to_staff"], "t": e["t"]}
                     for e in ev if e["type"] == "handoff"],
        "wait_alerts": [{"customer": e["customer"], "t": e["t"]} for e in rule if e["type"] == "wait_alert"],
    }


def export_mot(gt, out_dir):
    out = out_dir / gt["clip_id"]
    (out / "gt").mkdir(parents=True, exist_ok=True)
    lines = []
    for k, fr in enumerate(gt["frames"]):
        for p in fr["people"]:
            x1, y1, x2, y2 = p["box_xyxy"]
            lines.append(f"{k + 1},{p['id']},{x1},{y1},{x2 - x1},{y2 - y1},1,1,{p['visible_frac']:.4f}")
    (out / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    (out / "seqinfo.ini").write_text(
        f"[Sequence]\nname={gt['clip_id']}\nimDir=img1\nframeRate={gt['fps']}\nseqLength={len(gt['frames'])}\n"
        f"imWidth={gt['width']}\nimHeight={gt['height']}\nimExt=.jpg\n")
    return len(lines)


def export_coco(clip_dirs, splits):
    import cv2
    every = splits["sample_every_n_frames"]
    split_of = {s: k for k in ("train", "valid", "test") for s in splits[k]}
    cocos = {k: {"info": {"description": "synthetic showroom CCTV (sim/), person boxes = visible mask bbox",
                          "version": "1"}, "licenses": [],
                 "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
                 "images": [], "annotations": []} for k in ("train", "valid", "test")}
    for k in cocos:
        d = DATA / "coco" / k
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    roles_n = {}
    for cd in clip_dirs:
        gt = labels.read_gt(cd / "gt.json")
        split = split_of.get(gt["scene_id"])
        if split is None:
            continue
        coco = cocos[split]
        roles = {p["id"]: p["role"] for p in gt["persons"]}
        cap = cv2.VideoCapture(str(cd / f"{gt['clip_id']}.mp4"))
        for k, fr in enumerate(gt["frames"]):
            ok, img = cap.read()
            if not ok:
                break
            if k % every:
                continue
            name = f"{gt['clip_id']}_{k:04d}.jpg"
            cv2.imwrite(str(DATA / "coco" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            iid = len(coco["images"]) + 1
            coco["images"].append({"id": iid, "file_name": name, "width": gt["width"], "height": gt["height"],
                                   "clip_id": gt["clip_id"], "scene_id": gt["scene_id"], "lighting": gt["lighting"],
                                   "frame": k})
            for p in fr["people"]:
                x1, y1, x2, y2 = p["box_xyxy"]
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": iid, "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                    "attributes": {"role": roles[p["id"]], "visible_frac": p["visible_frac"], "track_id": p["id"]}})
                roles_n[(split, roles[p["id"]])] = roles_n.get((split, roles[p["id"]]), 0) + 1
        cap.release()
    stats = {}
    for k, coco in cocos.items():
        (DATA / "coco" / k / "_annotations.coco.json").write_text(json.dumps(coco))
        stats[k] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
                    "scenes": sorted({im["scene_id"] for im in coco["images"]}),
                    "staff": roles_n.get((k, "staff"), 0), "customer": roles_n.get((k, "customer"), 0)}
    return stats


def main():
    clip_dirs = sorted(d for d in (DATA / "clips").iterdir() if (d / "gt.json").exists() and
                       (d / f"{d.name}.mp4").exists())
    splits = json.loads((REPO / "sim" / "configs" / "splits.json").read_text())
    (DATA / "labels").mkdir(parents=True, exist_ok=True)
    n_mot = 0
    for cd in clip_dirs:
        gt = labels.read_gt(cd / "gt.json")
        (DATA / "labels" / f"{gt['clip_id']}.json").write_text(json.dumps(a_format(gt), indent=1))
        n_mot += export_mot(gt, DATA / "mot")
    stats = export_coco(clip_dirs, splits)
    stats["clips"] = len(clip_dirs)
    stats["mot_rows"] = n_mot
    (DATA / "export_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
