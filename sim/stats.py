"""Dataset and render statistics for reports/synth_status.md (measured from outputs).

python sim/stats.py   -> prints markdown tables, writes data/synthetic/dataset_stats.json
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"


def main():
    clips = sorted(d for d in (DATA / "clips").iterdir() if (d / "gt.json").exists())
    by_scen, by_light = Counter(), Counter()
    frames = person_boxes = 0
    roles = Counter()
    events = Counter()
    seen_scene_events = set()
    for c in clips:
        gt = json.loads((c / "gt.json").read_text())
        by_scen[gt["scenario"]] += 1
        by_light[gt["lighting"]] += 1
        frames += len(gt["frames"])
        r = {p["id"]: p["role"] for p in gt["persons"]}
        for fr in gt["frames"]:
            person_boxes += len(fr["people"])
            for p in fr["people"]:
                roles[r[p["id"]]] += 1
        if gt["scene_id"] not in seen_scene_events:
            seen_scene_events.add(gt["scene_id"])
            for e in gt["events_scripted"]:
                events[e["type"]] += 1
    scenes = sorted({json.loads((c / "gt.json").read_text())["scene_id"] for c in clips})
    persons = 0
    for s in scenes:
        cfg = json.loads((REPO / "sim" / "configs" / f"scene_{s[1:]}.json").read_text())
        persons += len(cfg["persons"])
    rs = defaultdict(list)
    gpus = set()
    for s in scenes:
        st = json.loads((DATA / "scenes" / s / "render_stats.json").read_text())
        gpus |= set(st.get("gpu_index", []))
        for cam in st["cams"].values():
            for k, v in cam.items():
                rs[k.split("_")[0] + "_" + k.split("_")[1] if k.startswith("rgb") else k].append(v["mean_s"])
    out = {"clips": len(clips), "scenes": len(scenes), "frames": frames, "person_boxes": person_boxes,
           "boxes_by_role": dict(roles), "clips_by_scenario": dict(by_scen), "clips_by_lighting": dict(by_light),
           "unique_persons": persons, "scripted_events_by_type (per scene)": dict(events),
           "gpu_index": sorted(gpus),
           "s_per_frame_mean_over_scenes": {k: round(statistics.mean(v), 3) for k, v in rs.items()}}
    exp = DATA / "export_stats.json"
    if exp.exists():
        out["export"] = json.loads(exp.read_text())
    (DATA / "dataset_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
