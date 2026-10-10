"""Build data/synthetic/share/: the best clips with mp4, gt.json, A-format labels, contact sheet and a
README per clip (what happens, expected events). Selection: sim/configs/share.json.

python sim/share.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import labels  # noqa: E402
from sim.export import a_format  # noqa: E402

DATA = REPO / "data" / "synthetic"
SHARE = DATA / "share"

STORY = {
    "greet_fast": "A customer walks in and goes to a car; a salesperson leaves the desk and greets them within ~15 s; they talk, then the salesperson returns to the desk.",
    "greet_slow": "A customer browses two cars for ~50 s while one salesperson stays idle at the desk ('missed while free'); a second salesperson finally greets them after 40-70 s. A wait alert fires at 30 s.",
    "walkout": "A customer browses two cars and leaves through the entrance without ever being greeted; the salesperson stays busy at the desk. Walk-out after 30-60 s.",
    "party": "Two customers arrive together (a party). The salesperson greets one of them; by the party rule that greets both.",
    "handoff": "A salesperson greets and talks with a customer, then a second salesperson takes over (handoff) and the first returns to the desk.",
    "crowded": "Two salespeople, five customers: a greet, a party greeted by the second salesperson, a walk-out and an unattended browser.",
    "dealership_long": "Four minutes: a long engagement (> 120 s) with one customer, while a second customer browses and walks out unattended.",
}


def _link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink() if (dst.is_symlink() or dst.is_file()) else shutil.rmtree(dst)
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent))


def readme(gt, hero):
    ev = gt["events_scripted"]
    preset = gt["eval_presets"][0]
    rule = gt["events_by_rule"][preset]
    roles = {p["id"]: p for p in gt["persons"]}

    def fmt(e):
        d = {k: v for k, v in e.items() if k not in ("type", "t")}
        return f"| {e['type']} | {e['t']:.2f} s | {json.dumps(d)} |"
    lines = [f"# {gt['clip_id']}" + ("  (HERO CLIP)" if hero else ""), "",
             f"Scenario **{gt['scenario']}**, scene {gt['scene_id']}, camera {gt['camera_id']}, lighting **{gt['lighting']}**. "
             f"{gt['duration_s']:.1f} s, {len(gt['frames'])} frames at {gt['fps']} fps, {gt['width']}x{gt['height']}.", "",
             STORY.get(gt["scenario"], ""), "",
             f"Camera: height {gt['camera']['height_m']} m, pitch {gt['camera']['pitch_deg']} deg, HFOV {gt['camera']['hfov_deg']} deg "
             f"(true values; A's pipeline assumes HFOV 70 deg and 1.70 m people).", "",
             "## People", "", "| id | role | height | character | hard negative | enters | leaves |", "|---|---|---|---|---|---|---|"]
    for p in gt["persons"]:
        lines.append(f"| {p['id']} | {p['role']} | {p['height_m']:.2f} m | {p['character']} | {p['hard_negative']} | "
                     f"{p['enter_t']} | {p['exit_t']} |")
    lines += ["", "## Expected events (ground truth = scripted)", "", "| type | t | details |", "|---|---|---|"]
    lines += [fmt(e) for e in ev]
    lines += ["", f"## Events from the rules on true 3D positions (`{preset}` preset, sim/configs/presets.json)", "",
              "| type | t | details |", "|---|---|---|"]
    lines += [fmt(e) for e in rule]
    lines += ["", "## Files", "",
              f"- `{gt['clip_id']}.mp4`: H.264 CCTV-look video (noise, blur, burned-in timestamp; no distortion, boxes align).",
              "- `gt.json`: full ground truth (per-frame boxes = visible mask bbox, visible_frac, positions in metres, yaw, action; camera K/R/t; zones in metres and pixels; scripted and rule events).",
              "- `labels.json`: A-format labels (adapter, FLAG: A's schema not available).",
              "- `contact_sheet.png`: 12 frames with GT boxes (staff red, customer blue), zones and event markers.",
              "", "Roles: staff wear the shared uniform (recoloured red suit); hard negatives are customers in suits / a burgundy blazer.",
              "Scripted greet time = staff arrival (stops 1.25 m away facing the customer); rule greet time = onset of the "
              "qualifying run (enters 1.5 m while facing). Licences: see DATA_SOURCES.md."]
    return "\n".join(lines) + "\n"


def main():
    sel = json.loads((REPO / "sim" / "configs" / "share.json").read_text())
    if SHARE.exists():
        shutil.rmtree(SHARE)
    SHARE.mkdir(parents=True)
    index = ["# Synthetic showroom share pack", "", "| clip | scenario | lighting | camera | duration | events |", "|---|---|---|---|---|---|"]
    for item in sel["clips"]:
        src = DATA / "clips" / item["clip"]
        gt = labels.read_gt(src / "gt.json")
        dst = SHARE / gt["clip_id"]
        dst.mkdir()
        for f in (f"{gt['clip_id']}.mp4", "gt.json", "contact_sheet.png"):
            shutil.copy(src / f, dst / f)
        for d in ("rgb", "mask", "solo"):
            _link(src / d, dst / d)
        (dst / "labels.json").write_text(json.dumps(a_format(gt), indent=1))
        (dst / "README.md").write_text(readme(gt, item.get("hero", False)))
        evs = ", ".join(f"{e['type']}@{e['t']:.1f}s" for e in gt["events_scripted"] if e["type"] != "engagement")
        index.append(f"| `{gt['clip_id']}`{' **HERO**' if item.get('hero') else ''} | {gt['scenario']} | {gt['lighting']} | "
                     f"{gt['camera_id']} | {gt['duration_s']:.0f} s | {evs} |")
        print("shared", gt["clip_id"])
    index += ["", "Each folder: mp4, gt.json, labels.json (A-format adapter), contact_sheet.png, README.md. "
              "`rgb/`, `mask/`, `solo/` are symlinks to the render outputs used by the QA gates."]
    (SHARE / "README.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
