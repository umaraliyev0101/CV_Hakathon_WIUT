"""Instance masks -> visible boxes, visible_frac, gt.json assembly (pure: numpy + cv2 for IO)."""
import json
from pathlib import Path

import numpy as np

# Mask colours: exact 8-bit values produced by Workbench FLAT/OBJECT colour with the Standard
# view transform (obj.color channels are 0 or 1, so sRGB encoding is exact).
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
           (255, 255, 255), (128, 0, 0), (0, 128, 0)]


def person_colour(pid):
    return PALETTE[(pid - 1) % len(PALETTE)]


def _match(mask, colour):
    return np.all(mask == np.asarray(colour, np.uint8), axis=-1)


def boxes_from_mask(mask_rgb, colours, min_px, min_h):
    """{id: {"box_xyxy": [x1, y1, x2, y2] (x2/y2 exclusive pixel edges), "mask_px": n}}"""
    out = {}
    for pid, col in colours.items():
        m = _match(mask_rgb, col)
        n = int(m.sum())
        if n < min_px:
            continue
        ys, xs = np.nonzero(m)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if y2 - y1 < min_h:
            continue
        out[pid] = {"box_xyxy": [x1, y1, x2, y2], "mask_px": n}
    return out


def visible_frac(mask_rgb, solo_rgb, colour):
    full = int(_match(solo_rgb, colour).sum())
    if full == 0:
        return 0.0
    return round(min(1.0, int(_match(mask_rgb, colour).sum()) / full), 4)


def unknown_colour_px(mask_rgb, colours):
    ok = np.all(mask_rgb == 0, axis=-1)
    for col in colours.values():
        ok |= _match(mask_rgb, col)
    return int((~ok).sum())


def read_png_rgb(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].copy()


def write_gt(gt, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gt, indent=1))


def read_gt(path):
    return json.loads(Path(path).read_text())


def build_gt(cfg, timeline, cam, lay, mask_dir, solo_dir, events_rule, git_sha, config_rel, qa_labels,
             cam_id, lighting, n_frames=None):
    """Assemble gt.json from the timeline + the camera's rendered instance masks and solo masks.

    A person is labelled in a frame when they are in the scene (rendered: inside or just outside
    the glass facade) and their mask passes the drop rule; `present` (inside the room) is kept per
    frame for the event rules."""
    from sim import projection, schema
    mask_dir, solo_dir = Path(mask_dir), Path(solo_dir)
    fps_out = schema.out_fps(cfg)
    colours = {p["id"]: person_colour(p["id"]) for p in cfg["persons"]}
    out_frames = schema.output_scene_frames(cfg)
    if n_frames:
        out_frames = out_frames[:n_frames]
    frames = []
    for k, sf in enumerate(out_frames):
        mask = read_png_rgb(mask_dir / f"{k:04d}.png")
        boxes = boxes_from_mask(mask, colours, qa_labels["min_mask_px"], qa_labels["min_box_h_px"])
        people = []
        for pid, b in sorted(boxes.items()):
            st = timeline["agents"][str(pid)]["frames"][sf]
            if not st["rendered"]:
                continue
            solo = read_png_rgb(solo_dir / f"p{pid}" / f"{k:04d}.png")
            people.append({"id": pid, "box_xyxy": b["box_xyxy"], "mask_px": b["mask_px"],
                           "visible_frac": visible_frac(mask, solo, colours[pid]),
                           "pos_m": st["pos_m"], "yaw_deg": st["yaw_deg"], "action": st["action"],
                           "present": st["present"]})
        frames.append({"t": round(k / fps_out, 6), "scene_frame": sf, "people": people})
    persons = []
    for p in cfg["persons"]:
        fr = timeline["agents"][str(p["id"])]["frames"]
        pres = [i for i, s in enumerate(fr) if s["present"]]
        enter_t = round(pres[0] / cfg["fps_scene"], 6) if pres else None
        exit_t = round((pres[-1] + 1) / cfg["fps_scene"], 6) if pres and pres[-1] + 1 < len(fr) else None
        persons.append({"id": p["id"], "role": p["role"], "height_m": p["height_m"], "outfit": p["outfit"],
                        "character": p.get("character"), "hard_negative": p["hard_negative"],
                        "enter_t": enter_t, "exit_t": exit_t,
                        "present_at_start": bool(fr[0]["present"]), "mask_rgb": list(colours[p["id"]])})
    zones = {name: {"floor_m": [[round(x, 4), round(y, 4)] for x, y in poly],
                    "image_px": projection.floor_polygon_to_image(cam, poly)}
             for name, poly in lay["zones"].items()}
    return {
        "clip_id": schema.clip_id(cfg, cam_id, lighting), "labeller": "synthetic-ground-truth",
        "scene_id": cfg["scene_id"], "scenario": cfg["scenario"], "camera_id": cam_id, "lighting": lighting,
        "generator": {"git_sha": git_sha, "config": config_rel, "seed": cfg["seed"]},
        "fps": fps_out, "width": cam.width, "height": cam.height,
        "duration_s": round(len(frames) / fps_out, 6),
        "camera": cam.to_dict(), "zones": zones, "persons": persons, "parties": cfg.get("parties", []),
        "frames": frames,
        "events_scripted": timeline["events_scripted"],
        "events_by_rule": events_rule,
        "eval_presets": cfg.get("eval_presets", ["short_clip"]),
        "notes": {"box_xyxy": "bounding box of mask pixels, x2/y2 exclusive",
                  "visible_frac": "mask px / solo-render mask px (person rendered alone, in-image only)",
                  "drop_rule": f"mask < {qa_labels['min_mask_px']} px or box < {qa_labels['min_box_h_px']} px tall",
                  "glass": "glass is hidden in the mask passes: people seen through the facade are labelled",
                  "distortion": "no barrel distortion is applied, so boxes align with the video",
                  "hair": "the mask pass ignores material alpha, so hair/eyelash cards count fully (boxes can be a few px larger at the head)",
                  "scripted_greet_time": "staff arrival (first talk frame after stopping 1.25 m away, facing the customer)",
                  "rule_event_time": "onset of the qualifying run (see sim/gt_events.py)"},
    }
