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


def build_gt(cfg, timeline, cam, lay, clip_dir, events_rule, git_sha, config_rel, qa_labels):
    """Assemble gt.json from the timeline + rendered masks in clip_dir/mask and clip_dir/solo."""
    from sim import projection, schema
    clip_dir = Path(clip_dir)
    fps_out = schema.out_fps(cfg)
    colours = {p["id"]: person_colour(p["id"]) for p in cfg["persons"]}
    frames = []
    for k, sf in enumerate(schema.output_scene_frames(cfg)):
        mask = read_png_rgb(clip_dir / "mask" / f"{k:04d}.png")
        boxes = boxes_from_mask(mask, colours, qa_labels["min_mask_px"], qa_labels["min_box_h_px"])
        people = []
        for pid, b in sorted(boxes.items()):
            st = timeline["agents"][str(pid)]["frames"][sf]
            if not st["present"]:
                continue
            solo = read_png_rgb(clip_dir / "solo" / f"p{pid}" / f"{k:04d}.png")
            people.append({"id": pid, "box_xyxy": b["box_xyxy"], "mask_px": b["mask_px"],
                           "visible_frac": visible_frac(mask, solo, colours[pid]),
                           "pos_m": st["pos_m"], "yaw_deg": st["yaw_deg"], "action": st["action"]})
        frames.append({"t": round(k / fps_out, 6), "scene_frame": sf, "people": people})
    persons = []
    for p in cfg["persons"]:
        persons.append({"id": p["id"], "role": p["role"], "height_m": p["height_m"], "outfit": p["outfit"],
                        "hard_negative": p["hard_negative"], "enter_t": p["enter_t"], "exit_t": None,
                        "present_at_start": p["present_at_start"], "mask_rgb": list(colours[p["id"]])})
    zones = {name: {"floor_m": [[round(x, 4), round(y, 4)] for x, y in poly],
                    "image_px": projection.floor_polygon_to_image(cam, poly)}
             for name, poly in lay["zones"].items()}
    return {
        "clip_id": schema.clip_id(cfg), "labeller": "synthetic-ground-truth",
        "generator": {"git_sha": git_sha, "config": config_rel, "seed": cfg["seed"]},
        "fps": fps_out, "width": cam.width, "height": cam.height, "duration_s": cfg["duration_s"],
        "camera": cam.to_dict(), "zones": zones, "persons": persons, "frames": frames,
        "events_scripted": timeline["events_scripted"],
        "events_by_rule": {"short_clip": events_rule},
        "notes": {"box_xyxy": "bounding box of mask pixels, x2/y2 exclusive",
                  "visible_frac": "mask px / solo-render mask px (person rendered alone, in-image only)",
                  "drop_rule": f"mask < {qa_labels['min_mask_px']} px or box < {qa_labels['min_box_h_px']} px tall"},
    }
