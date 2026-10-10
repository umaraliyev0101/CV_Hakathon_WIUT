"""12 evenly spaced frames from the final MP4 with GT boxes (role colour + id), zone outlines and
event markers, plus a timeline strip. python -m sim.qa.contact_sheets <clip_dir>"""
import sys
from pathlib import Path

import numpy as np

ROLE_BGR = {"staff": (0, 0, 255), "customer": (255, 160, 0)}  # staff red, customer blue
ZONE_BGR = {"car": (0, 220, 255), "desk": (200, 0, 200), "entrance": (0, 255, 0)}


def _zone_col(name):
    return ZONE_BGR["car"] if name.startswith("car_") else ZONE_BGR.get(name, (255, 255, 255))


def make_sheet(mp4, gt_path, out_png, n=12, cols=4, tile_w=640):
    import cv2
    from sim import labels
    gt = labels.read_gt(gt_path)
    roles = {p["id"]: p["role"] for p in gt["persons"]}
    events = [(e["t"], f"{e['type']} (scripted)") for e in gt["events_scripted"]]
    for preset, evs in gt["events_by_rule"].items():
        events += [(e["t"], f"{e['type']} (rule:{preset})") for e in evs]
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = [int(round(i)) for i in np.linspace(0, total - 1, n)]
    tiles = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((gt["height"], gt["width"], 3), np.uint8)
        fr = gt["frames"][k]
        for name, z in gt["zones"].items():
            if len(z["image_px"]) >= 3:
                pts = np.round(np.array(z["image_px"])).astype(np.int32)
                cv2.polylines(img, [pts], True, _zone_col(name), 1, cv2.LINE_AA)
                c = pts.mean(axis=0).astype(int)
                if 0 <= c[0] < gt["width"] and 0 <= c[1] < gt["height"]:
                    cv2.putText(img, name, tuple(int(v) for v in c), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                _zone_col(name), 1, cv2.LINE_AA)
        for p in fr["people"]:
            col = ROLE_BGR[roles[p["id"]]]
            x1, y1, x2, y2 = p["box_xyxy"]
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), col, 2)
            lab = f"#{p['id']} {roles[p['id']]} {p['action']} v={p['visible_frac']:.2f}"
            # label under the box when the box touches the burned-in timestamp band at the top
            ty = y1 - 8 if y1 > 60 else min(gt["height"] - 50, y2 + 26)
            cv2.putText(img, lab, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, lab, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        active = [lab for t, lab in events if t <= fr["t"]]
        head = f"frame {k}  t={fr['t']:.1f}s"
        if active:
            head += "  | after: " + ", ".join(sorted(set(active)))
        cv2.rectangle(img, (0, gt["height"] - 44), (gt["width"], gt["height"]), (0, 0, 0), -1)
        cv2.putText(img, head, (10, gt["height"] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        th = int(round(gt["height"] * tile_w / gt["width"]))
        tiles.append(cv2.resize(img, (tile_w, th), interpolation=cv2.INTER_AREA))
    cap.release()
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(int(np.ceil(n / cols)))]
    sheet = np.vstack(rows)
    # timeline strip: frames sampled (white ticks) and events (scripted green, rule orange)
    strip = np.full((70, sheet.shape[1], 3), 30, np.uint8)
    dur = gt["duration_s"]
    x_of = lambda t: int(10 + (sheet.shape[1] - 20) * t / dur)  # noqa: E731
    cv2.line(strip, (10, 35), (sheet.shape[1] - 10, 35), (200, 200, 200), 1)
    for k in idx:
        cv2.line(strip, (x_of(gt["frames"][k]["t"]), 28), (x_of(gt["frames"][k]["t"]), 42), (255, 255, 255), 1)
    for t, lab in events:
        col = (0, 255, 0) if "scripted" in lab else (0, 165, 255)
        y = 20 if "scripted" in lab else 62
        cv2.line(strip, (x_of(t), 10), (x_of(t), 60), col, 2)
        cv2.putText(strip, f"{lab} t={t:.2f}", (x_of(t) + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(strip, f"{gt['clip_id']}  boxes: staff=red customer=blue  zones: car=yellow desk=magenta "
                f"entrance=green", (10, 62 if not events else 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    sheet = np.vstack([sheet, strip])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return Path(out_png)


if __name__ == "__main__":
    from sim.qa.common import clip_arg
    from sim import labels as _l
    clip = clip_arg(sys.argv)
    g = _l.read_gt(clip / "gt.json")
    print(make_sheet(clip / f"{g['clip_id']}.mp4", clip / "gt.json", clip / "contact_sheet.png"))
