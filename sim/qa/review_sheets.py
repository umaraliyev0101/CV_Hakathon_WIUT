"""Zoomed review sheets for the visual review (reports/synth_visual_review.md).

python -m sim.qa.review_sheets <clip_dir> [out.png]
Row per sampled frame (12 evenly spaced): full-resolution crops around every labelled person (box
expanded 40 %, feet included) with the GT box. Final rows: 6 consecutive frames of the longest
walking stretch of each person (to judge sliding / moonwalking) and of a turn if any."""
import sys
from pathlib import Path

import numpy as np


def _crop(img, box, pad=0.4, h_out=260):
    import cv2
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(w, h) * (1 + pad) / 2
    X1, Y1, X2, Y2 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(img.shape[1], cx + s)), int(min(img.shape[0], cy + s))
    c = img[Y1:Y2, X1:X2].copy()
    cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1 - 1, y2 - Y1 - 1), (0, 255, 255), 1)
    if c.size == 0:
        c = np.zeros((10, 10, 3), np.uint8)
    return cv2.resize(c, (int(c.shape[1] * h_out / c.shape[0]), h_out))


def make(clip, out, motion_only=False, tile_h=230):
    import cv2
    from sim import labels
    gt = labels.read_gt(clip / "gt.json")
    roles = {p["id"]: p["role"] for p in gt["persons"]}
    cap = cv2.VideoCapture(str(clip / f"{gt['clip_id']}.mp4"))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    n = len(frames)
    tiles = []
    W, H = 1800, tile_h

    def tile(img, box, label, pad=0.4):
        t = _crop(img, box, pad=pad, h_out=H)
        cv2.rectangle(t, (0, 0), (t.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(t, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    for k in ([] if motion_only else [int(round(i)) for i in np.linspace(0, n - 1, 12)]):
        fr = gt["frames"][k]
        for p in fr["people"]:
            tile(frames[k], p["box_xyxy"], f"f{k} #{p['id']} {roles[p['id']][:4]} {p['action']}")
    for pers in gt["persons"]:
        seq = [(k, next((p for p in fr["people"] if p["id"] == pers["id"]), None)) for k, fr in enumerate(gt["frames"])]
        best, cur = [], []
        for k, p in seq:
            if p is not None and p["action"] == "walk":
                cur.append((k, p))
            else:
                best, cur = (cur if len(cur) > len(best) else best), []
        best = cur if len(cur) > len(best) else best
        if len(best) >= 6:
            mid = len(best) // 2
            for k, p in best[mid - 3: mid + 3]:
                tile(frames[k], p["box_xyxy"], f"{gt['clip_id'][:4]} WALK f{k} #{pers['id']}", pad=0.6)
    # in-place turns (stepped turn clips): 6 consecutive output frames around the longest turn
    import json
    tl_path = clip.resolve().parents[1] / "scenes" / gt["scene_id"] / "timeline.json"
    if not tl_path.exists():
        tl_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "scenes" / gt["scene_id"] / "timeline.json"
    tl = json.loads(tl_path.read_text())
    for pers in gt["persons"]:
        st = tl["agents"][str(pers["id"])]["frames"]
        turn = [k for k, fr in enumerate(gt["frames"]) if st[fr["scene_frame"]]["anim_clip"].startswith("turn")]
        runs, cur_r = [], []
        for k in turn:
            if cur_r and k != cur_r[-1] + 1:
                runs.append(cur_r)
                cur_r = []
            cur_r.append(k)
        if cur_r:
            runs.append(cur_r)
        if not runs:
            continue
        r = max(runs, key=len)
        ks = list(range(max(0, r[0] - 1), min(n, r[0] + 5)))
        for k in ks:
            p = next((q for q in gt["frames"][k]["people"] if q["id"] == pers["id"]), None)
            if p is not None:
                tile(frames[k], p["box_xyxy"], f"{gt['clip_id'][:4]} TURN f{k} #{pers['id']} {st[gt['frames'][k]['scene_frame']]['anim_clip']}", pad=0.6)
    rows, cur, x = [], np.zeros((H, W, 3), np.uint8), 0
    for t in tiles:
        if x + t.shape[1] > W:
            rows.append(cur)
            cur, x = np.zeros((H, W, 3), np.uint8), 0
        cur[:, x:x + t.shape[1]] = t[:, :min(t.shape[1], W - x)]
        x += t.shape[1] + 4
    rows.append(cur)
    cv2.imwrite(str(out), np.vstack(rows))
    return out, tiles


if __name__ == "__main__":
    clip = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else clip / "review_sheet.jpg"
    print(make(clip, out)[0])
