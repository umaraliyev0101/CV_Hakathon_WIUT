import numpy as np

from sim import labels


def test_boxes_from_mask_and_drop_rule():
    H, W = 72, 128
    mask = np.zeros((H, W, 3), np.uint8)
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    mask[10:40, 20:41] = colors[1]          # 30 x 21 = 630 px, h 30
    mask[50:52, 60:110] = colors[2]         # 2 x 50 = 100 px -> dropped (too few px)
    mask[0:72, 0:3] = colors[3]             # 216 px -> dropped
    out = labels.boxes_from_mask(mask, colors, min_px=400, min_h=20)
    assert set(out) == {1}
    assert out[1]["box_xyxy"] == [20, 10, 41, 40]
    assert out[1]["mask_px"] == 630


def test_short_box_dropped():
    mask = np.zeros((100, 400, 3), np.uint8)
    mask[10:29, 0:300] = (255, 0, 0)        # 19 px tall, 5700 px
    assert labels.boxes_from_mask(mask, {1: (255, 0, 0)}, 400, 20) == {}


def test_unknown_colours_detected():
    mask = np.zeros((10, 10, 3), np.uint8)
    mask[0, 0] = (255, 0, 0)
    mask[5, 5] = (12, 200, 7)
    assert labels.unknown_colour_px(mask, {1: (255, 0, 0)}) == 1


def test_visible_frac():
    full = np.zeros((10, 10, 3), np.uint8)
    full[0:10, 0:4] = (255, 0, 0)
    vis = full.copy()
    vis[0:5, 0:4] = 0
    assert labels.visible_frac(vis, full, (255, 0, 0)) == 0.5


def test_gt_roundtrip(tmp_path):
    gt = {"clip_id": "x", "frames": [{"t": 0.0, "people": [{"id": 1, "box_xyxy": [1, 2, 3, 4]}]}]}
    p = tmp_path / "gt.json"
    labels.write_gt(gt, p)
    assert labels.read_gt(p) == gt
