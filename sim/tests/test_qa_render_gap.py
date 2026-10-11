"""qa_render's track-gap check fires on a label dropout of a well-visible person, not on a gap
while the person is outside the image."""
import json

import cv2
import numpy as np

from sim.qa import common, qa_render


def _clip(tmp_path, n=20, drop=range(5, 13), visible_in_gap=True):
    for d in ("rgb", "mask", "solo/p1"):
        (tmp_path / d).mkdir(parents=True)
    frames = []
    for k in range(n):
        img = np.full((72, 128, 3), 120, np.uint8)
        cv2.imwrite(str(tmp_path / "rgb" / f"{k:04d}.png"), img)
        m = np.zeros((72, 128, 3), np.uint8)
        if k not in drop or visible_in_gap:
            m[10:50, 40:60] = (0, 0, 255)          # BGR of RGB (255, 0, 0)
        cv2.imwrite(str(tmp_path / "mask" / f"{k:04d}.png"), m)
        cv2.imwrite(str(tmp_path / "solo" / "p1" / f"{k:04d}.png"), m)
        people = [] if k in drop else [{"id": 1, "box_xyxy": [40, 10, 60, 50], "visible_frac": 1.0}]
        frames.append({"t": k / 10, "people": people})
    gt = {"clip_id": "x", "width": 128, "height": 72, "lighting": "day",
          "generator": {"config": "configs/scene_000.json"},
          "persons": [{"id": 1, "role": "staff", "mask_rgb": [255, 0, 0]}], "frames": frames}
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    return tmp_path


def _gap_fails(clip):
    qa = common.load_qa()
    qa["render"] = dict(qa["render"], expected_frames_per_clip=False, expected_frames=20, min_in_view_frac=0.0)
    fails, _ = qa_render.run(clip, qa)
    return [f for f in fails if "unlabelled" in f]


def test_dropout_of_visible_person_fails(tmp_path):
    assert _gap_fails(_clip(tmp_path, visible_in_gap=True))


def test_gap_while_out_of_image_passes(tmp_path):
    assert not _gap_fails(_clip(tmp_path, visible_in_gap=False))
