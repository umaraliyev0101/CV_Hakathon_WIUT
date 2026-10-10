import numpy as np

from pulse.datatypes import TrackObs
from pulse.perception.embed import Embedder, hsv_hist, torso_crop


def solid(bgr, n=40):
    return np.full((n, n, 3), bgr, np.uint8)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hsv_colour_similarity():
    red1, red2, blue = solid((0, 0, 255)), solid((10, 10, 230), 60), solid((255, 0, 0))
    e = Embedder(use_dino=False)
    r1, r2, b = e.embed([red1]), e.embed([red2]), e.embed([blue])
    assert abs(np.linalg.norm(r1) - 1) < 1e-6
    assert cos(r1, r2) > 0.95
    assert cos(r1, b) < 0.5
    assert hsv_hist(red1).shape == (64,)


def test_torso_crop_uses_confident_keypoints():
    frame = np.zeros((200, 200, 3), np.uint8)
    kps = [[0.0, 0.0, 0.0]] * 17
    kps = [list(k) for k in kps]
    kps[5], kps[6] = [80, 60, 0.9], [120, 60, 0.9]
    kps[11], kps[12] = [85, 110, 0.9], [115, 110, 0.9]
    o = TrackObs(0.0, 0, 1, (50, 20, 150, 190), 0.9, kps)
    c = torso_crop(frame, o, 0.3)
    assert c is not None and c.shape[0] == 50 and 30 <= c.shape[1] <= 45


def test_torso_crop_falls_back_to_box_rows():
    frame = np.zeros((200, 200, 3), np.uint8)
    o = TrackObs(0.0, 0, 1, (50, 0, 150, 100), 0.9, None)
    c = torso_crop(frame, o, 0.3)
    assert c is not None and c.shape[:2] == (40, 100)  # rows 20%..60% of the box
