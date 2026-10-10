import cv2
import numpy as np

from pulse.camera_motion import motion_score, segments


def textured(seed=0, w=320, h=180):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 128, np.uint8)
    for _ in range(250):
        c = (int(rng.integers(0, 256)),)
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(3, 12)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(rng.integers(4, 20)), p[1] + int(rng.integers(4, 20))), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def shift(img, dx, dy=0):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_shifted_frames_are_moving():
    a = textured()
    d, cut = motion_score(a, shift(a, 10))
    assert not cut
    assert 0.02 < d < 0.05  # ~10/320


def test_identical_frames_are_stable():
    a = textured()
    d, cut = motion_score(a, a.copy())
    assert d < 1e-3 and not cut


def test_random_noise_is_a_cut():
    a = textured()
    noise = np.random.default_rng(1).integers(0, 256, a.shape, dtype=np.uint8)
    _, cut = motion_score(a, noise)
    assert cut


def test_segments_shifted_then_still():
    a = textured()
    # frames 0-2 still, 3 jumps (moving), 4-7 still at the new place
    b = shift(a, 12)
    grays = [a, a, a, b, b, b, b, b]
    times = [i * 0.2 for i in range(8)]
    segs = segments(times, grays, key_idx=4)
    assert [s.stable for s in segs] == [True, False, True]
    assert segs[0].t0 == 0.0 and segs[-1].t1 == times[-1]
    assert segs[0].same_view is False      # first segment sits at a different view than the keyframe
    assert segs[2].same_view is True


def test_moving_segments_are_not_same_view():
    a = textured()
    grays = [a, shift(a, 12), shift(a, 24), shift(a, 24), shift(a, 24)]
    segs = segments([i * 0.2 for i in range(5)], grays, key_idx=3, move_thr=0.01)
    assert all(not s.same_view for s in segs if not s.stable)
