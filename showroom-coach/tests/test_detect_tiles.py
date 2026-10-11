"""Tiled detection and the pose size gate, tested with fake onnxruntime sessions (no weights)."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from showroom_coach.perception import detect, pose
from showroom_coach.perception import rtm_onnx as R


class FakeYolox:
    """Stands in for a YOLOX onnxruntime session: 'detects' every white rectangle in the
    letterboxed input and encodes it the way a real Megvii YOLOX export would (stride-8 anchor,
    raw dx, dy, log w, log h, sigmoid scores), so preprocess, decode and tile shifting are real."""

    def __init__(self, hw=(320, 320)):
        self.hw = hw
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=[1, 3, *self.hw])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, names, feed):
        self.calls += 1
        h, w = self.hw
        total = sum((h // s) * (w // s) for s in (8, 16, 32))
        out = np.zeros((1, total, 85), np.float32)
        mask = (feed["images"][0, 0] > 200).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, bw, bh = (float(v) for v in stats[i][:4])
            cx, cy = x + bw / 2, y + bh / 2
            gx, gy = int(cx // 8), int(cy // 8)
            a = gy * (w // 8) + gx
            out[0, a, :4] = [cx / 8 - gx, cy / 8 - gy, np.log(bw / 8), np.log(bh / 8)]
            out[0, a, 4] = 0.9
            out[0, a, 5] = 0.9  # class 0 = person
        return [out]


@pytest.fixture
def fake_yolox(monkeypatch):
    sess = FakeYolox()
    monkeypatch.setattr(detect, "ensure_model", lambda name: Path(f"{name}.onnx"))
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: sess)
    return sess


def _scene(w, h, rects):
    img = np.zeros((h, w, 3), np.uint8)
    for x1, y1, x2, y2 in rects:
        img[y1:y2, x1:x2] = 255
    return img


def _sorted_boxes(dets):
    return sorted((d.bbox for d in dets), key=lambda b: (b[0], b[1]))


# ---- parse_detector_spec ---------------------------------------------------------------


def test_parse_detector_spec():
    assert detect.parse_detector_spec("yolox_tiny@2") == ("yolox_tiny", 2)
    assert detect.parse_detector_spec("yolox_s") == ("yolox_s", 1)
    assert detect.parse_detector_spec("rfdetr_nano@1") == ("rfdetr_nano", 1)


@pytest.mark.parametrize("bad", ["", "@2", "yolox_s@", "yolox_s@x", "yolox_s@0", "yolox_s@-1",
                                 "yolox_s@3", "yolox_s@2@2", "yolox_s@1.5"])
def test_parse_detector_spec_rejects_garbage(bad):
    with pytest.raises(ValueError):
        detect.parse_detector_spec(bad)


# ---- tile windows -----------------------------------------------------------------------


def test_tile_windows_landscape_portrait_and_square():
    assert R.tile_windows(960, 540, 1) == [(0, 0, 960, 540)]
    assert R.tile_windows(960, 540, 2) == [(0, 0, 540, 540), (420, 0, 960, 540)]
    assert R.tile_windows(540, 960, 2) == [(0, 0, 540, 540), (0, 420, 540, 960)]  # top / bottom
    assert R.tile_windows(500, 500, 2) == [(0, 0, 500, 500)]  # square: one crop is the whole frame


@pytest.mark.parametrize("w,h", [(960, 540), (640, 480), (1920, 1080), (1280, 400), (960, 720)])
def test_tile_windows_overlap_and_coverage(w, h):
    (a, b) = R.tile_windows(w, h, 2)
    assert a[0] == 0 and b[2] == w and a[1] == b[1] == 0 and a[3] == b[3] == h  # full height
    assert a[2] - b[0] >= 0.1 * w - 1  # crops overlap by about 10% of W or more
    assert a[2] - a[0] == b[2] - b[0]  # equal widths


def test_tile_windows_rejects_unsupported_counts():
    with pytest.raises(ValueError):
        R.tile_windows(960, 540, 3)


# ---- merging tile detections -------------------------------------------------------------

WIN = R.tile_windows(960, 540, 2)  # [(0,0,540,540), (420,0,960,540)]: cut edges at x=540 (tile 0) and x=420 (tile 1)


def _merge(rows, windows=WIN, nms_thr=0.45):
    """rows: (x1, y1, x2, y2, score, tile[, cls]) -> sorted list of kept boxes (rounded tuples)."""
    a = np.array(rows, float)
    cls = a[:, 6].astype(int) if a.shape[1] > 6 else np.zeros(len(a), int)
    b, _, _ = R.merge_tiles(a[:, :4], a[:, 4], cls, a[:, 5].astype(int), windows, nms_thr)
    return sorted(tuple(round(v) for v in row) for row in b.tolist())


def test_merge_tiles_drops_truncated_copy_and_duplicates():
    boxes = np.array([
        [500, 100, 540, 300],  # tile 0: person cut by the seam at x=540 (area 8000)
        [500, 100, 600, 300],  # tile 1: the whole person (area 20000)
        [440, 300, 500, 420],  # tile 0: object fully in the overlap ...
        [440, 301, 500, 421],  # ... seen again by tile 1 (near-duplicate)
        [50, 50, 110, 200],    # tile 0 only
    ], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cls = np.zeros(5, int)
    tiles = np.array([0, 1, 0, 1, 0])
    b, s, c = R.merge_tiles(boxes, scores, cls, tiles, WIN, nms_thr=0.45)
    kept = sorted(map(tuple, b.tolist()))
    assert kept == [(50, 50, 110, 200), (440, 300, 500, 420), (500, 100, 600, 300)]
    assert len(s) == len(c) == 3


def test_merge_tiles_keeps_nested_people_seen_by_both_tiles():
    # Reviewer's case: a person standing in front of a bigger one, both inside the overlap strip and
    # seen by both tiles. Neither touches a cut edge, so neither copy of the small person may be
    # dropped as a "seam-truncated" box; only the exact duplicates collapse.
    big, small = (440, 100, 520, 300), (460, 150, 490, 250)
    assert _merge([(*big, 0.9, 0), (*small, 0.8, 0), (*big, 0.85, 1), (*small, 0.75, 1)]) == sorted([big, small])


def test_merge_tiles_nested_box_in_one_tile_only_survives():
    big, small = (440, 100, 520, 300), (460, 150, 490, 250)
    assert _merge([(*big, 0.9, 1), (*small, 0.8, 0)]) == sorted([big, small])


@pytest.mark.parametrize("gap,dropped", [(0, True), (2, True), (4, False)])
def test_merge_tiles_left_tile_box_must_touch_its_cut_within_2px(gap, dropped):
    whole = (500, 100, 600, 300)                      # tile 1 sees the whole person
    trunc = (500, 100, 540 - gap, 300)                # tile 0: cut at x=540
    got = _merge([(*trunc, 0.9, 0), (*whole, 0.8, 1)])
    assert got == ([whole] if dropped else sorted([trunc, whole]))


def test_merge_tiles_right_tile_box_touching_left_cut():
    whole = (380, 100, 460, 300)                      # tile 0 sees the whole person
    trunc = (420, 100, 460, 300)                      # tile 1: cut at x=420
    assert _merge([(*whole, 0.7, 0), (*trunc, 0.95, 1)]) == [whole]
    # the cut that counts is the box's OWN tile's: a tile-1 box ending at tile 0's cut line (x=540) is
    # not truncated by tile 1, so a nested tile-1 box there stays
    nested = (500, 100, 540, 200)
    assert _merge([(*nested, 0.9, 1), (400, 50, 545, 400, 0.8, 0)]) == sorted([nested, (400, 50, 545, 400)])


def test_merge_tiles_truncated_box_with_higher_score_does_not_beat_the_whole_box():
    # visible part is 54% of the whole box (above the old 50% area limit): NMS alone would keep the
    # higher-scoring truncated copy, so the border rule has to remove it
    whole = (470, 100, 600, 300)
    trunc = (470, 100, 540, 300)
    assert _merge([(*trunc, 0.95, 0), (*whole, 0.6, 1)]) == [whole]


def test_merge_tiles_truncated_box_without_a_counterpart_survives():
    trunc = (500, 100, 540, 300)  # touches the cut, nothing from the other tile covers it
    assert _merge([(*trunc, 0.9, 0), (50, 50, 110, 200, 0.5, 0)]) == sorted([trunc, (50, 50, 110, 200)])


def test_merge_tiles_portrait_windows_use_the_horizontal_cut_lines():
    win = R.tile_windows(540, 960, 2)  # [(0,0,540,540), (0,420,540,960)]
    whole = (100, 500, 300, 600)
    trunc = (100, 500, 300, 540)       # tile 0 (top): cut at y=540
    assert _merge([(*trunc, 0.9, 0), (*whole, 0.8, 1)], win) == [whole]
    whole2 = (100, 380, 300, 460)
    trunc2 = (100, 420, 300, 460)      # tile 1 (bottom): cut at y=420
    assert _merge([(*whole2, 0.8, 0), (*trunc2, 0.9, 1)], win) == [whole2]
    # x edges are irrelevant for portrait tiles: a small box ending at x=540 is not at a cut
    small, big = (500, 100, 540, 200), (400, 50, 545, 400)
    assert _merge([(*small, 0.9, 0), (*big, 0.8, 1)], win) == sorted([small, big])


def test_merge_tiles_keeps_different_classes_and_same_tile_boxes():
    # all of these touch the cut at x=540 and are covered by a bigger box, but the bigger box is of
    # another class (person vs car) or from the SAME tile, so none of them is a seam artefact
    rows = [(500, 0, 540, 100, 0.9, 0, 0), (500, 0, 600, 100, 0.8, 1, 2),
            (520, 200, 540, 300, 0.7, 0, 0), (440, 200, 540, 300, 0.6, 0, 0)]
    assert len(_merge(rows, nms_thr=0.99)) == 4


def test_merge_tiles_empty():
    b, s, c = R.merge_tiles(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int), WIN, 0.45)
    assert b.shape == (0, 4) and s.shape == (0,) and c.shape == (0,)


# ---- Detector with a fake session --------------------------------------------------------


def test_single_tile_unchanged(fake_yolox):
    det = detect.Detector("yolox_tiny", score_thr=0.35)
    img = _scene(960, 540, [(500, 100, 600, 300)])
    dets = det(img)
    assert fake_yolox.calls == 1 and det.tiles == 1
    assert len(dets) == 1 and dets[0].cls == 0 and 0.8 < dets[0].score < 0.82
    np.testing.assert_allclose(dets[0].bbox, (500, 100, 600, 300), atol=6)


def test_two_tiles_seam_box_once_and_shifted(fake_yolox):
    det = detect.Detector("yolox_tiny", score_thr=0.35, tiles=2)
    rects = [
        (500, 100, 600, 300),  # straddles the left crop's right edge (x=540); whole in the right crop
        (50, 50, 110, 200),    # left crop only
        (800, 200, 900, 450),  # right crop only: needs the +420 shift
        (440, 360, 500, 480),  # in the overlap: both crops see it (clear of the others)
    ]
    dets = det(_scene(960, 540, rects))
    assert fake_yolox.calls == 2
    boxes = _sorted_boxes(dets)
    expect = sorted(rects)
    assert len(boxes) == 4, boxes
    for got, want in zip(boxes, expect):
        np.testing.assert_allclose(got, want, atol=6)


def test_two_tiles_portrait_frame(fake_yolox):
    det = detect.Detector("yolox_tiny", tiles=2)
    rects = [(100, 500, 300, 600), (100, 700, 300, 900)]  # first straddles the top crop's bottom edge (540)
    dets = det(_scene(540, 960, rects))
    assert fake_yolox.calls == 2
    boxes = _sorted_boxes(dets)
    assert len(boxes) == 2, boxes
    for got, want in zip(boxes, sorted(rects)):
        np.testing.assert_allclose(got, want, atol=6)


def test_spec_string_and_tiles_argument(fake_yolox):
    assert detect.Detector("yolox_tiny@2").tiles == 2
    assert detect.Detector("yolox_tiny", tiles=2).tiles == 2
    assert detect.Detector("yolox_tiny@2", tiles=2).tiles == 2
    d = detect.Detector("yolox_s@2")
    assert d.name == "yolox_s" and d.spec == "yolox_s@2"
    assert detect.Detector("yolox_s").spec == "yolox_s"
    with pytest.raises(ValueError):
        detect.Detector("yolox_tiny@2", tiles=3)
    with pytest.raises(ValueError):
        detect.Detector("yolox_tiny", tiles=3)


def test_detector_default_follows_config(fake_yolox):
    from showroom_coach import config

    assert detect.Detector().spec == config.DEFAULT_DETECTOR


# ---- PoseEstimator size gate ---------------------------------------------------------------


class FakePose:
    """Batch-aware stand-in for the RTMPose session: `calls` counts session.run calls, `batches` records the batch
    size of each, so `sum(batches)` is how many boxes actually reached the model."""

    def __init__(self):
        self.calls = 0
        self.batches = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=["batch", 3, 256, 192])]

    def get_outputs(self):
        return [SimpleNamespace(name="simcc_x"), SimpleNamespace(name="simcc_y")]

    def peaks(self, blob):
        """Per input row: (x peak index, y peak index), by default the centre of the crop."""
        return [(192, 256)] * len(blob)

    def run(self, names, feed):
        blob = feed["input"]
        assert blob.ndim == 4 and blob.shape[1:] == (3, 256, 192)
        self.calls += 1
        self.batches.append(len(blob))
        sx = np.zeros((len(blob), 17, 384), np.float32)
        sy = np.zeros((len(blob), 17, 512), np.float32)
        for b, (px, py) in enumerate(self.peaks(blob)):
            sx[b, :, px] = 1.0
            sy[b, :, py] = 1.0
        return [sx, sy]


@pytest.fixture
def fake_pose(monkeypatch):
    sess = FakePose()
    monkeypatch.setattr(pose, "ensure_model", lambda name: Path(f"{name}.onnx"))
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: sess)
    return sess


def test_pose_skips_boxes_shorter_than_min_height(fake_pose):
    pe = pose.PoseEstimator()
    img = np.zeros((480, 640, 3), np.uint8)
    boxes = np.array([[10, 10, 40, 40], [100, 50, 200, 350]], float)  # heights 30 and 300
    out = pe(img, boxes)
    assert out.shape == (2, 17, 3)
    assert fake_pose.calls == 1  # the 30 px person never reached the model
    assert (out[0] == 0).all()
    np.testing.assert_allclose(out[1, :, 2], 1.0)
    np.testing.assert_allclose(out[1, 0, :2], (150, 200), atol=1.0)


def test_pose_min_height_is_a_parameter_with_default_40(fake_pose):
    assert inspect.signature(pose.PoseEstimator.__call__).parameters["min_height"].default == 40.0
    pe = pose.PoseEstimator()
    img = np.zeros((480, 640, 3), np.uint8)
    boxes = np.array([[10, 10, 40, 49], [10, 10, 40, 50]], float)  # heights 39 (skipped) and 40 (kept)
    out = pe(img, boxes)
    assert fake_pose.calls == 1 and out[0, :, 2].max() == 0 and out[1, :, 2].min() == 1.0
    fake_pose.calls, fake_pose.batches = 0, []
    out = pe(img, boxes, min_height=0)
    assert fake_pose.batches == [2]                      # both boxes reached the model, in one batch
    out = pe(img, np.array([[10, 10, 40, 400]], float), min_height=500)
    assert (out == 0).all() and fake_pose.batches == [2]


def test_pose_no_boxes(fake_pose):
    out = pose.PoseEstimator()(np.zeros((48, 64, 3), np.uint8), np.zeros((0, 4)))
    assert out.shape == (0, 17, 3) and fake_pose.calls == 0


class ContentPose(FakePose):
    """Peaks depend on each crop's pixels, so a crop run in the wrong batch row would give a different keypoint."""

    def peaks(self, blob):
        return [(int(abs(float(b.sum())) * 7) % 380, int(abs(float(b.mean())) * 900) % 500) for b in blob]


def _gradient_frame():
    ys, xs = np.indices((480, 640))
    return np.dstack([(xs * 0.4 + ys * 0.2) % 256, (xs * 0.1 + ys * 0.5) % 256, (xs + ys) % 256]).astype(np.uint8)


def test_pose_runs_every_eligible_box_in_one_session_call(fake_pose):
    pe = pose.PoseEstimator()
    boxes = np.array([[100, 50, 200, 350], [300, 100, 380, 300], [10, 10, 40, 40], [400, 200, 500, 470],
                      [50, 300, 120, 460]], float)       # the third is 30 px tall: gated out
    out = pe(_gradient_frame(), boxes)
    assert fake_pose.calls == 1 and fake_pose.batches == [4]
    assert out.shape == (5, 17, 3) and (out[2] == 0).all()
    for i in (0, 1, 3, 4):
        np.testing.assert_allclose(out[i, :, 2], 1.0)
        np.testing.assert_allclose(out[i, 0, :2], [(boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2],
                                   atol=1.0)


def test_batched_pose_equals_the_per_box_results(monkeypatch):
    sess = ContentPose()
    monkeypatch.setattr(pose, "ensure_model", lambda name: Path(f"{name}.onnx"))
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: sess)
    pe = pose.PoseEstimator()
    frame = _gradient_frame()
    boxes = np.array([[100, 50, 200, 350], [300, 100, 380, 300], [10, 10, 40, 40], [400, 200, 500, 470],
                      [50, 300, 120, 460]], float)
    out = pe(frame, boxes)
    assert sess.batches == [4]
    ref = ContentPose()
    expected = np.zeros((5, 17, 3), np.float32)
    for i, box in enumerate(boxes):                      # what the model did before batching: one box per run
        if box[3] - box[1] < 40:
            continue
        blob, center, scale = R.rtmpose_preprocess(frame, box, (192, 256))
        sx, sy = ref.run(None, {"input": blob})
        k, c = R.rtmpose_decode(sx, sy, center, scale, (192, 256))
        expected[i, :, :2], expected[i, :, 2] = k, c
    assert ref.batches == [1, 1, 1, 1]
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert len({tuple(np.round(out[i, 0, :2], 1)) for i in (0, 1, 3, 4)}) == 4      # the rows really differ


def test_pose_with_a_fixed_batch_model_runs_one_box_at_a_time(monkeypatch):
    class Fixed(FakePose):
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 256, 192])]

    sess = Fixed()
    monkeypatch.setattr(pose, "ensure_model", lambda name: Path(f"{name}.onnx"))
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: sess)
    boxes = np.array([[100, 50, 200, 350], [300, 100, 380, 300], [400, 200, 500, 470]], float)
    out = pose.PoseEstimator()(_gradient_frame(), boxes)
    assert sess.batches == [1, 1, 1] and (out[..., 2] == 1.0).all()
