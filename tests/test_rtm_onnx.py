from pathlib import Path

import numpy as np
import pytest

from showroom_coach.perception import rtm_onnx as R


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = R.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45)
    assert list(keep) == [0, 2]


def test_yolox_preprocess_letterbox():
    img = np.full((480, 640, 3), 200, np.uint8)
    blob, ratio = R.yolox_preprocess(img, (416, 416))
    assert blob.shape == (1, 3, 416, 416) and blob.dtype == np.float32
    assert abs(ratio - 0.65) < 1e-6
    assert blob[0, 0, 415, 0] == 114  # padding row below the resized image


def test_yolox_postprocess_decodes_one_box():
    hw = (64, 64)  # strides 8,16,32 -> 64+16+4 = 84 anchors
    out = np.zeros((1, 84, 85), np.float32)
    # anchor 0 at grid (0,0) stride 8: centre offset 0.5 -> 4 px, w=h=exp(1)*8
    out[0, 0, :4] = [0.5, 0.5, 1.0, 1.0]
    out[0, 0, 4] = 0.9
    out[0, 0, 5] = 0.95  # class 0
    boxes, scores, cls = R.yolox_postprocess(out, hw, ratio=1.0, score_thr=0.3, nms_thr=0.45)
    assert len(boxes) == 1 and cls[0] == 0
    w = np.exp(1.0) * 8
    np.testing.assert_allclose(boxes[0], [4 - w / 2, 4 - w / 2, 4 + w / 2, 4 + w / 2], rtol=1e-4)


def test_yolox_postprocess_does_not_mutate_input_and_handles_empty():
    out = np.zeros((1, 84, 85), np.float32)
    out[0, 0, :4] = [0.5, 0.5, 1.0, 1.0]
    before = out.copy()
    boxes, scores, cls = R.yolox_postprocess(out, (64, 64), 1.0, 0.3, 0.45)
    assert boxes.shape == (0, 4) and scores.shape == (0,) and cls.shape == (0,)
    np.testing.assert_array_equal(out, before)  # the model output buffer is left untouched


def test_yolox_postprocess_divides_by_ratio():
    out = np.zeros((1, 84, 85), np.float32)
    out[0, 0, :4] = [0.5, 0.5, 1.0, 1.0]
    out[0, 0, 4:6] = 0.9
    boxes, _, _ = R.yolox_postprocess(out, (64, 64), ratio=0.5, score_thr=0.3, nms_thr=0.45)
    w = np.exp(1.0) * 8 / 0.5
    np.testing.assert_allclose(boxes[0], [8 - w / 2, 8 - w / 2, 8 + w / 2, 8 + w / 2], rtol=1e-4)


def test_rtmpose_roundtrip_decode():
    # A peak at the centre of the SimCC vectors decodes to the bbox centre.
    bbox = np.array([100, 50, 200, 350], float)
    img = np.zeros((480, 640, 3), np.uint8)
    blob, center, scale = R.rtmpose_preprocess(img, bbox, (192, 256))
    assert blob.shape == (1, 3, 256, 192)
    sx = np.zeros((1, 17, 192 * 2), np.float32); sy = np.zeros((1, 17, 256 * 2), np.float32)
    sx[..., 192] = 1.0; sy[..., 256] = 1.0
    kpts, scores = R.rtmpose_decode(sx, sy, center, scale, (192, 256))
    np.testing.assert_allclose(kpts[0], center, atol=1.0)
    assert kpts.shape == (17, 2) and scores.shape == (17,)


def test_rtmpose_decode_off_centre_and_score_range():
    bbox = np.array([100, 50, 200, 350], float)
    _, center, scale = R.rtmpose_preprocess(np.zeros((480, 640, 3), np.uint8), bbox, (192, 256))
    sx = np.zeros((1, 17, 384), np.float32); sy = np.zeros((1, 17, 512), np.float32)
    sx[0, 0, 96] = 0.5; sy[0, 0, 128] = 0.5  # quarter way across / down the 192x256 input
    sx[0, 1, 10] = 9.0; sy[0, 1, 10] = 9.0   # raw SimCC peaks above 1 must clip to 1
    kpts, scores = R.rtmpose_decode(sx, sy, center, scale, (192, 256))
    exp_x = 48 / 192 * scale[0] + center[0] - scale[0] / 2
    exp_y = 64 / 256 * scale[1] + center[1] - scale[1] / 2
    np.testing.assert_allclose(kpts[0], [exp_x, exp_y], atol=1e-4)
    assert abs(scores[0] - 0.5) < 1e-6 and scores[1] == 1.0
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_rfdetr_preprocess_and_postprocess():
    img = np.full((100, 200, 3), 128, np.uint8)
    blob = R.rfdetr_preprocess(img, (384, 384))
    assert blob.shape == (1, 3, 384, 384) and blob.dtype == np.float32
    boxes = np.zeros((1, 300, 4), np.float32)
    logits = np.full((1, 300, 91), -10.0, np.float32)
    boxes[0, 5] = [0.5, 0.5, 0.2, 0.4]  # cx, cy, w, h (normalised)
    logits[0, 5, 1] = 4.0               # COCO-91 id 1 = person -> cls 0
    logits[0, 6, 0] = 8.0               # background column must never be selected
    boxes[0, 7] = [0.25, 0.25, 0.1, 0.1]
    logits[0, 7, 3] = 3.0               # COCO-91 id 3 = car -> COCO-80 index 2
    b, s, c = R.rfdetr_postprocess(boxes, logits, (100, 200), score_thr=0.3)
    assert sorted(c.tolist()) == [0, 2]
    i = int(np.where(c == 0)[0][0])
    np.testing.assert_allclose(b[i], [80, 30, 120, 70], atol=1e-3)  # scaled to (w=200, h=100)
    assert 0.97 < s[i] < 0.99


def _first_meva_clip() -> Path | None:
    from showroom_coach.config import repo_root

    clips = sorted((repo_root() / "data" / "meva" / "videos").glob("*.avi"))
    return clips[0] if clips else None


def _model_file(name: str) -> Path:
    from showroom_coach import models

    return models.models_dir() / name / models.MODELS[name].filename


@pytest.mark.slow
def test_real_frame_detect_and_pose():
    import cv2

    from showroom_coach.perception.detect import Detector
    from showroom_coach.perception.pose import PoseEstimator

    clip = _first_meva_clip()
    if clip is None:
        pytest.skip("no MEVA clip in data/meva/videos")
    for name in ("yolox_tiny", "rtmpose_s"):
        if not _model_file(name).is_file():
            pytest.skip(f"model {name} not downloaded")
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("cannot decode frame 300")
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (960, round(h * 960 / w)))

    people = Detector("yolox_tiny").persons(frame)
    assert len(people) >= 1
    boxes = np.array([d.bbox for d in people], float)
    assert boxes.shape == (len(people), 4)
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()

    kpts = PoseEstimator()(frame, boxes)
    assert kpts.shape == (len(people), 17, 3)
    assert np.isfinite(kpts).all()
    assert kpts[..., 2].min() >= 0.0 and kpts[..., 2].max() <= 1.0
    assert kpts[..., 2].max() > 0.3  # at least one confident keypoint on a real person

    # Decoding sanity on real data: confident keypoints sit inside their person box (25% + 4 px
    # margin: limbs may poke out a little, a wrong scale/centre would miss by far more) and,
    # wherever the nose and both ankles are confident, the nose is above the ankles.
    checked = 0
    for b, kp in zip(boxes, kpts):
        mx, my = 0.25 * (b[2] - b[0]) + 4, 0.25 * (b[3] - b[1]) + 4
        sure = kp[:, 2] > 0.5
        assert ((kp[sure, 0] >= b[0] - mx) & (kp[sure, 0] <= b[2] + mx)
                & (kp[sure, 1] >= b[1] - my) & (kp[sure, 1] <= b[3] + my)).all()
        if kp[0, 2] > 0.3 and (kp[[15, 16], 2] > 0.3).all():
            assert kp[0, 1] < kp[[15, 16], 1].mean()
            checked += 1
    assert checked >= 1
