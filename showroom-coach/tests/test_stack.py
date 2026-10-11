"""Perception stack: the fake test double, crop/embedding helpers, detector fallback (no weights needed),
and a slow end-to-end check on a real MEVA frame."""
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from showroom_coach.config import DEFAULT_DETECTOR, CameraConfig
from showroom_coach.geometry import Camera, facing_vector, position_m
from showroom_coach.models import ModelUnavailable
from showroom_coach.perception import detect, embed, pose, stack
from showroom_coach.perception import rtm_onnx as R
from showroom_coach.perception.embed import Embedder, hsv_hist, person_crop, torso_crop
from showroom_coach.perception.stack import fake_stack
from showroom_coach.perception.tracking import Tracker


# --------------------------------------------------------------------------- fake stack


def test_fake_stack_finds_coloured_people():
    img = np.full((540, 960, 3), 128, np.uint8)
    img[100:320, 100:180] = (0, 0, 255)
    img[150:370, 600:680] = (255, 0, 0)
    st = fake_stack()
    dets = [d for d in st.detect(img) if d.cls == 0]
    assert len(dets) == 2
    k = st.pose(img, np.array([d.bbox for d in dets]))
    assert k.shape == (2, 17, 3)
    e = st.embed([img[100:320, 100:180], img[150:370, 600:680]])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1)
    assert float(e[0] @ e[1]) < 0.9


def test_hsv_hist_and_torso_crop_fallback():
    img = np.zeros((200, 100, 3), np.uint8); img[:, :] = (0, 200, 0)
    h = hsv_hist(img)
    assert h.shape == (128,) and abs(np.linalg.norm(h) - 1) < 1e-6
    crop = torso_crop(img, (0, 0, 100, 200), None)
    assert crop is not None and crop.shape[0] < 200


def _frame_at(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"cannot read frame {index} of {path}"
    return frame


def _persons(frame):
    return sorted((d for d in fake_stack().detect(frame) if d.cls == 0), key=lambda d: d.bbox[0])


def _assert_boxes(dets, expected, tol=3.0):
    assert len(dets) == len(expected), [d.bbox for d in dets]
    for d, box in zip(dets, expected):
        assert d.score == pytest.approx(0.9) and d.cls == 0
        assert all(abs(a - b) <= tol for a, b in zip(d.bbox, box)), (d.bbox, box)


def test_fake_detector_on_synth_video(synth_video):
    # red walker at x = 1100 t / 6, y 60..280; blue stander at (600, 420) from t = 1 s; both 80 x 220
    _assert_boxes(_persons(_frame_at(synth_video, 15)), [(91.7, 60, 171.7, 280)])            # t = 0.5 s: red only
    _assert_boxes(_persons(_frame_at(synth_video, 60)),                                       # t = 2 s: both
                  [(366.7, 60, 446.7, 280), (600, 420, 680, 640)])
    _assert_boxes(_persons(_frame_at(synth_video, 150)),                                      # t = 5 s: both
                  [(600, 420, 680, 640), (916.7, 60, 996.7, 280)])


def test_fake_detector_on_meet_video(meet_video):
    # blue staff at x 600; red customer 4-8 s walking in to x 470, holds to 18 s, back out by 20 s; y 250..470
    _assert_boxes(_persons(_frame_at(meet_video, 60)), [(600, 250, 680, 470)])                # t = 2 s: staff only
    _assert_boxes(_persons(_frame_at(meet_video, 150)),                                       # t = 5 s: customer 117.5
                  [(117.5, 250, 197.5, 470), (600, 250, 680, 470)])
    _assert_boxes(_persons(_frame_at(meet_video, 300)),                                       # t = 10 s: waiting
                  [(470, 250, 550, 470), (600, 250, 680, 470)])
    _assert_boxes(_persons(_frame_at(meet_video, 585)),                                       # t = 19.5 s: leaving, still found
                  [(57.5, 250, 137.5, 470), (600, 250, 680, 470)])


def test_fake_detector_at_processing_size(meet_video):
    frame = cv2.resize(_frame_at(meet_video, 300), (960, 540))            # the app's max processing width
    _assert_boxes(_persons(frame), [(352.5, 187.5, 412.5, 352.5), (450, 187.5, 510, 352.5)], tol=3.0)


def test_fake_detector_ignores_grey_and_specks():
    grey = np.full((540, 960, 3), 128, np.uint8)
    assert fake_stack().detect(grey) == []
    img = grey.copy()
    img[100:320, 100:180] = (0, 0, 255)
    img[100:106, 500:506] = (255, 0, 0)                    # a 6 x 6 speck is below the 400 px area floor
    dets = fake_stack().detect(img)
    assert [d.bbox for d in dets] == [(100.0, 100.0, 180.0, 320.0)]


def test_fake_pose_is_a_standing_person_facing_the_camera():
    st = fake_stack()
    box = (100.0, 100.0, 180.0, 320.0)
    k = st.pose(np.zeros((540, 960, 3), np.uint8), np.array([box]))
    assert k.shape == (1, 17, 3) and np.allclose(k[0, :, 2], 0.9)
    cx, w, h = 140.0, 80.0, 220.0
    # nose 7% from the top, ankles 96%, left side of the body on the image right (person faces the camera)
    assert k[0, 0, 1] == pytest.approx(100 + 0.07 * h) and k[0, 15, 1] == pytest.approx(100 + 0.96 * h)
    assert k[0, 5, 0] == pytest.approx(cx + 0.4 * w) and k[0, 6, 0] == pytest.approx(cx - 0.4 * w)
    assert k[0, 11, 1] == pytest.approx(100 + 0.48 * h) and k[0, 13, 1] == pytest.approx(100 + 0.715 * h)
    cam = Camera.from_config(960, 540, CameraConfig())
    pos = position_m(box, k[0], cam)
    assert pos.source == "full"
    assert pos.z == pytest.approx(cam.f * 1.70 / h, rel=0.02)
    fx, fz = facing_vector(box, k[0], cam)
    assert fz < -0.9 and abs(fx) < 0.3


def test_fake_pose_skips_tiny_and_empty_boxes():
    st = fake_stack()
    k = st.pose(np.zeros((100, 100, 3), np.uint8), np.array([[10.0, 10.0, 20.0, 30.0], [0, 0, 50.0, 80.0]]))
    assert k.shape == (2, 17, 3)
    assert (k[0] == 0).all() and (k[1, :, 2] == 0.9).all()          # 20 px tall: below the 40 px gate
    assert st.pose(np.zeros((10, 10, 3), np.uint8), np.zeros((0, 4))).shape == (0, 17, 3)


def test_fake_embed_edge_cases():
    st = fake_stack()
    assert st.embed([]).shape[0] == 0
    out = st.embed([np.zeros((0, 5, 3), np.uint8), np.full((20, 10, 3), 200, np.uint8)])
    assert (out[0] == 0).all() and abs(np.linalg.norm(out[1]) - 1) < 1e-6      # unusable crop -> zero row
    assert st.name == "fake" and st.new_tracker(5.0).__class__ is Tracker


# --------------------------------------------------------------------------- crops and histograms


def test_hsv_hist_variants():
    assert hsv_hist(np.zeros((0, 4, 3), np.uint8)).shape == (128,)
    assert not hsv_hist(np.zeros((0, 4, 3), np.uint8)).any()                       # nothing to histogram -> zeros
    img = np.full((10, 10, 3), (0, 0, 255), np.uint8)
    h = hsv_hist(img, bins=(4, 2, 2))
    assert h.shape == (16,) and abs(np.linalg.norm(h) - 1) < 1e-6
    red, blue = hsv_hist(img), hsv_hist(np.full((10, 10, 3), (255, 0, 0), np.uint8))
    assert float(red @ blue) == pytest.approx(0.0, abs=1e-6)
    assert float(red @ hsv_hist(img.copy())) == pytest.approx(1.0)


def test_person_crop_pads_clips_and_copies():
    frame = np.zeros((200, 300, 3), np.uint8)
    crop = person_crop(frame, (100, 50, 160, 150))          # 60 x 100 box, 5% pad = 3 px x 5 px
    assert crop.shape == (110, 66, 3)
    crop[:] = 255
    assert frame.max() == 0                                  # a copy, not a view
    c2 = person_crop(frame, (-20, -10, 40, 80), pad=0.0)
    assert c2.shape == (80, 40, 3)
    assert person_crop(frame, (50, 50, 50, 90)) is None      # zero width
    assert person_crop(frame, (400, 10, 450, 90)) is None    # outside the frame
    assert person_crop(frame, (10, 10, float("nan"), 90)) is None


def test_torso_crop_uses_shoulders_and_hips_when_confident():
    frame = np.zeros((200, 100, 3), np.uint8)
    k = np.zeros((17, 3))
    k[5] = (30, 60, 0.9); k[6] = (70, 60, 0.9); k[11] = (35, 120, 0.9); k[12] = (65, 120, 0.9)
    crop = torso_crop(frame, (0, 0, 100, 200), k)            # x 30..70, y 60..120, +15% = 6 px / 9 px
    assert crop.shape == (78, 52, 3)
    k[11, 2] = 0.1                                            # a hip is unsure -> rows 20%..55% of the box
    crop = torso_crop(frame, (0, 0, 100, 200), k)
    assert crop.shape == (70, 100, 3)
    k[11, 2] = 0.9; k[11, 1] = k[12, 1] = 30                  # hips above the shoulders is nonsense -> box rows
    assert torso_crop(frame, (0, 0, 100, 200), k).shape == (70, 100, 3)
    assert torso_crop(frame, (0, 0, 100, 200), np.zeros((17, 3))).shape == (70, 100, 3)
    assert torso_crop(frame, (30, 30, 30, 90), None) is None


# --------------------------------------------------------------------------- Embedder with a fake DINOv2 session


class FakeDino:
    """Stands in for the DINOv2 onnxruntime session: the CLS token carries each crop's per-channel mean."""

    def __init__(self, outputs=("last_hidden_state",), batch_dim="batch_size"):
        self.outputs, self.batch_dim, self.calls = outputs, batch_dim, []

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values", shape=[self.batch_dim, 3, "height", "width"])]

    def get_outputs(self):
        shapes = {"last_hidden_state": ["batch_size", 257, 384], "pooler_output": ["batch_size", 384]}
        return [SimpleNamespace(name=n, shape=shapes[n]) for n in self.outputs]

    def run(self, names, feed):
        x = feed["pixel_values"]
        self.calls.append(x)
        b = len(x)
        hidden = np.zeros((b, 257, 384), np.float32)
        hidden[:, 0, :3] = x.mean(axis=(2, 3))
        hidden[:, 1:, 0] = 100.0                                       # patch tokens must not leak in
        pooled = np.zeros((b, 384), np.float32)
        pooled[:, 10] = 5.0
        return [{"last_hidden_state": hidden, "pooler_output": pooled}[n] for n in names]


def _embedder(monkeypatch, fake, batch=8):
    monkeypatch.setattr(embed, "ensure_model", lambda name, *a, **k: Path(name + ".onnx"))
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: fake)
    return Embedder(batch=batch)


def _colour_crops(n):
    return [np.full((40 + k, 20 + k, 3), (k * 12 % 256, k * 7 % 256, 255 - k * 9 % 256), np.uint8) for k in range(n)]


def test_embedder_preprocess_and_cls_token(monkeypatch):
    fake = FakeDino()
    emb = _embedder(monkeypatch, fake)
    out = emb([np.full((60, 30, 3), (0, 0, 255), np.uint8)])             # a pure red crop (BGR)
    blob = fake.calls[0]
    assert blob.shape == (1, 3, 224, 224) and blob.dtype == np.float32     # RGB, 224 x 224
    assert blob[0, 0].mean() == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)   # R plane: ImageNet mean/std
    assert blob[0, 1].mean() == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-4)
    assert blob[0, 2].mean() == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)
    assert out.shape == (1, 384) and out.dtype == np.float32
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-5)
    assert out[0, 3:].max() == 0.0                                         # CLS token only, patches ignored


def test_embedder_prefers_pooler_output(monkeypatch):
    fake = FakeDino(outputs=("last_hidden_state", "pooler_output"))
    out = _embedder(monkeypatch, fake)(_colour_crops(3))
    assert out.shape == (3, 384) and np.allclose(out[:, 10], 1.0) and np.allclose(np.linalg.norm(out, axis=1), 1)


def test_embedder_batches_and_keeps_order(monkeypatch):
    fake = FakeDino()
    emb = _embedder(monkeypatch, fake, batch=8)
    crops = _colour_crops(20)
    out = emb(crops)
    assert [len(c) for c in fake.calls] == [8, 8, 4]
    assert out.shape == (20, 384)
    for k in (0, 7, 8, 19):
        assert np.allclose(out[k], emb([crops[k]])[0], atol=1e-6)
    assert not np.allclose(out[0], out[19], atol=1e-3)


def test_embedder_fixed_batch_model_runs_one_crop_at_a_time(monkeypatch):
    fake = FakeDino(batch_dim=1)
    out = _embedder(monkeypatch, fake, batch=8)(_colour_crops(5))
    assert [len(c) for c in fake.calls] == [1] * 5 and out.shape == (5, 384)


def test_embedder_empty_and_unusable_crops(monkeypatch):
    fake = FakeDino()
    emb = _embedder(monkeypatch, fake)
    assert emb([]).shape == (0, 384) and fake.calls == []
    out = emb([np.zeros((0, 5, 3), np.uint8), _colour_crops(1)[0], None])
    assert out.shape == (3, 384) and not out[0].any() and not out[2].any()
    assert np.linalg.norm(out[1]) == pytest.approx(1.0, abs=1e-5) and [len(c) for c in fake.calls] == [1]


# --------------------------------------------------------------------------- default_stack and fallbacks


class FakeSession:
    """Enough of an onnxruntime session for the Detector / PoseEstimator / Embedder constructors."""

    def __init__(self, name):
        self.name = name

    def get_inputs(self):
        if "rtmpose" in self.name:
            return [SimpleNamespace(name="input", shape=[1, 3, 256, 192])]
        if "dinov2" in self.name:
            return [SimpleNamespace(name="pixel_values", shape=["batch_size", 3, "height", "width"])]
        return [SimpleNamespace(name="images", shape=[1, 3, 416, 416])]

    def get_outputs(self):
        if "dinov2" in self.name:
            return [SimpleNamespace(name="last_hidden_state", shape=["batch_size", 257, 384])]
        return [SimpleNamespace(name="output", shape=[1, 100, 85])]


@pytest.fixture
def offline(monkeypatch):
    """default_stack with no weights: ensure_model / make_session replaced, cache emptied.

    Call `offline(missing={"yolox_tiny"})` to make some models raise ModelUnavailable; the returned list records
    every model name that was asked for."""
    asked: list[str] = []

    def setup(missing=()):
        def ensure(name, *a, **k):
            asked.append(name)
            if name in missing:
                raise ModelUnavailable(f"{name}: download failed after 3 attempt(s): offline")
            return Path(f"{name}.onnx")

        for mod in (detect, pose, embed):                      # each module bound its own `ensure_model`
            monkeypatch.setattr(mod, "ensure_model", ensure)
        monkeypatch.setattr(R, "make_session", lambda path, threads=None: FakeSession(Path(path).stem))
        monkeypatch.setattr(stack, "_STACKS", {})
        return asked

    return setup


def test_default_stack_falls_back_when_detector_missing(offline, caplog):
    offline(missing={"yolox_tiny"})
    with caplog.at_level(logging.WARNING):
        st = stack.default_stack("yolox_tiny")
    assert "yolox_nano" in st.name and "yolox_tiny" not in st.name
    assert st.name == "yolox_nano+rtmpose_s+dinov2_small"
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "yolox_tiny" in warned and "yolox_nano" in warned


def test_default_stack_name_is_descriptive_and_stable(offline):
    offline()
    st = stack.default_stack()
    assert st.name == f"{DEFAULT_DETECTOR}+rtmpose_s+dinov2_small" == "yolox_nano@2+rtmpose_s+dinov2_small"
    assert callable(st.detect) and callable(st.pose) and callable(st.embed)
    assert st.embed.batch == 1              # one crop per run keeps embeddings independent of batch composition
    assert isinstance(st.new_tracker(5.0), Tracker)


def test_default_stack_follows_the_fallback_chain_with_tiles(offline):
    offline(missing={"yolox_nano"})                  # yolox_nano@2 -> yolox_tiny@2
    assert stack.default_stack("yolox_nano@2").name.startswith("yolox_tiny@2+")


def test_default_stack_falls_back_from_a_spec_missing_in_the_table(offline):
    offline(missing={"rfdetr_nano"})                 # "rfdetr_nano@2" is not a key: its base model's entry applies
    assert stack.default_stack("rfdetr_nano@2").name.startswith("yolox_nano@2+")


def test_default_stack_gives_up_when_the_chain_is_exhausted(offline):
    asked = offline(missing={"yolox_nano", "yolox_tiny"})
    with pytest.raises(ModelUnavailable) as err:
        stack.default_stack("yolox_nano@2")           # nano@2 -> tiny@2 -> nano -> tiny -> (nano seen) stop
    assert "yolox_nano@2" in str(err.value) and "yolox_tiny" in str(err.value)
    assert asked.count("yolox_nano") <= 2 and stack._STACKS == {}


def test_default_stack_runs_without_pose_and_embedder(offline, caplog):
    offline(missing={"rtmpose_s", "dinov2_small"})
    with caplog.at_level(logging.WARNING):
        st = stack.default_stack("yolox_nano@2")
    assert st.pose is None and st.embed is None and st.name == "yolox_nano@2"
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "rtmpose_s" in warned and "dinov2_small" in warned


def test_default_stack_is_cached_per_canonical_spec(offline):
    asked = offline(missing={"yolox_tiny"})
    a = stack.default_stack("yolox_nano")
    n = len(asked)
    assert stack.default_stack("yolox_nano") is a and stack.default_stack("yolox_nano@1") is a
    assert len(asked) == n                                       # no model was touched again
    assert stack.default_stack("yolox_nano@2") is not a
    fallen = stack.default_stack("yolox_tiny")                   # the requested spec is the key, even after a fallback
    n = len(asked)
    assert stack.default_stack("yolox_tiny") is fallen and len(asked) == n


def test_default_stack_rejects_a_bad_spec(offline):
    offline()
    with pytest.raises(ValueError):
        stack.default_stack("yolox_nano@3")


@pytest.mark.parametrize("spec", sorted(stack.DETECTOR_FALLBACKS))
def test_fallback_table_is_made_of_valid_specs(spec):
    detect.parse_detector_spec(spec)
    detect.parse_detector_spec(stack.DETECTOR_FALLBACKS[spec])


# --------------------------------------------------------------------------- real models (slow)


def _real_frame(*models: str) -> np.ndarray:
    """Frame 300 of the first MEVA clip, resized to width 960; skips when data or weights are missing."""
    from showroom_coach import models as M
    from showroom_coach.config import repo_root

    clips = sorted((repo_root() / "data" / "meva" / "videos").glob("*.avi"))
    if not clips:
        pytest.skip("no MEVA clip in data/meva/videos")
    for name in models:
        if not (M.models_dir() / name / M.MODELS[name].filename).is_file():
            pytest.skip(f"model {name} not downloaded")
    cap = cv2.VideoCapture(str(clips[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("cannot decode frame 300")
    h, w = frame.shape[:2]
    return cv2.resize(frame, (960, round(h * 960 / w)))


@pytest.mark.slow
def test_default_stack_on_a_real_frame():
    spec = DEFAULT_DETECTOR
    frame = _real_frame(spec.partition("@")[0], "rtmpose_s", "dinov2_small")
    st = stack.default_stack()
    assert st.name == f"{spec}+rtmpose_s+dinov2_small"
    dets = [d for d in st.detect(frame) if d.cls == 0]
    assert len(dets) >= 1
    boxes = np.array([d.bbox for d in dets], float)
    kpts = st.pose(frame, boxes)
    assert kpts.shape == (len(dets), 17, 3)
    crops = [person_crop(frame, d.bbox) for d in dets]
    crops = [c for c in crops if c is not None]
    torsos = [torso_crop(frame, d.bbox, k) for d, k in zip(dets, kpts)]
    assert crops and all(t is not None for t in torsos)
    emb = st.embed(crops + [t for t in torsos if t is not None])
    assert emb.shape == (len(crops) + len(torsos), 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    assert all(hsv_hist(t).shape == (128,) for t in torsos)
    tracked = st.new_tracker(5.0).update(dets)
    assert all(type(i) is int and i > 0 for i, _ in tracked) and all(any(d is x for x in dets) for _, d in tracked)
