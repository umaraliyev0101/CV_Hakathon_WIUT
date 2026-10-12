"""PULSE_RFDETR_FT: optional fine-tuned person-only RF-DETR for persons; vehicles stay on cfg.detector."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pulse import cache, engine as engine_mod
from pulse.config import get_config
from pulse.perception import finetuned
from pulse.perception.detect import Detector

REAL_FT = Path(__file__).resolve().parents[2] / "models" / "showroom_detector.onnx"


@pytest.fixture
def weights(tmp_path):
    p = tmp_path / "ft.onnx"
    p.write_bytes(b"weights v1")
    return p


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"])
def test_unset_or_blank_env_means_no_finetuned_detector(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(finetuned.ENV, raising=False)
    else:
        monkeypatch.setenv(finetuned.ENV, value)
    assert finetuned.path_from_env() is None
    assert finetuned.tag_from_env() is None


def test_env_path_is_stripped(monkeypatch, weights):
    monkeypatch.setenv(finetuned.ENV, f"  {weights}  ")
    assert finetuned.path_from_env() == weights
    assert finetuned.tag_from_env() == f"rfdetr_ft-{finetuned.short_sha(weights)}"
    assert len(finetuned.short_sha(weights)) == 12


def test_missing_weights_raise_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv(finetuned.ENV, str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match=finetuned.ENV):
        finetuned.tag_from_env()


def test_video_key_unchanged_when_blank_and_tracks_weights(monkeypatch, tmp_path, weights):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"video")
    cfg = get_config("short")
    monkeypatch.delenv(finetuned.ENV, raising=False)
    base = cache.video_key(v, cfg)
    monkeypatch.setenv(finetuned.ENV, "")
    assert cache.video_key(v, cfg) == base               # docker passes "" when not configured
    monkeypatch.setenv(finetuned.ENV, str(weights))
    k1 = cache.video_key(v, cfg)
    assert k1 != base
    weights.write_bytes(b"weights v2, retrained")
    k2 = cache.video_key(v, cfg)
    assert k2 not in (base, k1)


# ---------------------------------------------------------------- RF-DETR decode with a small head

class _Sess:
    def __init__(self, boxes, logits):
        self._out = {"dets": boxes, "labels": logits}

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self._out]

    def run(self, names, feed):
        return [self._out[n] for n in names]


def _small_head_detector(logits):
    boxes = np.array([[[0.25, 0.5, 0.2, 0.4], [0.75, 0.5, 0.2, 0.4], [0.5, 0.5, 0.1, 0.1]]], np.float32)
    d = object.__new__(Detector)
    d.name, d.kind, d.cfg = "rfdetr_ft-test", "rfdetr", get_config("short")
    d.in_hw, d.input_name = (32, 32), "input"
    d._mean = np.array([0.485, 0.456, 0.406], np.float32)
    d._std = np.array([0.229, 0.224, 0.225], np.float32)
    d.session = _Sess(boxes, np.asarray(logits, np.float32)[None])
    return d


def test_two_class_head_index1_is_person_and_never_a_vehicle():
    # C = 2: column 0 background, column 1 person (COCO id 1)
    d = _small_head_detector([[-9.0, 4.0], [-9.0, 3.0], [5.0, -9.0]])
    frame = np.zeros((100, 200, 3), np.uint8)
    persons = d.detect(frame, ("person",), score_thr=0.5)
    assert [p.cls for p in persons] == ["person", "person"]
    assert persons[0].bbox == pytest.approx((30.0, 30.0, 70.0, 70.0), abs=1e-3)
    assert d.detect(frame, ("car", "truck", "motorcycle", "bus"), score_thr=0.01) == []


# ---------------------------------------------------------------- engine wiring

class _FakeDetector:
    made: list = []

    def __init__(self, name, cfg, providers=("CPUExecutionProvider",), models_dir="models", *, path=None):
        self.name, self.path = name, path
        self.calls = []
        _FakeDetector.made.append(self)

    @classmethod
    def finetuned(cls, cfg, path, providers=("CPUExecutionProvider",)):
        return cls(finetuned.tag(Path(path)), cfg, providers, path=path)

    def detect(self, frame, classes=("person",), score_thr=None):
        self.calls.append(tuple(classes))
        return []


@pytest.fixture
def fake_engine_models(monkeypatch):
    _FakeDetector.made = []
    monkeypatch.setattr(engine_mod, "Detector", _FakeDetector)
    monkeypatch.setattr(engine_mod, "PoseEstimator", lambda *a, **k: SimpleNamespace())


@pytest.mark.parametrize("value", [None, "", "  "])
def test_engine_default_uses_one_detector_for_everything(monkeypatch, fake_engine_models, value):
    if value is None:
        monkeypatch.delenv(finetuned.ENV, raising=False)
    else:
        monkeypatch.setenv(finetuned.ENV, value)
    cfg = get_config("short")
    e = engine_mod.PerceptionEngine(cfg)
    assert len(_FakeDetector.made) == 1
    assert e.person_detector is e.detector and e.detector.name == cfg.detector


def test_engine_with_finetuned_routes_persons_and_vehicles(monkeypatch, fake_engine_models, weights):
    monkeypatch.setenv(finetuned.ENV, str(weights))
    cfg = get_config("short")
    e = engine_mod.PerceptionEngine(cfg)
    assert e.detector.name == cfg.detector and e.detector.path is None
    assert e.person_detector.name == finetuned.tag(weights) and e.person_detector.path == weights
    frame = np.zeros((40, 40, 3), np.uint8)
    e.process(0.0, frame, 0)
    e.detect_vehicles(frame)
    assert e.person_detector.calls == [("person",)]
    assert e.vehicle_detector.name == cfg.vehicle_detector
    assert e.vehicle_detector.calls == [("car", "truck", "motorcycle", "bus")]


# ---------------------------------------------------------------- the real export (stock COCO stand-in)

@pytest.mark.skipif(not REAL_FT.is_file(), reason="models/showroom_detector.onnx not present")
def test_real_onnx_loads_through_the_custom_path():
    cfg = get_config("short")
    d = Detector.finetuned(cfg, REAL_FT)
    assert d.kind == "rfdetr" and d.in_hw == (384, 384)
    assert d.name == f"rfdetr_ft-{finetuned.short_sha(REAL_FT)}"
    out = d.detect(np.zeros((240, 320, 3), np.uint8), ("person",))
    assert isinstance(out, list)


# --- our real fine-tuned head: person is column 0 (class_names=["person"]), not COCO id 1 -------------------------

_FT = Path(__file__).resolve().parents[2] / "models" / "showroom_detector_ft.onnx"
_CLIP = Path(__file__).resolve().parents[2] / "deploy" / "demo" / "s004_handoff_cam1_evening_dim.mp4"


@pytest.mark.skipif(not (_FT.is_file() and _CLIP.is_file()), reason="fine-tuned weights or demo clip not present")
def test_real_finetuned_weights_find_people():
    import cv2
    from pulse.config import get_config
    from pulse.perception.detect import Detector
    cap = cv2.VideoCapture(str(_CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    assert ok
    det = Detector.finetuned(get_config(), _FT)
    assert len(det.detect(frame, ("person",))) >= 2            # three people are in this frame
    assert det.detect(frame, ("car",)) == []                   # a person-only head never reports cars
