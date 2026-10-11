"""rfdetr_ft: our fine-tuned RF-DETR Nano (SHOWROOM_RFDETR_FT) as a detector spec, plus SHOWROOM_DETECTOR /
SHOWROOM_LIVE_DETECTOR overrides. Blank env values (docker-compose passes "") behave exactly like unset ones."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from showroom_coach import config, models
from showroom_coach.models import ModelUnavailable
from showroom_coach.perception import detect, embed, pose, stack
from showroom_coach.perception import rtm_onnx as R

REAL_FT = Path(__file__).resolve().parents[2] / "models" / "showroom_detector.onnx"
ENV = models.FINETUNED_ENV


# --------------------------------------------------------------------------- env parsing

@pytest.mark.parametrize("environ, expected", [
    ({}, "yolox_nano@2"),
    ({"X": ""}, "yolox_nano@2"),
    ({"X": "   "}, "yolox_nano@2"),
    ({"X": " rfdetr_ft@2 "}, "rfdetr_ft@2"),
])
def test_detector_from_env(environ, expected):
    assert config.detector_from_env("X", "yolox_nano@2", environ) == expected


def _config_in_subprocess(env_overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHOWROOM_")}
    env.update(env_overrides)
    code = "from showroom_coach import config; print(config.DEFAULT_DETECTOR, config.LIVE_DETECTOR)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return tuple(out.stdout.split())


def test_config_defaults_unchanged_when_unset_or_blank():
    assert _config_in_subprocess({}) == ("yolox_nano@2", "yolox_nano@2")
    assert _config_in_subprocess({"SHOWROOM_DETECTOR": "", "SHOWROOM_LIVE_DETECTOR": " ",
                                  ENV: ""}) == ("yolox_nano@2", "yolox_nano@2")


def test_config_live_falls_back_to_showroom_detector():
    assert _config_in_subprocess({"SHOWROOM_DETECTOR": "rfdetr_ft@2"}) == ("rfdetr_ft@2", "rfdetr_ft@2")
    assert _config_in_subprocess({"SHOWROOM_DETECTOR": "rfdetr_ft@2", "SHOWROOM_LIVE_DETECTOR": "rfdetr_ft"}) \
        == ("rfdetr_ft@2", "rfdetr_ft")


@pytest.mark.parametrize("environ", [{}, {ENV: ""}, {ENV: "  "}])
def test_finetuned_path_unset_or_blank_is_unavailable(environ):
    with pytest.raises(ModelUnavailable, match=ENV):
        models.finetuned_detector_path(environ)


def test_finetuned_path_missing_file_is_unavailable(tmp_path):
    with pytest.raises(ModelUnavailable, match="not a file"):
        models.finetuned_detector_path({ENV: str(tmp_path / "nope.onnx")})


def test_rfdetr_ft_is_not_in_the_download_registry():
    assert models.FINETUNED_DETECTOR not in models.MODELS


# --------------------------------------------------------------------------- postprocess with a small head

def test_postprocess_two_class_head_maps_index1_to_person():
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]]], np.float32)
    logits = np.array([[[-9.0, 4.0], [6.0, -9.0]]], np.float32)       # C = 2: background, person
    b, s, c = R.rfdetr_postprocess(boxes, logits, (100, 200), 0.5)
    assert c.tolist() == [0]                                           # person, COCO-80 index 0
    assert b[0] == pytest.approx([80.0, 30.0, 120.0, 70.0], abs=1e-3)
    assert s[0] == pytest.approx(1 / (1 + np.exp(-4.0)), abs=1e-6)


def test_postprocess_indices_beyond_coco_never_alias():
    boxes = np.full((1, 3, 4), 0.5, np.float32)
    logits = np.full((1, 3, 95), -9.0, np.float32)
    logits[0, 0, 91] = logits[0, 1, 94] = 5.0                          # extra classes past the COCO table
    logits[0, 2, 3] = 5.0                                               # COCO id 3 = car (COCO-80 index 2)
    _, _, c = R.rfdetr_postprocess(boxes, logits, (10, 10), 0.5)
    assert c.tolist() == [2]


def test_output_names_by_name_shape_and_order():
    def sess(*outs):
        return SimpleNamespace(get_outputs=lambda: [SimpleNamespace(name=n, shape=s) for n, s in outs])
    assert R.rfdetr_output_names(sess(("labels", [1, 300, 2]), ("dets", [1, 300, 4]))) == ["dets", "labels"]
    assert R.rfdetr_output_names(sess(("a", [1, 300, 2]), ("b", [1, 300, 4]))) == ["b", "a"]
    assert R.rfdetr_output_names(sess(("a", [1, 300, 4]), ("b", [1, 300, 91]))) == ["a", "b"]


# --------------------------------------------------------------------------- Detector spec and the stack

class _Sess:
    def __init__(self, path):
        self.path = path

    def get_inputs(self):
        if "rtmpose" in str(self.path):
            return [SimpleNamespace(name="input", shape=[1, 3, 256, 192])]
        if "dinov2" in str(self.path):
            return [SimpleNamespace(name="pixel_values", shape=["batch_size", 3, "height", "width"])]
        if "yolox" in str(self.path):
            return [SimpleNamespace(name="images", shape=[1, 3, 416, 416])]
        return [SimpleNamespace(name="input", shape=[1, 3, 384, 384])]

    def get_outputs(self):
        if "dinov2" in str(self.path):
            return [SimpleNamespace(name="last_hidden_state", shape=["batch_size", 257, 384])]
        if "yolox" in str(self.path) or "rtmpose" in str(self.path):
            return [SimpleNamespace(name="output", shape=[1, 100, 85])]
        return [SimpleNamespace(name="labels", shape=[1, 300, 2]), SimpleNamespace(name="dets", shape=[1, 300, 4])]


@pytest.fixture
def offline(monkeypatch):
    def ensure(name, *a, **k):
        assert name != models.FINETUNED_DETECTOR, "rfdetr_ft must never be downloaded"
        return Path(f"{name}.onnx")

    for mod in (detect, pose, embed):
        monkeypatch.setattr(mod, "ensure_model", ensure)
    monkeypatch.setattr(R, "make_session", lambda path, threads=None: _Sess(path))
    monkeypatch.setattr(stack, "_STACKS", {})


@pytest.fixture
def weights(tmp_path, monkeypatch):
    p = tmp_path / "ft.onnx"
    p.write_bytes(b"weights v1")
    monkeypatch.setenv(ENV, str(p))
    return p


def test_spec_carries_the_weights_sha_and_tiles(offline, weights):
    sha = models.short_sha256(weights)
    assert len(sha) == 12
    d = detect.Detector("rfdetr_ft")
    assert d.name == "rfdetr_ft" and d.kind == "rfdetr" and d.spec == f"rfdetr_ft-{sha}"
    assert d.input_hw == (384, 384) and d._out_names == ["dets", "labels"]
    d2 = detect.Detector("rfdetr_ft@2")
    assert d2.tiles == 2 and d2.spec == f"rfdetr_ft-{sha}@2"


def test_stack_name_changes_when_weights_change(offline, weights):
    n1 = stack.default_stack("rfdetr_ft@2").name
    assert n1.startswith(f"rfdetr_ft-{models.short_sha256(weights)}@2+")
    weights.write_bytes(b"weights v2, retrained")
    stack._STACKS.clear()
    n2 = stack.default_stack("rfdetr_ft@2").name
    assert n2 != n1 and n2.startswith("rfdetr_ft-") and n2.endswith("+rtmpose_s+dinov2_small")


@pytest.mark.parametrize("spec", ["rfdetr_ft", "rfdetr_ft@2"])
@pytest.mark.parametrize("value", [None, "", "  ", "/no/such/file.onnx"])
def test_stack_falls_back_when_weights_unset_blank_or_missing(offline, monkeypatch, caplog, spec, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    with caplog.at_level(logging.WARNING):
        st = stack.default_stack(spec)
    assert st.name.startswith("yolox_nano@2+")
    assert "rfdetr_ft" in " ".join(r.getMessage() for r in caplog.records)


def test_stack_falls_back_when_weights_do_not_load(offline, weights, monkeypatch):
    def broken(path, threads=None):
        if Path(path) == weights:
            raise RuntimeError("[ONNXRuntimeError] INVALID_PROTOBUF")
        return _Sess(path)
    monkeypatch.setattr(R, "make_session", broken)
    assert stack.default_stack("rfdetr_ft@2").name.startswith("yolox_nano@2+")


# --------------------------------------------------------------------------- the real export (stock COCO stand-in)

@pytest.mark.skipif(not REAL_FT.is_file(), reason="models/showroom_detector.onnx not present")
def test_real_onnx_runs_through_rfdetr_ft(monkeypatch):
    monkeypatch.setenv(ENV, str(REAL_FT))
    d = detect.Detector("rfdetr_ft@2", threads=2)
    assert d.input_hw == (384, 384) and d._out_names == ["dets", "labels"]
    assert d.spec == f"rfdetr_ft-{models.short_sha256(REAL_FT)}@2"
    assert isinstance(d.persons(np.zeros((240, 320, 3), np.uint8)), list)
