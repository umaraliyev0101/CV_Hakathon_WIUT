import json

import pytest

from pulse.perception import models
from pulse.perception.models import REGISTRY, ensure_model

BANNED = ("ultralytics", "boxmot", "market", "duke", "msmt")


def test_registry_metadata_complete():
    assert {"yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "rfdetr_n", "rfdetr_s",
            "rtmpose_s", "rtmpose_m", "dinov2_small"} <= set(REGISTRY)
    for name, s in REGISTRY.items():
        assert s.name == name
        assert s.url.startswith("https://")
        assert s.licence and s.trained_on and s.kind and s.file


def test_no_banned_names():
    for s in REGISTRY.values():
        low = (s.url + s.name + s.trained_on).lower()
        assert not any(b in low for b in BANNED), s.name


def test_fallbacks_point_to_registry_or_none():
    for s in REGISTRY.values():
        assert s.fallback is None or s.fallback in REGISTRY
    assert REGISTRY["yolox_tiny"].fallback == "yolox_nano"
    assert REGISTRY["dinov2_small"].fallback is None


def test_ensure_model_falls_back_after_three_failures(tmp_path):
    calls = []

    def dl(url, dest):
        calls.append(url)
        if "yolox_tiny" in url:
            raise OSError("boom")
        dest.write_bytes(b"dummy-onnx")

    p = ensure_model("yolox_tiny", models_dir=tmp_path, downloader=dl, sleep=lambda s: None)
    assert p == tmp_path / "yolox_nano" / "yolox_nano.onnx"
    assert p.read_bytes() == b"dummy-onnx"
    assert sum("yolox_tiny" in u for u in calls) == 3
    man = json.loads((tmp_path / "MANIFEST.json").read_text())
    ent = {m["name"]: m for m in man["models"]}
    assert "yolox_tiny" not in ent
    e = ent["yolox_nano"]
    assert len(e["sha256"]) == 64 and e["bytes"] == len(b"dummy-onnx")
    assert e["licence"] == "Apache-2.0" and e["downloaded"][:2] == "20"


def test_ensure_model_no_fallback_raises(tmp_path):
    def dl(url, dest):
        raise OSError("down")

    with pytest.raises(RuntimeError):
        ensure_model("dinov2_small", models_dir=tmp_path, downloader=dl, sleep=lambda s: None)


def test_ensure_model_cached_does_not_download(tmp_path):
    def dl(url, dest):
        dest.write_bytes(b"x")

    p = ensure_model("yolox_nano", models_dir=tmp_path, downloader=dl, sleep=lambda s: None)

    def dl2(url, dest):
        raise AssertionError("should not download")

    assert ensure_model("yolox_nano", models_dir=tmp_path, downloader=dl2) == p


def test_zip_extraction_rejects_traversal(tmp_path):
    import zipfile
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.onnx", b"x")
    with pytest.raises(ValueError):
        models.safe_extract(z, tmp_path / "out")
    z2 = tmp_path / "abs.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.writestr("/abs.onnx", b"x")
    with pytest.raises(ValueError):
        models.safe_extract(z2, tmp_path / "out2")


def test_zip_kind_finds_end2end(tmp_path):
    import zipfile
    def dl(url, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("sub/end2end.onnx", b"onnxbytes")
            zf.writestr("sub/deploy.json", b"{}")
    p = ensure_model("rtmpose_s", models_dir=tmp_path, downloader=dl, sleep=lambda s: None)
    assert p.name == "end2end.onnx" and p.read_bytes() == b"onnxbytes"
    assert tmp_path.resolve() in p.resolve().parents
