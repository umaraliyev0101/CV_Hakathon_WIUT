import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from pulse.config import get_config
from pulse.render import (FrameOverlay, blur_faces, contact_sheet, draw_overlay, face_box,
                          render, render_timeline, thumbnail)


def _kps(face=True):
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    if face:
        k[0] = [50.0, 30.0, 0.9]
        k[1] = [44.0, 26.0, 0.9]
        k[2] = [56.0, 26.0, 0.9]
    return k


def test_face_box_from_keypoints():
    x1, y1, x2, y2 = face_box((30, 10, 70, 110), _kps())
    assert x1 < 44 and x2 > 56 and y1 < 26 and y2 > 30
    assert (x1 + x2) / 2 == pytest.approx(50, abs=3)


def test_face_box_fallback_top_20_percent():
    assert face_box((30, 10, 70, 110), _kps(False)) == (30, 10, 70, 30)
    low = _kps()
    for p in low[:5]:
        p[2] = 0.1
    assert face_box((30, 10, 70, 110), low) == (30, 10, 70, 30)
    assert face_box((30, 10, 70, 110), None) == (30, 10, 70, 30)


def test_blur_faces_changes_inside_only():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (120, 100, 3), dtype=np.uint8)
    orig = frame.copy()
    bbox = (30, 10, 70, 110)
    out = blur_faces(frame, [(bbox, None)])
    assert np.array_equal(frame, orig)
    x1, y1, x2, y2 = face_box(bbox, None)
    inside = out[y1:y2, x1:x2].astype(float)
    assert inside.var() < 0.5 * orig[y1:y2, x1:x2].astype(float).var()
    assert np.array_equal(out[y2 + 2:], orig[y2 + 2:])
    assert np.array_equal(out[:, :x1 - 2], orig[:, :x1 - 2])


def _make_video(path, secs=2, fps=10, w=160, h=120):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(1)
    for _ in range(secs * fps):
        wr.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    wr.release()


def _overlays(cfg, n=10):
    out = {}
    for k in range(n):
        t = round(k / cfg.sample_fps, 3)
        persons = [
            {"person_id": 1, "bbox": (20, 20, 60, 110), "keypoints": None, "role": "customer",
             "zone_id": "z", "dwell": 1.5, "alert": k % 2 == 0},
            {"person_id": 2, "bbox": (90, 20, 130, 110), "keypoints": _kps(), "role": "staff",
             "zone_id": None, "dwell": 0.0, "alert": False},
        ]
        zones = [{"zone_id": "z", "name": "Car A", "polygon": [(0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.1, 0.9)],
                  "now": 1, "total": 3}]
        out[t] = FrameOverlay(t, persons, zones)
    return out


def test_render_h264_10_frames(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    out = render(vid, _overlays(cfg), tmp_path / "o.mp4", cfg, {})
    assert out.exists()
    assert b"ftyp" in out.read_bytes()[:32]
    n, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert n == 10
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    assert fourcc in ("avc1", "h264")


def test_render_blurs_frames_without_overlay(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    obs = {round(k / 5, 3): [((30, 10, 90, 110), None)] for k in range(10)}
    out = render(vid, {}, tmp_path / "o.mp4", cfg, obs)
    cap = cv2.VideoCapture(str(out))
    ok, f = cap.read()
    cap.release()
    x1, y1, x2, y2 = face_box((30, 10, 90, 110), None)
    assert f[y1:y2, x1:x2].astype(float).var() < 0.5 * f[y2 + 5:, :].astype(float).var()


def test_render_timeline_wrapper(tmp_path):
    from pulse.datatypes import Sample
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    tl = {1: [Sample(round(k / 5, 3), 1, (0.3, 0.8), (20, 20, 60, 110), None, 0.0) for k in range(10)]}
    out = render_timeline(vid, tl, [], tmp_path / "o.mp4", cfg)
    assert imageio_ffmpeg.count_frames_and_secs(str(out))[0] == 10


def test_thumbnail_blurred_and_sized():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (120, 100, 3), dtype=np.uint8)
    th = thumbnail(frame, (30, 10, 70, 110), None)
    assert th.shape == (256, 128, 3)


def test_draw_overlay_does_not_mutate_and_blurs():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    orig = frame.copy()
    ov = _overlays(get_config())[0.0]
    out = draw_overlay(frame, ov, (160, 120))
    assert np.array_equal(frame, orig) and out.shape == frame.shape
    assert not np.array_equal(out, orig)


def test_contact_sheet_grid(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    moments = [{"t": 0.1}, {"t": 0.5}, {"t": 1.0}, {"t": 1.6}]
    out = contact_sheet(vid, moments, _overlays(cfg), tmp_path / "c.jpg", cfg, {}, cols=2)
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[1] == 2 * 320
    assert img.shape[0] == 2 * 240


# ---- fix round 1 ----
def _noise(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_render_requires_observations_by_t(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    with pytest.raises(TypeError):
        render(vid, {}, tmp_path / "o.mp4", cfg)


def test_render_timeline_blurs_detected_people(tmp_path):
    from pulse.datatypes import Sample
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    tl = {1: [Sample(round(k / 5, 3), 1, (0.3, 0.8), (30, 10, 90, 110), None, 0.0) for k in range(10)]}
    out = render_timeline(vid, tl, [], tmp_path / "o.mp4", cfg)
    cap = cv2.VideoCapture(str(out))
    ok, f = cap.read()
    cap.release()
    x1, y1, x2, y2 = face_box((30, 10, 90, 110), None)
    assert f[y1 + 2:y2 - 2, x1 + 2:x2 - 2].astype(float).var() < 0.5 * f[y2 + 5:, :].astype(float).var()


def test_numpy_keypoints_ok():
    kp = np.asarray(_kps())
    assert face_box((30, 10, 70, 110), kp) == face_box((30, 10, 70, 110), _kps())
    blur_faces(_noise(), [((30, 10, 70, 110), kp)])
    assert face_box((30, 10, 70, 110), np.zeros((0, 3))) == (30, 10, 70, 30)


def test_render_failure_leaves_no_output(tmp_path, monkeypatch):
    import pulse.render as r
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    calls = {"n": 0}
    real = r.draw_overlay

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(r, "draw_overlay", boom)
    out = tmp_path / "o.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        render(vid, _overlays(cfg), out, cfg, {})
    assert not out.exists() and not list(tmp_path.glob("*.part.mp4"))


def test_thumbnail_face_region_blurred():
    frame = _noise(200, 100)
    bbox = (10, 10, 90, 190)
    th = thumbnail(frame, bbox, None)
    raw = cv2.resize(frame[10:190, 10:90], (128, 256), interpolation=cv2.INTER_AREA)
    fh = int(256 * 0.2)
    assert th[:fh - 8].astype(float).var() < 0.5 * raw[:fh - 8].astype(float).var()


def test_contact_sheet_tile_face_blurred(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    obs = {round(k / 5, 3): [((30, 10, 90, 110), None)] for k in range(10)}
    out = contact_sheet(vid, [{"t": 0.0}], {}, tmp_path / "c.jpg", cfg, obs, cols=1)
    img = cv2.imread(str(out))
    x1, y1, x2, y2 = face_box((30, 10, 90, 110), None)  # tile is 320 wide, frame 160: x2 scale
    face = img[2 * y1 + 4:2 * y2 - 4, 2 * x1 + 4:2 * x2 - 4].astype(float)
    assert face.var() < 0.5 * img[2 * y2 + 20:2 * 118 - 20].astype(float).var()


def test_keypoint_face_box_blurred():
    frame = _noise()
    bbox = (30, 10, 70, 110)
    out = blur_faces(frame, [(bbox, _kps())])
    x1, y1, x2, y2 = face_box(bbox, _kps())
    assert out[y1:y2, x1:x2].astype(float).var() < 0.5 * frame[y1:y2, x1:x2].astype(float).var()


def test_edge_and_degenerate_boxes():
    frame = _noise(60, 80)
    for b in [(-50, -50, -10, -10), (70, 50, 200, 200), (5, 5, 5, 5), (10, 10, 11, 11), (10, 10, 12, 60)]:
        out = blur_faces(frame, [(b, None)])
        assert out.shape == frame.shape
    out = blur_faces(frame, [((60, 40, 200, 200), None)])  # clipped at the edge, still blurred
    assert out[40:46, 60:80].astype(float).var() < 0.5 * frame[40:46, 60:80].astype(float).var()


def test_odd_frame_size_renders(tmp_path):
    cfg = get_config(sample_fps=5.0, max_width=960)
    vid = tmp_path / "odd.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 539))
    for _ in range(10):
        wr.write(_noise(539, 960))
    wr.release()
    out = render(vid, {}, tmp_path / "o.mp4", cfg, {})
    assert imageio_ffmpeg.count_frames_and_secs(str(out))[0] >= 4


def test_overlay_colours_present():
    frame = np.full((120, 160, 3), 128, np.uint8)
    ov = FrameOverlay(0.0, [
        {"person_id": 1, "bbox": (10, 30, 40, 110), "keypoints": None, "role": "customer",
         "zone_id": None, "dwell": 0.0, "alert": False},
        {"person_id": 2, "bbox": (60, 30, 90, 110), "keypoints": None, "role": "staff",
         "zone_id": None, "dwell": 0.0, "alert": False},
        {"person_id": 3, "bbox": (110, 30, 140, 110), "keypoints": None, "role": "customer",
         "zone_id": None, "dwell": 0.0, "alert": True}], [])
    out = draw_overlay(frame, ov, (160, 120))
    assert (out == np.array([0, 140, 255], np.uint8)).all(axis=2).any()
    assert (out == np.array([0, 0, 255], np.uint8)).all(axis=2).any()
    blue = out[60, 59:62].astype(int)  # staff edge, 50% alpha over grey
    assert (blue[:, 0] > blue[:, 2]).any()


def test_contact_sheet_single_moment_has_no_filler_tile(tmp_path):
    cfg = get_config(sample_fps=5.0)
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    out = contact_sheet(vid, [{"t": 0.5}], _overlays(cfg), tmp_path / "c1.jpg", cfg, {}, cols=2)
    img = cv2.imread(str(out))
    assert img.shape[1] == 320 and img.shape[0] == 240


def test_zone_names_transliterated_for_putText():
    from pulse.render import _ascii
    assert _ascii("Машина 1") == "Mashina 1"
    assert _ascii("Car A") == "Car A"
    assert _ascii("Zone \u4e2d") == "Zone ?"
