import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pulse import cache, pipeline, report
from pulse.config import get_config
from pulse.datatypes import StableSegment, TrackObs, Zone

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 320, 180, 5.0


def _check_module():
    spec = importlib.util.spec_from_file_location("check_outputs", ROOT / "scripts" / "check_outputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _video(path: Path, seconds=12, w=W, h=H) -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)       # noisy: blur changes it visibly
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _ in range(int(seconds * FPS)):
        vw.write(base)
    vw.release()
    cv2.imwrite(str(path.with_suffix(".png")), base)


def _setup(tmp_path, src=(W, H), frame=(W, H), kf="ok"):
    """src: video size; frame: analysis-frame size the boxes live in; kf: cache keyframe ok|none|raw."""
    video = tmp_path / "v.mp4"
    _video(video, w=src[0], h=src[1])
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    kp[0], kp[1], kp[2] = [150.0, 60.0, 0.9], [146.0, 56.0, 0.9], [154.0, 56.0, 0.9]
    obs = []
    for i in range(60):                                          # customer standing in the car zone
        obs.append(TrackObs(round(i / FPS, 3), i, 1, (130.0, 50.0, 170.0, 150.0), 0.9, kp))
    for i in range(40):                                          # a second person, short
        obs.append(TrackObs(round(i / FPS, 3), i, 2, (20.0, 60.0, 50.0, 150.0), 0.9, None))
    meta = {"video": str(video), "sha1": "x", "key": "k", "src_w": W, "src_h": H, "src_fps": FPS,
            "duration": 12.0, "analysed_duration": 12.0, "trimmed": False, "frame_w": frame[0],
            "frame_h": frame[1],
            "sample_fps": FPS, "n_samples": 60, "detector": "test", "pose_model": "test",
            "providers": ["CPUExecutionProvider"], "perception_seconds": 1.0, "rtf": 12.0}
    cdir = tmp_path / "cache"
    cdir.mkdir()
    raw = cv2.imread(str(video.with_suffix(".png")))
    if kf != "none":                                   # "raw": full source size, not analysis size
        cv2.imwrite(str(cdir / "keyframe.jpg"), raw)
    pc = cache.PerceptionCache(cdir, meta, obs, [], 6.0, [StableSegment(0, 12.0, True, True)], {})
    zones = [Zone("car-1", "Car 1", "car", [(0.3, 0.1), (0.7, 0.1), (0.7, 0.95), (0.3, 0.95)], "user")]
    return video, pc, zones


@pytest.fixture
def built(tmp_path):
    video, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones)
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=False)
    return video, pc, a, out


def test_every_file_written_and_check_outputs_passes(built, capsys):
    _, _, a, out = built
    rep = json.loads((out / "report.json").read_text())
    for k in ("banner", "meta", "store", "per_car", "per_zone", "zones", "people", "insights",
              "journey_flow", "charts", "files", "warnings"):
        assert k in rep
    for name in ("report.json", "stats.json", "insights.json", "zones.json", "roles.json", "people.json",
                 "visits.csv", "per_car.csv", "occupancy.csv", "heatmap.png", "occupancy.svg",
                 "waiting.svg", "journey_flow.json", "keyframe.jpg"):
        assert (out / name).is_file(), name
    assert (out / "thumbs" / "person_1.jpg").is_file()
    assert all((out / p).is_file() for p in rep["files"].values())
    flow = json.loads((out / "journey_flow.json").read_text())
    assert {"Start", "End"} <= {n["id"] for n in flow["nodes"]}
    assert _check_module().main([str(out), "--no-video"]) == 0
    assert "OK report.json" in capsys.readouterr().out


def test_check_outputs_fails_on_broken_file(built, capsys):
    out = built[3]
    (out / "visits.csv").write_text("a,b\n")
    assert _check_module().main([str(out), "--no-video"]) == 1


def test_keyframe_output_is_face_blurred(built):
    _, pc, _, out = built
    raw = cv2.imread(str(pc.keyframe_path)).astype(float)
    blurred = cv2.imread(str(out / "keyframe.jpg")).astype(float)
    face = (slice(45, 80), slice(135, 165))
    assert np.abs(raw[face] - blurred[face]).mean() > 15
    # the second person has no keypoints: the top 20% of the box is blurred too
    top = (slice(60, 78), slice(20, 50))
    assert np.abs(raw[top] - blurred[top]).mean() > 15
    assert report.observations_by_t(pc)[0.0]


def test_video_rendered_and_checked(tmp_path):
    video, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones)
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=True)
    assert (out / "annotated.mp4").is_file()
    assert _check_module().main([str(out)]) == 0


def _lap(img, box=(53, 63, 143, 157)):
    y0, y1, x0, x1 = box
    return cv2.Laplacian(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def _ref_frame(video, t, width=960):
    from pulse import sources
    for tt, fr in sources.iter_frames(video, FPS, width):
        if tt >= t:
            return fr


@pytest.mark.parametrize("kf", ["none", "raw"])
def test_keyframe_blurred_for_source_wider_than_analysis_frame(tmp_path, kf):
    video, pc, zones = _setup(tmp_path, src=(1280, 720), frame=(960, 540), kf=kf)
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones)
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=False)
    key = cv2.imread(str(out / "keyframe.jpg"))
    assert key.shape[:2] == (540, 960)
    assert _lap(key) < 0.1 * _lap(_ref_frame(video, 6.0))
    heat = cv2.imread(str(out / "heatmap.png"))
    assert _lap(heat) < 0.1 * _lap(_ref_frame(video, 6.0))


def test_unreconcilable_keyframe_becomes_grey(tmp_path):
    video, pc, zones = _setup(tmp_path)
    cv2.imwrite(str(pc.keyframe_path), np.random.default_rng(1).integers(0, 255, (300, 100, 3), np.uint8))
    a = pipeline.score(pc, get_config("short"), zones=zones)
    pipeline.write_outputs(pc, a, tmp_path / "out", get_config("short"), video=None, render_video=False)
    key = cv2.imread(str(tmp_path / "out" / "keyframe.jpg"))
    assert key.shape[:2] == (H, W) and key.std() < 5


def test_thumbnail_heatmap_and_contact_sheet_faces_blurred(tmp_path):
    from pulse.datatypes import Insight
    video, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones)
    a.insights.append(Insight("R1", "t", {}, "a", ["Car 1"], None, "low", 1,
                              [{"t": 6.0, "person_id": 1, "zone_id": "car-1"}]))
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=False)
    ref = _lap(_ref_frame(video, 6.0, W))
    sheet = cv2.imread(str(out / "contact_sheets" / "insight_0.jpg"))
    assert _lap(sheet) < 0.1 * ref
    heat = cv2.imread(str(out / "heatmap.png"))
    assert _lap(heat) < 0.1 * ref
    thumb = cv2.imread(str(out / "thumbs" / "person_1.jpg"))      # box (130,50,170,150) -> 128x256
    face = cv2.cvtColor(thumb[8:25, 45:75], cv2.COLOR_BGR2GRAY)
    unblurred = cv2.resize(_ref_frame(video, 6.0, W)[50:150, 130:170], (128, 256), interpolation=cv2.INTER_AREA)
    ref_face = cv2.cvtColor(unblurred[8:25, 45:75], cv2.COLOR_BGR2GRAY)
    assert cv2.Laplacian(face, cv2.CV_64F).var() < 0.1 * cv2.Laplacian(ref_face, cv2.CV_64F).var()


def test_keyframe_without_exact_detections_warns(tmp_path):
    video, pc, zones = _setup(tmp_path)
    pc.keyframe_t = 6.1                                  # not a sampled time, > half a period from none? 6.1 is exact-ish
    pc.obs = [o for o in pc.obs if not (5.9 < o.t < 6.3)]
    a = pipeline.score(pc, get_config("short"), zones=zones)
    rep = pipeline.write_outputs(pc, a, tmp_path / "out", get_config("short"), video=None, render_video=False)
    warnings = json.loads((tmp_path / "out" / "report.json").read_text())["warnings"]
    assert any("keyframe" in w.lower() for w in warnings)


def test_check_outputs_rejects_bad_file_entries(built):
    out = built[3]
    rp = out / "report.json"
    rep = json.loads(rp.read_text())
    for bad in ("../escape.txt", "/etc/hosts", "x.bin"):
        rep2 = {**rep, "files": {**rep["files"], "evil": bad}}
        rp.write_text(json.dumps(rep2))
        assert _check_module().main([str(out), "--no-video"]) == 1, bad


def test_stale_outputs_cleaned_but_foreign_files_kept(tmp_path):
    video, pc, zones = _setup(tmp_path)
    out = tmp_path / "out"
    (out / "thumbs").mkdir(parents=True)
    (out / "thumbs" / "person_999.jpg").write_bytes(b"x")
    (out / "annotated.mp4").write_bytes(b"x")
    (out / "mine.txt").write_text("keep")
    cfg = get_config("short")
    pipeline.write_outputs(pc, pipeline.score(pc, cfg, zones=zones), out, cfg, video=video, render_video=False)
    assert not (out / "thumbs" / "person_999.jpg").exists() and not (out / "annotated.mp4").exists()
    assert (out / "mine.txt").exists()
