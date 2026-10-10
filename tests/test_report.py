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


def _video(path: Path, seconds=12) -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)       # noisy: blur changes it visibly
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(int(seconds * FPS)):
        vw.write(base)
    vw.release()
    cv2.imwrite(str(path.with_suffix(".png")), base)


def _setup(tmp_path):
    video = tmp_path / "v.mp4"
    _video(video)
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    kp[0], kp[1], kp[2] = [150.0, 60.0, 0.9], [146.0, 56.0, 0.9], [154.0, 56.0, 0.9]
    obs = []
    for i in range(60):                                          # customer standing in the car zone
        obs.append(TrackObs(round(i / FPS, 3), i, 1, (130.0, 50.0, 170.0, 150.0), 0.9, kp))
    for i in range(40):                                          # a second person, short
        obs.append(TrackObs(round(i / FPS, 3), i, 2, (20.0, 60.0, 50.0, 150.0), 0.9, None))
    meta = {"video": str(video), "sha1": "x", "key": "k", "src_w": W, "src_h": H, "src_fps": FPS,
            "duration": 12.0, "analysed_duration": 12.0, "trimmed": False, "frame_w": W, "frame_h": H,
            "sample_fps": FPS, "n_samples": 60, "detector": "test", "pose_model": "test",
            "providers": ["CPUExecutionProvider"], "perception_seconds": 1.0, "rtf": 12.0}
    cdir = tmp_path / "cache"
    cdir.mkdir()
    raw = cv2.imread(str(video.with_suffix(".png")))
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
