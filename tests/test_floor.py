"""Floor calibration (spec 4.3): zone-file round trip, validation, score in metres, top-down heatmap."""
import json
import math

import pytest

from pulse import pipeline, zones as Z
from pulse.app.main import ZoneError, validate_zones
from pulse.config import get_config
from pulse.datatypes import Zone
from tests.test_report import W, H, _check_module, _setup

# image corners -> a 10 m x 5.625 m floor: 32 px per metre in the 320x180 analysis frame
FLOOR = {"img": [[0, 0], [1, 0], [1, 1], [0, 1]], "world": [[0, 0], [10, 0], [10, 5.625], [0, 5.625]]}
ZONE = {"zone_id": "z", "name": "Z", "kind": "car", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}


def test_zone_file_roundtrip_with_floor(tmp_path):
    zs = [Zone("desk", "Desk", "desk", [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], "user")]
    p = tmp_path / "z.json"
    Z.save_zones(p, zs, (320, 180), floor=FLOOR)
    assert json.loads(p.read_text())["floor"] == FLOOR
    assert Z.load_floor(p) == FLOOR
    assert [z.zone_id for z in Z.load_zones(p)] == ["desk"]


def test_old_zone_file_without_floor(tmp_path):
    p = tmp_path / "z.json"
    Z.save_zones(p, [Zone("d", "D", "desk", [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], "user")])
    assert "floor" not in json.loads(p.read_text())
    assert Z.load_floor(p) is None
    assert validate_zones({"zones": [ZONE]})


def _bad(**kw):
    f = json.loads(json.dumps(FLOOR))
    f.update(kw)
    return f


@pytest.mark.parametrize("floor", [
    "x", {"img": FLOOR["img"]}, _bad(img=FLOOR["img"][:3]), _bad(world=FLOOR["world"] + [[1, 1]]),
    _bad(img=[[0, 0], [1, 0], [1, 1], [1.5, 1]]), _bad(img=[[0, 0], [1, 0], [1, 1], [-0.1, 1]]),
    _bad(world=[[0, 0], [10, 0], [10, 5], [0, 2000]]), _bad(world=[[0, 0], [10, 0], [10, 5], [0, None]]),
    _bad(world=[[0, 0], [10, 0], [10, 5], [0, "a"]]), _bad(world=[[0, 0], [10, 0], [10, 5], [0, math.nan]]),
    _bad(img=[[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.9, 0.9]]),       # collinear
    _bad(img=[[0.1, 0.1], [0.1, 0.1], [0.5, 0.5], [0.9, 0.2]]),       # coincident
    _bad(world=[[0, 0], [1, 1], [2, 2], [3, 3]]),
])
def test_validation_rejects(floor):
    with pytest.raises(ZoneError):
        validate_zones({"zones": [ZONE], "floor": floor})


def test_validation_accepts():
    assert validate_zones({"zones": [ZONE], "floor": FLOOR})


def test_score_with_floor_gives_metric_positions(tmp_path):
    _, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones, floor=FLOOR)
    s1, s2 = a.timeline[1][0], a.timeline[2][0]
    assert s1.dist_method == s2.dist_method == "floor"
    assert s1.pos_m == pytest.approx((150 / 32, 150 / 32), abs=1e-3)       # foot = bottom-centre of box
    assert math.dist(s1.pos_m, s2.pos_m) == pytest.approx(115 / 32, abs=1e-3)
    assert a.floor == FLOOR
    assert pipeline.score(pc, cfg, zones=zones).timeline[1][0].dist_method != "floor"


def test_topdown_heatmap_only_when_calibrated(tmp_path, capsys):
    video, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    out = tmp_path / "plain"
    pipeline.write_outputs(pc, pipeline.score(pc, cfg, zones=zones), out, cfg, video=video, render_video=False)
    assert not (out / "heatmap_topdown.png").exists()
    assert "heatmap_topdown" not in json.loads((out / "report.json").read_text())["files"]

    out2 = tmp_path / "cal"
    pipeline.write_outputs(pc, pipeline.score(pc, cfg, zones=zones, floor=FLOOR), out2, cfg, video=video,
                           render_video=False)
    rep = json.loads((out2 / "report.json").read_text())
    assert rep["files"]["heatmap_topdown"] == "heatmap_topdown.png"
    assert (out2 / "heatmap_topdown.png").stat().st_size > 0
    assert json.loads((out2 / "zones.json").read_text())["floor"] == FLOOR
    assert _check_module().main([str(out2), "--no-video"]) == 0
    assert "OK heatmap_topdown.png" in capsys.readouterr().out
    (out2 / "heatmap_topdown.png").write_bytes(b"junk")
    assert _check_module().main([str(out2), "--no-video"]) == 1


# ---------------------------------------------------------------- fix round 1
from pulse import geometry as G  # noqa: E402
from pulse.datatypes import TrackObs  # noqa: E402
from pulse.timeline import build_timeline  # noqa: E402

# perspective floor: a trapezoid in the lower half of the image; the horizon lies near v = 0.25
PERSP = {"img": [[0.4, 0.4], [0.6, 0.4], [0.9, 0.9], [0.1, 0.9]],
         "world": [[0, 40], [4, 40], [4, 0], [0, 0]]}


def _walker(x0, step_px, n=20, y=150.0, fps=5.0, tid=1):
    return [TrackObs(round(i / fps, 3), i, tid, (x0 + i * step_px - 10, y - 100, x0 + i * step_px + 10, y),
                     0.9, None) for i in range(n)]


def test_floor_speed_is_metres_per_second():
    h = Z.floor_homography(FLOOR, (W, H))
    cfg = get_config("short")
    tl = build_timeline(_walker(20, 20.48), {1: 1}, (W, H), cfg, h)       # 20.48 px / 0.2 s = 3.2 m/s
    sp = [s.speed for s in tl[1]]
    assert all(s.dist_method == "floor" for s in tl[1])
    assert sp[10] == pytest.approx(3.2, rel=0.02)
    assert sp[0] == pytest.approx(3.2, rel=0.05) and sp[-1] == pytest.approx(3.2, rel=0.05)


def test_floor_speed_independent_of_world_origin():
    cfg = get_config("short")
    shifted = {**FLOOR, "world": [[x - 50, y - 20] for x, y in FLOOR["world"]]}
    a = build_timeline(_walker(20, 20.48), {1: 1}, (W, H), cfg, Z.floor_homography(FLOOR, (W, H)))
    b = build_timeline(_walker(20, 20.48), {1: 1}, (W, H), cfg, Z.floor_homography(shifted, (W, H)))
    assert [s.speed for s in a[1]] == pytest.approx([s.speed for s in b[1]])


def test_bowtie_calibration_rejected():
    bow_img = {**FLOOR, "img": [[0, 0], [1, 1], [1, 0], [0, 1]]}
    bow_world = {**FLOOR, "world": [[0, 0], [10, 5.625], [10, 0], [0, 5.625]]}
    for f in (bow_img, bow_world):
        with pytest.raises(Z.FloorError):
            Z.validate_floor(f)
        with pytest.raises(ZoneError):
            validate_zones({"zones": [ZONE], "floor": f})


def test_mirrored_but_consistent_calibration_accepted():
    Z.validate_floor({**FLOOR, "world": FLOOR["world"][::-1]})             # opposite winding, both convex


def test_to_floor_behind_horizon_is_nan():
    h = Z.floor_homography(PERSP, (W, H))
    x, y = G.to_floor(h, (0.5 * W, 0.7 * H))
    assert math.isfinite(x) and 0 < y < 40
    assert all(math.isnan(v) for v in G.to_floor(h, (0.5 * W, 0.0)))        # above the horizon
    assert all(math.isnan(v) for v in G.to_floor(h, (0.5 * W, 0.2 * H)))


def test_foot_above_horizon_invalid_and_excluded(tmp_path):
    h = Z.floor_homography(PERSP, (W, H))
    cfg = get_config("short")
    above = [TrackObs(round(i / 5, 3), i, 1, (150, 0, 170, 6), 0.9, None) for i in range(10)]
    tl = build_timeline(above, {1: 1}, (W, H), cfg, h)
    assert all(s.pos_m is None and s.dist_method == "invalid" for s in tl[1])
    assert all(s.speed is None for s in tl[1])
