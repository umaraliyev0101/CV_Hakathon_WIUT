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
