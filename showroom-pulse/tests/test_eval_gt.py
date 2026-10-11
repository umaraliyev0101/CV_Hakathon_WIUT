import json

import pytest

from pulse.config import get_config
from pulse.eval import run_eval as R
from pulse.eval.meva import MevaClip
from pulse.eval.scenarios import SCENARIOS
from pulse.insights import recommend


def test_sampled_frame_indices_match_iter_frames_rule():
    assert R.sampled_frame_indices(5, 30.0, 5.0) == [0, 6, 12, 18, 24]
    assert R.sampled_frame_indices(4, 25.0, 5.0) == [0, 5, 10, 15]
    assert R.sampled_frame_indices(3, 29.97, 5.0) == [int(round(k * 29.97 / 5.0)) for k in range(3)]


def test_build_gt_samples_scales_and_normalises():
    # source 1920x1080 analysed at 960x540 (scale 0.5); 30 fps sampled at 5 fps -> frames 0, 6, 12
    boxes = {0: (100.0, 200.0, 200.0, 400.0), 6: (110.0, 200.0, 210.0, 400.0), 7: (999.0, 1.0, 1000.0, 2.0),
             12: (120.0, 200.0, 220.0, 400.0)}
    car = {5: (0.0, 0.0, 10.0, 10.0)}
    meva = MevaClip("c", {1: boxes, 2: car}, {1: "person", 2: "vehicle"}, [])
    tl, frames = R.build_gt(meva, 3, 30.0, 5.0, (1920, 1080), (960, 540))
    assert list(tl) == [1]                                   # vehicles are not people; off-sample frame 7 dropped
    s = tl[1]
    assert [x.t for x in s] == [0.0, 0.2, 0.4]
    assert s[0].bbox == (50.0, 100.0, 100.0, 200.0)          # scaled by 0.5
    assert s[0].foot == pytest.approx((75 / 960, 200 / 540))  # bottom centre, normalised
    assert frames[1][1] == (55.0, 100.0, 105.0, 200.0)       # keyed by sample index, not source frame
    assert s[0].pos_m is None and s[0].speed is None


def test_build_gt_skips_degenerate_boxes():
    meva = MevaClip("c", {1: {0: (10.0, 10.0, 10.0, 50.0)}}, {1: "person"}, [])
    tl, frames = R.build_gt(meva, 1, 30.0, 5.0, (960, 540), (960, 540))
    assert tl == {} and frames == {}


def test_insight_scenarios_fire_exactly_as_expected():
    cfg = get_config("short")
    for sc in SCENARIOS:
        st, eps = sc.build()
        fired = any(i.rule == sc.rule for i in recommend(st, eps, cfg))
        assert fired == sc.fires, sc.key


def test_aggregate_pools_counts_without_cancelling_clips():
    def clip(pe, ge, px, gx):
        return {"entries": {"pred": pe, "gt": ge, "timing": [1.0], "n_matched": 1},
                "exits": {"pred": px, "gt": gx, "timing": [], "n_matched": 0},
                "occupancy": {"abs_err_sum": 10.0, "n": 20}, "mot_raw": _mot(), "mot_merged": _mot(),
                "dwell_pairs": [[10.0, 12.0], [5.0, 5.0]], "journey_acc": [1.0, 0.5],
                "n_gt_persons": 4, "n_pred_persons": 3, "n_matched": 2, "coverage": {"hit": 3, "total": 4}}
    a = R.aggregate([clip(8, 10, 10, 10), clip(12, 10, 10, 10)])
    assert a["entries"]["pred"] == 20 and a["entries"]["gt"] == 20
    assert a["entry_exit_error"] == pytest.approx(4 / 40 * 100)    # |8-10| + |12-10| over 40 GT events
    assert a["occupancy_mae"] == pytest.approx(0.5)
    assert a["dwell"]["n_gt_positive"] == 4 and a["dwell"]["median_abs_pct"] == pytest.approx(10.0)
    assert a["journey_mean"] == pytest.approx(0.75) and a["coverage"] == {"hit": 6, "total": 8}


def _mot():
    from pulse.eval import metrics as M
    return M.mot({0: {1: (0, 0, 10, 10)}}, {0: {1: (0, 0, 10, 10)}}, [0])


def test_main_returns_1_when_inputs_are_missing(tmp_path, monkeypatch):
    split = tmp_path / "labels"
    split.mkdir()
    (split / "split.json").write_text(json.dumps({
        "dev": [{"clip": "nope", "video": "data/meva/videos/nope.avi", "kpf": "data/meva/annotations/nope"}],
        "test": []}))
    monkeypatch.setattr(R, "ROOT", tmp_path)
    assert R.main(tmp_path / "EVAL.md") == 1
    assert not (tmp_path / "EVAL.md").exists()
