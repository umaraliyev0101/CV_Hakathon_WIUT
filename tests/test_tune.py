"""Dev-only tuning: the selection rule on synthetic numbers, then the grid on a synthetic clip and the command line.

The selection tests need no video: rows are plain dicts shaped like ``tune.evaluate`` rows. The end-to-end tests use the
model-free fake stack on the shared ``meet_video`` with hand-written KPF files, as ``test_meva_eval`` does.
"""
import dataclasses
import json
import sys

import pytest

from showroom_coach.evaluation import meva_eval as me
from showroom_coach.evaluation import tune
from showroom_coach.perception.pipeline import cached_perception
from showroom_coach.perception.stack import fake_stack
from tests.test_meva_eval import _kpf_geom


# ------------------------------------------------------------------------------------------- selection on numbers

def row(merge, dist, min_dur, f1, idf1):
    return {"merge_thr": merge, "dist": dist, "min_dur": min_dur, "gap_tol": 1.5,
            "twin_aware": {"f1": f1}, "persons_idf1": idf1}


def test_f1_is_the_harmonic_mean_and_zero_when_either_side_is_missing_or_zero():
    assert tune.f1(0.5, 1.0) == pytest.approx(2 / 3)
    assert tune.f1(0.6, 0.6) == pytest.approx(0.6)
    assert tune.f1(None, 0.9) == 0.0 and tune.f1(0.9, None) == 0.0 and tune.f1(None, None) == 0.0
    assert tune.f1(0.0, 0.0) == 0.0 and tune.f1(0.0, 1.0) == 0.0


def test_choose_maximises_the_twin_aware_talk_f1():
    rows = [row(0.3, 2.0, 8.0, 0.40, 0.90), row(0.2, 2.5, 4.0, 0.55, 0.10), row(0.4, 3.0, 2.0, 0.50, 0.99)]
    assert tune.choose(rows) is rows[1]                       # F1 wins even with the worst IDF1


def test_choose_breaks_an_f1_tie_by_idf1():
    rows = [row(0.2, 2.0, 8.0, 0.5, 0.60), row(0.4, 2.0, 8.0, 0.5, 0.75), row(0.5, 2.0, 8.0, 0.5, 0.70)]
    assert tune.choose(rows) is rows[1]


def test_choose_ignores_float_noise_when_comparing():
    rows = [row(0.2, 2.0, 8.0, 0.5 + 1e-13, 0.60), row(0.4, 2.0, 8.0, 0.5, 0.75)]
    assert tune.choose(rows) is rows[1]                       # the F1s are equal for any practical purpose


def test_choose_treats_a_missing_idf1_as_zero():
    rows = [row(0.2, 2.0, 8.0, 0.5, None), row(0.4, 2.0, 8.0, 0.5, 0.01)]
    assert tune.choose(rows) is rows[1]


def test_choose_falls_back_to_the_settings_nearest_the_untuned_defaults():
    rows = [row(0.5, 3.0, 2.0, 0.5, 0.7), row(0.3, 2.5, 8.0, 0.5, 0.7), row(0.3, 2.0, 6.0, 0.5, 0.7),
            row(0.2, 2.0, 8.0, 0.5, 0.7)]
    assert tune.choose(rows) is rows[2]                       # merge 0.3 is the default, then dist 2.0, then min_dur
    assert tune.choose(list(reversed(rows))) is rows[2]       # independent of the order of the rows


def test_choose_needs_rows():
    with pytest.raises(ValueError):
        tune.choose([])


def test_merge_beats_default_needs_a_better_f1_and_no_worse_idf1():
    base = row(0.3, 2.0, 8.0, 0.50, 0.70)
    assert tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.55, 0.70), base)
    assert tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.55, 0.80), base)
    assert not tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.55, 0.60), base)      # F1 up but IDF1 down: keep 0.30
    assert not tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.45, 0.90), base)      # IDF1 up but F1 down: keep 0.30
    assert not tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.50, 0.70), base)      # a tie is not a win
    assert tune.merge_beats_default(row(0.4, 2.0, 8.0, 0.50, 0.75), base)          # equal F1, better IDF1
    assert not tune.merge_beats_default(base, base)


def test_find_picks_the_row_of_given_settings():
    rows = [row(0.3, 2.0, 8.0, 0.1, 0.1), row(0.4, 2.0, 8.0, 0.2, 0.2)]
    assert tune.find(rows, 0.4, 2.0, 8.0) is rows[1]
    assert tune.find(rows, 0.5, 2.0, 8.0) is None


# ----------------------------------------------------------------------------------------- a synthetic clip

def write_meet_annotations(directory, name="meet.G1"):
    """KPF files for ``meet_video``: blue actor 1 stands at x 600, red actor 2 walks in at 4 s, leaves at 18 s; they talk
    from 8 s to 18 s (frames 240..540)."""
    from tests.conftest import _customer_x1
    boxes = []
    for f in range(600):
        boxes.append((f, 1, (600, 250, 680, 470)))
        x1 = _customer_x1(f / 30.0)
        if x1 is not None:
            boxes.append((f, 2, (x1, 250, x1 + 80, 470)))
    prefix = directory / name
    _kpf_geom(directory / f"{name}.geom.yml", boxes)
    (directory / f"{name}.types.yml").write_text(
        "".join(f"- {{'types': {{'cset3': {{'person': 1.0}}, 'id1': {i}}}}}\n" for i in (1, 2)), encoding="utf-8")
    span = "{'tsr0': [240, 540]}"
    (directory / f"{name}.activities.yml").write_text(
        "- {'act': {'act2': {'person_talks_to_person': 1.0}, 'actors': [{'id1': 1, 'timespan': [" + span + "]}, "
        "{'id1': 2, 'timespan': [" + span + "]}], 'id2': 4, 'src_status': 'good', 'timespan': [" + span + "]}}\n"
        "- {'act': {'act2': {'person_enters_scene_through_structure': 1.0}, 'actors': [{'id1': 2, 'timespan': "
        "[{'tsr0': [120, 150]}]}], 'id2': 5, 'src_status': 'good', 'timespan': [{'tsr0': [120, 150]}]}}\n",
        encoding="utf-8")
    return prefix


@pytest.fixture
def meet_clip(meet_video, tmp_path):
    prefix = write_meet_annotations(tmp_path)
    cache = tmp_path / "cache"
    cached_perception(meet_video, fake_stack(), cache)            # what a previous evaluation run left behind
    return meet_video, prefix, cache


def test_load_dev_clip_refuses_to_run_perception(meet_video, tmp_path):
    prefix = write_meet_annotations(tmp_path)

    def boom(*a, **k):
        raise AssertionError("perception must not run during tuning")

    stack = dataclasses.replace(fake_stack(), detect=boom)
    with pytest.raises(RuntimeError, match="cache"):
        tune.load_dev_clip(meet_video, prefix, stack, tmp_path / "cache", None)
    assert not (tmp_path / "cache").exists()


def test_the_grid_matches_eval_clip_at_the_same_settings(meet_clip):
    video, prefix, cache = meet_clip
    clip = tune.load_dev_clip(video, prefix, fake_stack(), cache, None)
    out = tune.evaluate([clip], merges=(0.3, 0.5), dists=(1.5, 2.0), min_durs=(4.0, 8.0), gap_tol=1.5, preset="short")
    assert len(out["rows"]) == 8
    assert [(r["merge_thr"], r["dist"], r["min_dur"]) for r in out["rows"]][:3] == [(0.3, 1.5, 4.0), (0.3, 1.5, 8.0),
                                                                                  (0.3, 2.0, 4.0)]
    ref = me.eval_clip(video, prefix, fake_stack(), cache, preset="short", dist=2.0, min_dur=8.0, gap_tol=1.5,
                       merge_thr=0.3)
    got = tune.find(out["rows"], 0.3, 2.0, 8.0)
    for variant in me.VARIANTS:
        for key in ("n_gt", "n_gt_tracked", "n_det", "n_det_mapped", "n_matched", "recall_all", "precision_mapped"):
            assert got[variant]["talk"][key] == ref[variant]["talk"][key], (variant, key)
        assert "start_errs_s" not in got[variant]["talk"]
    assert got["persons_idf1"] == pytest.approx(ref["persons"]["tracking"]["idf1"])
    assert got["enter_exit_recall"]["twin_aware"] == ref["twin_aware"]["enter_exit"]["all"]["recall_all"]
    f1 = tune.f1(ref["twin_aware"]["talk"]["recall_all"], ref["twin_aware"]["talk"]["precision_mapped"])
    assert got["twin_aware"]["f1"] == pytest.approx(f1) and f1 > 0
    assert set(out["per_merge"]) == {"0.3", "0.5"} and out["per_merge"]["0.3"]["n_persons"] == [ref["n_persons"]]
    json.dumps(out)                                                # the whole result is JSON


def test_a_looser_distance_never_finds_fewer_pairs(meet_clip):
    video, prefix, cache = meet_clip
    clip = tune.load_dev_clip(video, prefix, fake_stack(), cache, None)
    out = tune.evaluate([clip], merges=(0.3,), dists=(0.5, 3.0), min_durs=(4.0,), gap_tol=1.5, preset="short")
    tight, loose = out["rows"]
    assert tight["dist"] == 0.5 and loose["twin_aware"]["talk"]["n_det"] >= tight["twin_aware"]["talk"]["n_det"]


def test_the_table_marks_the_chosen_row(meet_clip):
    video, prefix, cache = meet_clip
    clip = tune.load_dev_clip(video, prefix, fake_stack(), cache, None)
    out = tune.evaluate([clip], merges=(0.3, 0.4), dists=(2.0,), min_durs=(4.0, 8.0), gap_tol=1.5, preset="short")
    chosen = tune.choose(out["rows"])
    text = tune.format_grid(out["rows"], out["per_merge"], chosen)
    marked = [line for line in text.splitlines() if line.startswith("*")]
    assert len(marked) == 1 and f"{chosen['merge_thr']:.2f}" in marked[0]
    assert "twin" in text.lower() and "IDF1" in text


# ------------------------------------------------------------------------------------------------ command line

def make_meva_tree(root, meet_video, split):
    """root/clips.csv, root/videos/meet.G1.r13.avi, root/annotations/meet.G1.*, plus a split file."""
    (root / "videos").mkdir()
    (root / "videos" / "meet.G1.r13.avi").write_bytes(meet_video.read_bytes())
    (root / "annotations").mkdir()
    write_meet_annotations(root / "annotations")
    (root / "clips.csv").write_text(
        "clip,camera,location,video_url,annotation_path,talk_count,enter_count,exit_count,video_bytes,fps_guess\n"
        "meet.G1,G1,school,http://x,data/meva/annotations/meet.G1,1,1,0,10,30.0\n"
        "held.G2,G2,school,http://x,data/meva/annotations/held.G2,1,0,0,10,30.0\n", encoding="utf-8")
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")


def test_main_tunes_on_the_dev_clips_only_and_writes_the_json(meet_video, tmp_path, monkeypatch, capsys):
    root = tmp_path / "meva"
    root.mkdir()
    make_meva_tree(root, meet_video, {"limit_seconds": 12, "dev": ["meet.G1"], "test": ["held.G2"]})
    cache = tmp_path / "cache"
    cached_perception(root / "videos" / "meet.G1.r13.avi", fake_stack(), cache, max_seconds=12.0)
    loaded = []
    real_load = tune.load_dev_clip
    monkeypatch.setattr(tune, "load_dev_clip", lambda video, *a, **k: (loaded.append(video.name), real_load(video, *a, **k))[1])
    monkeypatch.setattr(me, "_default_stack", fake_stack)
    monkeypatch.setattr(tune, "GRID_MERGE", (0.3, 0.4))
    monkeypatch.setattr(tune, "GRID_DIST", (2.0,))
    monkeypatch.setattr(tune, "GRID_MIN_DUR", (4.0, 8.0))
    out = tmp_path / "out" / "tune_dev.json"
    rc = tune.main(["--clips", str(root / "clips.csv"), "--split-file", str(root / "split.json"), "--out", str(out),
                    "--cache", str(cache)])
    assert rc == 0
    assert loaded == ["meet.G1.r13.avi"]                           # never the held-out clip
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["clips"] == ["meet.G1"] and data["limit_seconds"] == 12
    assert len(data["rows"]) == 4 and data["chosen"] in data["rows"]
    assert data["selection_rule"] and "adopt" in data["merge_decision"]
    assert data["default_untuned"]["min_dur"] == 8.0 and data["default_untuned"]["merge_thr"] == 0.3
    text = capsys.readouterr().out
    assert "chosen" in text.lower() and str(out) in text


def test_main_rejects_a_missing_cache_without_detecting(meet_video, tmp_path, monkeypatch, capsys):
    root = tmp_path / "meva"
    root.mkdir()
    make_meva_tree(root, meet_video, {"limit_seconds": 12, "dev": ["meet.G1"], "test": ["held.G2"]})
    monkeypatch.setattr(me, "_default_stack", lambda: dataclasses.replace(
        fake_stack(), detect=lambda f: pytest.fail("perception must not run")))
    rc = tune.main(["--clips", str(root / "clips.csv"), "--split-file", str(root / "split.json"),
                    "--out", str(tmp_path / "o.json"), "--cache", str(tmp_path / "empty-cache")])
    assert rc == 1 and "cache" in capsys.readouterr().err.lower()
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("split", [{"dev": [], "test": ["held.G2"]}, "not json"])
def test_main_usage_errors_return_2(meet_video, tmp_path, monkeypatch, capsys, split):
    root = tmp_path / "meva"
    root.mkdir()
    make_meva_tree(root, meet_video, {"dev": ["meet.G1"], "test": []})
    (root / "split.json").write_text(split if isinstance(split, str) else json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(me, "_default_stack", lambda: pytest.fail("must not build the stack"))
    rc = tune.main(["--clips", str(root / "clips.csv"), "--split-file", str(root / "split.json"),
                    "--out", str(tmp_path / "o.json")])
    assert rc == 2 and capsys.readouterr().err


def test_main_without_motmetrics_says_how_to_install_it(meet_video, tmp_path, monkeypatch, capsys):
    root = tmp_path / "meva"
    root.mkdir()
    make_meva_tree(root, meet_video, {"dev": ["meet.G1"], "test": ["held.G2"]})
    monkeypatch.setitem(sys.modules, "motmetrics", None)
    monkeypatch.setattr(me, "_default_stack", lambda: pytest.fail("must not build the stack"))
    rc = tune.main(["--clips", str(root / "clips.csv"), "--split-file", str(root / "split.json"),
                    "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "motmetrics" in capsys.readouterr().err
