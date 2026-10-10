import pytest

from showroom_coach import cli
from showroom_coach.perception import bench

GEOM = """\
- {'geom': {'g0': '10 20 50 120', 'id0': 1, 'id1': 7, 'keyframe': True, 'ts0': 100}}
- {'geom': {'g0': '12 20 52 120', 'id0': 2, 'id1': 7, 'keyframe': False, 'ts0': 101}}
- {'geom': {'g0': '300 40 340 90', 'id0': 1, 'id1': 8, 'keyframe': True, 'ts0': 100}}
- {'geom': {'g0': '5 5 15 25', 'id0': 1, 'id1': 9, 'keyframe': True, 'ts0': 100}}
- {'geom': {'g0': '1 1 2 2', 'id0': 1, 'id1': 7, 'keyframe': True, 'ts0': 999}}
"""
TYPES = """\
- {'types': {'cset3': {'person': 1.0}, 'id1': 7}}
- {'types': {'cset3': {'other': 1.0}, 'id1': 8}}
- {'types': {'cset3': {'person': 1.0}, 'id1': 9}}
"""


@pytest.fixture
def ann(tmp_path):
    g, t = tmp_path / "c.geom.yml", tmp_path / "c.types.yml"
    g.write_text(GEOM, encoding="utf-8")
    t.write_text(TYPES, encoding="utf-8")
    return g, t


def test_load_person_boxes_keeps_only_person_actors_at_wanted_frames(ann):
    g, t = ann
    got = bench._load_person_boxes(g, t, {100, 101})
    assert sorted(got) == [100, 101]
    assert sorted(got[100]) == [(5.0, 5.0, 15.0, 25.0), (10.0, 20.0, 50.0, 120.0)]  # actor 8 is 'other'
    assert got[101] == [(12.0, 20.0, 52.0, 120.0)]


def test_load_person_boxes_ignores_garbage_lines(tmp_path):
    g, t = tmp_path / "g.yml", tmp_path / "t.yml"
    g.write_text("not yaml at all\n- {'geom': {'g0': 'a b c d', 'id1': 1, 'ts0': 5}}\n\x00\x01\n", encoding="utf-8")
    t.write_text("- {'types': {'cset3': {'person': 1.0}, 'id1': 1}}\n", encoding="utf-8")
    assert bench._load_person_boxes(g, t, {5}) == {}


def test_sample_indices_even_and_in_range():
    idx = bench.sample_indices(9000, 60)
    assert len(idx) == 60 and idx == sorted(set(idx))
    assert idx[0] < 150 and idx[-1] > 8850 and idx[-1] < 9000
    assert bench.sample_indices(5, 60) == [0, 1, 2, 3, 4]  # never more frames than the clip has


def test_greedy_match_counts():
    gt = [(0, 0, 10, 10), (100, 100, 150, 200), (300, 300, 320, 340)]
    det = [(1, 1, 11, 11, 0.9), (100, 100, 150, 200, 0.8), (500, 500, 520, 540, 0.7), (2, 2, 12, 12, 0.6)]
    tp, fp, fn = bench.greedy_match(gt, [d[:4] for d in det], [d[4] for d in det])
    assert (tp, fp, fn) == (2, 2, 1)  # the second box over gt0 is a duplicate -> false positive


def test_greedy_match_empty():
    assert bench.greedy_match([], [], []) == (0, 0, 0)
    assert bench.greedy_match([(0, 0, 10, 10)], [], []) == (0, 0, 1)
    assert bench.greedy_match([], [(0, 0, 10, 10)], [0.9]) == (0, 1, 0)


def test_greedy_flags_and_height_buckets():
    gt = [(0, 0, 10, 30), (100, 100, 150, 220), (300, 300, 320, 440)]
    det = [(0, 0, 10, 30), (100, 100, 150, 220)]
    assert bench.greedy_flags(gt, det, [0.9, 0.8]) == [True, True, False]
    assert [bench.height_bucket(b[3] - b[1]) for b in gt] == ["20-40px", "80px+", "80px+"]
    assert bench.height_bucket(40.0) == "40-80px" and bench.height_bucket(79.9) == "40-80px"


def test_choose_detector_fastest_within_five_points():
    rows = [
        {"model": "a", "ms_median": 30.0, "recall": 0.60},
        {"model": "b", "ms_median": 50.0, "recall": 0.66},
        {"model": "c", "ms_median": 90.0, "recall": 0.71},
        {"model": "d", "ms_median": 20.0, "recall": 0.64},  # 7 points below best -> excluded
    ]
    assert bench.choose_detector(rows) == "b"  # best 0.71; within 5 points: b (0.66) and c; fastest is b
    rows[0]["recall"] = 0.67
    assert bench.choose_detector(rows) == "a"


def test_default_and_live_detectors_are_benchmarked_specs():
    from showroom_coach import config, models
    from showroom_coach.perception.detect import parse_detector_spec

    for spec in (config.DEFAULT_DETECTOR, config.LIVE_DETECTOR):
        assert spec in bench.CANDIDATES
        name, tiles = parse_detector_spec(spec)
        assert name in models.MODELS and tiles in (1, 2)


def test_candidates_include_tiled_variants():
    assert bench.CANDIDATES == ("yolox_nano", "yolox_tiny", "yolox_s", "rfdetr_nano",
                                "yolox_nano@2", "yolox_tiny@2", "yolox_s@2")


def _rows():
    def row(model, ms, recall, big):
        return {"model": model, "ms_median": ms, "recall": recall,
                "recall_by_gt_height": {"20-40px": {"n": 5, "recall": 0.0}, "40-80px": {"n": 5, "recall": 0.1},
                                        "80px+": {"n": 100, "recall": big}}}
    return [row("a", 20.0, 0.10, 0.50), row("b", 40.0, 0.14, 0.64), row("c", 90.0, 0.20, 0.70),
            row("d", 30.0, 0.19, 0.60)]


def test_choose_live_detector_uses_the_80px_bucket():
    rows = _rows()
    # best 80px+ recall 0.70 -> need >= 0.65 : only c; the all-sizes rule would pick d (0.19, fastest within 5 points)
    assert bench.choose_detector(rows) == "d"
    assert bench.choose_live_detector(rows) == "c"
    rows[1]["recall_by_gt_height"]["80px+"]["recall"] = 0.66  # b is now within 5 points and faster than c
    assert bench.choose_live_detector(rows) == "b"


def test_choose_live_detector_without_large_gt_falls_back_to_all_sizes_rule():
    rows = _rows()
    for r in rows:
        r["recall_by_gt_height"]["80px+"] = {"n": 0, "recall": None}
    assert bench.choose_live_detector(rows) == bench.choose_detector(rows)


def test_cli_bench_accepts_tiled_specs_and_rejects_bad_ones(tmp_path, capsys):
    for name in ("v.avi", "g.yml", "t.yml"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    base = ["bench-detectors", "--video", str(tmp_path / "v.avi"),
            "--geom", str(tmp_path / "g.yml"), "--types", str(tmp_path / "t.yml")]
    for bad in ("yolox_tiny@3", "yolox_tiny@", "ultralytics@2", "rtmpose_s@2"):
        assert cli.main(base + ["--models", bad]) == 2, bad
        assert "unknown detector" in capsys.readouterr().err
    # a valid spec passes validation and reaches the (here: undecodable) video
    assert cli.main(base + ["--models", "yolox_tiny@2", "--frames", "2"]) == 1
    assert "bench-detectors failed" in capsys.readouterr().err


def test_cli_registers_bench_detectors():
    args = cli.build_parser().parse_args(
        ["bench-detectors", "--video", "v.avi", "--geom", "g.yml", "--types", "t.yml", "--frames", "7"])
    assert args.frames == 7 and args.video == "v.avi" and callable(args.handler)


def test_cli_bench_missing_file_is_a_clean_error(tmp_path, capsys):
    rc = cli.main(["bench-detectors", "--video", str(tmp_path / "nope.avi"),
                   "--geom", str(tmp_path / "g"), "--types", str(tmp_path / "t")])
    assert rc == 2 and "nope.avi" in capsys.readouterr().err


def test_cli_bench_rejects_bad_frames_and_models(tmp_path, capsys):
    for name in ("v.avi", "g.yml", "t.yml"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    base = ["bench-detectors", "--video", str(tmp_path / "v.avi"),
            "--geom", str(tmp_path / "g.yml"), "--types", str(tmp_path / "t.yml")]
    assert cli.main(base + ["--frames", "0"]) == 2
    assert cli.main(base + ["--models", "ultralytics_yolo"]) == 2
    assert "unknown detector" in capsys.readouterr().err
