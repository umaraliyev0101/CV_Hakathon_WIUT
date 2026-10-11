"""`showroom-coach bench` (file-mode real-time factor, live fps) and `eval-meva` (a thin wrapper), plus serve's logging.

Everything here runs on the model-free fake stack and the shared synthetic videos; no real model is loaded.
"""
import json
import logging
import sys

import pytest

from showroom_coach import config
from showroom_coach.cli import build_parser, main
from showroom_coach.perception import pipeline
from showroom_coach.perception import stack as stacks


def run_bench(video, tmp_path, *extra):
    out = tmp_path / "bench.json"
    rc = main(["bench", "--video", str(video), "--fake", "--out", str(out), *extra])
    return rc, out


# ---------------------------------------------------------------------------------------------------------- bench

def test_bench_reports_file_and_live_modes_and_writes_json(synth_video, tmp_path, capsys):
    rc, out = run_bench(synth_video, tmp_path, "--frames", "12")
    assert rc == 0
    text = capsys.readouterr().out.lower()
    for needle in ("real-time factor", "perception", "detect", "render", "analysis", "fps", "people per frame"):
        assert needle in text, needle
    data = json.loads(out.read_text(encoding="utf-8"))
    f = data["file"]
    assert f["duration_s"] == pytest.approx(6.0, abs=0.3) and f["perception_wall_s"] > 0
    assert f["rtf"] == pytest.approx(f["duration_s"] / f["perception_wall_s"])
    assert set(f["stages_s"]) >= {"decode", "detect", "track", "pose", "embed"}
    assert f["render_s"] > 0 and f["analysis_s"] >= 0 and f["n_people"] >= 1
    live = data["live"]
    assert live["frames"] == 12 and live["fps_mean"] > 0 and live["fps_median"] > 0
    assert live["people_mean"] >= 1.0 and live["stack"] == "fake"
    assert data["fake"] is True and data["machine"]["cpu_count"] >= 1


def test_bench_file_mode_runs_perception_fresh_and_never_uses_the_cache(synth_video, tmp_path, monkeypatch):
    calls = []
    real = pipeline.run_perception
    monkeypatch.setattr(pipeline, "run_perception", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    monkeypatch.setattr(pipeline, "cached_perception", lambda *a, **k: pytest.fail("bench must not use the cache"))
    rc, _ = run_bench(synth_video, tmp_path, "--mode", "file")
    assert rc == 0 and len(calls) == 1
    assert calls[0].get("progress") is None and "crops_dir" not in calls[0]


def test_bench_mode_file_has_no_live_part_and_mode_live_no_file_part(synth_video, tmp_path):
    rc, out = run_bench(synth_video, tmp_path, "--mode", "file", "--no-render")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and "live" not in data and data["file"]["render_s"] is None
    rc, out = run_bench(synth_video, tmp_path, "--mode", "live", "--frames", "5")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and "file" not in data and data["live"]["frames"] == 5


def test_bench_live_uses_the_live_video_and_stops_at_the_end_of_a_short_one(synth_video, meet_video, tmp_path):
    rc, out = run_bench(synth_video, tmp_path, "--mode", "live", "--frames", "500", "--live-video", str(meet_video))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and data["live"]["video"].endswith("meet.avi")
    assert data["live"]["frames"] == 100                       # 20 s at 5 fps; asked for more than the clip has


def test_bench_max_seconds_limits_the_file_mode(meet_video, tmp_path):
    rc, out = run_bench(meet_video, tmp_path, "--mode", "file", "--no-render", "--max-seconds", "8")
    assert rc == 0 and json.loads(out.read_text(encoding="utf-8"))["file"]["duration_s"] == pytest.approx(8.0, abs=0.3)


def test_bench_picks_the_detector_of_each_mode_from_the_config(synth_video, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(stacks, "default_stack", lambda spec=None: (asked.append(spec), stacks.fake_stack())[1])
    monkeypatch.setattr(config, "DEFAULT_DETECTOR", "yolox_tiny@1")
    monkeypatch.setattr(config, "LIVE_DETECTOR", "yolox_s@2")
    out = tmp_path / "bench.json"
    assert main(["bench", "--video", str(synth_video), "--frames", "3", "--no-render", "--out", str(out)]) == 0
    assert asked == ["yolox_tiny@1", "yolox_s@2"]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["file"]["detector"] == "yolox_tiny@1" and data["live"]["detector"] == "yolox_s@2"


def test_bench_loads_the_model_stack_once_when_both_modes_use_the_same_detector(synth_video, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(stacks, "default_stack", lambda spec=None: (asked.append(spec), stacks.fake_stack())[1])
    monkeypatch.setattr(config, "DEFAULT_DETECTOR", "yolox_nano@2")
    monkeypatch.setattr(config, "LIVE_DETECTOR", "yolox_nano@2")
    assert main(["bench", "--video", str(synth_video), "--frames", "3", "--no-render",
                 "--out", str(tmp_path / "b.json")]) == 0
    assert asked == ["yolox_nano@2"]


def test_bench_errors(synth_video, tmp_path, capsys):
    assert main(["bench", "--video", str(tmp_path / "nope.mp4"), "--fake", "--out", str(tmp_path / "b.json")]) == 2
    assert "cannot read video" in capsys.readouterr().err.lower()
    bad = tmp_path / "bad.mp4"
    bad.write_text("x")
    assert main(["bench", "--video", str(bad), "--fake", "--out", str(tmp_path / "b.json")]) == 2
    assert "cannot read video" in capsys.readouterr().err.lower()
    assert main(["bench", "--video", str(synth_video), "--fake", "--frames", "0", "--out", str(tmp_path / "b.json")]) == 2
    assert main(["bench", "--video", str(synth_video), "--fake", "--max-seconds", "0",
                 "--out", str(tmp_path / "b.json")]) == 2
    assert main(["bench", "--video", str(synth_video), "--fake", "--mode", "live", "--live-video",
                 str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "b.json")]) == 2
    assert not (tmp_path / "b.json").exists()


def test_bench_reports_unavailable_models(synth_video, tmp_path, monkeypatch, capsys):
    from showroom_coach import models

    def unavailable(spec=None):
        raise models.ModelUnavailable("yolox_nano is not downloaded")

    monkeypatch.setattr(stacks, "default_stack", unavailable)
    assert main(["bench", "--video", str(synth_video), "--out", str(tmp_path / "b.json")]) == 1
    assert "fetch-models" in capsys.readouterr().err


def test_bench_default_output_is_runs_bench_bench_json(synth_video, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    assert main(["bench", "--video", str(synth_video), "--fake", "--mode", "file", "--no-render"]) == 0
    assert (tmp_path / "runs" / "bench" / "bench.json").is_file()


# ------------------------------------------------------------------------------------------------------ eval-meva

def test_eval_meva_hands_every_flag_to_the_evaluation_and_returns_its_exit_code(monkeypatch):
    from showroom_coach.evaluation import meva_eval
    seen = []
    monkeypatch.setattr(meva_eval, "main", lambda argv=None: (seen.append(list(argv)), 7)[1])
    argv = ["--split", "test", "--split-file", "x.json", "--dist", "2.5", "--min-dur", "6", "--out", "o.json"]
    assert main(["eval-meva", *argv]) == 7
    assert seen == [argv]
    assert main(["eval-meva"]) == 7 and seen[-1] == []


def test_eval_meva_help_is_the_evaluations_own(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval-meva", "--help"])
    assert exc.value.code == 0
    assert "--merge-thr" in capsys.readouterr().out


def test_eval_meva_is_listed_in_the_top_level_help(capsys):
    assert "eval-meva" in build_parser().format_help()


def test_eval_meva_usage_error_comes_back_as_exit_code_2(tmp_path, capsys):
    rc = main(["eval-meva", "--clips", str(tmp_path / "missing.csv"), "--split", "dev"])
    assert rc == 2 and capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------- serve

def test_serve_configures_logging_before_uvicorn_runs(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    order = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: order.append(("logging", kw)))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: order.append(("run", k)))
    assert main(["serve", "--fake", "--port", "8123"]) == 0
    assert [name for name, _ in order] == ["logging", "run"]
    kw = order[0][1]
    assert kw["level"] == logging.INFO
    assert kw["format"] == "%(asctime)s %(levelname)s %(name)s: %(message)s"
