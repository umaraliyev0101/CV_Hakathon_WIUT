import json
import sys
import types

import pytest

from showroom_coach.cli import build_parser, main


def test_analyze_cli_writes_every_output(meet_video, tmp_path):
    roles = tmp_path / "roles.json"; roles.write_text(json.dumps({"roles": {"1": "staff"}, "names": {"1": "Aziz"}}))
    out = tmp_path / "out"
    assert main(["analyze", str(meet_video), "--out", str(out), "--fake", "--roles", str(roles),
                 "--cache", str(tmp_path / "cache")]) == 0
    for name in ["annotated.mp4", "timeline.png", "results.json", "events.csv", "scorecard.csv"]:
        assert (out / name).exists(), name
    assert list((out / "people").glob("*.jpg")) and list((out / "contact_sheets").glob("*.jpg"))


def test_analyze_cli_bad_video(tmp_path, capsys):
    bad = tmp_path / "bad.mp4"; bad.write_text("x")
    assert main(["analyze", str(bad), "--out", str(tmp_path / "o"), "--fake"]) == 2
    assert "cannot read video" in capsys.readouterr().err.lower()


def test_analyze_cli_missing_video(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o"), "--fake",
                 "--cache", str(tmp_path / "cache")]) == 2
    assert "cannot read video" in capsys.readouterr().err.lower()


def test_analyze_cli_summary_and_roles_applied(meet_video, tmp_path, capsys):
    roles = tmp_path / "roles.json"; roles.write_text(json.dumps({"roles": {"1": "staff"}, "names": {"1": "Aziz"}}))
    out = tmp_path / "out"
    assert main(["analyze", str(meet_video), "--out", str(out), "--fake", "--roles", str(roles), "--no-video",
                 "--cache", str(tmp_path / "cache")]) == 0
    text = capsys.readouterr().out
    for needle in ("people", "suggested", "events", "greet", "real-time factor", "total wall", "perception"):
        assert needle in text.lower(), needle
    assert not (out / "annotated.mp4").exists()
    res = json.loads((out / "results.json").read_text())
    assert res["roles"]["1"] == "staff" and res["names"]["1"] == "Aziz"
    assert res["preset"] == "short"
    assert "Aziz" in (out / "scorecard.csv").read_text()


def test_analyze_cli_second_run_uses_the_cache(meet_video, tmp_path, capsys):
    args = ["analyze", str(meet_video), "--fake", "--no-video", "--cache", str(tmp_path / "cache")]
    assert main(args + ["--out", str(tmp_path / "o1")]) == 0
    assert "saved" not in capsys.readouterr().out.lower()
    assert main(args + ["--out", str(tmp_path / "o2"), "--preset", "dealership"]) == 0
    text = capsys.readouterr().out.lower()
    assert "saved" in text                                   # perception came from the cache
    assert json.loads((tmp_path / "o2" / "results.json").read_text())["preset"] == "dealership"


@pytest.mark.parametrize("content", ["{not json", json.dumps([1, 2]), json.dumps({"roles": "staff"})])
def test_analyze_cli_bad_roles_file(meet_video, tmp_path, capsys, content):
    roles = tmp_path / "roles.json"; roles.write_text(content)
    assert main(["analyze", str(meet_video), "--out", str(tmp_path / "o"), "--fake", "--roles", str(roles),
                 "--cache", str(tmp_path / "cache")]) == 2
    assert "roles" in capsys.readouterr().err.lower()
    assert not (tmp_path / "o" / "results.json").exists()


def test_analyze_cli_missing_roles_file(meet_video, tmp_path, capsys):
    assert main(["analyze", str(meet_video), "--out", str(tmp_path / "o"), "--fake",
                 "--roles", str(tmp_path / "none.json"), "--cache", str(tmp_path / "cache")]) == 2
    assert "roles" in capsys.readouterr().err.lower()


def test_analyze_cli_rejects_unknown_preset_and_detector(meet_video, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["analyze", str(meet_video), "--out", str(tmp_path / "o"), "--fake", "--preset", "nope"])
    assert exc.value.code == 2
    capsys.readouterr()
    assert main(["analyze", str(meet_video), "--out", str(tmp_path / "o"), "--detector", "not_a_model",
                 "--cache", str(tmp_path / "cache")]) == 2
    assert "detector" in capsys.readouterr().err.lower()


def test_analyze_cli_rejects_bad_numbers(meet_video, tmp_path, capsys):
    for flag, value in (("--fps", "0"), ("--max-seconds", "-3")):
        assert main(["analyze", str(meet_video), "--out", str(tmp_path / "o"), "--fake", flag, value,
                     "--cache", str(tmp_path / "cache")]) == 2
        assert flag in capsys.readouterr().err


def test_analyze_parser_defaults():
    ns = build_parser().parse_args(["analyze", "v.mp4", "--out", "o"])
    assert (ns.preset, ns.fps, ns.max_seconds, ns.fake, ns.no_video) == ("short", 5.0, 300.0, False, False)
    assert ns.roles is None and ns.cache is None and ns.detector is None


# ---------------------------------------------------------------------------------------------------- serve

@pytest.fixture
def serve_fakes(monkeypatch):
    """A stand-in for showroom_coach.app.main and uvicorn.run that record how they were called."""
    calls = {}

    def create_app(**kw):
        calls["create_app"] = kw
        return "APP"

    module = types.ModuleType("showroom_coach.app.main")
    module.create_app = create_app
    monkeypatch.setitem(sys.modules, "showroom_coach.app.main", module)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(run=(app, kw)))
    return calls


def test_serve_defaults(serve_fakes, capsys):
    assert main(["serve"]) == 0
    assert serve_fakes["create_app"] == {}
    assert serve_fakes["run"] == ("APP", {"host": "127.0.0.1", "port": 8000, "log_level": "info"})
    assert "Showroom Coach running at http://127.0.0.1:8000" in capsys.readouterr().out


def test_serve_fake_host_port(serve_fakes, capsys):
    from showroom_coach.perception.stack import fake_stack
    assert main(["serve", "--fake", "--host", "0.0.0.0", "--port", "8123"]) == 0
    assert serve_fakes["create_app"] == {"stack_factory": fake_stack}
    assert serve_fakes["run"] == ("APP", {"host": "0.0.0.0", "port": 8123, "log_level": "info"})
    assert "http://0.0.0.0:8123" in capsys.readouterr().out


def test_printable_never_breaks_on_a_narrow_console():
    from showroom_coach.cli import _printable
    text = "nobody marked as staff — Азиз"
    assert _printable(text, "ascii") == "nobody marked as staff - ????"
    assert _printable(text, "cp437").startswith("nobody marked as staff - ")
    assert _printable(text, "utf-8") == "nobody marked as staff - Азиз"
    assert _printable(text, None).isascii() and _printable(text, "no-such-codec").isascii()


def test_summary_stays_one_screen_with_many_staff():
    from pathlib import Path
    from types import SimpleNamespace as NS

    from showroom_coach.cli import _summary
    from showroom_coach.types import StaffCard, StoreCard

    staff = [StaffCard(i, None, 1, 2.0, 1, 10.0, 10.0, 0, 0) for i in range(1, 13)]
    store = StoreCard(5, 5, 3, 0.6, 2.0, 1, 4.0, 0, 2)
    an = NS(preset="short", persons=list(range(12)), suggestions={}, roles={}, events=[], warnings=["w1"],
            scorecard=NS(staff=staff, store=store))
    res = NS(video=NS(duration=60.0, width=960, height=540, sample_fps=5.0), stack_name="fake",
             timings={"total": 30.0, "decode": 1.0})
    text = _summary(Path("v.mp4"), res, an, False, {"perception": 30.0, "analysis": 1.0}, 40.0, 2.0, 30.0, None, "",
                    "out", ["results.json"])
    lines = text.splitlines()
    assert len(lines) <= 24, text
    assert sum(1 for ln in lines if ln.startswith("  staff #")) <= 5
    assert "7 more" in text and "scorecard.csv" in text


def test_analyze_cli_cut_off_video_gives_partial_results_with_a_warning(meet_video, tmp_path, capsys):
    damaged = tmp_path / "damaged.avi"
    data = meet_video.read_bytes()
    damaged.write_bytes(data[: len(data) * 2 // 5])              # the file stops in the middle
    out = tmp_path / "o"
    assert main(["analyze", str(damaged), "--out", str(out), "--fake", "--no-video",
                 "--cache", str(tmp_path / "cache")]) == 0
    text = capsys.readouterr().out
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert 0 < res["duration"] < 0.9 * 20
    (warning,) = [w for w in res["warnings"] if w.startswith("video ended at")]
    assert warning.endswith(f"results cover the first {res['duration']:.0f} s")
    assert f"warning      {warning.replace('—', '-')}" in text                            # in the summary too
    assert f"video        {res['duration']:.1f} s analysed" in text


def test_analyze_cli_longer_audio_track(tmp_path, make_video, long_audio, capsys):
    mover = {"color": (0, 0, 255), "w": 40, "h": 100, "pos": lambda t: (20 + 20 * t, 60)}
    clip = make_video(tmp_path / "clip.avi", seconds=4, fps=10, size=(320, 240), movers=[mover])
    mkv = long_audio(clip, tmp_path / "clip.mkv", 12)
    out = tmp_path / "o"
    assert main(["analyze", str(mkv), "--out", str(out), "--fake", "--cache", str(tmp_path / "cache")]) == 0
    text = capsys.readouterr().out
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["duration"] == pytest.approx(4.0)
    assert ("video ended at 4 s of an expected 12 s — the file may be damaged or have a longer audio track; "
            "results cover the first 4 s") in res["warnings"]
    assert "warning      video ended at 4 s of an expected 12 s" in text
    assert (out / "annotated.mp4").is_file()


def test_analyze_cli_zero_frame_video_exits_2(tmp_path, capsys):
    import cv2
    zero = tmp_path / "zero.avi"
    cv2.VideoWriter(str(zero), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)).release()   # a header, no frames
    assert main(["analyze", str(zero), "--out", str(tmp_path / "o"), "--fake", "--no-video",
                 "--cache", str(tmp_path / "cache")]) == 2
    assert "cannot read video: zero.avi" in capsys.readouterr().err
