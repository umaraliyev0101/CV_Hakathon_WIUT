"""Rendering: the face-blurred H.264 video, the timeline image, contact sheets, thumbnails and the overlay drawing."""
import dataclasses
import re

import cv2
import numpy as np
import pytest

from showroom_coach import render
from showroom_coach.analysis import analyze, rescore
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack
from showroom_coach.render import (CUSTOMER_BGR, STAFF_BGR, H264Writer, contact_sheets, draw_overlay, face_region,
                                   render_video, save_thumbnails, timeline_png)
from showroom_coach.sources import VideoError
from showroom_coach.types import Event


def test_render_video_is_h264_and_playable(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"})
    out = render_video(meet_video, res, an, tmp_path / "annotated.mp4", fps_out=10)
    head = out.read_bytes()[:4096]
    assert b"ftyp" in head and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 150
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore").lower()
    assert fourcc in ("avc1", "h264")


def test_timeline_and_contact_sheets(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"})
    assert cv2.imread(str(timeline_png(an, tmp_path / "timeline.png"))) is not None
    sheets = contact_sheets(meet_video, res, an, tmp_path / "sheets")
    assert sheets and all(p.exists() for p in sheets)


def test_draw_overlay_colours():
    img = np.zeros((200, 300, 3), np.uint8)
    out = draw_overlay(img, [{"bbox": (10, 10, 60, 150), "role": "staff", "label": "S1", "state": "free", "wait_s": None}],
                       [], "t=0", None)
    assert tuple(out[80, 10]) == (230, 120, 20)


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def meet(meet_video):
    res = run_perception(meet_video, fake_stack())
    return res, rescore(analyze(res, "short"), res, roles={1: "staff"}, names={1: "Aziz"})


BLUE, GREEN, GREY = (255, 0, 0), (0, 255, 0), (128, 128, 128)
SIZE = (640, 360)
BOX_W, BOX_H, BOX_Y1, X0, SPEED = 70, 200, 80, 100.0, 100.0   # a person box moving right at 100 px/s
FPS_SRC = 30


def _box_x1(i):
    return int(round(X0 + SPEED * i / FPS_SRC))


def _write_clip(path, seconds=4.0, texture=True, seed=7):
    """One blue person box moving right. With `texture`, the box's face region (where the fake pose puts the nose,
    eyes and ears) is filled with pixel noise of two strongly saturated colours: the fake detector still sees one blob,
    and unblurred it would show up as a high-variance patch."""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(FPS_SRC), SIZE)
    assert writer.isOpened()
    stack = fake_stack()
    for i in range(int(round(seconds * FPS_SRC))):
        frame = np.full((SIZE[1], SIZE[0], 3), GREY, np.uint8)
        x1 = _box_x1(i)
        box = (x1, BOX_Y1, x1 + BOX_W, BOX_Y1 + BOX_H)
        frame[BOX_Y1:BOX_Y1 + BOX_H, x1:x1 + BOX_W] = BLUE
        if texture:
            kpts = stack.pose(frame, np.array([box], np.float64))[0]
            fx1, fy1, fx2, fy2 = face_region(box, kpts)
            fx1, fy1, fx2, fy2 = max(fx1, box[0]), max(fy1, box[1]), min(fx2, box[2]), min(fy2, box[3])
            noise = rng.random((fy2 - fy1, fx2 - fx1)) < 0.5
            frame[fy1:fy2, fx1:fx2] = np.where(noise[..., None], np.array(GREEN, np.uint8), np.array(BLUE, np.uint8))
        writer.write(frame)
    writer.release()
    return path


def _face_inner(x1):
    """Rows and columns well inside the face region of the box at left edge x1 (clear of the box outline and chips)."""
    return slice(BOX_Y1 + 6, BOX_Y1 + 28), slice(x1 + 20, x1 + 50)


def _std(img_bgr, rows, cols):
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[rows, cols].astype(np.float64).std())


def _video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


@pytest.fixture(scope="module")
def textured(tmp_path_factory):
    path = _write_clip(tmp_path_factory.mktemp("textured") / "face_texture.avi")
    res = run_perception(path, fake_stack())
    return path, res, analyze(res, "short")


# --------------------------------------------------------------------------------------------- privacy guards

def test_rendered_video_has_no_unblurred_face(textured, tmp_path):
    path, res, an = textured
    assert [p.pid for p in an.persons] == [1]
    out = render_video(path, res, an, tmp_path / "a.mp4", fps_out=10)
    frames = _video_frames(out)
    assert len(frames) >= 38
    stds = [_std(f, *_face_inner(_box_x1(3 * k))) for k, f in enumerate(frames)]       # output k is source frame 3k
    assert max(stds) < 12, stds


def test_the_guard_can_fail(textured, tmp_path, monkeypatch):
    """Control: with blurring switched off the same measurement sees the noise, so the guard above means something."""
    path, res, an = textured
    monkeypatch.setattr(render, "blur_faces", lambda frame, bboxes, kpts_list=None: frame.copy())
    frames = _video_frames(render_video(path, res, an, tmp_path / "a.mp4", fps_out=10))
    stds = [_std(f, *_face_inner(_box_x1(3 * k))) for k, f in enumerate(frames)]
    assert np.median(stds) > 30, stds


def test_contact_sheets_blur_faces(textured, tmp_path):
    path, res, an = textured
    an = dataclasses.replace(an, events=[Event("wait_alert", 2.0, customer=1, data={"wait_s": 12.0, "free_staff": []})])
    (sheet,) = contact_sheets(path, res, an, tmp_path / "sheets")
    img = cv2.imread(str(sheet))
    assert not (img[:240] == np.array(render.ALERT_BGR, np.uint8)).all(axis=2).any()      # the caption says it; no banner
    scale = 240 / SIZE[1]
    panel_w = img.shape[1] // 3
    for k, t in enumerate((0.0, 2.0, 4.0 - 1 / FPS_SRC)):                                # t-2 s clipped, t, t+2 s clipped
        x1 = _box_x1(int(round(t * FPS_SRC)))
        rows, cols = _face_inner(x1)
        r = slice(int(rows.start * scale), int(rows.stop * scale))
        c = slice(k * panel_w + int(cols.start * scale), k * panel_w + int(cols.stop * scale))
        assert _std(img, r, c) < 12, (k, _std(img, r, c))


def test_fallback_thumbnail_is_blurred(textured, tmp_path):
    path, res, an = textured
    res = dataclasses.replace(res, crops_dir=None)                  # no pipeline crops: cut the person from the video
    names = save_thumbnails(res, an, tmp_path / "people")
    assert names == {1: "1.jpg"}
    thumb = cv2.imread(str(tmp_path / "people" / "1.jpg"))
    pad = 0.05
    s = thumb.shape[0] / (BOX_H * (1 + 2 * pad))                      # the crop is box + 5 % each side, shrunk to <= 160 px
    rows = slice(int((pad * BOX_H + 6) * s), int((pad * BOX_H + 28) * s))
    cols = slice(int((pad * BOX_W + 20) * s), int((pad * BOX_W + 50) * s))
    assert thumb.shape[0] <= 160 and _std(thumb, rows, cols) < 12


# ---------------------------------------------------------------------------------------- render_video details

@pytest.fixture(scope="module")
def meet_rendered(meet, meet_video, tmp_path_factory):
    res, an = meet
    calls = []
    out = render_video(meet_video, res, an, tmp_path_factory.mktemp("rendered") / "a.mp4",
                       progress=lambda f, m: calls.append((f, m)))
    return out, calls, _video_frames(out)


def test_render_matches_processed_size_and_clip_length(meet, meet_rendered):
    res, an = meet
    out, _, frames = meet_rendered
    cap = cv2.VideoCapture(str(out))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (res.video.width, res.video.height)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
    assert len(frames) == pytest.approx(200, abs=2)
    assert [p.name for p in out.parent.iterdir()] == ["a.mp4"]                  # no temp file left behind


def test_render_draws_staff_blue_and_customer_orange(meet_rendered):
    frames = meet_rendered[2]
    mid = frames[100]                                                      # t = 10 s: both people are in view

    def count(colour, tol=40):
        return int((np.abs(mid.astype(int) - np.array(colour)).max(axis=2) <= tol).sum())

    assert count(STAFF_BGR) > 150 and count(CUSTOMER_BGR) > 150            # the box outlines (and chips)
    assert not np.array_equal(mid, frames[10])


def test_render_progress_is_monotonic_and_ends_at_one(meet_rendered):
    calls = meet_rendered[1]
    fracs = [f for f, _ in calls]
    assert len(calls) >= 3 and fracs == sorted(fracs) and fracs[-1] == 1.0 and all(isinstance(m, str) for _, m in calls)


def test_render_with_nobody_in_the_video(make_video, tmp_path):
    v = make_video(tmp_path / "empty.avi", seconds=3, movers=[])
    res = run_perception(v, fake_stack())
    an = analyze(res, "short")
    out = render_video(v, res, an, tmp_path / "a.mp4")
    assert len(_video_frames(out)) >= 25


def test_render_of_a_low_fps_source_keeps_real_time(make_video, tmp_path):
    mover = {"color": BLUE, "w": 80, "h": 220, "pos": lambda t: (100 + 50 * t, 100)}
    v = make_video(tmp_path / "low.avi", seconds=4, fps=5, size=(640, 360), movers=[mover])
    res = run_perception(v, fake_stack())
    out = render_video(v, res, analyze(res, "short"), tmp_path / "a.mp4", fps_out=10)
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(5.0, abs=0.1)      # not played back at double speed
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(20, abs=1)


def test_render_respects_the_perception_trim(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack(), max_seconds=5.0)
    out = render_video(meet_video, res, analyze(res, "short"), tmp_path / "a.mp4", fps_out=10)
    assert len(_video_frames(out)) == pytest.approx(50, abs=2)


def test_render_of_an_unreadable_video_fails_cleanly(textured, tmp_path):
    path, res, an = textured
    bad = tmp_path / "bad.avi"
    bad.write_bytes(path.read_bytes()[:200])                                      # not even a whole header
    with pytest.raises(VideoError):
        render_video(bad, res, an, tmp_path / "out" / "a.mp4")
    assert not list((tmp_path / "out").glob("*"))                                 # neither the file nor a temp file


def test_render_of_a_cut_off_video_draws_the_part_that_decodes(textured, tmp_path):
    path, res, an = textured
    cut = tmp_path / "cut.avi"
    cut.write_bytes(path.read_bytes()[: path.stat().st_size // 10])               # decoding stops early
    out = render_video(cut, res, an, tmp_path / "out" / "a.mp4", fps_out=10)
    assert 0 < len(_video_frames(out)) < len(_video_frames(render_video(path, res, an, tmp_path / "full.mp4",
                                                                         fps_out=10)))


# -------------------------------------------------------------------------------------------------- H264Writer

def test_h264_writer_publishes_only_a_finished_file(tmp_path):
    out = tmp_path / "w.mp4"
    out.write_bytes(b"old")
    with H264Writer(out, (64, 48), 10) as w:
        for i in range(20):
            w.write(np.full((48, 64, 3), i * 10, np.uint8))
        assert out.read_bytes() == b"old"                                    # still the previous file while writing
    assert b"ftyp" in out.read_bytes()[:64]
    assert [p.name for p in tmp_path.iterdir()] == ["w.mp4"]
    assert len(_video_frames(out)) == 20


def test_h264_writer_failure_leaves_nothing_behind(tmp_path):
    out = tmp_path / "w.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        with H264Writer(out, (64, 48), 10) as w:
            w.write(np.zeros((48, 64, 3), np.uint8))
            raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_h264_writer_refuses_to_publish_a_truncated_encode(tmp_path, monkeypatch):
    import imageio_ffmpeg

    real = imageio_ffmpeg.write_frames

    def encoder_that_loses_frames(*args, **kwargs):          # only the first 5 of the frames sent reach the file
        inner = real(*args, **kwargs)
        inner.send(None)
        n = 0
        try:
            while True:
                data = yield
                n += 1
                if n <= 5:
                    inner.send(data)
        finally:
            inner.close()

    out = tmp_path / "w.mp4"
    out.write_bytes(b"old")
    monkeypatch.setattr(imageio_ffmpeg, "write_frames", encoder_that_loses_frames)
    with pytest.raises(OSError, match=r"5 of 20 frames"):
        with H264Writer(out, (64, 48), 10) as w:
            for i in range(20):
                w.write(np.full((48, 64, 3), i * 10, np.uint8))
    assert out.read_bytes() == b"old" and [p.name for p in tmp_path.iterdir()] == ["w.mp4"]


def test_h264_writer_tolerates_one_frame_fewer_than_written(tmp_path, monkeypatch):
    out = tmp_path / "w.mp4"
    monkeypatch.setattr(render, "_probe_frame_count", lambda path: 19)
    with H264Writer(out, (64, 48), 10) as w:
        for i in range(20):
            w.write(np.full((48, 64, 3), i * 10, np.uint8))
    assert out.is_file()
    monkeypatch.setattr(render, "_probe_frame_count", lambda path: 18)
    with pytest.raises(OSError):
        with H264Writer(tmp_path / "x.mp4", (64, 48), 10) as w:
            for i in range(20):
                w.write(np.full((48, 64, 3), i * 10, np.uint8))
    assert not (tmp_path / "x.mp4").exists()


def test_h264_writer_rejects_a_wrong_frame_and_an_empty_video(tmp_path):
    with pytest.raises(ValueError):
        with H264Writer(tmp_path / "w.mp4", (64, 48), 10) as w:
            w.write(np.zeros((40, 64, 3), np.uint8))
    with pytest.raises(ValueError):
        with H264Writer(tmp_path / "w.mp4", (64, 48), 10):
            pass
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------------------------- draw_overlay

def _person(bbox, role="customer", label="C2", state="browsing", wait_s=None, **kw):
    return {"bbox": bbox, "role": role, "label": label, "state": state, "wait_s": wait_s, **kw}


def test_overlay_does_not_modify_its_input_and_draws_nothing_for_nobody():
    img = np.full((120, 200, 3), 77, np.uint8)
    before = img.copy()
    out = draw_overlay(img, [_person((20, 30, 80, 110))], [], "0:01 | short", None)
    assert np.array_equal(img, before) and not np.array_equal(out, before) and out is not img
    assert np.array_equal(draw_overlay(img, [], [], "", None), before)


def test_overlay_customer_box_is_orange():
    out = draw_overlay(np.zeros((200, 300, 3), np.uint8), [_person((10, 40, 60, 150))], [], "", None)
    assert tuple(out[100, 10]) == CUSTOMER_BGR and tuple(out[150, 35]) == CUSTOMER_BGR


def test_overlay_label_sits_above_the_box_and_falls_inside_when_there_is_no_room():
    img = np.zeros((200, 300, 3), np.uint8)
    above = draw_overlay(img, [_person((100, 80, 200, 190), label="C7")], [], "", None)
    assert (above[40:78, 100:140] != 0).any() and not (above[0:20] != 0).any()
    top = draw_overlay(img, [_person((100, 0, 200, 150), label="C7")], [], "", None)
    assert (top[3:16, 102:130] != 0).any()                                  # chip inside the box, not clipped away


def test_overlay_engaged_pair_gets_a_white_line_with_a_dark_outline():
    img = np.full((200, 400, 3), 100, np.uint8)
    people = [_person((20, 60, 70, 180), "staff", "S1", "busy", pid=1), _person((300, 60, 350, 180), label="C2", state="engaged", pid=2)]
    out = draw_overlay(img, people, [(1, 2, "engaged")], "", None)
    assert (out[120, 185] > 235).all()                                      # white core midway between the boxes
    assert (out[116:124, 185].max(axis=1) < 90).any()                       # dark outline just outside the white
    plain = draw_overlay(img, people, [], "", None)
    assert not (plain[120, 185] > 235).all()
    only_close = draw_overlay(img, people, [(1, 2, "close")], "", None)
    assert np.array_equal(only_close, plain)                                # only "engaged" pairs are joined


def test_overlay_pairs_by_index_when_people_have_no_pid():
    img = np.full((200, 400, 3), 100, np.uint8)
    people = [_person((20, 60, 70, 180), "staff", "S1", "busy"), _person((300, 60, 350, 180), label="C2")]
    out = draw_overlay(img, people, [(0, 1)], "", None)
    assert (out[120, 185] > 235).all()
    draw_overlay(img, people, [(0, 5), (7, 1)], "", None)                   # a pair pointing at nobody is ignored


def test_overlay_waiting_customer_gets_a_timer_above_the_label():
    img = np.zeros((220, 300, 3), np.uint8)

    def waiting(wait_s):
        return draw_overlay(img, [_person((100, 100, 160, 200), state="waiting", wait_s=wait_s)], [], "", None)

    diff_rows = np.flatnonzero((waiting(42.0) != waiting(None)).any(axis=(1, 2)))
    assert len(diff_rows) and diff_rows.max() < 100                          # only above the box, on top of the label
    assert not np.array_equal(waiting(42.0), waiting(75.0))                  # the text carries the time
    staff = draw_overlay(img, [_person((100, 100, 160, 200), "staff", "S1", "busy", wait_s=42.0)], [], "", None)
    assert np.array_equal(staff, draw_overlay(img, [_person((100, 100, 160, 200), "staff", "S1", "busy")], [], "", None))


def test_overlay_greeted_customers_get_a_tick_and_waiting_ones_do_not():
    img = np.zeros((220, 300, 3), np.uint8)

    def has_tick(state):
        out = draw_overlay(img, [_person((100, 100, 160, 200), state=state)], [], "", None)
        return bool((out == np.array(render.TICK_BGR, np.uint8)).all(axis=2).any())

    assert [has_tick(s) for s in ("waiting", "greeted", "engaged", "browsing")] == [False, True, True, True]
    staff = draw_overlay(img, [_person((100, 100, 160, 200), "staff", "S1", "busy")], [], "", None)
    assert not (staff == np.array(render.TICK_BGR, np.uint8)).all(axis=2).any()


def test_overlay_banner_spans_the_top_and_hud_moves_below_it():
    img = np.zeros((200, 300, 3), np.uint8)
    out = draw_overlay(img, [], [], "0:05 | short", "Long wait: C5 - free: Aziz")
    assert tuple(out[2, 295]) == render.ALERT_BGR and tuple(out[2, 3]) == render.ALERT_BGR
    assert (out[0, :] == np.array(render.ALERT_BGR)).all(axis=1).all()
    no_alert = draw_overlay(img, [], [], "0:05 | short", None)
    assert tuple(no_alert[2, 295]) == (0, 0, 0) and (no_alert[2:20, 2:60] != 0).any()      # HUD at the top left
    assert (out[0:2, 0:300] != no_alert[0:2, 0:300]).any()


def test_overlay_survives_odd_text_and_boxes():
    img = np.zeros((120, 200, 3), np.uint8)
    draw_overlay(img, [_person((10, 10, 90, 110), "staff", "S1 Zoë Хабиб — x" * 4, "busy"),
                       _person((float("nan"), 0, 5, 5)), _person((-30, -30, 500, 500), label=""),
                       _person((50, 50, 50, 50))], [], "µ · ü", "—" * 80)


def test_ascii_text_folds_accents_and_marks_the_rest():
    assert render._ascii("Zoë Ünal") == "Zoe Unal" and render._ascii("Хаб—") == "????"
    assert render._ascii("a\tb") == "a?b"


# ------------------------------------------------------------------------------------- the scene (who is where)

@pytest.fixture(scope="module")
def plain_clip(tmp_path_factory):
    path = _write_clip(tmp_path_factory.mktemp("plain") / "plain.avi", seconds=6.0, texture=False)
    res = run_perception(path, fake_stack())
    return path, res, analyze(res, "short")


def _track_times(res, an, pid=1):
    return res.frame_times[res.obs_frame[an.obs_pid == pid]]


def test_scene_interpolates_between_sampled_frames(plain_clip):
    path, res, an = plain_clip
    ts = _track_times(res, an)
    scene = render._Scene(res, an)
    a, b = float(ts[4]), float(ts[5])
    (p,) = scene.people((a + b) / 2)
    assert p["bbox"][0] == pytest.approx(X0 + SPEED * (a + b) / 2, abs=1.5)
    assert p["bbox"][2] - p["bbox"][0] == pytest.approx(BOX_W, abs=1e-6)
    (p,) = scene.people(a)
    assert p["bbox"][0] == pytest.approx(X0 + SPEED * a, abs=1.0)
    assert p["kpts"].shape == (17, 3) and p["pid"] == 1 and p["label"] == "C1"


def _without_frames(res, an, frames):
    pid = an.obs_pid.copy()
    pid[np.isin(res.obs_frame, frames)] = 0
    return dataclasses.replace(an, obs_pid=pid)


def test_scene_rejects_an_analysis_of_another_video(plain_clip):
    path, res, an = plain_clip
    with pytest.raises(ValueError, match="does not belong"):
        render._Scene(res, dataclasses.replace(an, obs_pid=an.obs_pid[:-1]))


def test_scene_gap_rules(plain_clip):
    path, res, an = plain_clip
    kept_frames = np.unique(res.obs_frame[an.obs_pid == 1])
    f = kept_frames[5:9]
    t = lambda k: float(res.frame_times[f[k]])
    # one sampled frame missing: the neighbours are 0.4 s apart, so the person is still interpolated
    one = render._Scene(res, _without_frames(res, an, [f[1]]))
    (p,) = one.people(t(0) + 0.2)
    assert p["bbox"][0] == pytest.approx(X0 + SPEED * (t(0) + 0.2), abs=1.5)
    # two missing: 0.6 s apart. Within 0.25 s of an observation the person is shown at that observation ...
    two = render._Scene(res, _without_frames(res, an, [f[1], f[2]]))
    (p,) = two.people(t(0) + 0.1)
    assert p["bbox"][0] == pytest.approx(X0 + SPEED * t(0), abs=1.0)
    # ... and otherwise not drawn at all
    assert two.people(t(0) + 0.3) == []
    # the undrawn person's face is still blurred at the nearest observation (held)
    (held,) = two.held(t(0) + 0.3, drawn=set())
    assert held[0][0] == pytest.approx(X0 + SPEED * t(0), abs=1.0)
    assert two.held(t(0) + 0.3, drawn={1}) == []
    assert two.held(t(0) + 30.0, drawn=set()) == []


def test_scene_blur_covers_raw_detections_of_both_neighbouring_sampled_frames(plain_clip):
    path, res, an = plain_clip
    nobody = dataclasses.replace(an, obs_pid=np.zeros_like(an.obs_pid))     # nothing tracked: only raw boxes remain
    scene = render._Scene(res, nobody)
    frame = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    rng = np.random.default_rng(0)
    frame[:] = rng.integers(0, 255, frame.shape, dtype=np.uint8)
    t = float(res.frame_times[10]) + 0.1
    boxes = scene.raw_near(t)
    assert len(boxes) == 2                                                  # the person in the frame before and after t
    out = scene.blur(frame, t, [])
    for x1, y1, x2, y2 in boxes:
        region = out[int(y1):int(y1 + 0.18 * (y2 - y1)), int(x1) + 2:int(x2) - 2]
        assert region.astype(float).std() < 25 < frame[int(y1):int(y1 + 0.18 * (y2 - y1))].astype(float).std()
    assert len(scene.raw_near(float(res.frame_times[10]))) == 1             # exactly on a sampled frame: just that one


def test_scene_states_from_events(meet):
    res, an = meet
    idx = render._Scene(res, an).events
    by = {e.type: e for e in an.events if e.type in ("enter", "greet", "engagement", "leave")}
    enter, eng = by["enter"], by["engagement"]
    mid = (eng.t + eng.t_end) / 2
    assert idx.state_of(2, "customer", enter.t)[0] in ("waiting", "greeted")
    assert idx.state_of(2, "customer", mid)[0] == "engaged"
    assert idx.state_of(1, "staff", mid) == ("busy", None)
    assert idx.state_of(1, "staff", 0.5) == ("free", None)
    assert idx.state_of(2, "customer", eng.t_end + 0.5)[0] == "browsing"
    assert idx.pairs(0.0, {1, 2}) == [] and idx.pairs(mid, {1, 2}) == [(1, 2, "engaged")]
    assert idx.pairs(mid, {1}) == []                                        # a pair needs both people on screen


def test_scene_waiting_timer_counts_from_a_known_entry():
    idx = render._EventIndex([
        Event("enter", 10.0, customer=5, data={"entry_known": True}),
        Event("enter", 11.0, customer=6, data={"entry_known": False}),
        Event("greet", 25.0, customer=5, staff=1, data={"time_to_greet": 15.0}),
        Event("leave", 40.0, customer=5, data={}),
        Event("enter", 50.0, customer=5, data={"entry_known": True}),           # a second visit starts a new wait
    ], duration=60.0, names={})
    assert idx.state_of(5, "customer", 12.5) == ("waiting", pytest.approx(2.5))
    assert idx.state_of(6, "customer", 12.5) == ("waiting", None)           # entry unknown: no timer
    assert idx.state_of(5, "customer", 30.0) == ("greeted", None)
    assert idx.state_of(5, "customer", 25.0)[0] == "greeted"
    assert idx.state_of(5, "customer", 52.0) == ("waiting", pytest.approx(2.0))
    assert idx.state_of(9, "customer", 5.0) == ("waiting", None)            # no visit on record: plain, no timer


def test_scene_banner_text_and_duration():
    idx = render._EventIndex([
        Event("wait_alert", 20.0, customer=5, data={"wait_s": 61.0, "free_staff": [1, 2]}),
        Event("wait_alert", 40.0, customer=7, data={"wait_s": 61.0, "free_staff": []}),
        Event("walkout", 50.0, customer=6, data={"free_staff": [1]}),
    ], duration=60.0, names={1: "Aziz", 2: "Bobur"})
    assert idx.alert(19.9) is None
    assert idx.alert(20.0) == "Long wait: C5 - free: Aziz, Bobur"
    assert idx.alert(22.9) == "Long wait: C5 - free: Aziz, Bobur" and idx.alert(23.1) is None
    assert idx.alert(41.0) == "Long wait: C7 - free: none"
    assert idx.alert(51.0) == "Walk-out: C6"


# ---------------------------------------------------------------------------------------------- timeline image

def test_timeline_is_1200_wide_and_uses_the_state_colours(meet, tmp_path):
    res, an = meet
    img = cv2.imread(str(timeline_png(an, tmp_path / "t.png")))
    assert img.shape[1] == 1200 and img.shape[0] > 150
    plot = img[: img.shape[0] - render.TIMELINE_LEGEND_H]                      # the legend shows every colour: leave it out

    def has(colour):
        return bool((plot == np.array(colour, np.uint8)).all(axis=2).any())

    assert has(render.PRESENT_BGR) and has(STAFF_BGR) and has(CUSTOMER_BGR) and has(render.GREET_BGR)


def test_timeline_marks_alerts_in_red_and_respects_width(meet, tmp_path):
    res, an = meet
    quiet = dataclasses.replace(an, events=[e for e in an.events if e.type not in ("wait_alert", "walkout")])

    def red(analysis, name):
        img = cv2.imread(str(timeline_png(analysis, tmp_path / name, width=800)))
        assert img.shape[1] == 800
        return int((img == np.array(render.ALERT_BGR, np.uint8)).all(axis=2).sum())

    base = red(quiet, "t0.png")                                                # the legend has red markers too
    loud = dataclasses.replace(quiet, events=quiet.events + [
        Event("wait_alert", 12.0, customer=2, data={"wait_s": 8.0, "free_staff": [1]}),
        Event("walkout", 18.0, customer=2, data={"free_staff": [1]})])
    assert red(loud, "t1.png") > base + 40                                     # a triangle and a cross more


def test_timeline_rows_are_sorted_by_first_appearance(meet):
    res, an = meet
    late, early = dataclasses.replace(an.persons[0], pid=7, t_first=9.0), dataclasses.replace(an.persons[1], pid=3, t_first=2.0)
    assert [p.pid for p in render._timeline_order(dataclasses.replace(an, persons=[late, early]))] == [3, 7]


def test_timeline_tick_spacing_depends_on_the_clip_length():
    assert render._tick_step(60.0) == 10.0 and render._tick_step(120.0) == 10.0
    assert render._tick_step(121.0) == 60.0 and render._tick_step(300.0) == 60.0


def test_timeline_of_an_empty_and_of_a_long_analysis(meet, tmp_path, make_video):
    v = make_video(tmp_path / "empty.avi", seconds=3, movers=[])
    empty = analyze(run_perception(v, fake_stack()), "short")
    img = cv2.imread(str(timeline_png(empty, tmp_path / "empty.png")))
    assert img.shape[1] == 1200 and img.shape[0] >= 80
    res, an = meet
    long = dataclasses.replace(an, duration=290.0)
    assert cv2.imread(str(timeline_png(long, tmp_path / "long.png"))) is not None


# --------------------------------------------------------------------------------------------- contact sheets

def _many_events():                                                          # all inside the 4 s textured clip
    ev = [Event("greet", 1.0 + i * 0.1, customer=1, staff=2, data={"time_to_greet": 3.0, "via": "direct"}) for i in range(20)]
    ev += [Event("engagement", 1.5 + i * 0.1, customer=1, staff=2, t_end=3.0 + i * 0.05) for i in range(10)]
    ev += [Event("wait_alert", 0.5 + i * 0.1, customer=1, data={"wait_s": 12.0, "free_staff": [2]}) for i in range(6)]
    ev += [Event("walkout", 2.5 + i * 0.1, customer=1, data={"free_staff": []}) for i in range(3)]
    ev += [Event("enter", 0.5, customer=1, data={"entry_known": True}), Event("leave", 3.5, customer=1)]
    return ev


def test_contact_sheets_pick_a_balanced_dozen_and_name_files(textured, tmp_path):
    path, res, an = textured
    an = dataclasses.replace(an, events=sorted(_many_events(), key=lambda e: e.t))
    sheets = contact_sheets(path, res, an, tmp_path / "sheets")
    assert len(sheets) == 12 and all(p.parent == tmp_path / "sheets" for p in sheets)
    names = [p.name for p in sheets]
    assert all(re.fullmatch(r"\d{2}_(greet|engagement|wait_alert|walkout)_\d+s\.jpg", n) for n in names)
    assert [n[:2] for n in names] == [f"{i:02d}" for i in range(1, 13)]
    kinds = [re.fullmatch(r"\d{2}_(\w+)_\d+s\.jpg", n).group(1) for n in names]
    assert {k: kinds.count(k) for k in set(kinds)} == {"greet": 3, "engagement": 3, "wait_alert": 3, "walkout": 3}
    assert len(contact_sheets(path, res, an, tmp_path / "few", max_events=2)) == 2
    assert contact_sheets(path, res, an, tmp_path / "zero", max_events=0) == []


def test_contact_sheet_layout_and_caption(textured, tmp_path):
    path, res, an = textured
    an = dataclasses.replace(an, names={1: "Aziz"}, events=[
        Event("greet", 2.0, customer=1, staff=1, data={"time_to_greet": 7.4, "via": "direct"})])
    (sheet,) = contact_sheets(path, res, an, tmp_path / "s")
    assert sheet.name == "01_greet_2s.jpg"
    img = cv2.imread(str(sheet))
    panel_w = round(res.video.width * 240 / res.video.height)
    assert img.shape[1] == 3 * panel_w and img.shape[0] == 240 + render.SHEET_CAPTION_H
    bar = img[240:]
    assert bar.mean() < 90 and (bar > 200).any()                              # dark caption bar with light text
    assert not np.array_equal(img[:240, :panel_w], img[:240, panel_w:2 * panel_w])      # three different moments
    assert not np.array_equal(img[:240, panel_w:2 * panel_w], img[:240, 2 * panel_w:])
    assert render._caption(an.events[0], render._EventIndex(an.events, 4.0, {1: "Aziz"})) ==         "Greet at 0:02: C1 greeted by Aziz after 7 s"


def _blue_left_edge(img_bgr):
    """Column of the left-most strongly blue pixel (the person's fill), or None."""
    blue = (img_bgr[..., 0] > 200) & (img_bgr[..., 1] < 60) & (img_bgr[..., 2] < 60)
    cols = np.flatnonzero(blue.any(axis=0))
    return int(cols[0]) if len(cols) else None


@pytest.fixture(scope="module")
def slow_clip(tmp_path_factory):
    """10 s at only 2 fps (frames at 0, 0.5, ..., 9.5 s), 640x360; the blue person's left edge is 50 + 50 t px."""
    path = tmp_path_factory.mktemp("slow") / "slow.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, SIZE)
    assert writer.isOpened()
    for i in range(20):
        frame = np.full((SIZE[1], SIZE[0], 3), GREY, np.uint8)
        x1 = int(round(50 + 50 * i / 2.0))
        frame[100:320, x1:x1 + 80] = BLUE
        writer.write(frame)
    writer.release()
    res = run_perception(path, fake_stack())
    return path, res, analyze(res, "short")


def test_grab_frames_returns_the_nearest_frame_even_when_the_source_is_slow(slow_clip):
    path, res, _ = slow_clip
    targets = [0.0, 2.2, 4.7, 9.9]                     # the source has frames only every 0.5 s: no frame is "near" 2.2
    got = render._grab_frames(path, res, targets)
    assert {tg: got[tg][0] for tg in targets} == {0.0: 0.0, 2.2: 2.0, 4.7: 4.5, 9.9: 9.5}
    assert render._grab_frames(path, res, [2.25])[2.25][0] in (2.0, 2.5)     # the review's repro (it returned t = 9.5)
    for tg in targets:
        t, frame = got[tg]
        assert _blue_left_edge(frame) == pytest.approx(50 + 50 * t, abs=2)          # and it really is that moment's picture


def test_grab_frames_is_independent_of_the_order_of_the_targets_and_stops_early(textured, monkeypatch):
    path, res, _ = textured                             # 4 s at 30 fps, decoded at 10 fps = 40 frames
    seen = []
    real = render.iter_frames
    monkeypatch.setattr(render, "iter_frames", lambda *a, **k: (seen.append(1) or f for f in real(*a, **k)))
    got = render._grab_frames(path, res, [1.0, 0.5])
    assert {tg: got[tg][0] for tg in (0.5, 1.0)} == pytest.approx({0.5: 0.5, 1.0: 1.0}, abs=1 / 30)
    assert len(seen) <= 14                              # it stopped one frame after the last target, not at the end


def test_contact_sheet_panels_of_a_slow_source_show_the_right_moments(slow_clip, tmp_path):
    path, res, an = slow_clip
    an = dataclasses.replace(an, events=[Event("wait_alert", 4.2, customer=1, data={"wait_s": 12.0, "free_staff": []})])
    (sheet,) = contact_sheets(path, res, an, tmp_path / "s")
    img = cv2.imread(str(sheet))
    panel_w, scale = img.shape[1] // 3, 240 / SIZE[1]
    for k, t in enumerate((2.0, 4.0, 6.0)):          # the frames nearest t-2 = 2.2, t = 4.2 and t+2 = 6.2 (not the last frame)
        edge = _blue_left_edge(img[:240, k * panel_w:(k + 1) * panel_w])
        assert edge == pytest.approx((50 + 50 * t) * scale, abs=4), (k, edge)


def test_contact_sheet_captions_for_every_type():
    idx = render._EventIndex([], 60.0, {3: "Aziz"})
    cap = lambda e: render._caption(e, idx)
    assert cap(Event("engagement", 65.0, customer=2, staff=3, t_end=95.0)) == "Engagement 1:05-1:35: C2 with Aziz (30 s)"
    assert cap(Event("wait_alert", 20.0, customer=2, data={"wait_s": 61.4, "free_staff": [3, 4]})) ==         "Long wait at 0:20: C2 waiting 61 s, free staff: Aziz, S4"
    assert cap(Event("walkout", 20.0, customer=2, data={"free_staff": []})) ==         "Walk-out at 0:20: C2 left without a greet, free staff: none"
    assert cap(Event("greet", 5.0, customer=2, staff=3, data={"time_to_greet": None})) == "Greet at 0:05: C2 greeted by Aziz"


def test_contact_sheet_files_reads_type_and_time_from_the_names(tmp_path):
    for name in ("02_walkout_31s.jpg", "01_greet_9s.jpg", "10_wait_alert_120s.jpg", "notes.txt", "1_greet_3s.jpg",
                 "03_party_5s.jpg"):
        (tmp_path / name).write_bytes(b"x")
    got = render.contact_sheet_files(tmp_path)
    assert [g["name"] for g in got] == ["01_greet_9s.jpg", "02_walkout_31s.jpg", "10_wait_alert_120s.jpg"]
    assert got[0] == {"name": "01_greet_9s.jpg", "type": "greet", "t": 9.0, "caption": "Greet at 0:09"}
    assert got[2] == {"name": "10_wait_alert_120s.jpg", "type": "wait_alert", "t": 120.0,
                      "caption": "Long wait at 2:00"}
    assert render.contact_sheet_files(tmp_path / "missing") == []


def test_contact_sheets_ignore_other_events_clip_times_and_drop_stale_sheets(textured, tmp_path):
    path, res, an = textured
    only_enter = dataclasses.replace(an, events=[Event("enter", 1.0, customer=1, data={"entry_known": True}),
                                                  Event("leave", 3.0, customer=1), Event("party", 1.0)])
    assert contact_sheets(path, res, only_enter, tmp_path / "none") == []
    edge = dataclasses.replace(an, events=[Event("greet", 0.2, customer=1, staff=2, data={"time_to_greet": None}),
                                           Event("walkout", 3.95, customer=1, data={"free_staff": [2]})])
    out = tmp_path / "edge"
    sheets = contact_sheets(path, res, edge, out)
    assert len(sheets) == 2 and all(cv2.imread(str(p)) is not None for p in sheets)
    (out / "other.txt").write_text("keep me")
    again = contact_sheets(path, res, dataclasses.replace(edge, events=edge.events[:1]), out)    # fewer events now
    assert [p.name for p in again] == ["01_greet_0s.jpg"]
    assert sorted(p.name for p in out.iterdir()) == ["01_greet_0s.jpg", "other.txt"]               # the old walkout sheet is gone


# ------------------------------------------------------------------------------------------------ thumbnails

def test_save_thumbnails_copies_the_pipeline_crops(textured, tmp_path):
    path, _, _ = textured
    res = run_perception(path, fake_stack(), crops_dir=tmp_path / "crops")
    an = analyze(res, "short")
    names = save_thumbnails(res, an, tmp_path / "people")
    assert names == {1: "1.jpg"}
    (p,) = an.persons
    assert (tmp_path / "people" / "1.jpg").read_bytes() == (tmp_path / "crops" / p.thumb).read_bytes()
    assert save_thumbnails(res, an, tmp_path / "people") == names                      # again: replaced, not an error


def test_save_thumbnails_cuts_missing_ones_from_the_video(meet, meet_video, tmp_path):
    res, an = meet
    res = dataclasses.replace(res, crops_dir=None)
    names = save_thumbnails(res, an, tmp_path / "people")
    assert set(names) == {1, 2}
    for pid in (1, 2):
        img = cv2.imread(str(tmp_path / "people" / f"{pid}.jpg"))
        assert img is not None and 20 < img.shape[0] <= 160 and img.shape[1] < img.shape[0]
    mix = cv2.imread(str(tmp_path / "people" / "1.jpg")).reshape(-1, 3).mean(axis=0)
    assert mix[0] > mix[2]                                                     # person 1 is the blue one


def _write_ramp_clip(path, seconds=3.0, fps=10):
    """Background grey level 20 + 8 i on source frame i; one blue person box that grows, so the best observation is the
    last tracked frame and its crop's corner shows which frame was cut."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), SIZE)
    assert writer.isOpened()
    for i in range(int(round(seconds * fps))):
        frame = np.full((SIZE[1], SIZE[0], 3), 20 + 8 * i, np.uint8)
        frame[60:60 + 100 + 3 * i, 300:370] = BLUE
        writer.write(frame)
    writer.release()
    return path


def test_fallback_thumbnail_is_cut_from_the_frame_the_observation_came_from(tmp_path, monkeypatch):
    path = _write_ramp_clip(tmp_path / "ramp.avi")
    res = run_perception(path, fake_stack())
    res = dataclasses.replace(res, crops_dir=None)
    an = analyze(res, "short")
    idx = np.flatnonzero(an.obs_pid == 1)
    b = res.obs_bbox[idx]
    best = idx[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * res.obs_score[idx])]
    src_i = int(res.src_frame_idx[res.obs_frame[best]])
    assert src_i > 10                                            # a late frame: a seek that is off shows in the grey level
    calls = []
    real = render.iter_frames
    monkeypatch.setattr(render, "iter_frames", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert save_thumbnails(res, an, tmp_path / "people") == {1: "1.jpg"}
    assert calls, "thumbnails must be cut by decoding sequentially, not by seeking"
    thumb = cv2.imread(str(tmp_path / "people" / "1.jpg"))
    assert float(thumb[1:4, 1:4].mean()) == pytest.approx(20 + 8 * src_i, abs=4)    # the padding around the box is background


def test_fallback_thumbnail_is_skipped_when_the_frame_is_not_there(tmp_path, caplog):
    path = _write_ramp_clip(tmp_path / "ramp.avi")
    res = run_perception(path, fake_stack())
    an = analyze(res, "short")
    res = dataclasses.replace(res, crops_dir=None, src_frame_idx=res.src_frame_idx + 1000)     # no such source frames
    with caplog.at_level("WARNING", logger="showroom_coach.render"):
        assert save_thumbnails(res, an, tmp_path / "people") == {}
    assert "person 1" in caplog.text and not list((tmp_path / "people").glob("*.jpg"))


def test_save_thumbnails_skips_people_it_cannot_cut(meet, tmp_path):
    res, an = meet
    res = dataclasses.replace(res, crops_dir=None, video=dataclasses.replace(res.video, path=str(tmp_path / "gone.avi")))
    assert save_thumbnails(res, an, tmp_path / "people") == {}
