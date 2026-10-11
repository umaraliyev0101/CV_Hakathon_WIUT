import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from showroom_coach.sources import StreamEnd, VideoError, decode_jpeg, iter_frames, probe, resize_max_width

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "meva_a_60s.mp4"


def _box(frame, bgr):
    """Bounding box (x1, y1, x2, y2) of pixels close to a pure colour, or None."""
    close = np.abs(frame.astype(int) - np.array(bgr)).max(axis=2) < 60
    ys, xs = np.nonzero(close)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def test_probe_and_resize(synth_video):
    info = probe(synth_video)
    assert (info.src_width, info.src_height) == (1280, 720)
    assert info.width == 960 and info.height == 540 and info.scale == pytest.approx(0.75)
    assert info.src_fps == pytest.approx(30, abs=0.5) and not info.trimmed


def test_sampling_5fps(synth_video):
    frames = list(iter_frames(synth_video, sample_fps=5.0))
    ts = [t for _, t, _ in frames]
    assert len(frames) == 30
    assert np.allclose(np.diff(ts), 0.2, atol=0.034)
    assert frames[0][2].shape == (540, 960, 3)


def test_trimming(synth_video):
    info = probe(synth_video, max_seconds=2.0)
    assert info.trimmed and info.duration == pytest.approx(2.0, abs=0.05)
    assert len(list(iter_frames(synth_video, sample_fps=5.0, max_seconds=2.0))) == 10


def test_garbage_file_raises(tmp_path):
    p = tmp_path / "x.mp4"; p.write_text("not a video")
    with pytest.raises(VideoError):
        probe(p)


def test_decode_jpeg_roundtrip():
    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))
    assert decode_jpeg(buf.tobytes()).shape == (540, 960, 3)
    with pytest.raises(VideoError):
        decode_jpeg(b"nope")


def test_probe_fields(synth_video):
    info = probe(synth_video)
    assert info.src_frames == 180
    assert info.src_duration == pytest.approx(6.0, abs=0.05)
    assert info.duration == pytest.approx(info.src_duration) and not info.trimmed
    assert info.sample_fps == 5.0 and info.max_seconds == 300.0
    assert info.path == str(synth_video)


def test_frame_indices_and_times_follow_source_fps(synth_video):
    frames = list(iter_frames(synth_video, sample_fps=5.0))
    assert [i for i, _, _ in frames[:4]] == [0, 6, 12, 18]
    assert frames[0][1] == 0.0
    assert all(t == pytest.approx(i / 30.0) for i, t, _ in frames)
    assert all(t < 6.0 for _, t, _ in frames)


def test_sampling_follows_requested_rate_and_never_exceeds_source(synth_video):
    assert len(list(iter_frames(synth_video, sample_fps=2.0))) == 12
    assert len(list(iter_frames(synth_video, sample_fps=1000.0))) == 180   # every frame, no duplicates


def test_frames_carry_the_content(synth_video):
    """Red walker is there from t=0 (top-left); the blue stander only appears at t=1 s (MJPG is lossy: match by colour)."""
    frames = {i: f for i, _, f in iter_frames(synth_video, sample_fps=5.0)}
    first, later = frames[0], frames[36]     # t = 0 s and t = 1.2 s
    red, blue = (0, 0, 255), (255, 0, 0)
    assert _box(first, red) is not None and _box(first, blue) is None
    assert _box(later, blue) is not None


def test_small_frames_are_not_upscaled(tmp_path, make_video):
    path = make_video(tmp_path / "small.avi", seconds=1, fps=10, size=(640, 360), movers=[])
    info = probe(path)
    assert (info.width, info.height, info.scale) == (640, 360, 1.0)
    frame = next(iter_frames(path, sample_fps=5.0))[2]
    assert frame.shape == (360, 640, 3)


def test_even_dimensions_for_odd_aspect(tmp_path, make_video):
    path = make_video(tmp_path / "odd.avi", seconds=1, fps=10, size=(1920, 1072), movers=[])
    info = probe(path)
    assert (info.width, info.height) == (960, 536)
    assert info.width % 2 == 0 and info.height % 2 == 0


def test_resize_max_width_keeps_aspect_and_even_dims():
    out = resize_max_width(np.zeros((1080, 1920, 3), np.uint8), 960)
    assert out.shape == (540, 960, 3)
    out = resize_max_width(np.zeros((481, 641, 3), np.uint8), 960)     # odd sizes are made even, never upscaled
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0 and out.shape[1] <= 641
    out = resize_max_width(np.zeros((1000, 3001, 3), np.uint8), 960)
    assert out.shape[1] <= 960 and out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0
    assert out.shape[0] / out.shape[1] == pytest.approx(1000 / 3001, abs=0.01)


def test_iter_frames_garbage_and_missing_raise(tmp_path):
    p = tmp_path / "x.mp4"; p.write_text("not a video")
    with pytest.raises(VideoError):
        list(iter_frames(p))
    with pytest.raises(VideoError, match="cannot read video: nope.mp4"):
        probe(tmp_path / "nope.mp4")


def test_decode_jpeg_empty_and_small_widths():
    with pytest.raises(VideoError):
        decode_jpeg(b"")
    ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
    assert decode_jpeg(buf.tobytes()).shape == (240, 320, 3)
    assert decode_jpeg(buf.tobytes(), max_width=160).shape == (120, 160, 3)


def test_meet_video_script(meet_video):
    """The shared 'meet' fixture: blue staff fixed, red customer walks in at 4 s, waits, leaves by 20 s."""
    info = probe(meet_video)
    assert (info.src_width, info.src_height) == (1280, 720)
    assert info.src_duration == pytest.approx(20.0, abs=0.05)
    wanted = {int(round(s * 30)): s for s in (2.0, 6.0, 12.0, 19.9)}
    seen = {}
    cap = cv2.VideoCapture(str(meet_video))
    try:
        for i in range(max(wanted) + 1):
            ok, frame = cap.read()
            assert ok
            if i in wanted:
                seen[wanted[i]] = frame
    finally:
        cap.release()
    blue, red = (255, 0, 0), (0, 0, 255)
    bx = _box(seen[2.0], blue)
    assert _box(seen[2.0], red) is None                         # customer not there yet
    assert abs(bx[0] - 600) <= 3 and abs(bx[1] - 250) <= 3
    assert abs(_box(seen[6.0], red)[0] - 235) <= 4              # half-way through the walk-in
    rx = _box(seen[12.0], red)                                  # waiting, 50 px left of staff
    assert abs(rx[0] - 470) <= 3 and _box(seen[12.0], blue)[0] - rx[2] >= 40
    out = _box(seen[19.9], red)                                 # almost out of the left edge
    assert out is None or out[2] <= 30


def _damaged_copy(src, dst, how, frac=0.4, seed=0):
    """Copy ``src`` to ``dst`` and damage it at ``frac`` of the file: cut it off, or overwrite 3000 random bytes."""
    data = bytearray(Path(src).read_bytes())
    at = int(len(data) * frac)
    if how == "truncate":
        del data[at:]
    else:
        rng = random.Random(seed)
        data[at:at + 3000] = bytes(rng.randrange(256) for _ in range(3000))
    Path(dst).write_bytes(bytes(data))
    return Path(dst)


@pytest.fixture
def small_video(tmp_path, make_video):
    """4 s, 30 fps, 320x240 AVI with one walker (fast to write and to damage)."""
    mover = {"color": (0, 0, 255), "w": 40, "h": 100, "pos": lambda t: (20 * t, 20)}
    return make_video(tmp_path / "small.avi", seconds=4, fps=30, size=(320, 240), movers=[mover])


@pytest.mark.parametrize("how", ["truncate", "overwrite"])
def test_damaged_video_ends_early_and_says_so(small_video, tmp_path, how):
    """Decoding stops part-way: the readable part is delivered and `end` reports the shortfall (no exception)."""
    bad = _damaged_copy(small_video, tmp_path / f"{how}.avi", how)
    end = StreamEnd()
    got = [t for _, t, _ in iter_frames(bad, sample_fps=5.0, end=end)]
    assert 0 < len(got) < 0.9 * 20                         # the readable part only
    assert end.early and end.expected_s == pytest.approx(4.0)
    assert got[-1] < end.read_s < 0.9 * 4.0


def test_intact_video_does_not_trip_the_damage_check(small_video):
    end = StreamEnd()
    frames = list(iter_frames(small_video, sample_fps=5.0, end=end))
    assert len(frames) == 20 and frames[-1][1] == pytest.approx(3.8)
    assert not end.early and end.read_s == pytest.approx(4.0) and end.expected_s == pytest.approx(4.0)


def test_damage_beyond_the_trim_point_is_not_our_problem(small_video, tmp_path):
    bad = _damaged_copy(small_video, tmp_path / "late.avi", "truncate", frac=0.6)    # readable up to ~2.4 s
    end = StreamEnd()
    frames = list(iter_frames(bad, sample_fps=5.0, max_seconds=1.0, end=end))
    assert len(frames) == 5 and not end.early
    end = StreamEnd()
    frames = list(iter_frames(bad, sample_fps=5.0, max_seconds=3.5, end=end))
    assert 0 < len(frames) < 0.9 * 3.5 * 5 and end.early and end.expected_s == pytest.approx(3.5)


def test_container_with_a_longer_audio_track_ends_early(small_video, tmp_path, long_audio):
    """A 4 s picture in a 10 s Matroska container: the frame count is derived from the 10 s, the decoder finds 4 s."""
    mkv = long_audio(small_video, tmp_path / "long_audio.mkv", 10)
    info = probe(mkv)
    assert info.src_frames == pytest.approx(300, abs=2) and info.duration == pytest.approx(10.0, abs=0.1)
    end = StreamEnd()
    frames = list(iter_frames(mkv, sample_fps=5.0, end=end))
    assert len(frames) == 20 and frames[-1][1] == pytest.approx(3.8)
    assert end.early and end.read_s == pytest.approx(4.0) and end.expected_s == pytest.approx(10.0, abs=0.1)


def test_zero_frame_video_still_raises(tmp_path):
    path = tmp_path / "zero.avi"
    cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)).release()   # a header, no frames
    with pytest.raises(VideoError, match="cannot read video: zero.avi"):
        probe(path)
    with pytest.raises(VideoError, match="cannot read video: zero.avi"):
        list(iter_frames(path, end=StreamEnd()))


def test_probe_rejects_rateless_container(tmp_path, make_video):
    """Containers with no real frame rate report 1000 or 90000 fps: refuse them instead of timing events wrongly."""
    path = make_video(tmp_path / "rateless.avi", seconds=0.05, fps=1000, size=(64, 48), movers=[])
    with pytest.raises(VideoError, match="cannot read video: rateless.avi"):
        probe(path)
    with pytest.raises(VideoError):
        list(iter_frames(path))


def test_probe_accepts_high_but_real_frame_rates(tmp_path, make_video):
    path = make_video(tmp_path / "hfr.avi", seconds=0.5, fps=240, size=(64, 48), movers=[])
    assert probe(path).src_fps == pytest.approx(240)


@pytest.mark.slow
def test_real_meva_sample_5fps():
    if not SAMPLE.exists():
        pytest.skip(f"missing {SAMPLE}")
    info = probe(SAMPLE)
    assert info.src_fps == pytest.approx(30, abs=0.5) and not info.trimmed
    assert info.width <= 960 and info.width % 2 == 0 and info.height % 2 == 0
    assert info.duration == pytest.approx(60.0, abs=0.5)
    n = 0
    last_t = -1.0
    for idx, t, frame in iter_frames(SAMPLE, sample_fps=5.0):
        assert frame.shape == (info.height, info.width, 3)
        assert t > last_t
        last_t = t
        n += 1
    assert 295 <= n <= 305
    assert last_t == pytest.approx(60.0, abs=0.5)


@pytest.mark.slow
def test_real_meva_sample_damaged_copy_ends_early(tmp_path):
    if not SAMPLE.exists():
        pytest.skip(f"missing {SAMPLE}")
    bad = _damaged_copy(SAMPLE, tmp_path / "damaged.mp4", "overwrite")
    end = StreamEnd()
    frames = list(iter_frames(bad, sample_fps=5.0, end=end))
    assert frames and end.early and end.read_s < 0.9 * end.expected_s
