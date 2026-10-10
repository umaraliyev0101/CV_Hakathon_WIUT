import numpy as np, imageio_ffmpeg, pytest
from pulse import sources


def write_video(path, w, h, fps, seconds):
    gen = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                      macro_block_size=1, output_params=["-loglevel", "error"])
    gen.send(None)
    for i in range(int(fps * seconds)):
        f = np.full((h, w, 3), 90, np.uint8); x = (i * 5) % (w - 40)
        f[h // 2 - 20:h // 2 + 20, x:x + 40] = 255
        gen.send(f)
    gen.close()


def test_iter_frames_samples_at_5fps(tmp_path):
    p = tmp_path / "a.mp4"; write_video(p, 640, 360, 30, 2)
    frames = list(sources.iter_frames(p, 5.0, 960))
    assert [t for t, _ in frames] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    assert frames[0][1].shape == (360, 640, 3)


def test_portrait_video_resized_and_timed(tmp_path):
    p = tmp_path / "p.mp4"; write_video(p, 1080, 1920, 29.97, 1.2)
    frames = list(sources.iter_frames(p, 5.0, 960))
    assert frames[0][1].shape == (1707, 960, 3)
    assert frames[1][0] == pytest.approx(0.2)


def test_max_duration_trims_and_flags(tmp_path):
    p = tmp_path / "l.mp4"; write_video(p, 320, 240, 10, 3)
    assert sources.probe(p).duration == pytest.approx(3.0, abs=0.15)
    frames = list(sources.iter_frames(p, 5.0, 960, max_duration=1.0))
    assert frames[-1][0] < 1.0 + 1e-9


def test_garbage_raises_video_error(tmp_path):
    p = tmp_path / "x.mp4"; p.write_text("not a video")
    with pytest.raises(sources.VideoError):
        sources.probe(p)


def test_decode_jpeg_rejects_garbage():
    with pytest.raises(ValueError):
        sources.decode_jpeg(b"nope")


def test_sampled_frame_size_and_jpeg_roundtrip(tmp_path):
    import cv2
    p = tmp_path / "s.mp4"; write_video(p, 1080, 1920, 30, 0.5)
    info = sources.probe(p)
    assert (info.width, info.height) == (1080, 1920)
    assert sources.sampled_frame_size(info, 960) == (960, 1707)
    img = np.full((100, 2000, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert sources.decode_jpeg(buf.tobytes(), 960).shape == (48, 960, 3)
