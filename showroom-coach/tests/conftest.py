"""Shared fixtures: synthetic videos built from coloured rectangles on a grey background.

A "mover" is a dict ``{"color": (B, G, R), "w": int, "h": int, "pos": f(t) -> (x1, y1) | None}``.
``pos`` returns the top-left corner at time ``t`` seconds, or ``None`` while the rectangle is absent.
Rectangles are clipped to the frame. Keep rectangles apart in the fixtures below: a fake colour-blob
detector would merge boxes that touch.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

GREY = (128, 128, 128)
RED = (0, 0, 255)      # BGR
BLUE = (255, 0, 0)     # BGR


def _write_video(path, seconds, fps, size, movers):
    path = Path(path)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    background = np.full((height, width, 3), GREY, np.uint8)
    try:
        for i in range(int(round(seconds * fps))):
            t = i / fps
            frame = background.copy()
            for m in movers:
                pos = m["pos"](t)
                if pos is None:
                    continue
                x1, y1 = int(round(pos[0])), int(round(pos[1]))
                xa, xb = max(x1, 0), min(x1 + m["w"], width)
                ya, yb = max(y1, 0), min(y1 + m["h"], height)
                if xa < xb and ya < yb:
                    frame[ya:yb, xa:xb] = m["color"]
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def make_video():
    """Factory: ``make_video(path, seconds=6.0, fps=30, size=(1280, 720), movers=())`` -> Path of an MJPG .avi."""
    def _make(path, seconds=6.0, fps=30, size=(1280, 720), movers=()):
        return _write_video(path, seconds, fps, size, movers)
    return _make


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory):
    """6 s, 30 fps, 1280x720: red walker (x 0 -> 1100, y 60..280) and a blue stander (x 600, y 420..640) from t=1 s."""
    movers = [
        {"color": RED, "w": 80, "h": 220, "pos": lambda t: (1100.0 * t / 6.0, 60)},
        {"color": BLUE, "w": 80, "h": 220, "pos": lambda t: (600, 420) if t >= 1.0 else None},
    ]
    path = tmp_path_factory.mktemp("synth") / "synth.avi"
    return _write_video(path, 6.0, 30, (1280, 720), movers)


def _customer_x1(t):
    """Customer left edge: absent < 4 s; 0 -> 470 over 4..8 s; holds to 18 s; back out to -80 by 20 s."""
    if t < 4.0:
        return None
    if t < 8.0:
        return 470.0 * (t - 4.0) / 4.0
    if t < 18.0:
        return 470.0
    x1 = 470.0 - 550.0 * (t - 18.0) / 2.0
    return x1 if x1 + 80 > 0 else None


@pytest.fixture(scope="session")
def meet_video(tmp_path_factory):
    """20 s, 30 fps, 1280x720: blue staff stands at x 600 all along; red customer walks in (4-8 s), waits, leaves (18-20 s).

    Both boxes are 80x220 at y1=250. The customer stops 50 px left of the staff box and never touches it.
    """
    def customer(t):
        x1 = _customer_x1(t)
        return None if x1 is None else (x1, 250)

    movers = [
        {"color": BLUE, "w": 80, "h": 220, "pos": lambda t: (600, 250)},
        {"color": RED, "w": 80, "h": 220, "pos": customer},
    ]
    path = tmp_path_factory.mktemp("meet") / "meet.avi"
    return _write_video(path, 20.0, 30, (1280, 720), movers)


@pytest.fixture
def long_audio():
    """Factory: ``long_audio(video, dst, seconds)`` -> Path of an .mkv with `video`'s frames (stream-copied) and a silent
    audio track `seconds` long. Matroska has no frame count of its own, so OpenCV derives one from the container's
    duration: the file then claims `seconds` worth of video frames while holding only those of `video`."""
    def _mux(video, dst, seconds):
        import subprocess

        import imageio_ffmpeg

        dst = Path(dst)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
               "-f", "lavfi", "-t", f"{float(seconds):g}", "-i", "anullsrc=r=8000:cl=mono",
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        return dst
    return _mux
