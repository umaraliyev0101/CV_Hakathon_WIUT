"""`showroom-coach webcam-check` with a fake OpenCV capture device (no camera needed)."""
import json

import cv2
import numpy as np
import pytest

from showroom_coach import webcam
from showroom_coach.cli import main
from showroom_coach.perception.stack import fake_stack


def _scene():
    """640x480 grey frame with one saturated 80x200 block: the fake detector sees one person."""
    img = np.full((480, 640, 3), 128, np.uint8)
    img[120:320, 280:360] = (255, 0, 0)
    # high-frequency detail (grey noise: unsaturated, so it is not a person) so a blur is measurable
    noise = np.random.default_rng(1).integers(0, 255, (60, 200, 1), dtype=np.uint8)
    img[0:60, 0:200] = np.repeat(noise, 3, axis=2)
    return img


def _fake_capture(frames, opens=(0,), log=None):
    """A cv2.VideoCapture stand-in: only the indices in `opens` open; read() returns `frames` in turn, then the last."""

    class Fake:
        def __init__(self, index=0, api=None):
            self.index, self.api = index, api
            self._open = index in opens
            self._reads = 0
            if log is not None:
                log.append(("open", index, api))

        def isOpened(self):
            return self._open

        def read(self):
            if not self._open:
                return False, None
            frame = frames[min(self._reads, len(frames) - 1)]
            self._reads += 1
            return (frame is not None), frame

        def release(self):
            self._open = False
            if log is not None:
                log.append(("release", self.index, self.api))

    return Fake


def test_webcam_check_writes_blurred_copy_and_prints_overlay(monkeypatch, tmp_path, capsys):
    scene = _scene()
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([scene]))
    out = tmp_path / "grab.jpg"

    code = webcam.webcam_check(out=out, stack_factory=fake_stack)

    assert code == 0
    text = capsys.readouterr().out
    assert "camera: index 0 (DSHOW), 640x480" in text and "mean brightness" in text
    overlay = json.loads(text.strip().splitlines()[-1])
    assert overlay["type"] == "overlay" and overlay["w"] == 640 and overlay["h"] == 480
    assert len(overlay["people"]) == 1
    saved = cv2.imread(str(out))
    assert saved is not None and saved.shape[1] == 480 and saved.shape[0] == 360    # downscaled from 640x480
    # the copy is blurred: far less fine detail than a plain downscale of the same frame
    small = cv2.resize(scene, (saved.shape[1], saved.shape[0]), interpolation=cv2.INTER_AREA)
    assert float(np.abs(cv2.Laplacian(saved, cv2.CV_64F)).mean()) < 0.2 * float(np.abs(cv2.Laplacian(small, cv2.CV_64F)).mean())


def test_webcam_check_through_the_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([_scene()]))
    monkeypatch.setattr("showroom_coach.perception.stack.default_stack", lambda *a, **k: fake_stack())
    out = tmp_path / "nested" / "g.jpg"

    assert main(["webcam-check", "--out", str(out)]) == 0
    assert out.is_file() and "live step: 1 people" in capsys.readouterr().out


def test_black_warm_up_frames_are_skipped_and_empty_reads_ignored(monkeypatch):
    black = np.zeros((480, 640, 3), np.uint8)
    frames = [None, black, black, _scene()]
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture(frames))

    grab = webcam.grab_frame()

    assert grab is not None and grab.reads == 4 and grab.brightness > 100


def test_an_all_black_camera_still_counts_as_a_capture(monkeypatch):
    black = np.zeros((240, 320, 3), np.uint8)
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([black]))

    grab = webcam.grab_frame()

    assert grab is not None and grab.reads == webcam.MAX_READS and grab.brightness == 0.0


def test_falls_back_to_the_next_index_and_then_to_msmf(monkeypatch):
    log = []
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([_scene()], opens=(2,), log=log))
    grab = webcam.grab_frame()
    assert grab.index == 2 and grab.backend == "DSHOW"
    assert [e[1] for e in log if e[0] == "open"] == [0, 1, 2]
    assert [e[1] for e in log if e[0] == "release"] == [0, 1, 2]       # every device is released

    # a device that only opens through Media Foundation
    class OnlyMsmf(_fake_capture([_scene()])):
        def __init__(self, index=0, api=None):
            super().__init__(index, api)
            self._open = api == cv2.CAP_MSMF and index == 1

    monkeypatch.setattr(cv2, "VideoCapture", OnlyMsmf)
    grab = webcam.grab_frame()
    assert (grab.index, grab.backend) == (1, "MSMF")


def test_no_camera_exits_2_and_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([_scene()], opens=()))
    out = tmp_path / "grab.jpg"

    assert webcam.webcam_check(out=out, stack_factory=fake_stack) == 2
    assert not out.exists()
    assert "no camera" in capsys.readouterr().err


def test_a_camera_that_opens_but_never_delivers_a_frame_exits_2(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([None], opens=(0, 1, 2)))

    assert webcam.webcam_check(out=tmp_path / "g.jpg", stack_factory=fake_stack) == 2


def test_a_failing_live_step_exits_1_but_keeps_the_blurred_copy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cv2, "VideoCapture", _fake_capture([_scene()]))

    def broken():
        raise RuntimeError("no models")

    out = tmp_path / "g.jpg"
    assert webcam.webcam_check(out=out, stack_factory=broken) == 1
    assert out.is_file() and "live step FAILED" in capsys.readouterr().err
