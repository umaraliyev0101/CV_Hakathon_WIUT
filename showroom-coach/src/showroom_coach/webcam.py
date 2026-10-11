"""Real-camera check (`showroom-coach webcam-check`, `scripts/webcam_grab.py`).

Proves that live mode can see a real device: open the webcam with OpenCV (DirectShow first, then Media Foundation, on
indices 0 to 2), read up to ten frames until a usable one arrives, report the device, resolution and mean brightness,
save a downscaled and strongly blurred copy (nobody is identifiable in it; the raw frame is never written to disk) and
push the raw frame once through a real `LiveSession`, printing the overlay it returns.

Exit codes: 0 a frame was captured (and the live step ran), 2 no camera could be opened or read, 1 a frame was
captured but the live step failed (for example the model files are missing).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from . import config
from .sources import resize_max_width

__all__ = ["EXIT_OK", "EXIT_STEP_FAILED", "EXIT_NO_CAMERA", "Grab", "grab_frame", "blurred_copy", "webcam_check",
           "default_out"]

EXIT_OK = 0
EXIT_STEP_FAILED = 1
EXIT_NO_CAMERA = 2

INDICES = (0, 1, 2)
MAX_READS = 10                 # frames read per device while waiting for a usable one
BLACK_MEAN = 1.0               # a frame darker than this is a camera still waking up: read on while reads remain
THUMB_WIDTH = 480
JPEG_QUALITY = 85
# (name, OpenCV backend id). The ids are the documented constants, used when the build lacks the attribute.
BACKENDS = (("DSHOW", getattr(cv2, "CAP_DSHOW", 700)), ("MSMF", getattr(cv2, "CAP_MSMF", 1400)))


@dataclass
class Grab:
    frame: np.ndarray          # BGR, as the device delivered it
    index: int
    backend: str
    reads: int                 # which read gave this frame (1 to MAX_READS)

    @property
    def brightness(self) -> float:
        return float(self.frame.mean())


def default_out() -> Path:
    return config.repo_root() / "runs" / "webcam_grab.jpg"


def grab_frame(indices: Iterable[int] = INDICES) -> Grab | None:
    """First usable frame from any device: each backend in turn (DirectShow, then Media Foundation), each index in
    turn. Up to `MAX_READS` reads per device; empty reads are skipped, and so are all-black frames while reads remain
    (many cameras send black for the first frames). Returns None when nothing could be read."""
    indices = tuple(indices)
    for backend, api in BACKENDS:
        for index in indices:
            cap = cv2.VideoCapture(index, api)
            try:
                if not cap.isOpened():
                    continue
                last: tuple[int, np.ndarray] | None = None
                for n in range(1, MAX_READS + 1):
                    ok, frame = cap.read()
                    if not ok or frame is None or getattr(frame, "size", 0) == 0:
                        continue
                    last = (n, frame)
                    if float(frame.mean()) >= BLACK_MEAN:
                        break
                if last is not None:
                    return Grab(last[1], index, backend, last[0])
            finally:
                cap.release()
    return None


def blurred_copy(frame: np.ndarray, width: int = THUMB_WIDTH) -> np.ndarray:
    """The frame shrunk to at most `width` px wide, then blurred hard over the whole picture (sigma 5% of the width).
    Shape and light survive; faces and clothes do not."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    small = resize_max_width(frame[..., :3], width)
    return cv2.GaussianBlur(small, (0, 0), sigmaX=max(8.0, small.shape[1] * 0.05))


def _save_jpeg(image: np.ndarray, path: Path) -> None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cannot encode the blurred copy")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def webcam_check(out: Path | str | None = None, indices: Iterable[int] = INDICES,
                 stack_factory: Callable[[], object] | None = None) -> int:
    """Run the check and print the report; returns the exit code. `out` is where the blurred copy goes (default
    `runs/webcam_grab.jpg`); `stack_factory` builds the perception stack (default: the real one, with
    `config.LIVE_DETECTOR`), so tests can pass the fake."""
    out = Path(out) if out else default_out()
    indices = tuple(indices)
    grab = grab_frame(indices)
    if grab is None:
        print("no camera: could not read a frame from device index "
              f"{', '.join(str(i) for i in indices)} with {' or '.join(b for b, _ in BACKENDS)}", file=sys.stderr)
        return EXIT_NO_CAMERA
    h, w = grab.frame.shape[:2]
    print(f"camera: index {grab.index} ({grab.backend}), {w}x{h}, frame {grab.reads} of up to {MAX_READS}, "
          f"mean brightness {grab.brightness:.1f} / 255")
    thumb = blurred_copy(grab.frame)
    _save_jpeg(thumb, out)
    print(f"saved blurred {thumb.shape[1]}x{thumb.shape[0]} copy: {out}")

    try:
        from .live import LiveSession

        if stack_factory is None:
            from .perception.stack import default_stack

            def stack_factory():
                return default_stack(config.LIVE_DETECTOR)

        session = LiveSession(stack_factory())
        t0 = time.perf_counter()
        overlay = session.process(resize_max_width(grab.frame[..., :3], config.MAX_WIDTH), 0.0)
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # the frame grab itself worked; say what broke in the live step
        print(f"live step FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_STEP_FAILED
    print(f"live step: {len(overlay['people'])} people in a {overlay['w']}x{overlay['h']} frame, "
          f"{ms:.0f} ms for the first frame (includes warm-up)")
    print(json.dumps(overlay))
    return EXIT_OK
