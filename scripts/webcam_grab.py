#!/usr/bin/env python
"""Webcam sanity check: grab one frame, verify it is not blank, store a PRIVACY-SAFE copy.

The raw frame only ever lives in memory. What is saved to runs/verify/webcam.jpg is the frame
pixelated to 24 px wide (nearest-neighbour upscale), so nobody's face is stored legibly.
Prints size/mean/std and `WEBCAM OK`; exits 1 otherwise.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "verify" / "webcam.jpg"
WARMUP = 15
WARMUP_MAX = 120
PIXEL_W = 24


def grab_opencv() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for i in range(WARMUP_MAX):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
            # auto-exposure on macOS needs a moment: after the warm-up keep reading until not dark
            if i + 1 >= WARMUP and frame is not None and is_ok(frame):
                break
        return frame
    finally:
        cap.release()


def grab_ffmpeg() -> np.ndarray | None:
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    system = platform.system()
    if system == "Darwin":
        src = ["-f", "avfoundation", "-framerate", "30", "-i", "0"]
    elif system == "Windows":
        src = ["-f", "dshow", "-i", "video=0"]
    else:
        src = ["-f", "v4l2", "-i", "/dev/video0"]
    cmd = [exe, "-y", "-loglevel", "error", *src, "-frames:v", "1", "-f", "image2pipe",
           "-vcodec", "mjpeg", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def stats(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def is_ok(frame: np.ndarray) -> bool:
    mean, std = stats(frame)
    return std > 5 and 5 < mean < 250


def pixelate(frame: np.ndarray, width: int = PIXEL_W) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    frame = grab_opencv()
    if frame is None:
        print("OpenCV could not read the webcam; trying ffmpeg", file=sys.stderr)
        frame = grab_ffmpeg()
    if frame is None:
        print("WEBCAM FAILED: no frame (is the camera free and permission granted?)", file=sys.stderr)
        return 1
    mean, std = stats(frame)
    h, w = frame.shape[:2]
    print(f"frame {w}x{h} mean={mean:.1f} std={std:.1f}")
    if not is_ok(frame):
        print("WEBCAM FAILED: frame looks blank (covered lens or no light)", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUT), pixelate(frame), [cv2.IMWRITE_JPEG_QUALITY, 80]):
        print("WEBCAM FAILED: could not write the privacy-safe image", file=sys.stderr)
        return 1
    print(f"saved pixelated ({PIXEL_W}px) copy to {OUT.relative_to(ROOT)}; the raw frame was not stored")
    print("WEBCAM OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
