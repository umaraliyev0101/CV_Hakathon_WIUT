"""CCTV look (noise, blur, timestamp) and H.264 encode via ffmpeg (pure: numpy/cv2/subprocess)."""
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np

DEFAULTS = {"noise_sigma": 3.0, "blur_sigma": 0.7, "crf": 26, "start_time": "2026-10-11T10:00:00"}


def degrade(img_bgr, rng, noise_sigma, blur_sigma):
    import cv2
    out = cv2.GaussianBlur(img_bgr, (0, 0), blur_sigma) if blur_sigma > 0 else img_bgr
    noise = rng.normal(0.0, noise_sigma, out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def burn_timestamp(img_bgr, when, cam_label):
    import cv2
    text = f"{when:%Y-%m-%d %H:%M:%S}.{when.microsecond // 100000}  {cam_label}"
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img_bgr, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    return img_bgr


def encode(frames_dir, out_mp4, fps, cam_label, seed=0, opts=None):
    import cv2
    o = dict(DEFAULTS, **(opts or {}))
    files = sorted(Path(frames_dir).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    h, w = cv2.imread(str(files[0])).shape[:2]
    rng = np.random.default_rng(seed)
    t0 = dt.datetime.fromisoformat(o["start_time"])
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(o["crf"]),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", str(fps), str(out_mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k, f in enumerate(files):
        img = degrade(cv2.imread(str(f)), rng, o["noise_sigma"], o["blur_sigma"])
        img = burn_timestamp(img, t0 + dt.timedelta(seconds=k / fps), cam_label)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return len(files)
