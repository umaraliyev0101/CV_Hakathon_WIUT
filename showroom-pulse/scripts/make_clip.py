#!/usr/bin/env python
"""Cut and re-encode a clip (H.264) with the bundled imageio-ffmpeg binary.

usage: make_clip.py SRC DST --start S --dur D [--width W] [--rotate-portrait]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def make_clip(src, dst, start: float, dur: float, width: int | None = None,
              rotate_portrait: bool = False) -> Path:
    vf = []
    if rotate_portrait:
        vf.append("transpose=1")
    if width:
        vf.append(f"scale={int(width)}:-2")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-ss", str(start), "-i", str(src), "-t", str(dur)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--width", type=int)
    ap.add_argument("--rotate-portrait", action="store_true")
    a = ap.parse_args(argv)
    make_clip(a.src, a.dst, a.start, a.dur, a.width, a.rotate_portrait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
