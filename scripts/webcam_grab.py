"""Grab one real webcam frame and run the live engine on it once. Same as `showroom-coach webcam-check`.

    .venv/Scripts/python scripts/webcam_grab.py [--out runs/webcam_grab.jpg]

Saves a downscaled, strongly blurred copy (never the raw frame), prints the device, resolution, mean brightness and the
overlay the live engine returned. Exit code 0 = captured, 2 = no camera, 1 = captured but the live step failed.
"""
import argparse
import sys

from showroom_coach.webcam import webcam_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="where to save the blurred copy (default runs/webcam_grab.jpg)")
    return webcam_check(out=parser.parse_args(argv).out)


if __name__ == "__main__":
    sys.exit(main())
