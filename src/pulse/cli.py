import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pulse")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("bench", help="benchmarks")
    bs = b.add_subparsers(dest="what")
    d = bs.add_parser("detectors", help="detector benchmark (person recall, cars, speed)")
    d.add_argument("--clip", required=True, type=Path)
    d.add_argument("--kpf", required=True, type=Path, help="MEVA annotation dir")
    d.add_argument("--showroom", required=True, nargs="+", type=Path, help="showroom image(s)")
    d.add_argument("--out", type=Path, default=Path("runs/bench/detectors.json"))
    d.add_argument("--frames", type=int, default=60)
    args = p.parse_args(argv)
    if args.cmd == "bench" and getattr(args, "what", None) == "detectors":
        from pulse.bench import run_detector_bench
        run_detector_bench(args.clip, args.kpf, list(args.showroom), args.out, args.frames)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
