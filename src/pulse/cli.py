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
    a = sub.add_parser("analyze", help="analyse a video and write every result file")
    a.add_argument("video", type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--preset", choices=["short", "dealership"], default="short")
    a.add_argument("--zones", type=Path, default=None)
    a.add_argument("--roles", type=Path, default=None)
    a.add_argument("--no-video", action="store_true")
    a.add_argument("--cache", type=Path, default=Path("runs/cache"))
    args = p.parse_args(argv)
    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "bench" and getattr(args, "what", None) == "detectors":
        from pulse.bench import run_detector_bench
        run_detector_bench(args.clip, args.kpf, list(args.showroom), args.out, args.frames)
        return 0
    p.print_help()
    return 0


def _analyze(args) -> int:
    from pulse.sources import VideoError
    from pulse import pipeline
    try:
        res = pipeline.analyze(args.video, args.out, args.preset, args.zones, args.roles, args.cache,
                               render_video=not args.no_video)
    except VideoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    s = res.stats.store
    print(f"visitors: {s['visitors']}  staff: {s['staff']}  insights: {len(res.insights)}  "
          f"RTF: {res.meta['rtf']:.1f}x")
    for i in res.insights:
        print(f"  [{i.rule}] {i.title} ({i.confidence})")
    for w in res.stats.warnings:
        print(f"  warning: {w}")
    print(f"outputs: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
