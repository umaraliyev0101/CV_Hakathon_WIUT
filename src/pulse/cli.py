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
    sp = bs.add_parser("speed", help="file-mode RTF (60 s 720p excerpt) and live engine fps")
    sp.add_argument("--clip", required=True, type=Path, help="MEVA dev clip")
    sp.add_argument("--out", type=Path, default=Path("runs/bench/speed.json"))
    sp.add_argument("--no-coreml", action="store_true")
    ev = sub.add_parser("eval", help="score against MEVA ground truth and write docs/EVAL.md")
    ev.add_argument("--out", type=Path, default=Path("docs/EVAL.md"))
    ev.add_argument("--clips", nargs="*", default=None, help="restrict to these clip names")
    a = sub.add_parser("analyze", help="analyse a video and write every result file")
    a.add_argument("video", type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--preset", choices=["short", "dealership"], default="short")
    a.add_argument("--zones", type=Path, default=None)
    a.add_argument("--roles", type=Path, default=None)
    a.add_argument("--sales", type=Path, default=None, help="optional sales CSV: car_model,period,sold")
    a.add_argument("--no-video", action="store_true")
    a.add_argument("--cache", type=Path, default=Path("runs/cache"))
    args = p.parse_args(argv)
    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "bench" and getattr(args, "what", None) == "detectors":
        from pulse.bench import run_detector_bench
        run_detector_bench(args.clip, args.kpf, list(args.showroom), args.out, args.frames)
        return 0
    if args.cmd == "bench" and getattr(args, "what", None) == "speed":
        from pulse.bench import run_speed_bench
        run_speed_bench(args.clip, args.out, coreml=not args.no_coreml)
        return 0
    if args.cmd == "eval":
        from pulse.eval import run_eval
        return run_eval.main(args.out, args.clips or None)
    p.print_help()
    return 0


def _analyze(args) -> int:
    from pulse.sources import VideoError
    from pulse import pipeline
    from pulse.sales import SalesError, parse_sales_bytes
    if args.sales:
        try:
            parse_sales_bytes(args.sales.read_bytes())
        except OSError as e:
            print(f"error: cannot read sales file ({e.strerror})", file=sys.stderr)
            return 2
        except SalesError as e:
            print(f"error: sales file invalid: {'; '.join(e.reasons)}", file=sys.stderr)
            return 2
    try:
        res = pipeline.analyze(args.video, args.out, args.preset, args.zones, args.roles, args.cache,
                               render_video=not args.no_video, sales_path=args.sales)
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
