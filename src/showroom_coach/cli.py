"""Command line entry point (`showroom-coach` / `python -m showroom_coach`).

Subcommands are registered in `build_parser()` as later tasks add them.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_fetch_models(args: argparse.Namespace) -> int:
    from . import models

    names = args.names or list(models.MODELS)
    unknown = [n for n in names if n not in models.MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}; available: {', '.join(models.MODELS)}",
              file=sys.stderr)
        return 2
    failed = 0
    for name in names:
        print(f"fetching {name} ...", flush=True)
        try:
            path = models.ensure_model(name)
        except models.ModelUnavailable as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"manifest: {models.write_manifest()}")
    return 1 if failed else 0


def _bench_candidates() -> tuple[str, ...]:
    from .perception import bench

    return bench.CANDIDATES


def _cmd_bench_detectors(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .perception import bench

    for label, p in (("video", args.video), ("geom", args.geom), ("types", args.types)):
        if not Path(p).is_file():
            print(f"{label} file not found: {p}", file=sys.stderr)
            return 2
    if args.frames < 1:
        print("--frames must be at least 1", file=sys.stderr)
        return 2
    names = tuple(n.strip() for n in args.models.split(",") if n.strip())
    unknown = [n for n in names if not bench.is_valid_spec(n)]
    if not names or unknown:
        print(f"unknown detector(s): {', '.join(unknown) or args.models!r}; each must be "
              f"<model> or <model>@<tiles> (tiles 1 or 2) with model one of "
              f"{', '.join(bench.BASE_DETECTORS)}", file=sys.stderr)
        return 2
    try:
        result = bench.bench_detectors(Path(args.video), Path(args.geom), Path(args.types),
                                       frames=args.frames, models=names,
                                       out=Path(args.out) if args.out else None)
    except RuntimeError as exc:
        print(f"bench-detectors failed: {exc}", file=sys.stderr)
        return 1
    print(bench.format_table(result))
    print(f"wrote {result['out']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="showroom-coach",
        description="Camera-based coaching for car-showroom salespeople.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("fetch-models", help="download model weights into models/ and write MANIFEST.json")
    p.add_argument("names", nargs="*", help="model names (default: every registered model)")
    p.set_defaults(handler=_cmd_fetch_models)

    p = sub.add_parser("bench-detectors",
                       help="benchmark person detectors against MEVA geom boxes (recall, precision, ms/frame)")
    p.add_argument("--video", required=True, help="MEVA .avi clip")
    p.add_argument("--geom", required=True, help="MEVA <clip>.geom.yml")
    p.add_argument("--types", required=True, help="MEVA <clip>.types.yml")
    p.add_argument("--frames", type=int, default=60, help="frames sampled evenly across the clip (default 60)")
    p.add_argument("--models", default=",".join(_bench_candidates()),
                   help="comma-separated detector specs <model>[@<tiles>] to compare "
                        "(default: every benchmark candidate)")
    p.add_argument("--out", help="result JSON (default runs/bench/detectors.json)")
    p.set_defaults(handler=_cmd_bench_detectors)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"showroom-coach {__version__}")
        return 0
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
