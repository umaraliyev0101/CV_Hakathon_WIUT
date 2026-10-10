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
