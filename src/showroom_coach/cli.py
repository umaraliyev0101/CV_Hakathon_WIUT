"""Command line entry point (`showroom-coach` / `python -m showroom_coach`).

Subcommands are registered in `build_parser()` as later tasks add them.
"""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="showroom-coach",
        description="Camera-based coaching for car-showroom salespeople.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_subparsers(dest="command", metavar="command")
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
