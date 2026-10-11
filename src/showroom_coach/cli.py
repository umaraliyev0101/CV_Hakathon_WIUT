"""Command line entry point (`showroom-coach` / `python -m showroom_coach`).

Subcommands are registered in `build_parser()` as later tasks add them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

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
    names = (bench.CANDIDATES if args.models is None  # default resolved here, not at parser build
             else tuple(n.strip() for n in args.models.split(",") if n.strip()))
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


def _cmd_webcam_check(args: argparse.Namespace) -> int:
    from .webcam import webcam_check

    return webcam_check(out=args.out)


class _Progress:
    """Progress callback for a long stage: one stderr line each time another 10% is done."""

    def __init__(self, stage: str):
        self.stage, self.step = stage, -1

    def __call__(self, fraction: float, message: str) -> None:
        step = int(max(0.0, min(1.0, fraction)) * 10)
        if step > self.step:
            self.step = step
            print(f"  [{self.stage}] {step * 10:3d}%  {message}", file=sys.stderr, flush=True)


def _load_roles(path: str) -> tuple[dict, dict]:
    """`roles.json` ({"roles": {"3": "staff"}, "names": {"3": "Aziz"}}) -> (roles, names); ValueError says what is wrong."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"cannot read roles file {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:                                    # bad JSON or bad UTF-8
        raise ValueError(f"roles file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f'roles file {path} must hold an object like {{"roles": {{"3": "staff"}}, "names": {{...}}}}')
    out = []
    for key in ("roles", "names"):
        part = data.get(key, {})
        if not isinstance(part, dict):
            raise ValueError(f'roles file {path}: "{key}" must be an object mapping person ids to text')
        out.append(part)
    return out[0], out[1]


MAX_STAFF_LINES = 5      # the summary shows this many salespeople; the rest are in scorecard.csv


def _secs(v) -> str:
    return "n/a" if v is None else f"{v:.1f} s"


def _printable(text: str, encoding: str | None) -> str:
    """`text` with what the console's encoding cannot show replaced by '?', so a name from roles.json (or the em
    dash in a warning) never makes the summary crash a cp437/cp1252 console."""
    text = text.replace("—", "-")
    for enc in (encoding or "ascii", "ascii"):
        try:
            return text.encode(enc, "replace").decode(enc)
        except LookupError:
            continue
    return text


def _summary(video, res, an, cached: bool, laps: dict, wall: float, rtf: float, perc_s: float,
             roles_file: str | None, stack_note: str, out_dir, wrote: list[str]) -> str:
    from collections import Counter

    sug = Counter(s.role for s in an.suggestions.values())
    used = Counter(an.roles.values())
    kinds = Counter(e.type for e in an.events)
    store = an.scorecard.store
    lines = [f"Showroom Coach analysis of {video.name}"]
    lines.append(f"  video        {res.video.duration:.1f} s analysed ({res.video.width}x{res.video.height}, "
                 f"{res.video.sample_fps:g} fps sampling), preset {an.preset}, stack {res.stack_name}{stack_note}")
    lines.append(f"  people       {len(an.persons)} found; suggested roles: "
                 f"{sug.get('staff', 0)} staff, {sug.get('customer', 0)} customer, {sug.get('unsure', 0)} unsure")
    lines.append(f"  roles used   {used.get('staff', 0)} staff, {used.get('customer', 0)} customer "
                 f"({'from ' + roles_file if roles_file else 'suggestions'})")
    lines.append("  events       " + (", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) or "none"))
    greet_rate = "n/a" if store.greet_rate is None else f"{100 * store.greet_rate:.0f}%"
    lines.append(f"  store        customers {store.customers}, greeted {store.greeted} ({greet_rate}), median time to "
                 f"greet {_secs(store.median_time_to_greet)}, walk-outs {store.walkouts}, unattended wait "
                 f"{store.unattended_wait_s:.1f} s, wait alerts {store.wait_alerts}")
    for c in an.scorecard.staff[:MAX_STAFF_LINES]:
        who = f"#{c.staff}" + (f" {c.name}" if c.name else "")
        lines.append(f"  staff {who:<5}  greets {c.greets}, median time to greet {_secs(c.median_time_to_greet)}, "
                     f"engaged {c.customers_engaged} customer(s) for {c.engaged_total_s:.1f} s, "
                     f"missed while free {c.missed_while_free}, walk-outs while free {c.walkouts_while_free}")
    if len(an.scorecard.staff) > MAX_STAFF_LINES:
        lines.append(f"  staff        ... and {len(an.scorecard.staff) - MAX_STAFF_LINES} more in scorecard.csv")
    for w in an.warnings:
        lines.append(f"  warning      {w}")
    t = res.timings
    stages = ", ".join(f"{k} {t[k]:.1f}" for k in ("decode", "detect", "track", "pose", "embed") if k in t)
    if cached:
        lines.append(f"  perception   saved run reused (loaded in {laps['perception']:.1f} s); the original run "
                     f"took {perc_s:.1f} s: {stages}")
    else:
        lines.append(f"  perception   {perc_s:.1f} s ({stages})")
    rest = ", ".join(f"{k} {laps[k]:.1f} s" for k in ("models", "analysis", "exports", "video") if k in laps)
    lines.append(f"  other stages {rest}")
    lines.append(f"  real-time factor  {rtf:.2f}x  ({res.video.duration:.1f} s of video / {perc_s:.1f} s perception"
                 f"{', from the saved run' if cached else ''}; above 1 is faster than real time)")
    lines.append(f"  total wall   {wall:.1f} s (loading models, perception, analysis, rendering and exports)")
    lines.append(f"  output       {out_dir}: {', '.join(wrote)}")
    return "\n".join(lines)


def _cmd_analyze(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import analysis, config, export, models, render
    from .perception import pipeline, stack as stacks
    from .sources import VideoError
    from .types import to_jsonable

    if not args.fps > 0:
        print("--fps must be greater than 0", file=sys.stderr)
        return 2
    if not args.max_seconds > 0:
        print("--max-seconds must be greater than 0", file=sys.stderr)
        return 2
    roles = names = None
    if args.roles:
        try:
            roles, names = _load_roles(args.roles)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    video, out = Path(args.video), Path(args.out)
    cache = Path(args.cache) if args.cache else config.cache_dir()
    detector = args.detector or config.DEFAULT_DETECTOR
    laps: dict[str, float] = {}
    t_wall = last = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        laps[name] = laps.get(name, 0.0) + now - last
        last = now

    try:
        if args.fake:
            stack, note = stacks.fake_stack(), " (fake stack: tests only)"
        else:
            print(f"loading models ({detector}) ...", file=sys.stderr, flush=True)
            try:
                stack, note = stacks.default_stack(detector), ""
            except ValueError as exc:            # a malformed detector spec or an unknown model name
                print(f"cannot use detector {detector!r}: {exc}", file=sys.stderr)
                return 2
        lap("models")
        res, cached = pipeline.cached_perception(video, stack, cache, progress=_Progress("perception"),
                                                 sample_fps=args.fps, max_seconds=args.max_seconds)
        lap("perception")
        perc_s = float(res.timings["total"]) if cached else laps["perception"]   # a cache hit reports the original run
        rtf = res.video.duration / perc_s if perc_s > 0 else float("inf")

        an = analysis.analyze(res, args.preset, roles, names)
        lap("analysis")

        out.mkdir(parents=True, exist_ok=True)
        thumbs = render.save_thumbnails(res, an, out / "people")
        extra = {"video": {"name": video.name, "duration": res.video.duration, "width": res.video.width,
                           "height": res.video.height, "sample_fps": res.video.sample_fps,
                           "trimmed": res.video.trimmed, "stack": res.stack_name},
                 "persons": [{**to_jsonable(p), "thumb": f"people/{thumbs[p.pid]}" if p.pid in thumbs else None}
                             for p in an.persons]}
        export.write_all(an, out, extra=extra)                     # the numbers first: they survive a later failure
        render.timeline_png(an, out / "timeline.png")
        sheets = render.contact_sheets(video, res, an, out / "contact_sheets")
        lap("exports")

        if not args.no_video:
            render.render_video(video, res, an, out / "annotated.mp4", progress=_Progress("video"))
            lap("video")
    except VideoError as exc:
        text = str(exc)
        print(text if text.startswith("cannot read video") else f"cannot read video: {text}", file=sys.stderr)
        return 2
    except models.ModelUnavailable as exc:
        print(f"models unavailable: {exc}\nrun 'showroom-coach fetch-models' while online, or try --fake for a dry run",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"cannot write the results to {out}: {exc}", file=sys.stderr)
        return 1

    wrote = ["results.json", "events.csv", "scorecard.csv", "timeline.png"]
    if not args.no_video:
        wrote.insert(0, "annotated.mp4")
    wrote += [f"people/ ({len(thumbs)} thumbnails)", f"contact_sheets/ ({len(sheets)})"]
    print(_printable(_summary(video, res, an, cached, laps, time.perf_counter() - t_wall, rtf, perc_s, args.roles,
                              note, out, wrote), sys.stdout.encoding))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if not 0 < args.port < 65536:
        print("--port must be between 1 and 65535", file=sys.stderr)
        return 2
    try:
        import uvicorn

        from .app.main import create_app
    except ImportError as exc:
        print(f"the web app is not available: {exc}", file=sys.stderr)
        return 1
    kwargs = {}
    if args.fake:
        from .perception.stack import fake_stack

        kwargs["stack_factory"] = fake_stack
    app = create_app(**kwargs)
    print(f"Showroom Coach running at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
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
    p.add_argument("--models",
                   help="comma-separated detector specs <model>[@<tiles>] to compare "
                        "(default: every benchmark candidate)")
    p.add_argument("--out", help="result JSON (default runs/bench/detectors.json)")
    p.set_defaults(handler=_cmd_bench_detectors)

    p = sub.add_parser("webcam-check",
                       help="grab one real webcam frame (saved blurred) and run the live engine on it once")
    p.add_argument("--out", help="where to save the blurred copy (default runs/webcam_grab.jpg)")
    p.set_defaults(handler=_cmd_webcam_check)
    from .config import MAX_SECONDS, PRESETS, SAMPLE_FPS

    p = sub.add_parser("analyze",
                       help="analyse a video: annotated video, timeline, scorecards, CSV/JSON exports")
    p.add_argument("video", help="video file (mp4, avi, mov, ...)")
    p.add_argument("--out", required=True, help="output directory (created if missing)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="short",
                   help="threshold preset: short (10-60 s clips, default) or dealership (real shifts)")
    p.add_argument("--roles", help='roles.json: {"roles": {"3": "staff"}, "names": {"3": "Aziz"}}; '
                                   "without it the suggested roles are used")
    p.add_argument("--fps", type=float, default=SAMPLE_FPS, help=f"sampling rate in frames/s (default {SAMPLE_FPS:g})")
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS,
                   help=f"analyse only the first N seconds (default {MAX_SECONDS:g})")
    p.add_argument("--detector", help="detector spec <model>[@<tiles>] (default: the configured default, yolox_nano@2)")
    p.add_argument("--fake", action="store_true", help="use the model-free test stack (tests only)")
    p.add_argument("--no-video", action="store_true", help="skip rendering annotated.mp4")
    p.add_argument("--cache", help="perception cache directory (default runs/cache)")
    p.set_defaults(handler=_cmd_analyze)

    p = sub.add_parser("serve", help="start the web app (upload, results, live webcam)")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    p.add_argument("--fake", action="store_true", help="use the model-free test stack (tests only)")
    p.set_defaults(handler=_cmd_serve)
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
