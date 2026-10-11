"""Stream a video file to a running /ws/live endpoint, the way the browser page does, and report what comes back.

    .venv/Scripts/python scripts/live_file_client.py ws://127.0.0.1:8000/ws/live data/samples/meva_a_60s.mp4 [--fps 5]
                                                      [--frames N] [--preset short] [--no-pace]

For every frame the client sends `{"type":"time","t":<video seconds>}` and then the JPEG (quality 70), and waits for
the overlay before sending the next (stop-and-wait, like the page). Frames go out no faster than `--fps` unless
`--no-pace`. Each second it prints the server-reported processing fps and the client's round-trip fps, then the
people count of every frame in that second; every event is printed as it arrives. At the end it asks for the
session `summary` and prints the totals. Exit code 0 when every frame got an overlay, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import cv2
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from showroom_coach.sources import VideoError, iter_frames

JPEG_QUALITY = 70
REPLY_TIMEOUT_S = 120.0       # the first frame also pays for model loading


def _with_preset(url: str, preset: str | None) -> str:
    if not preset:
        return url
    parts = urlparse(url)
    query = parse_qs(parts.query)
    query["preset"] = [preset]
    return urlunparse(parts._replace(query=urlencode(query, doseq=True)))


def _event_line(e: dict) -> str:
    bits = [f"{k}={e[k]}" for k in ("customer", "staff") if e.get(k) is not None]
    if e.get("t_end") is not None:
        bits.append(f"t_end={e['t_end']}")
    data = ", ".join(f"{k}={v}" for k, v in (e.get("data") or {}).items())
    return f"    event t={e['t']:.1f}s {e['type']} {' '.join(bits)}" + (f" ({data})" if data else "")


def run(url: str, video: str, fps: float = 5.0, frames: int | None = None, preset: str | None = None,
        pace: bool = True) -> int:
    if fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 2
    sent = 0
    errors = 0
    server_fps: list[float] = []
    rtts: list[float] = []
    people_counts: list[int] = []
    events: Counter = Counter()
    window_people: list[int] = []
    window_server: list[float] = []
    window_rtt: list[float] = []
    started = time.perf_counter()
    window_start = started
    summary = None
    try:
        with connect(_with_preset(url, preset), max_size=None, open_timeout=15, close_timeout=5) as ws:
            started = window_start = time.perf_counter()
            for _, t, frame in iter_frames(video, sample_fps=fps):
                if frames is not None and sent >= frames:
                    break
                if pace:                                   # no faster than `fps`
                    due = started + sent / fps
                    while (wait := due - time.perf_counter()) > 0:
                        time.sleep(min(wait, 0.05))
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError("cannot encode a frame")
                ws.send(json.dumps({"type": "time", "t": round(t, 3)}))
                t_send = time.perf_counter()
                ws.send(jpg.tobytes())
                reply = json.loads(ws.recv(timeout=REPLY_TIMEOUT_S))
                rtt = time.perf_counter() - t_send
                sent += 1
                if reply.get("type") != "overlay":
                    errors += 1
                    print(f"  frame {sent}: server said {reply}", file=sys.stderr)
                    continue
                rtts.append(rtt)
                server_fps.append(float(reply["fps"]))
                people_counts.append(len(reply["people"]))
                window_people.append(len(reply["people"]))
                window_server.append(float(reply["fps"]))
                window_rtt.append(rtt)
                for e in reply["events"]:
                    events[e["type"]] += 1
                    print(_event_line(e))
                now = time.perf_counter()
                if now - window_start >= 1.0:
                    print(f"[{now - started:5.1f}s] frame {sent:4d} t={t:6.1f}s  "
                          f"server {sum(window_server) / len(window_server):4.1f} fps  "
                          f"round trip {len(window_rtt) / (now - window_start):4.1f} fps "
                          f"(mean {1000 * sum(window_rtt) / len(window_rtt):4.0f} ms)  people per frame {window_people}")
                    window_start, window_people, window_server, window_rtt = now, [], [], []
            if sent:
                ws.send(json.dumps({"type": "summary"}))
                summary = json.loads(ws.recv(timeout=REPLY_TIMEOUT_S))
    except (OSError, WebSocketException, TimeoutError, VideoError, RuntimeError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not rtts:
        print("FAILED: no overlay received", file=sys.stderr)
        return 1
    wall = time.perf_counter() - started
    print()
    print(f"frames sent {sent}, overlays {len(rtts)}, errors {errors}, wall {wall:.1f} s")
    print(f"mean server fps {sum(server_fps) / len(server_fps):.2f} (last {server_fps[-1]:.1f}), "
          f"mean round trip {1000 * sum(rtts) / len(rtts):.0f} ms = {len(rtts) / sum(rtts):.2f} fps, "
          f"end-to-end {len(rtts) / wall:.2f} fps")
    print(f"people per frame: min {min(people_counts)}, mean {sum(people_counts) / len(people_counts):.1f}, "
          f"max {max(people_counts)}")
    print("events: " + (", ".join(f"{k} {v}" for k, v in sorted(events.items())) or "none"))
    if summary and summary.get("type") == "summary":
        store = summary["scorecard"]["store"]
        print(f"summary: {len(summary['events'])} events so far; store {json.dumps(store)}")
    return 1 if errors else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="e.g. ws://127.0.0.1:8000/ws/live")
    ap.add_argument("video", help="video file to stream")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="frames sampled per video second, and the send-rate cap (default 5)")
    ap.add_argument("--frames", type=int, help="stop after this many frames")
    ap.add_argument("--preset", help="thresholds preset to ask for (adds ?preset= to the URL)")
    ap.add_argument("--no-pace", action="store_true",
                    help="send as fast as the server answers instead of capping at --fps")
    a = ap.parse_args(argv)
    return run(a.url, a.video, a.fps, a.frames, a.preset, pace=not a.no_pace)


if __name__ == "__main__":
    sys.exit(main())
