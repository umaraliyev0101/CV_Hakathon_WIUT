#!/usr/bin/env python
"""Live-mode smoke test: real models, real clip, in-process (fastapi TestClient websocket).

Streams JPEG frames of a MEVA dev-clip window to /ws/live at 5 fps in real time (the server clock
supplies `t`, so dwell timers advance in wall-clock seconds), with one full-frame car zone and the
`short` preset (alert after 8 s). Asserts overlays with people, a running dwell timer and an
unserved-interest alert. A second, unpaced run measures the engine's maximum sustainable fps.
Prints `processed fps: X.XX` and `SMOKE LIVE OK`; writes runs/bench/live_smoke.json.

usage: smoke_live.py [--clip PATH] [--seconds 30] [--max-seconds 10]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bench" / "live_smoke.json"
SEND_FPS = 5.0
# Window of the first MEVA dev clip that starts with a quiet scene and then has a person standing in
# view for > 15 s from ~44 s (found by scanning 30 s to 130 s with the live session, full-frame zone).
START_S = 40.0
ZONE = {"zone_id": "car-demo", "name": "Car: Demo", "kind": "car",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "source": "user"}


def step(msg: str) -> None:
    print(f"[smoke_live] {msg}", flush=True)


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_frames(clip: Path | None, seconds: float) -> list[bytes]:
    sys.path.insert(0, str(ROOT / "src"))
    from pulse.sources import iter_frames
    if clip is None:
        spec = importlib.util.spec_from_file_location("make_clip", ROOT / "scripts" / "make_clip.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        split = json.loads((ROOT / "labels" / "split.json").read_text())
        src = ROOT / split["dev"][0]["video"]
        clip = ROOT / "runs" / "fixtures" / f"live_{int(START_S)}s_{int(seconds) + 2}s.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        if not clip.is_file():
            step(f"cutting {clip.name} from {src.name}")
            mod.make_clip(src, clip, start=START_S, dur=seconds + 2, width=960)
    out = []
    for _, frame in iter_frames(clip, SEND_FPS, 960, max_duration=seconds):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        check(ok, "jpeg encode failed")
        out.append(buf.tobytes())
    check(out, f"no frames read from {clip}")
    return out


class Collector(threading.Thread):
    """Drains overlay messages from the websocket while the main thread sends."""

    def __init__(self, ws):
        super().__init__(daemon=True)
        self.ws, self.msgs, self.errors = ws, [], []
        self.cv = threading.Condition()

    def run(self):
        try:
            while True:
                m = self.ws.receive_json()
                with self.cv:
                    (self.msgs if m.get("type") == "overlay" else self.errors).append(m)
                    self.cv.notify_all()
        except Exception:  # noqa: BLE001 - socket closed
            pass

    def wait_for(self, n: int, timeout: float) -> bool:
        end = time.monotonic() + timeout
        with self.cv:
            while len(self.msgs) < n:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self.cv.wait(left)
        return True


def paced_run(client, frames: list[bytes], seconds: float) -> dict:
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "preset", "name": "short"}))
        ws.send_text(json.dumps({"type": "zones", "zones": [ZONE]}))
        col = Collector(ws)
        col.start()
        ws.send_bytes(frames[0])                       # warm-up: first frame loads/warms the models
        check(col.wait_for(1, 120), "no overlay for the first frame")
        base = len(col.msgs)
        sent, t_start = 0, time.monotonic()
        n = int(seconds * SEND_FPS)
        for i in range(n):
            due = t_start + i / SEND_FPS
            time.sleep(max(0.0, due - time.monotonic()))
            ws.send_bytes(frames[(i + 1) % len(frames)])
            sent += 1
        end = time.monotonic() + 15                     # every frame is either answered or dropped
        while time.monotonic() < end:
            last = col.msgs[-1] if col.msgs else {"dropped": 0}
            if (len(col.msgs) - base) + last["dropped"] >= sent:
                break
            time.sleep(0.05)
        wall = time.monotonic() - t_start
    msgs = col.msgs[base:]
    return {"msgs": msgs, "sent": sent, "wall": wall, "errors": col.errors}


def max_run(client, frames: list[bytes], seconds: float) -> float:
    """One frame in flight at a time: the engine's sustainable rate, nothing dropped."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "zones", "zones": [ZONE]}))
        ws.send_bytes(frames[0])
        ws.receive_json()                               # warm-up
        n, t0, i = 0, time.monotonic(), 0
        while time.monotonic() - t0 < seconds:
            ws.send_bytes(frames[i % len(frames)])
            m = ws.receive_json()
            check(m["type"] == "overlay", f"unexpected message {m}")
            n += 1
            i += 1
        return n / (time.monotonic() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from fastapi.testclient import TestClient
    from pulse.app.main import create_app

    frames = make_frames(a.clip, a.seconds)
    step(f"{len(frames)} JPEG frames ready; streaming {a.seconds:.0f} s at {SEND_FPS:.0f} fps (real time)")
    with tempfile.TemporaryDirectory() as td:
        client = TestClient(create_app(jobs_dir=Path(td) / "jobs"))
        res = paced_run(client, frames, a.seconds)
        msgs = res["msgs"]
        check(not res["errors"], f"server errors: {res['errors'][:2]}")
        check(msgs, "no overlays received")
        people_max = max(len(m["persons"]) for m in msgs)
        dwell_max = max((p["dwell"] for m in msgs for p in m["persons"]), default=0.0)
        alert_msgs = [m for m in msgs if m["alerts"]]
        check(people_max >= 1, "no overlay contained a person")
        check(dwell_max > 0, "no person had a dwell timer > 0")
        check(alert_msgs, "no unserved-interest alert fired")
        fps = len(msgs) / res["wall"]
        dropped = msgs[-1]["dropped"]
        step(f"people_max={people_max} dwell_max={dwell_max:.1f}s alert_overlays={len(alert_msgs)} "
             f"dropped={dropped}; first alert: {alert_msgs[0]['alerts'][0]['message']}")
        step(f"engine step mean {sum(m['step_ms'] for m in msgs) / len(msgs):.0f} ms")
        step(f"measuring maximum sustainable fps for {a.max_seconds:.0f} s")
        fps_max = max_run(client, frames, a.max_seconds)

    print(f"processed fps: {fps:.2f}")
    print(f"max sustainable fps: {fps_max:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fps": round(fps, 2), "fps_max": round(fps_max, 2), "frames": len(msgs), "sent": res["sent"],
        "dropped": dropped, "people_max": people_max, "dwell_max_s": round(dwell_max, 1),
        "alerts": max(len(m["alerts"]) for m in msgs), "alert_overlays": len(alert_msgs),
        "step_ms_mean": round(sum(m["step_ms"] for m in msgs) / len(msgs), 1),
        "seconds": a.seconds, "clip_start_s": None if a.clip else START_S}, indent=2) + "\n")
    print("SMOKE LIVE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
