#!/usr/bin/env python
"""End-to-end API smoke test: real models, real clip, in-process (fastapi TestClient).

upload -> progress -> zones -> roles -> score -> edit zones -> re-score without re-detection.
Prints each step and `SMOKE API OK`; exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "runs" / "fixtures" / "meva_20s.mp4"
POLL_TIMEOUT_S = 300


def step(msg: str) -> None:
    print(f"[smoke_api] {msg}", flush=True)


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def ensure_clip() -> Path:
    if CLIP.is_file():
        return CLIP
    spec = importlib.util.spec_from_file_location("make_clip", ROOT / "scripts" / "make_clip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    split = json.loads((ROOT / "labels" / "split.json").read_text())
    src = ROOT / split["dev"][0]["video"]
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    step(f"making {CLIP.relative_to(ROOT)} from {src.name}")
    return mod.make_clip(src, CLIP, start=40, dur=20, width=960)


def rect(zid, name, kind, x0, y0, x1, y1):
    return {"zone_id": zid, "name": name, "kind": kind,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "source": "user"}


def poll(c, jid, want: set[str], timeout: float = POLL_TIMEOUT_S) -> dict:
    end = time.time() + timeout
    last = None
    while time.time() < end:
        st = c.get(f"/api/jobs/{jid}").json()
        if st != last and st["state"] != "ready_for_zones":
            step(f"  state={st['state']} progress={st['progress']:.2f} {st['message']}")
        last = st
        check(st["state"] != "failed", f"job failed: {st['message']}")
        if st["state"] in want:
            return st
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {want}; last status {last}")


def main() -> int:
    try:
        return _run()
    except AssertionError as e:
        print(f"SMOKE API FAILED: {e}", file=sys.stderr)
        return 1


def _run() -> int:
    warnings.filterwarnings("ignore", message="Using .httpx. with .starlette.testclient")
    from fastapi.testclient import TestClient

    from pulse import cache
    from pulse.app.main import create_app

    clip = ensure_clip()
    with tempfile.TemporaryDirectory(prefix="pulse-smoke-") as td:
        jobs_dir = Path(td) / "jobs"
        with TestClient(create_app(jobs_dir=jobs_dir)) as c:
            step("health")
            check(c.get("/api/health").json()["ok"] is True, "health not ok")

            step(f"upload {clip.name}")
            with open(clip, "rb") as f:
                r = c.post("/api/jobs", files={"file": (clip.name, f, "video/mp4")},
                           data={"preset": "dealership"})
            check(r.status_code == 200, f"upload failed: {r.text}")
            jid = r.json()["job_id"]
            step(f"  job {jid}")

            step("poll until ready_for_zones")
            poll(c, jid, {"ready_for_zones"})
            check(c.get(f"/api/jobs/{jid}/keyframe.jpg").status_code == 200, "no keyframe")

            step("GET zones (auto)")
            zdoc = c.get(f"/api/jobs/{jid}/zones").json()
            check(zdoc["version"] == 1 and isinstance(zdoc["zones"], list), "bad zones doc")
            step(f"  {len(zdoc['zones'])} auto zone(s)")

            step("PUT zones: Car: Demo + Desk")
            doc = {"version": 1, "frame_size": zdoc["frame_size"],
                   "zones": [rect("car-demo", "Car: Demo", "car", 0.0, 0.0, 1.0, 1.0),
                             rect("desk", "Desk", "desk", 0.0, 0.0, 0.1, 0.1)]}
            check(c.put(f"/api/jobs/{jid}/zones", json=doc).status_code == 200, "PUT zones failed")

            step("GET people")
            people = c.get(f"/api/jobs/{jid}/people").json()
            check(len(people) >= 1, "no people found")
            step(f"  {len(people)} people, first: {people[0]['person_id']} "
                 f"suggested={people[0]['suggested_role']}")
            check(c.get(people[0]["thumb_url"]).status_code == 200, "thumbnail missing")

            step("PUT roles: first person -> staff")
            pid = people[0]["person_id"]
            r = c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(pid): "staff"}, "skip": False})
            check(r.status_code == 200, f"PUT roles failed: {r.text}")

            cdir = next((jobs_dir / jid / "cache").iterdir())
            tracks = cdir / "tracks.npz"
            mtime, runs = tracks.stat().st_mtime_ns, cache.perception_runs()

            step("POST score (with background video render)")
            r = c.post(f"/api/jobs/{jid}/score", json={"render_video": True})
            check(r.status_code == 200, f"score failed: {r.text}")
            res1 = c.get(f"/api/jobs/{jid}/results").json()
            for key in ("store", "per_car", "per_zone", "insights", "warnings", "banner"):
                check(key in res1, f"results missing {key}")
            s1 = res1["store"]
            check(s1["visitors"] + s1["staff"] >= 1, "nobody counted")
            check(s1["visitors"] >= 1, "visitors < 1")
            check(res1["per_car"][0]["visitors"] >= 1, "no visitor in the demo car zone")
            step(f"  visitors={s1['visitors']} staff={s1['staff']} "
                 f"per_car={[(z['name'], z['visitors']) for z in res1['per_car']]}")

            step("wait for annotated video")
            end = time.time() + POLL_TIMEOUT_S
            while c.get(f"/api/jobs/{jid}").json()["video_state"] == "rendering" and time.time() < end:
                time.sleep(0.5)
            check(c.get(f"/api/jobs/{jid}").json()["video_state"] == "ready", "video render failed")
            check(c.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 200, "no annotated.mp4")

            step("PUT zones: move the car rectangle, POST score again")
            doc["zones"][0] = rect("car-demo", "Car: Demo", "car", 0.9, 0.9, 0.99, 0.99)
            check(c.put(f"/api/jobs/{jid}/zones", json=doc).status_code == 200, "PUT zones 2 failed")
            r = c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
            check(r.status_code == 200, f"re-score failed: {r.text}")
            res2 = c.get(f"/api/jobs/{jid}/results").json()
            check(res2["per_car"] != res1["per_car"] or res2["per_zone"] != res1["per_zone"],
                  "results did not change after moving the zone")
            step(f"  per_car={[(z['name'], z['visitors']) for z in res2['per_car']]}")

            step("assert no re-detection")
            check(cache.perception_runs() == runs, "perception ran again")
            check(tracks.stat().st_mtime_ns == mtime, "tracks.npz was rewritten")
            check(c.get("/api/health").json()["perception_runs"] == runs, "health run count changed")
    print("SMOKE API OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
