"""Real HTTP smoke test of the Showroom Coach web API (the file-upload flow of the page, without a browser).

    .venv/Scripts/python scripts/smoke_api.py http://127.0.0.1:8000 data/samples/meva_b_60s.mp4
                                              [--preset short] [--timeout 600] [--budget SECONDS]

Flow: GET /api/presets, POST /api/jobs (multipart), poll GET /api/jobs/{id} every second until needs_confirm,
GET /people, pick the people with the longest presence as staff (top 1 when there are 4 people or fewer, else top 2;
the first is named "Demo"), POST /roles, poll until the annotated video is ready, then GET /results, the video
(with a Range request, expecting 206), events.csv, scorecard.csv, timeline.png and one person thumbnail.

It prints every HTTP status, the timings and the counts, then "SMOKE OK" (exit 0) or "SMOKE FAILED: <reason>" (exit 1).
With --budget SECONDS it also prints "BUDGET MET" or "BUDGET MISSED (x s > budget)" for the time from the start of the
upload to needs_confirm, and a miss makes the run fail (exit 1) even when every other check passed.
Needs the server to be running already (run.bat / run.sh, or `showroom-coach serve`). The only third-party package it
uses is httpx, which is a main dependency of the project, so it runs in the venv that run.bat / run.sh build.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

POLL_S = 1.0
UPLOAD_TIMEOUT_S = 300.0       # a 1 GB upload to localhost; the server also stores the file before it answers
REQUEST_TIMEOUT_S = 60.0
DEMO_NAME = "Demo"


class SmokeFailed(Exception):
    """A check failed; the message is the reason printed after 'SMOKE FAILED:'."""


def say(text: str = "") -> None:
    """Print as plain ASCII: server messages hold an ellipsis or an en dash that a Windows console may not show."""
    text = text.replace("\u2026", "...").replace("\u2013", "-").replace("\u2014", "-")
    print(text.encode("ascii", "replace").decode("ascii"), flush=True)


class Smoke:
    def __init__(self, base_url: str, deadline: float):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=REQUEST_TIMEOUT_S)
        self.deadline = deadline
        self.t_start = time.perf_counter()
        self.statuses: list[tuple[str, int]] = []

    def request(self, method: str, path: str, expect: int | tuple[int, ...] = 200, label: str | None = None,
                **kwargs) -> httpx.Response:
        label = label or f"{method} {path}"
        expected = (expect,) if isinstance(expect, int) else expect
        t0 = time.perf_counter()
        try:
            resp = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise SmokeFailed(f"{label}: request failed: {type(exc).__name__}: {exc}") from exc
        dt = time.perf_counter() - t0
        say(f"  HTTP {resp.status_code}  {label}  ({dt:.2f} s, {len(resp.content)} bytes)")
        self.statuses.append((label, resp.status_code))
        if resp.status_code not in expected:
            detail = resp.text[:300].replace("\n", " ")
            raise SmokeFailed(f"{label}: expected HTTP {'/'.join(map(str, expected))}, got {resp.status_code}: {detail}")
        return resp

    def json(self, method: str, path: str, **kwargs):
        resp = self.request(method, path, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise SmokeFailed(f"{method} {path}: the response is not JSON: {resp.text[:200]!r}") from exc

    def wait_job(self, job_id: str, label: str, done) -> dict:
        """Poll the job status once a second, printing each change of status/progress/message, until done(job)."""
        last = None
        path = f"/api/jobs/{job_id}"
        while True:
            try:
                resp = self.client.get(path)
            except httpx.HTTPError as exc:
                raise SmokeFailed(f"polling {path}: {type(exc).__name__}: {exc}") from exc
            if resp.status_code != 200:
                raise SmokeFailed(f"polling {path}: HTTP {resp.status_code}: {resp.text[:200]}")
            job = resp.json()
            self.statuses.append((f"poll {path}", resp.status_code))
            key = (job["status"], f"{100 * float(job.get('progress') or 0):.0f}", job.get("message"), job.get("video_status"))
            if key != last:
                last = key
                say(f"  [{time.perf_counter() - self.t_start:6.1f} s] {job['status']:<13} {key[1]:>3}%  "
                    f"video={job.get('video_status')}  {job.get('message') or ''}")
            if job["status"] == "failed":
                raise SmokeFailed(f"the job failed while waiting for {label}: {job.get('error') or job.get('message')}")
            if job.get("video_status") == "failed":
                raise SmokeFailed(f"the annotated video failed to render: {job.get('error') or job.get('message')}")
            if done(job):
                return job
            if time.monotonic() > self.deadline:
                raise SmokeFailed(f"timed out waiting for {label} (last status {job['status']}, {key[1]}%, "
                                  f"video {job.get('video_status')}: {job.get('message')})")
            time.sleep(POLL_S)


def choose_staff(people: list[dict]) -> list[int]:
    """The longest-present people: the top 1 when there are 4 people or fewer, else the top 2."""
    ranked = sorted(people, key=lambda p: (-(float(p["t_last"]) - float(p["t_first"])), int(p["pid"])))
    return [int(p["pid"]) for p in ranked[: 1 if len(people) <= 4 else 2]]


def csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))          # the exports start with a BOM for Excel


def run(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.is_file():
        raise SmokeFailed(f"video file not found: {video}")
    t0 = time.perf_counter()
    smoke = Smoke(args.url, deadline=time.monotonic() + args.timeout)
    timings: dict[str, float] = {}

    say(f"Smoke test of {args.url} with {video} ({video.stat().st_size / 1e6:.1f} MB), preset {args.preset}, "
        f"timeout {args.timeout:g} s")

    say("1. page and presets")
    page = smoke.request("GET", "/")
    if "text/html" not in page.headers.get("content-type", ""):
        raise SmokeFailed(f"GET / is not HTML (content-type {page.headers.get('content-type')!r})")
    presets = smoke.json("GET", "/api/presets")
    if args.preset not in presets:
        raise SmokeFailed(f"preset {args.preset!r} is not offered by the server (offers {', '.join(presets)})")
    say("  presets: " + ", ".join(f"{k} ({v.get('label')})" for k, v in presets.items()))

    say("2. upload")
    t_up = time.perf_counter()
    with video.open("rb") as fh:
        created = smoke.json("POST", "/api/jobs", label="POST /api/jobs (multipart upload)",
                             files={"file": (video.name, fh, "video/mp4")}, data={"preset": args.preset},
                             timeout=UPLOAD_TIMEOUT_S)
    timings["upload"] = time.perf_counter() - t_up
    job_id = created.get("job_id")
    if not job_id:
        raise SmokeFailed(f"POST /api/jobs returned no job_id: {created}")
    say(f"  job {job_id}")
    t_job = time.perf_counter()

    say("3. perception and analysis (polling every second)")
    job = smoke.wait_job(job_id, "needs_confirm", lambda j: j["status"] == "needs_confirm")
    timings["to_needs_confirm"] = time.perf_counter() - t_job
    timings["upload_to_needs_confirm"] = time.perf_counter() - t_up        # what --budget is checked against
    for w in job.get("warnings") or []:
        say(f"  warning: {w}")

    say("4. people and roles")
    people_resp = smoke.json("GET", f"/api/jobs/{job_id}/people")
    people = people_resp.get("people") or []
    if not people:
        raise SmokeFailed("no people were detected in the video")
    staff = choose_staff(people)
    for p in sorted(people, key=lambda p: int(p["pid"])):
        sug = p.get("suggestion") or {}
        say(f"  person {p['pid']:>3}: {p['t_first']:6.1f}-{p['t_last']:6.1f} s "
            f"(present {float(p['t_last']) - float(p['t_first']):5.1f} s), suggested {sug.get('role')} "
            f"{float(sug.get('confidence') or 0):.2f}, chosen {'staff' if int(p['pid']) in staff else 'customer'}")
    body = {"roles": {str(int(p["pid"])): ("staff" if int(p["pid"]) in staff else "customer") for p in people},
            "names": {str(staff[0]): DEMO_NAME}, "preset": args.preset}
    t_roles = time.perf_counter()
    posted = smoke.json("POST", f"/api/jobs/{job_id}/roles", json=body)
    timings["roles"] = time.perf_counter() - t_roles
    if not isinstance(posted, dict) or "scorecard" not in posted:
        raise SmokeFailed("POST /roles did not return a results object")

    say("5. annotated video (polling every second)")
    smoke.wait_job(job_id, "the annotated video", lambda j: j.get("video_status") == "ready")
    timings["to_video_ready"] = time.perf_counter() - t_roles

    say("6. results and files")
    res = smoke.json("GET", f"/api/jobs/{job_id}/results")
    problems = []
    if res.get("video_status") != "ready":
        problems.append(f"results.video_status is {res.get('video_status')!r}, not 'ready'")
    for pid in staff:
        if res.get("roles", {}).get(str(pid)) != "staff":
            problems.append(f"person {pid} is not staff in the results")
    if res.get("names", {}).get(str(staff[0])) != DEMO_NAME:
        problems.append(f"person {staff[0]} is not named {DEMO_NAME!r} in the results")
    cards = res.get("scorecard", {}).get("staff", [])
    if len(cards) != len(staff):
        problems.append(f"{len(cards)} staff scorecards for {len(staff)} staff")
    elif cards and cards[0].get("name") != DEMO_NAME:
        problems.append(f"the first scorecard is named {cards[0].get('name')!r}, not {DEMO_NAME!r}")
    for key in ("video_url", "timeline_url"):
        if not res.get(key):
            problems.append(f"results.{key} is missing")
    export = res.get("export") or {}
    for key in ("events_csv", "scorecard_csv"):
        if not export.get(key):
            problems.append(f"results.export.{key} is missing")
    if problems:
        raise SmokeFailed("; ".join(problems))

    video_resp = smoke.request("GET", res["video_url"], expect=206, headers={"Range": "bytes=0-1023"},
                               label=f"GET {res['video_url']} (Range: bytes=0-1023)")
    if not video_resp.headers.get("content-range", "").startswith("bytes 0-"):
        raise SmokeFailed(f"the video Content-Range header is {video_resp.headers.get('content-range')!r}")
    if b"ftyp" not in video_resp.content[:16]:
        raise SmokeFailed("the first bytes of the video are not an MP4 header (no ftyp box)")
    say(f"  video Content-Range: {video_resp.headers['content-range']}, {len(video_resp.content)} bytes, "
        f"content-type {video_resp.headers.get('content-type')}")

    events_resp = smoke.request("GET", export["events_csv"])
    events_rows = csv_rows(events_resp.text)
    if not events_rows or len(events_rows[0]) < 3:
        raise SmokeFailed("events.csv has no header row")
    scorecard_resp = smoke.request("GET", export["scorecard_csv"])
    scorecard_rows = csv_rows(scorecard_resp.text)
    if not scorecard_rows or len(scorecard_rows[0]) < 3:
        raise SmokeFailed("scorecard.csv has no header row")
    timeline = smoke.request("GET", res["timeline_url"])
    if not timeline.content.startswith(b"\x89PNG"):
        raise SmokeFailed("timeline.png is not a PNG file")
    thumb_url = next((p["thumb_url"] for p in people if p.get("thumb_url")), None)      # first person that has one
    if not thumb_url:
        raise SmokeFailed(f"none of the {len(people)} people has a thumb_url")
    thumb = smoke.request("GET", thumb_url)
    if not thumb.content.startswith(b"\xff\xd8"):
        raise SmokeFailed("the person thumbnail is not a JPEG file")
    say(f"  events.csv: header {events_rows[0]}, {len(events_rows) - 1} row(s); scorecard.csv: "
        f"{len(scorecard_rows) - 1} row(s); timeline.png {len(timeline.content)} bytes; "
        f"thumbnail {len(thumb.content)} bytes")

    total = time.perf_counter() - t0
    kinds = Counter(e["type"] for e in res.get("events", []))
    store = res["scorecard"]["store"]
    say("")
    say("Timings")
    say(f"  upload                     {timings['upload']:7.2f} s")
    say(f"  to needs_confirm           {timings['to_needs_confirm']:7.2f} s  "
        f"({res['duration']:.1f} s of video)")
    say(f"  roles round-trip           {timings['roles']:7.2f} s")
    say(f"  to video ready (from roles){timings['to_video_ready']:7.2f} s")
    say(f"  total                      {total:7.2f} s")
    budget_missed = None
    if args.budget is not None:
        took = timings["upload_to_needs_confirm"]
        if took <= args.budget:
            say(f"  BUDGET MET (upload to needs_confirm {took:.1f} s <= {args.budget:g} s)")
        else:
            budget_missed = f"BUDGET MISSED ({took:.1f} s > {args.budget:g} s)"
            say(f"  {budget_missed}")
    say("Counts")
    say(f"  people {len(people)} (staff {len(staff)}: {', '.join(map(str, staff))}), "
        f"persons in results {len(res.get('persons', []))}")
    say("  events " + (", ".join(f"{k} {n}" for k, n in sorted(kinds.items())) or "none")
        + f" ({sum(kinds.values())} total)")
    say(f"  staff scorecards {len(cards)}; store: customers {store.get('customers')}, greeted {store.get('greeted')}, "
        f"walkouts {store.get('walkouts')}, wait alerts {store.get('wait_alerts')}")
    for w in res.get("warnings", []):
        say(f"  warning: {w}")
    say(f"  {len(smoke.statuses)} HTTP requests, all as expected")
    if budget_missed:
        raise SmokeFailed(budget_missed)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")            # a server message must never crash a cp1252 console
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description="HTTP smoke test of a running Showroom Coach server.")
    p.add_argument("url", help="server base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("video", help="video file to upload")
    p.add_argument("--preset", default="short", help="threshold preset (default short)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="give up after this many seconds in total, polling included (default 600)")
    p.add_argument("--budget", type=float, default=None, metavar="SECONDS",
                   help="also require upload -> needs_confirm to take at most this many seconds: prints BUDGET MET "
                        "or BUDGET MISSED and exits 1 on a miss (default: no budget)")
    args = p.parse_args(argv)
    if not args.timeout > 0:
        p.error("--timeout must be greater than 0")
    if args.budget is not None and not args.budget > 0:
        p.error("--budget must be greater than 0")
    try:
        run(args)
    except SmokeFailed as exc:
        say(f"SMOKE FAILED: {exc}")
        return 1
    except KeyboardInterrupt:
        say("SMOKE FAILED: interrupted")
        return 1
    except (httpx.InvalidURL, OSError) as exc:              # a malformed URL, or the video file cannot be read
        say(f"SMOKE FAILED: {type(exc).__name__}: {exc}")
        return 1
    except (KeyError, TypeError, ValueError) as exc:        # a response that does not match the API contract
        say(f"SMOKE FAILED: unexpected response shape: {type(exc).__name__}: {exc}")
        return 1
    say("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
