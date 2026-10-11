"""FastAPI app: upload a video, follow progress, edit zones and roles, read results, re-score.

Perception (models) runs once per video in a single background worker. Everything after it
(zones, roles, presets) is `pipeline.score` + `write_outputs` on the cached perception output, so
editing zones never re-runs a model.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from pulse import cache as cache_mod
from pulse import pipeline, render as render_mod, report as report_mod, roles as roles_mod
from pulse import sources, zones as zones_mod
from pulse.config import PRESETS as PRESET_DEFS, get_config
from pulse.datatypes import Zone

from . import jobs as jobs_mod
from .jobs import ALLOWED_EXT, PRESETS, ApiError, Job, JobStore, atomic_write_text

log = logging.getLogger("pulse.app")
STATIC = Path(__file__).parent / "static"
BANNER = report_mod.BANNER
MAX_UPLOAD = 1024 ** 3
CHUNK = 1024 * 1024
ZONE_KINDS = ("car", "desk", "entrance", "lounge", "other")
ROLES = ("staff", "customer")
MIN_ZONE_AREA = 1e-4
MAX_ZONES = 50
MAX_POLY_POINTS = 200
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOSTS = ("localhost", "127.0.0.1", "[::1]", "testserver")
WS_MAX_FRAME = 2 * 1024 * 1024        # live JPEG frame
WS_MAX_TEXT = 64 * 1024               # live control message
WS_MAX_ZONE_FILE = 1_000_000
WS_IDLE_S = 120.0                     # no message from the client for this long: close
WS_SEND_S = 10.0                      # a client that stops reading is dropped
WS_MAX_BAD = 5                        # consecutive oversize messages before the socket is closed
WS_MAX_SIZE = 2 * WS_MAX_FRAME        # pass as uvicorn ws_max_size (--ws-max-size) when serving
SKIP_WARNING = "Roles skipped: everyone counted as a customer"


class ZoneError(ValueError):
    pass


def validate_zones(doc) -> list[Zone]:
    """Zone file dict -> Zones, or ZoneError with a human-readable reason."""
    if not isinstance(doc, dict) or not isinstance(doc.get("zones"), list):
        raise ZoneError("Body must be an object with a 'zones' list")
    if len(doc["zones"]) > MAX_ZONES:
        raise ZoneError(f"At most {MAX_ZONES} zones are allowed")
    zones, seen = [], set()
    for i, z in enumerate(doc["zones"], 1):
        if not isinstance(z, dict):
            raise ZoneError(f"Zone {i}: must be an object")
        zid, name, kind, poly = z.get("zone_id"), z.get("name"), z.get("kind"), z.get("polygon")
        if not isinstance(zid, str) or not zid.strip():
            raise ZoneError(f"Zone {i}: zone_id must be a non-empty string")
        label = f"Zone '{zid}'"
        if zid in seen:
            raise ZoneError(f"{label}: duplicate zone_id")
        seen.add(zid)
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ZoneError(f"{label}: name must be 1 to 80 characters")
        if kind not in ZONE_KINDS:
            raise ZoneError(f"{label}: kind must be one of {', '.join(ZONE_KINDS)}")
        if not isinstance(poly, list) or len(poly) < 3:
            raise ZoneError(f"{label}: polygon needs at least 3 points")
        if len(poly) > MAX_POLY_POINTS:
            raise ZoneError(f"{label}: at most {MAX_POLY_POINTS} points per polygon")
        pts = []
        for p in poly:
            ok = (isinstance(p, (list, tuple)) and len(p) == 2
                  and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p))
            if not ok or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in p):
                raise ZoneError(f"{label}: every point must be [x, y] with finite values in [0, 1]")
            pts.append((float(p[0]), float(p[1])))
        if zones_mod.polygon_area(pts) <= MIN_ZONE_AREA:
            raise ZoneError(f"{label}: polygon area is too small")
        zones.append(Zone(zid, name.strip(), kind, pts, "user"))
    return zones


def host_name(hostport: str) -> str:
    """'Localhost:8000' -> 'localhost', '[::1]:8000' -> '[::1]'."""
    h = (hostport or "").strip().lower()
    if h.startswith("["):
        return h[:h.find("]") + 1] if "]" in h else h
    return h.split(":")[0]


def allowed_hosts() -> set[str]:
    extra = {host_name(x) for x in os.environ.get("PULSE_ALLOWED_HOSTS", "").split(",") if x.strip()}
    return set(DEFAULT_HOSTS) | extra


class HostOriginGuard:
    """HTTP only: unknown Host header -> 400 (DNS rebinding); unsafe method with a foreign Origin
    -> 403 (cross-site form posts). WebSockets are checked inside the route, before accept()."""

    def __init__(self, app, hosts: set[str]):
        self.app, self.hosts = app, hosts

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            host = headers.get("host", "")
            resp = None
            if host_name(host) not in self.hosts:
                resp = JSONResponse({"error": "Host not allowed"}, status_code=400)
            elif scope["method"] not in ("GET", "HEAD", "OPTIONS") and "origin" in headers \
                    and urlparse(headers["origin"]).netloc.lower() != host.lower():
                resp = JSONResponse({"error": "Cross-origin request rejected"}, status_code=403)
            if resp is not None:
                return await resp(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(jobs_dir: Path = Path("jobs"), cache_root: Path | None = None,
               engine_factory=None) -> FastAPI:
    jobs_dir = Path(jobs_dir)
    store = JobStore(jobs_dir)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pulse-worker")
    engine_lock = threading.Lock()
    engine_box: list = []

    def get_engine():
        with engine_lock:                       # one shared engine, created on first job
            if not engine_box:
                if engine_factory is None:
                    from pulse.engine import PerceptionEngine
                    engine_box.append(PerceptionEngine(get_config("short")))
                elif inspect.signature(engine_factory).parameters:
                    engine_box.append(engine_factory(get_config("short")))
                else:
                    engine_box.append(engine_factory())
            return engine_box[0]

    live_lock = threading.Lock()
    live_box: list = []
    live_state = {"active": False}

    def get_live_engine():
        # live mode owns its engine: a running upload job never shares tracker state with it
        with live_lock:
            if not live_box:
                if engine_factory is None:
                    from pulse.engine import PerceptionEngine
                    live_box.append(PerceptionEngine(get_config("short")))
                elif inspect.signature(engine_factory).parameters:
                    live_box.append(engine_factory(get_config("short")))
                else:
                    live_box.append(engine_factory())
            return live_box[0]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        jobs_mod.cleanup(jobs_dir, max_age_s=jobs_mod.MAX_AGE_S)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Showroom Pulse", lifespan=lifespan)
    hosts = allowed_hosts()
    app.state.allowed_hosts = hosts
    app.add_middleware(HostOriginGuard, hosts=hosts)
    app.state.jobs_dir = jobs_dir
    app.state.store = store

    @app.middleware("http")
    async def _early_size_check(request: Request, call_next):
        # reject oversized uploads before Starlette spools the multipart body to disk
        if request.method == "POST" and request.url.path == "/api/jobs":
            try:
                declared = int(request.headers.get("content-length", "0"))
            except ValueError:
                declared = 0
            if declared > MAX_UPLOAD:
                return JSONResponse({"error": "File too large (limit 1 GB)"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse({"error": e.message}, status_code=e.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException):
        return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, e: RequestValidationError):
        return JSONResponse({"error": "Invalid request: " + "; ".join(
            str(x.get("msg", "")) for x in e.errors())}, status_code=422)

    def job_of(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise ApiError(404, "Unknown job")
        return job

    def need_ready(job: Job) -> None:
        if job.state == "failed":
            raise ApiError(409, job.message or "Job failed")
        if job.state not in ("ready_for_zones", "scored"):
            raise ApiError(409, "Video is still being analysed")

    # ------------------------------------------------------------------ background work
    def perceive_task(job: Job) -> None:
        try:
            with job.lock:
                job.state, job.message = "perceiving", "Detecting and tracking people"
                job.save()
            cfg = get_config(job.preset)
            engine = get_engine()
            root = Path(cache_root) if cache_root is not None else job.dir / "cache"

            def progress(f: float) -> None:
                job.progress = min(0.95, max(0.0, f) * 0.95)

            # `embedder`: dependency-injection hook for tests (stub engines carry a light HSV-only
            # Embedder); the real PerceptionEngine has no such attribute, so perceive builds DINOv2.
            pc = pipeline.perceive(job.input_path, root, cfg, progress=progress, engine=engine,
                                   embedder=getattr(engine, "embedder", None))
            job.progress, job.message = 0.96, "Preparing preview"
            if pc.meta.get("trimmed"):
                job.base_warnings.append(f"Video is longer than {int(cfg.max_duration_s)} s; only the "
                                    f"first {int(cfg.max_duration_s)} s were analysed")
            # preview outputs (keyframe, blurred thumbnails) with automatic zones and suggested roles
            analysis = pipeline.score(pc, cfg)
            pipeline.write_outputs(pc, analysis, job.out_dir, cfg, video=job.input_path,
                                   render_video=False)
            with job.lock:
                job.set_analysis_warnings(analysis.stats.warnings)
                job.pc, job.cache_dir = pc, str(pc.dir)
                job.state, job.progress, job.message = "ready_for_zones", 1.0, "Ready for zones"
                job.save()
        except sources.VideoError as e:
            fail(job, str(e).replace(str(job.input_path), "the uploaded file"))
        except Exception as e:  # noqa: BLE001 - any failure becomes a readable job state
            log.exception("job %s failed", job.job_id)
            fail(job, f"Analysis failed unexpectedly ({type(e).__name__}). Try another video.")

    def fail(job: Job, message: str) -> None:
        with job.lock:
            job.state, job.message = "failed", message
            job.save()

    def render_task(job: Job, analysis, pc, cfg, generation: int) -> None:
        out = job.out_dir / "annotated.mp4"
        try:
            overlays = report_mod.build_overlays(analysis, pc, cfg)
            render_mod.render(job.input_path, overlays, out, cfg, report_mod.observations_by_t(pc))
            with job.lock:
                if job.generation != generation:           # zones changed meanwhile: result is stale
                    out.unlink(missing_ok=True)
                    return
                rp = job.out_dir / "report.json"
                rep = json.loads(rp.read_text())
                rep["files"]["annotated_video"] = "annotated.mp4"
                atomic_write_text(rp, json.dumps(rep, indent=2))
                job.video_state = "ready"
                job.save()
        except Exception:  # noqa: BLE001
            log.exception("render failed for %s", job.job_id)
            with job.lock:
                if job.generation == generation:
                    job.video_state = "failed"
                    job.save()

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/live", include_in_schema=False)
    def live():
        return FileResponse(STATIC / "live.html")

    def live_zone_file(raw) -> Path:
        """A zone file path from a live client: an existing .json (<= 1 MB) under labels/zones/ of the
        repo or under the jobs dir. Relative paths are taken from the repo root, never the CWD."""
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ZoneError("load_zones needs a path")
        allowed = [(REPO_ROOT / "labels" / "zones").resolve(), jobs_dir.resolve()]
        try:
            p = (Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw).resolve()  # follows '..', symlinks
            ok = (any(p.is_relative_to(a) for a in allowed) and p.suffix == ".json" and p.is_file()
                  and p.stat().st_size <= WS_MAX_ZONE_FILE)
        except OSError:
            ok = False
        if not ok:
            raise ZoneError("Zone files must be .json files (up to 1 MB) under labels/zones/ or jobs/")
        return p

    def step_locked(session, t, frame):
        with live_lock:                          # the engine is shared: one step at a time
            return session.step(t, frame)

    def make_session(engine):
        from pulse.live import LiveSession
        with live_lock:
            return LiveSession(get_config("short"), engine)   # resets the engine once

    def release_slot_later() -> None:
        """A cancelled step keeps running in its thread: free the slot only once it has finished."""
        def run():
            with live_lock:
                live_state["active"] = False
        threading.Thread(target=run, name="pulse-live-release", daemon=True).start()

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        host = ws.headers.get("host", "")
        origin = ws.headers.get("origin")
        if host_name(host) not in hosts or (origin is not None
                                            and urlparse(origin).netloc.lower() != host.lower()):
            await ws.close(code=1008)               # cross-site / rebinding attempt: never accepted
            return
        await ws.accept()
        if live_state["active"]:
            await ws.send_json({"type": "error", "message": "Another live session is already running"})
            await ws.close(code=1013)
            return
        live_state["active"] = True
        worker_task = None

        async def send(msg: dict) -> None:
            await asyncio.wait_for(ws.send_json(msg), WS_SEND_S)

        try:
            try:
                engine = await run_in_threadpool(get_live_engine)
                session = await run_in_threadpool(make_session, engine)
            except Exception:  # noqa: BLE001 - reported to the client, logged here
                log.exception("live engine unavailable")
                await send({"type": "error", "message": "Live engine unavailable"})
                await ws.close(code=1011)
                return
            t0 = time.monotonic()
            pending: dict = {"data": None, "t": 0.0}
            wake = asyncio.Event()
            bad = {"n": 0}

            async def reject(message: str) -> bool:
                bad["n"] += 1
                await send({"type": "error", "message": message})
                if bad["n"] >= WS_MAX_BAD:
                    await ws.close(code=1009)
                    return True
                return False

            async def control(text: str) -> None:
                try:
                    msg = json.loads(text)
                    if not isinstance(msg, dict):
                        raise ValueError
                except ValueError:
                    return await send({"type": "error", "message": "Control message is not valid JSON"})
                kind = msg.get("type")
                try:
                    if kind == "zones":
                        zs = validate_zones({"zones": msg.get("zones")})
                        await run_in_threadpool(session.set_zones, zs)
                    elif kind == "load_zones":
                        zs = validate_zones(json.loads(live_zone_file(msg.get("path")).read_text()))
                        await run_in_threadpool(session.set_zones, zs)
                        await send({"type": "zones", "zones": [z.to_dict() for z in zs]})
                    elif kind == "preset":
                        if msg.get("name") not in PRESET_DEFS:
                            raise ZoneError("preset must be 'short' or 'dealership'")
                        await run_in_threadpool(session.set_preset, msg["name"])
                    elif kind == "toggle":
                        x, y = msg.get("x"), msg.get("y")
                        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                   and math.isfinite(v) for v in (x, y)):
                            raise ZoneError("toggle needs numeric x and y")
                        await run_in_threadpool(session.toggle_role_at, float(x), float(y))
                    else:
                        raise ZoneError(f"Unknown message type {kind!r}")
                except (ZoneError, OSError, ValueError) as e:
                    await send({"type": "error", "message": str(e) if isinstance(e, ZoneError)
                                else "Could not read that zone file"})

            async def reader() -> None:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), WS_IDLE_S)
                    except asyncio.TimeoutError:
                        await ws.close(code=1001)       # idle client must not hold the only slot
                        return
                    if msg["type"] == "websocket.disconnect":
                        return
                    data, text = msg.get("bytes"), msg.get("text")
                    if data is not None:
                        if len(data) > WS_MAX_FRAME:
                            if await reject("Frame too large (limit 2 MB)"):
                                return
                            continue
                        bad["n"] = 0
                        if pending["data"] is not None:
                            session.dropped += 1          # newest frame wins; the stale one is dropped
                        pending["data"], pending["t"] = data, time.monotonic() - t0
                        wake.set()
                    elif text is not None:
                        if len(text) > WS_MAX_TEXT:
                            if await reject("Control message too large (limit 64 KB)"):
                                return
                            continue
                        bad["n"] = 0
                        await control(text)

            async def worker() -> None:
                try:
                    while True:
                        await wake.wait()
                        wake.clear()
                        data, t = pending["data"], pending["t"]
                        pending["data"] = None
                        if data is None:
                            continue
                        try:
                            frame = await run_in_threadpool(sources.decode_jpeg, data,
                                                            session.cfg.max_width)
                        except ValueError:
                            await send({"type": "error", "message": "Frame is not a decodable JPEG"})
                            continue
                        try:
                            overlay = await asyncio.to_thread(step_locked, session, t, frame)
                        except Exception:  # noqa: BLE001 - one bad frame must not end the session
                            log.exception("live step failed")
                            await send({"type": "error", "message": "Frame analysis failed"})
                            continue
                        await send(overlay)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - a dead socket ends the worker, never silently
                    log.warning("live send failed; closing session", exc_info=True)
                    try:
                        await ws.close(code=1008)
                    except Exception:  # noqa: BLE001
                        pass

            worker_task = asyncio.create_task(worker())
            try:
                await reader()               # returns when the client disconnects or is dropped
            finally:
                worker_task.cancel()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("live session crashed")
        finally:
            release_slot_later()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/health")
    def health():
        return {"ok": True, "perception_runs": cache_mod.perception_runs(), "banner": BANNER}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), preset: str = Form("short")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise ApiError(400, "Unsupported file type; use " + ", ".join(e[1:] for e in ALLOWED_EXT))
        if preset not in PRESETS:
            raise ApiError(400, "preset must be 'short' or 'dealership'")
        job = store.create(preset, ext)
        size = 0
        try:
            with open(job.input_path, "wb") as f:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise ApiError(413, "File too large (limit 1 GB)")
                    f.write(chunk)
        except BaseException:
            store.discard(job)
            raise
        if size == 0:
            store.discard(job)
            raise ApiError(400, "Empty file")
        job.save()
        executor.submit(perceive_task, job)
        return {"job_id": job.job_id, "banner": BANNER}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return {**job_of(job_id).status(), "banner": BANNER}

    @app.get("/api/jobs/{job_id}/keyframe.jpg")
    def keyframe(job_id: str):
        job = job_of(job_id)
        p = job.out_dir / "keyframe.jpg"
        if not p.is_file():
            raise ApiError(404, "Keyframe not ready")
        return FileResponse(p, media_type="image/jpeg")

    # ------------------------------------------------------------------ zones
    @app.get("/api/jobs/{job_id}/zones")
    def get_zones(job_id: str):
        job = job_of(job_id)
        need_ready(job)
        with job.lock:
            if not job.zones_path.is_file():
                pc = job.get_pc()
                size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
                zones_mod.save_zones(job.zones_path, zones_mod.auto_zones(pc.cars, (size[1], size[0])),
                                     size)
            return json.loads(job.zones_path.read_text())

    @app.put("/api/jobs/{job_id}/zones")
    async def put_zones(job_id: str, request: Request):
        job = job_of(job_id)
        need_ready(job)
        try:
            doc = await request.json()
            zones = validate_zones(doc)
        except ZoneError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        except ValueError:
            return JSONResponse({"error": "Body is not valid JSON"}, status_code=422)
        pc = job.get_pc()
        size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
        with job.lock:
            zones_mod.save_zones(job.zones_path, zones, size)
            return json.loads(job.zones_path.read_text())

    # ------------------------------------------------------------------ people / roles
    def read_people(job: Job) -> list[dict]:
        p = job.out_dir / "people.json"
        if not p.is_file():
            return []
        out = []
        for e in json.loads(p.read_text())["persons"]:
            pid = e["person_id"]
            role = job.overrides.get(pid) or job.imported.get(pid) or e["suggested_role"] or e["role"]
            thumb = e.get("thumbnail")
            out.append({"person_id": pid, "role": role, "suggested_role": e["suggested_role"],
                        "confidence": e["confidence"], "reasons": e["reasons"],
                        "thumb_url": f"/api/jobs/{job.job_id}/files/{thumb}" if thumb else None,
                        "first_t": e["first_t"], "last_t": e["last_t"]})
        return out

    @app.get("/api/jobs/{job_id}/people")
    def get_people(job_id: str):
        job = job_of(job_id)
        need_ready(job)
        return read_people(job)

    @app.put("/api/jobs/{job_id}/roles")
    async def put_roles(job_id: str, request: Request):
        job = job_of(job_id)
        need_ready(job)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "Body is not valid JSON"}, status_code=422)
        if not isinstance(body, dict) or not isinstance(body.get("overrides", {}), dict):
            return JSONResponse({"error": "Expected {\"overrides\": {...}, \"skip\": bool}"},
                                status_code=422)
        known = {p["person_id"] for p in read_people(job)}
        parsed: dict[int, str] = {}
        for k, v in body.get("overrides", {}).items():
            try:
                pid = int(k)
            except (TypeError, ValueError):
                return JSONResponse({"error": f"Bad person id {k!r}"}, status_code=422)
            if pid not in known:
                return JSONResponse({"error": f"Unknown person {pid}"}, status_code=422)
            if v not in ROLES:
                return JSONResponse({"error": f"Role for person {pid} must be staff or customer"},
                                    status_code=422)
            parsed[pid] = v
        with job.lock:
            job.roles_skipped = bool(body.get("skip"))     # skip = everyone is a customer
            job.overrides = {} if job.roles_skipped else parsed
            job.save()
        return {"overrides": {str(k): v for k, v in job.overrides.items()}, "skip": bool(body.get("skip"))}

    @app.post("/api/jobs/{job_id}/roles/import")
    async def import_roles(job_id: str, file: UploadFile = File(...)):
        job = job_of(job_id)
        need_ready(job)
        data = await file.read(CHUNK + 1)
        if len(data) > CHUNK:
            raise ApiError(413, "roles.json is too large")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "roles.json"
            p.write_bytes(data)
            try:
                imported = roles_mod.load_roles_json(p)
            except ValueError as e:
                return JSONResponse({"error": str(e).replace(str(p), "roles.json")}, status_code=422)
        with job.lock:
            job.imported = imported
            job.save()
        return {"imported": len(imported)}

    # ------------------------------------------------------------------ score / results
    def read_report(job: Job) -> dict:
        p = job.out_dir / "report.json"
        with job.lock:                       # a score / render rewrite is never read half-done
            if job.state != "scored" or not p.is_file():
                raise ApiError(404, "Not scored yet")
            return json.loads(p.read_text())

    @app.post("/api/jobs/{job_id}/score")
    async def score_job(job_id: str, request: Request):
        job = job_of(job_id)
        need_ready(job)
        raw = await request.body()
        # scoring is CPU/disk heavy: run it in the threadpool so status polling is not blocked
        return await run_in_threadpool(_score, job, raw)

    def _score(job: Job, raw: bytes):
        job_id = job.job_id
        try:
            body = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return JSONResponse({"error": "Body is not valid JSON"}, status_code=422)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Body must be an object"}, status_code=422)
        preset = body.get("preset", job.preset)
        if preset not in PRESETS:
            return JSONResponse({"error": "preset must be 'short' or 'dealership'"}, status_code=422)
        render_video = bool(body.get("render_video", True))
        with job.lock:
            try:
                cfg = get_config(preset)
                pc = job.get_pc()
                zones = zones_mod.load_zones(job.zones_path) if job.zones_path.is_file() else None
                if job.roles_skipped:     # spec 3: skip means everyone is a customer
                    everyone = {p["person_id"]: "customer" for p in read_people(job)}
                    analysis = pipeline.score(pc, cfg, zones=zones, overrides=everyone)
                    analysis.stats.warnings.append(SKIP_WARNING)
                else:
                    analysis = pipeline.score(pc, cfg, zones=zones, overrides=job.overrides or None,
                                              imported_roles=job.imported or None)
                pipeline.write_outputs(pc, analysis, job.out_dir, cfg, video=job.input_path,
                                       render_video=False)
            except Exception as e:  # noqa: BLE001
                log.exception("scoring %s failed", job_id)
                raise ApiError(500, f"Scoring failed ({type(e).__name__})") from e
            job.set_analysis_warnings(analysis.stats.warnings)
            job.preset, job.state, job.generation = preset, "scored", job.generation + 1
            job.video_state = "rendering" if render_video else "none"
            job.save()
            gen = job.generation
            if render_video:
                executor.submit(render_task, job, analysis, pc, cfg, gen)
            return read_report(job)

    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str):
        return read_report(job_of(job_id))

    @app.get("/api/jobs/{job_id}/export.json")
    def export_json(job_id: str):
        job = job_of(job_id)
        read_report(job)
        return FileResponse(job.out_dir / "report.json", media_type="application/json",
                            filename="showroom-pulse-report.json")

    @app.get("/api/jobs/{job_id}/export.csv")
    def export_csv(job_id: str):
        job = job_of(job_id)
        read_report(job)
        return FileResponse(job.out_dir / "per_car.csv", media_type="text/csv",
                            filename="showroom-pulse-per-car.csv")

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def files(job_id: str, path: str):
        job = job_of(job_id)
        out = job.out_dir.resolve()
        parts = path.replace("\\", "/").split("/")
        if (not path or path.startswith("/") or "\x00" in path or ".." in parts
                or "\\" in path):
            raise ApiError(404, "Not found")
        target = (out / path).resolve()
        if not target.is_relative_to(out) or not target.is_file():   # resolve() follows symlinks
            raise ApiError(404, "Not found")
        return FileResponse(target)

    return app


app = create_app()
