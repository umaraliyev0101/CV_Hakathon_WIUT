"""FastAPI app: upload a video, follow progress, edit zones and roles, read results, re-score.

Perception (models) runs once per video in a single background worker. Everything after it
(zones, roles, presets) is `pipeline.score` + `write_outputs` on the cached perception output, so
editing zones never re-runs a model.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from pulse import cache as cache_mod
from pulse import pipeline, render as render_mod, report as report_mod, roles as roles_mod
from pulse import sources, zones as zones_mod
from pulse.config import get_config
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


class ZoneError(ValueError):
    pass


def validate_zones(doc) -> list[Zone]:
    """Zone file dict -> Zones, or ZoneError with a human-readable reason."""
    if not isinstance(doc, dict) or not isinstance(doc.get("zones"), list):
        raise ZoneError("Body must be an object with a 'zones' list")
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        jobs_mod.cleanup(jobs_dir, max_age_s=jobs_mod.MAX_AGE_S)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Showroom Pulse", lifespan=lifespan)
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
            job.overrides = {} if body.get("skip") else parsed     # skip = keep suggested roles only
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
