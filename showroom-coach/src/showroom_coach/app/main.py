"""The FastAPI app: REST job API, static pages and the live WebSocket (api-contract.md is the binding reference).

    GET  /                               static/index.html (the upload -> confirm -> results app)
    GET  /live                           static/live.html (the live camera page)
    GET  /static/...                     the frontend's files
    GET  /api/presets                    {name: {label, thresholds}}
    POST /api/jobs                       multipart `file` + form `preset` -> {"job_id"}; 411 / 413 / 415 / 400
    GET  /api/jobs/{id}                  job status; 404 unknown
    GET  /api/jobs/{id}/people           confirm-view people; 409 before needs_confirm
    POST /api/jobs/{id}/roles            {"roles", "names", "preset"} -> results (instant re-score, video re-rendered)
    GET  /api/jobs/{id}/results          results; 409 before needs_confirm
    POST /api/jobs/{id}/sales            multipart `file` = sales CSV (salesperson,timestamp,sold; <= 1 MB) -> results
                                         with sold/conversion per salesperson; 400 / 409 / 411 / 413 / 415
    DELETE /api/jobs/{id}/sales          forget the sales CSV -> results; 409 before needs_confirm
    GET  /api/jobs/{id}/files/{name}     whitelisted outputs (HTTP Range supported, so the video can seek)
    WS   /ws/live?preset=short           live camera protocol (live_ws.serve_live)

Everything runs on this computer. Uploads are bounded before a byte of the body is read: a request without a declared
Content-Length, or sent with chunked Transfer-Encoding, is refused with 411 (browsers always declare the length, and
h11 never delivers more body than declared), and a declared length over 1 GB (plus 1 MiB of multipart framing) with
413. Starlette then parses the multipart body, spooling the file part to a temporary file (bounded by that length), so
the file type (415) and the preset (400) are checked only after the body has arrived, but before anything is kept in
the jobs directory. The optional sales CSV (at most 1 MB) is read into memory the same way and kept only as parsed rows. The file is copied there in 1 MB chunks with the 1 GB cap counted again, under an ASCII name
(`<id>/upload<ext>`), never under the client's file name. Job directories older than 24 h are deleted when the app
starts.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from .. import config
from ..live import LiveSession
from ..live_ws import serve_live
from ..metrics import SalesCsvError, parse_sales_csv
from ..perception.stack import PerceptionStack, default_stack
from ..types import to_jsonable
from .jobs import ALLOWED_EXTENSIONS, JobConflict, JobManager, upload_extension

log = logging.getLogger(__name__)

__all__ = ["create_app", "MAX_UPLOAD_BYTES"]

STATIC_DIR = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 1 << 30            # 1 GB; read at request time
MULTIPART_SLACK = 1 << 20             # form fields and boundaries on top of the file itself
CHUNK = 1 << 20
MAX_SALES_BYTES = 1 << 20             # a sales CSV; read at request time
SALES_EXTENSIONS = (".csv", ".txt")
SALES_TYPES = ("text/csv", "application/csv", "application/vnd.ms-excel", "text/plain")   # browsers differ on Windows
MAX_NAME_CHARS = 40
ROLES = ("staff", "customer")
NO_STORE = {"Cache-Control": "no-store"}
NO_CACHE = {"Cache-Control": "no-cache"}
MEDIA_TYPES = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
               ".csv": "text/csv; charset=utf-8"}
class RolesBody(BaseModel):
    roles: dict[str, str] = {}
    names: dict[str, str | None] = {}
    preset: str | None = None


class _TooLarge(Exception):
    pass


class _NoCacheStatic(StaticFiles):
    """The frontend changes between runs during the hackathon; make browsers revalidate (ETag) instead of guessing."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


def _copy_capped(src, dst: Path, limit: int) -> int:
    """Copy the file object `src` to `dst` in 1 MB chunks; _TooLarge as soon as more than `limit` bytes came."""
    src.seek(0)
    n = 0
    with open(dst, "wb") as out:
        while chunk := src.read(CHUNK):
            n += len(chunk)
            if n > limit:
                raise _TooLarge()
            out.write(chunk)
    return n


def _pid_map(values: dict, what: str) -> dict[int, object]:
    out = {}
    for key, value in values.items():
        try:
            out[int(key)] = value
        except ValueError:
            raise HTTPException(400, f"{what}: {key!r} is not a person id") from None
    return out


def create_app(stack_factory: Callable[[], PerceptionStack] | None = None, jobs_root: Path | None = None,
               live_session_factory: Callable[[str], LiveSession] | None = None) -> FastAPI:
    """The web app. `stack_factory()` gives the perception stack for file jobs (default: the real stack with
    `config.DEFAULT_DETECTOR`); `live_session_factory(preset)` gives a `LiveSession` per WebSocket connection (default:
    the real stack with `config.LIVE_DETECTOR`, or `stack_factory`'s stack when one was injected, e.g. in tests);
    `jobs_root` defaults to `config.jobs_dir()`."""
    if live_session_factory is None:
        if stack_factory is None:
            def live_session_factory(preset: str) -> LiveSession:
                return LiveSession(default_stack(config.LIVE_DETECTOR), preset=preset)
        else:
            injected = stack_factory

            def live_session_factory(preset: str) -> LiveSession:
                return LiveSession(injected(), preset=preset)
    if stack_factory is None:
        def stack_factory() -> PerceptionStack:
            return default_stack(config.DEFAULT_DETECTOR)

    manager = JobManager(Path(jobs_root) if jobs_root is not None else config.jobs_dir(), stack_factory)
    live_factory = live_session_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        removed = await run_in_threadpool(manager.cleanup)
        log.info("startup cleanup: %d job director%s older than %g h deleted from %s", removed,
                 "y" if removed == 1 else "ies", manager.max_age_h, manager.root)
        try:
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(title="Showroom Coach", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.jobs = manager

    # ------------------------------------------------------------------------------------------------ pages

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html", headers=NO_STORE)

    @app.get("/live", include_in_schema=False)
    def live_page():
        page = STATIC_DIR / "live.html"
        if not page.is_file():
            raise HTTPException(404, "The live camera page is not installed.")
        return FileResponse(page, media_type="text/html", headers=NO_STORE)

    app.mount("/static", _NoCacheStatic(directory=STATIC_DIR), name="static")

    # -------------------------------------------------------------------------------------------------- API

    @app.get("/api/presets")
    def presets():
        return {name: {"label": config.PRESET_LABELS.get(name, name), "thresholds": to_jsonable(th)}
                for name, th in config.PRESETS.items()}

    @app.post("/api/jobs")
    async def create_job(request: Request):
        limit = MAX_UPLOAD_BYTES
        length = request.headers.get("content-length", "").strip()
        chunked = "chunked" in request.headers.get("transfer-encoding", "").lower()
        if chunked or not length.isdigit():
            # Without a declared length nothing bounds what Starlette would spool to disk while parsing the form.
            raise HTTPException(411, "The upload must declare its size (Content-Length); chunked uploads are refused.")
        if int(length) > limit + MULTIPART_SLACK:
            raise HTTPException(413, "The file is larger than 1 GB.")
        async with request.form(max_files=1, max_fields=10) as form:
            upload = form.get("file")
            preset = form.get("preset") or "short"
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "No video file was sent (form field 'file').")
            if upload_extension(upload.filename) is None:
                raise HTTPException(415, f"Unsupported file type. Use {', '.join(e[1:].upper() for e in ALLOWED_EXTENSIONS)}.")
            if not isinstance(preset, str) or preset not in config.PRESETS:
                raise HTTPException(400, f"Unknown preset; choose one of {', '.join(sorted(config.PRESETS))}.")
            part = manager.incoming_path()
            try:
                await run_in_threadpool(_copy_capped, upload.file, part, limit)
            except _TooLarge:
                part.unlink(missing_ok=True)
                raise HTTPException(413, "The file is larger than 1 GB.") from None
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            filename = upload.filename
        try:
            job = await run_in_threadpool(manager.create, filename, part, preset)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        log.info("job %s created (%s, preset %s)", job.id, job.filename, preset)
        return {"job_id": job.id}

    def _job_or_404(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job. It may have been deleted, or the server was restarted.") from None

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return _job_or_404(job_id).public()

    @app.get("/api/jobs/{job_id}/people")
    def job_people(job_id: str):
        _job_or_404(job_id)
        try:
            return {"people": manager.people(job_id)}
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/jobs/{job_id}/roles")
    def job_roles(job_id: str, body: RolesBody):
        _job_or_404(job_id)
        roles = _pid_map(body.roles, "roles")
        bad = sorted({r for r in roles.values() if r not in ROLES})
        if bad:
            raise HTTPException(400, f"roles must be 'staff' or 'customer', got {', '.join(map(repr, bad))}")
        names = {pid: ("" if n is None else str(n).strip()[:MAX_NAME_CHARS])
                 for pid, n in _pid_map(body.names, "names").items()}
        if body.preset is not None and body.preset not in config.PRESETS:
            raise HTTPException(400, f"Unknown preset; choose one of {', '.join(sorted(config.PRESETS))}.")
        try:
            manager.set_roles(job_id, roles, names, body.preset)
            return manager.results(job_id)
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/jobs/{job_id}/results")
    def job_results(job_id: str):
        _job_or_404(job_id)
        try:
            return manager.results(job_id)
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/jobs/{job_id}/sales")
    async def job_sales(job_id: str, request: Request):
        _job_or_404(job_id)
        limit = MAX_SALES_BYTES
        length = request.headers.get("content-length", "").strip()
        if "chunked" in request.headers.get("transfer-encoding", "").lower() or not length.isdigit():
            raise HTTPException(411, "The upload must declare its size (Content-Length); chunked uploads are refused.")
        if int(length) > limit + MULTIPART_SLACK:
            raise HTTPException(413, "The sales CSV is larger than 1 MB.")
        async with request.form(max_files=1, max_fields=10) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "No file was sent (form field 'file').")
            kind = (upload.content_type or "").split(";")[0].strip().lower()
            if not ((upload.filename or "").lower().endswith(SALES_EXTENSIONS) or kind in SALES_TYPES):
                raise HTTPException(415, "Unsupported file type. Use a CSV file (salesperson,timestamp,sold).")
            data = await upload.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "The sales CSV is larger than 1 MB.")
        try:
            rows = parse_sales_csv(data.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise HTTPException(400, "The sales CSV must be UTF-8 text. In Excel use Save As > CSV UTF-8.") from None
        except SalesCsvError as exc:
            raise HTTPException(400, f"Sales CSV: {exc}") from None

        def store() -> dict:
            manager.set_sales(job_id, rows, rows.skipped)
            return manager.results(job_id)
        try:
            return await run_in_threadpool(store)
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.delete("/api/jobs/{job_id}/sales")
    def job_sales_clear(job_id: str):
        _job_or_404(job_id)
        try:
            manager.clear_sales(job_id)
            return manager.results(job_id)
        except JobConflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str):
        try:
            path = manager.file(job_id, name)
        except KeyError:
            raise HTTPException(404, "No such file.") from None
        # Versioned videos and person thumbnails never change under their name; everything else is rewritten in place
        # by a re-score (and annotated.mp4 is an alias of the latest video), so browsers must revalidate those.
        stable = name.startswith("people/") or (name.startswith("annotated-") and name.endswith(".mp4"))
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower()),
                            headers=None if stable else NO_CACHE)

    # ------------------------------------------------------------------------------------------------- live

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        await serve_live(websocket, live_factory)

    return app
