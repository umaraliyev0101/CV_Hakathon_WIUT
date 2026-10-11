"""Background jobs for the web app: one uploaded video each, from perception to an annotated video.

    queued -> perceiving -> analyzing -> needs_confirm -> rendering -> done          (or failed at any point)

A job lives in `<root>/<id>/`: `upload<ext>` (the video, stored under an ASCII name), `cache/` (its perception cache),
`people/<pid>.jpg` (face-blurred thumbnails), `results.json`, `events.csv`, `scorecard.csv`, `timeline.png`,
`annotated-<rev>.mp4` and `contact_sheets/`. Everything a job writes stays in its directory, and `cleanup` (run when the
app starts) deletes directories older than `max_age_h` hours, so uploads never outlive a day on this computer.

The worker thread of a job probes the file first (so an unreadable one fails at once, even while another job is
running), then runs the cached perception pass (progress 0-0.9: finding people over the frames, then describing them),
analyses it with suggested roles (0.9-1.0), saves thumbnails and the first exports and stops at `needs_confirm`. Only
one perception pass runs at a time in the whole process (`PERCEPTION_SLOT`), so two uploads never fight over the CPU;
the second waits as `queued`. Any exception fails the job (message "failed", `error` for the user) and is logged with
its traceback; the server keeps running, and a job never keeps partial results.

After that, `set_roles` re-scores in place (`analysis.rescore`: events and metrics only, never detection), rewrites the
exports and the timeline at once and draws the annotated video in a background thread. Each call bumps the job's
render revision and writes `annotated-<rev>.mp4`, a new name each time, because a browser on Windows keeps the file it
is playing open and an open file cannot be replaced there. A newer call supersedes an older render still in flight
(it stops at its next progress report), and only the latest revision may mark the video ready.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .. import config
from ..analysis import Analysis, analyze, rescore
from ..export import results_dict, write_all
from ..perception import pipeline
from ..render import contact_sheets, render_video, save_thumbnails, timeline_png
from ..sources import VideoError, probe

if TYPE_CHECKING:
    from ..perception.pipeline import PerceptionResult
    from ..perception.stack import PerceptionStack

log = logging.getLogger(__name__)

__all__ = ["ALLOWED_EXTENSIONS", "Job", "JobConflict", "JobManager", "PERCEPTION_SLOT"]

PERCEPTION_SLOT = threading.Semaphore(1)     # one perception pass at a time, process-wide
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
PERCEPTION_SHARE = 0.9                       # perception fills progress 0..0.9; analysis the rest
ACTIVE = ("queued", "perceiving", "analyzing", "rendering")
CONFIRMED_OR_LATER = ("needs_confirm", "rendering", "done")
INCOMING_PREFIX = ".incoming-"               # uploads being received, before they become a job
MAX_FILENAME_CHARS = 200
CLOSE_TIMEOUT_S = 15.0
_JOB_ID = re.compile(r"[0-9a-f]{12}")
# Files a client may fetch from a job directory; anything else (the upload, the cache, partial files) is not served.
_FILES = re.compile(r"(annotated(-\d+)?\.mp4|timeline\.png|results\.json|events\.csv|scorecard\.csv"
                    r"|people/\d+\.jpg|contact_sheets/[\w.-]+\.jpg)", re.ASCII)

M_QUEUED = "waiting for a free worker…"
M_READING = "reading the video…"
M_ANALYZING = "working out who is staff and who is a customer…"
M_THUMBS = "saving blurred thumbnails…"
M_CONFIRM = "ready to confirm who is staff"
M_RENDERING = "drawing the annotated video…"
M_DONE = "done"
M_FAILED = "failed"


class JobConflict(Exception):
    """The job is not in a state that allows this yet (the web layer answers 409)."""


class _Cancelled(Exception):
    """Raised from a progress callback to stop work: the server is shutting down or a newer render took over."""


@dataclass(eq=False)
class Job:
    id: str
    dir: Path
    filename: str
    preset: str
    status: str = "queued"      # queued | perceiving | analyzing | needs_confirm | rendering | done | failed
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    video_status: str = "none"  # none | rendering | ready | failed
    created: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)
    upload: Path | None = None
    # internal state, guarded by `lock`
    res: PerceptionResult | None = field(default=None, repr=False)
    an: Analysis | None = field(default=None, repr=False)
    thumbs: dict[int, str] = field(default_factory=dict, repr=False)
    rev: int = 0                         # render revision; bumped by every set_roles
    video_file: str | None = None        # the latest ready annotated video
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    render_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict:
        """The job status object of the web API (api-contract.md, `GET /api/jobs/{id}`)."""
        with self.lock:
            return {"id": self.id, "filename": self.filename, "preset": self.preset, "status": self.status,
                    "progress": round(float(self.progress), 3), "message": self.message, "error": self.error,
                    "warnings": list(self.warnings), "video_status": self.video_status}


def display_name(filename) -> str:
    """The client's file name for display only: no directories, no control characters, at most 200 characters."""
    name = re.split(r"[\\/]", str(filename or ""))[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:MAX_FILENAME_CHARS] or "video"


def upload_extension(filename) -> str | None:
    """The lower-case extension of `filename` when it is an accepted video type, else None."""
    ext = Path(display_name(filename)).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else None


def _video_message(exc: VideoError, job: Job) -> str:
    text = str(exc) or "the file is not a readable video"
    if job.upload is not None:
        text = text.replace(job.upload.name, job.filename)       # the user knows their name, not ours
    return text if text.lower().startswith("cannot read video") else f"cannot read video: {text}"


class JobManager:
    def __init__(self, root: Path, stack_factory: Callable[[], PerceptionStack], max_age_h: float = 24.0):
        self.root = Path(root)
        self.stack_factory = stack_factory
        self.max_age_h = float(max_age_h)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._closing = threading.Event()

    # ------------------------------------------------------------------------------------------- housekeeping

    def cleanup(self) -> int:
        """Delete job directories (and abandoned partial uploads) last modified more than `max_age_h` hours ago.
        A job that is still working is left alone. Returns the number of job directories removed."""
        if not self.root.is_dir():
            return 0
        cutoff = time.time() - self.max_age_h * 3600.0
        removed = 0
        for p in list(self.root.iterdir()):
            try:
                if p.stat().st_mtime >= cutoff:
                    continue
                if p.is_dir():
                    with self._lock:
                        job = self._jobs.get(p.name)
                    if job is not None and job.status in ACTIVE:
                        continue
                    shutil.rmtree(p, ignore_errors=True)
                    if p.exists():
                        log.warning("could not fully delete old job directory %s", p)
                        continue
                    removed += 1
                    with self._lock:
                        self._jobs.pop(p.name, None)
                elif p.name.startswith(INCOMING_PREFIX):
                    p.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("cleanup skipped %s: %s", p, exc)
        if removed:
            log.info("deleted %d job director%s older than %g h in %s", removed, "y" if removed == 1 else "ies",
                     self.max_age_h, self.root)
        return removed

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> None:
        """Stop: running perception and renders end at their next progress report; waits up to `timeout` s."""
        self._closing.set()
        deadline = time.monotonic() + timeout
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))

    def incoming_path(self) -> Path:
        """A fresh path in the jobs root to receive an upload into before `create` moves it into its job."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{INCOMING_PREFIX}{secrets.token_hex(8)}.part"

    # ------------------------------------------------------------------------------------------------ jobs

    def create(self, filename: str, data_path: Path, preset: str) -> Job:
        """Make a job of the video at `data_path` (moved into the job directory as `upload<ext>`) and start its
        worker thread. ValueError for an unknown preset or a file type that is not accepted."""
        config.get_preset(preset)
        ext = upload_extension(filename)
        if ext is None:
            raise ValueError(f"unsupported file type; use one of {', '.join(ALLOWED_EXTENSIONS)}")
        with self._lock:
            jid = secrets.token_hex(6)
            while jid in self._jobs or (self.root / jid).exists():
                jid = secrets.token_hex(6)
            d = self.root / jid
            d.mkdir(parents=True)
            upload = d / f"upload{ext}"
            try:
                shutil.move(str(data_path), str(upload))
            except BaseException:
                shutil.rmtree(d, ignore_errors=True)
                raise
            job = Job(id=jid, dir=d, filename=display_name(filename), preset=preset, upload=upload, message=M_QUEUED)
            self._jobs[jid] = job
        self._spawn(self._work, job, name=f"job-{jid}")
        return job

    def get(self, job_id: str) -> Job:
        """The job with this id; KeyError if there is none."""
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        with self._lock:
            return self._jobs[job_id]

    def analysis(self, job_id: str) -> Analysis:
        """The latest analysis of the job (from `needs_confirm` on); JobConflict before that."""
        job = self.get(job_id)
        with job.lock:
            return self._analysis_of(job)

    def people(self, job_id: str) -> list[dict]:
        """One entry per person for the confirm view (api-contract.md, `GET /api/jobs/{id}/people`)."""
        job = self.get(job_id)
        with job.lock:
            an = self._analysis_of(job)
            thumbs = dict(job.thumbs)
        out = []
        for p in an.persons:
            s = an.suggestions[p.pid]
            out.append({
                "pid": p.pid,
                "thumb_url": self._url(job, f"people/{thumbs[p.pid]}") if p.pid in thumbs else None,
                "suggestion": {"role": s.role, "confidence": float(s.confidence), "reasons": list(s.reasons)},
                "role": an.roles.get(p.pid, "customer"), "name": an.names.get(p.pid),
                "t_first": float(p.t_first), "t_last": float(p.t_last),
            })
        return out

    def set_roles(self, job_id: str, roles: dict[int, str], names: dict[int, str], preset: str | None) -> Analysis:
        """Re-score with these roles, names and preset (people left out keep theirs; a "" name removes it), write
        the exports and the timeline, then draw the annotated video in a background thread. Never re-detects.
        KeyError for an unknown job, JobConflict before `needs_confirm`, ValueError for an unknown preset."""
        job = self.get(job_id)
        if preset is not None:
            config.get_preset(preset)
        with job.lock:
            if job.an is None or job.res is None or job.status not in CONFIRMED_OR_LATER:
                raise JobConflict(f"the job is {job.status}; roles can be set once people have been found")
            an = rescore(job.an, job.res, roles=roles, names=names, preset=preset)
            self._write_exports(job, an)
            job.an, job.preset = an, an.preset
            job.rev += 1
            rev = job.rev
            job.status, job.video_status, job.message, job.error = "rendering", "rendering", M_RENDERING, None
        self._spawn(self._render, job, rev, an, name=f"render-{job.id}-{rev}")
        return an

    def results(self, job_id: str) -> dict:
        """The results object of the web API: the analysis plus video status, URLs and every warning."""
        job = self.get(job_id)
        with job.lock:
            an = self._analysis_of(job)
            ready = job.video_status == "ready" and job.video_file is not None
            extra = {
                "video_status": job.video_status,
                "video_url": self._url(job, job.video_file) if ready else None,
                "timeline_url": self._url(job, "timeline.png"),
                "export": {"json": self._url(job, "results.json"), "events_csv": self._url(job, "events.csv"),
                           "scorecard_csv": self._url(job, "scorecard.csv")},
                "warnings": self._warnings(job, an),
            }
        return results_dict(an, extra)

    def file(self, job_id: str, name: str) -> Path:
        """The path of a downloadable file of the job; KeyError unless `name` is on the whitelist and exists inside
        the job directory. `annotated.mp4` stands for the latest ready video."""
        job = self.get(job_id)
        if not isinstance(name, str) or not _FILES.fullmatch(name):
            raise KeyError(name)
        if name == "annotated.mp4":
            with job.lock:
                if job.video_status != "ready" or job.video_file is None:
                    raise KeyError(name)
                name = job.video_file
        base = job.dir.resolve()
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise KeyError(name)
        return path

    # ------------------------------------------------------------------------------------------- internals

    @staticmethod
    def _analysis_of(job: Job) -> Analysis:
        """`job.an` (call with `job.lock` held); JobConflict before needs_confirm."""
        if job.an is None:
            raise JobConflict(f"the job is {job.status}; results are not ready")
        return job.an

    @staticmethod
    def _url(job: Job, name: str) -> str:
        return f"/api/jobs/{job.id}/files/{name}"

    @staticmethod
    def _warnings(job: Job, an: Analysis) -> list[str]:
        return list(an.warnings) + [w for w in job.warnings if w not in an.warnings]

    def _write_exports(self, job: Job, an: Analysis) -> None:
        write_all(an, job.dir, extra={"warnings": self._warnings(job, an)})
        timeline_png(an, job.dir / "timeline.png")

    def _spawn(self, target, *args, name: str) -> None:
        t = threading.Thread(target=self._run_thread, args=(target, *args), name=name, daemon=True)
        with self._lock:
            self._threads.add(t)
        t.start()

    def _run_thread(self, target, *args) -> None:
        try:
            target(*args)
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _check_open(self) -> None:
        if self._closing.is_set():
            raise _Cancelled()

    @staticmethod
    def _set(job: Job, **changes) -> None:
        with job.lock:
            for k, v in changes.items():
                setattr(job, k, v)

    @staticmethod
    def _fail(job: Job, message: str) -> None:
        with job.lock:
            job.status, job.message, job.error = "failed", M_FAILED, message

    def _work(self, job: Job) -> None:
        try:
            self._set(job, status="queued", message=M_READING)
            # Probed before queueing: a garbage upload fails now instead of after the job ahead of it.
            info = probe(job.upload, config.MAX_WIDTH, config.SAMPLE_FPS, float(config.MAX_SECONDS))
            if info.trimmed:
                with job.lock:
                    job.warnings.append(f"video trimmed to the first {info.max_seconds:g} s")
            self._set(job, message=M_QUEUED)
            while not PERCEPTION_SLOT.acquire(timeout=0.25):
                self._check_open()
            try:
                self._perceive_and_analyze(job, info)
            finally:
                PERCEPTION_SLOT.release()
        except _Cancelled:
            log.info("job %s stopped: the server is shutting down", job.id)
            self._fail(job, "the server stopped before the analysis finished")
        except VideoError as exc:
            log.exception("job %s: unreadable video", job.id)
            self._fail(job, _video_message(exc, job))
        except Exception as exc:
            log.exception("job %s failed", job.id)
            self._fail(job, f"analysis failed: {type(exc).__name__}: {exc}")

    def _perceive_and_analyze(self, job: Job, info) -> None:
        self._check_open()
        duration = float(info.duration)
        self._set(job, status="perceiving", progress=0.0, message=f"detecting people… 0 s of {duration:.0f} s")

        def progress(fraction: float, message: str) -> None:
            self._check_open()
            frac = min(max(float(fraction), 0.0), 1.0)
            if not message.startswith("describing people"):         # the frame loop fills 0..FRAMES_PROGRESS
                seen = min(frac / pipeline.FRAMES_PROGRESS, 1.0) * duration
                message = f"detecting people… {seen:.0f} s of {duration:.0f} s"
            self._set(job, progress=PERCEPTION_SHARE * frac, message=message)

        stack = self.stack_factory()
        res, _cached = pipeline.cached_perception(job.upload, stack, job.dir / "cache", progress=progress,
                                                  max_seconds=info.max_seconds)
        self._check_open()
        self._set(job, status="analyzing", progress=PERCEPTION_SHARE, message=M_ANALYZING)
        an = analyze(res, preset=job.preset)
        self._set(job, progress=0.95, message=M_THUMBS)
        thumbs = save_thumbnails(res, an, job.dir / "people")
        # The results show each person's own thumbnail (people/<pid>.jpg), not the track crop analysis named.
        an = dataclasses.replace(an, persons=[dataclasses.replace(p, thumb=thumbs.get(p.pid)) for p in an.persons])
        self._write_exports(job, an)
        with job.lock:
            job.res, job.an, job.thumbs = res, an, dict(thumbs)
            job.status, job.progress, job.message = "needs_confirm", 1.0, M_CONFIRM

    def _current(self, job: Job, rev: int) -> bool:
        return job.rev == rev and not self._closing.is_set()

    def _render(self, job: Job, rev: int, an: Analysis) -> None:
        with job.render_lock:                       # one render per job at a time; a newer one waits for this to stop
            if not self._current(job, rev):
                return
            out = job.dir / f"annotated-{rev}.mp4"
            duration = float(an.duration)

            def progress(fraction: float, _message: str) -> None:
                if not self._current(job, rev):
                    raise _Cancelled()
                with job.lock:
                    if job.rev == rev:
                        job.message = (f"drawing the annotated video… {min(max(fraction, 0.0), 1.0) * duration:.0f} s"
                                       f" of {duration:.0f} s")

            try:
                render_video(job.upload, job.res, an, out, progress=progress)
                if self._current(job, rev):
                    try:
                        contact_sheets(job.upload, job.res, an, job.dir / "contact_sheets")
                    except Exception:
                        log.exception("job %s: contact sheets failed (the video is fine)", job.id)
            except _Cancelled:
                log.info("job %s: render %d superseded or stopped", job.id, rev)
                return
            except Exception as exc:
                log.exception("job %s: rendering the annotated video failed", job.id)
                with job.lock:
                    if job.rev == rev:
                        job.status, job.video_status, job.message = "done", "failed", M_DONE
                        job.error = f"the annotated video could not be drawn: {type(exc).__name__}: {exc}"
                return
            with job.lock:
                current = job.rev == rev
                if current:
                    job.video_file = out.name
                    job.status, job.video_status, job.message, job.error = "done", "ready", M_DONE, None
            # Still under render_lock, so no newer render of this job has written a video yet.
            for p in (job.dir.glob("annotated-*.mp4") if current else [out]):
                if p.name != out.name or not current:
                    _unlink_quietly(p)


def _unlink_quietly(path: Path) -> None:
    """Best effort: on Windows a browser may still hold an old video open; the 24 h cleanup takes it later."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
