"""Job records for the web app: one directory per job under `jobs_dir`.

jobs/<id>/input.<ext>   uploaded video
jobs/<id>/job.json      persisted status (so a restarted server can still answer)
jobs/<id>/cache/        perception cache (unless a shared cache_root is configured)
jobs/<id>/zones.json    the user's current zones
jobs/<id>/roles.json    {"overrides": {...}, "imported": {...}}
jobs/<id>/out/          every generated result file (the only thing served by /files)

Privacy (spec section 7): uploads and job data are deleted after 24 h by `cleanup`, run on app start.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXT = (".mp4", ".mov", ".avi", ".webm", ".mkv")
PRESETS = ("short", "dealership")
MAX_AGE_S = 86400


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Job:
    job_id: str
    dir: Path
    ext: str = ".mp4"
    preset: str = "short"
    state: str = "queued"          # queued | perceiving | ready_for_zones | scored | failed
    progress: float = 0.0
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    video_state: str = "none"      # none | rendering | ready | failed
    cache_dir: str | None = None
    overrides: dict[int, str] = field(default_factory=dict)
    imported: dict[int, str] = field(default_factory=dict)
    generation: int = 0            # bumped on every score; stale background renders are discarded
    pc: object = None              # PerceptionCache, loaded lazily
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def input_path(self) -> Path:
        return self.dir / f"input{self.ext}"

    @property
    def out_dir(self) -> Path:
        return self.dir / "out"

    @property
    def zones_path(self) -> Path:
        return self.dir / "zones.json"

    def status(self) -> dict:
        return {"job_id": self.job_id, "state": self.state, "progress": round(self.progress, 4),
                "message": self.message, "preset": self.preset, "warnings": list(self.warnings),
                "video_state": self.video_state}

    def get_pc(self):
        from pulse import cache
        if self.pc is None:
            if not self.cache_dir:
                raise RuntimeError("perception not finished")
            self.pc = cache.load(Path(self.cache_dir))
        return self.pc

    def to_json(self) -> dict:
        return {"job_id": self.job_id, "ext": self.ext, "preset": self.preset, "state": self.state,
                "progress": self.progress, "message": self.message, "warnings": self.warnings,
                "video_state": self.video_state, "cache_dir": self.cache_dir,
                "overrides": {str(k): v for k, v in self.overrides.items()},
                "imported": {str(k): v for k, v in self.imported.items()},
                "generation": self.generation}

    def save(self) -> None:
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / "job.json.tmp"
            tmp.write_text(json.dumps(self.to_json()))
            os.replace(tmp, self.dir / "job.json")


class JobStore:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, preset: str, ext: str) -> Job:
        job = Job(new_id(), Path(), ext, preset)
        job.dir = self.jobs_dir / job.job_id
        job.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def discard(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            path = self.jobs_dir / job_id / "job.json"
            if not path.is_file():
                return None
            try:
                d = json.loads(path.read_text())
            except (OSError, ValueError):
                return None
            job = Job(job_id, self.jobs_dir / job_id, d.get("ext", ".mp4"), d.get("preset", "short"),
                      d.get("state", "failed"), float(d.get("progress", 0.0)), d.get("message", ""),
                      list(d.get("warnings", [])), d.get("video_state", "none"), d.get("cache_dir"),
                      {int(k): v for k, v in d.get("overrides", {}).items()},
                      {int(k): v for k, v in d.get("imported", {}).items()}, int(d.get("generation", 0)))
            if job.state in ("queued", "perceiving"):
                job.state, job.message = "failed", "Interrupted by a server restart; please upload again."
            if job.video_state == "rendering":
                job.video_state = "failed"
            self._jobs[job_id] = job
            return job


def cleanup(jobs_dir: Path, max_age_s: float = MAX_AGE_S) -> list[str]:
    """Delete job directories (uploads, caches, outputs) not touched for `max_age_s`. Only directories
    whose name is a job id are considered. Returns the removed ids."""
    jobs_dir = Path(jobs_dir)
    removed = []
    if not jobs_dir.is_dir():
        return removed
    now = time.time()
    for d in jobs_dir.iterdir():
        if not (d.is_dir() and not d.is_symlink() and JOB_ID_RE.match(d.name)):
            continue
        try:
            marker = d / "job.json"
            mtime = max(d.stat().st_mtime, marker.stat().st_mtime if marker.exists() else 0.0)
        except OSError:
            continue
        if now - mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed
