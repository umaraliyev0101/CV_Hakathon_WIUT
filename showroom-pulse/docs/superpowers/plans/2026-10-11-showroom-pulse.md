# Showroom Pulse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one showroom camera (uploaded video or live webcam) into anonymous customer-experience statistics (footfall, occupancy, dwell and stop rate per car, journeys, unserved interest) and evidence-backed recommendations, served by a local FastAPI web app and a `pulse` CLI.

**Architecture:** One perception engine (detect → track → pose, all ONNX on CPU) feeds both file mode and live mode. File mode caches perception per video (`.npz`/`.json`), then an offline pass (identity merge → geometry → roles) builds a per-person *timeline*; pure, test-first units (`zones`, `journey`, `interest`, `stats`, `insights`) turn timeline + zones + roles into statistics, so editing zones or roles re-scores in seconds without re-detection. Live mode runs the same engine per WebSocket frame and the same pure units over a rolling window.

**Tech Stack:** Python 3.12 (`/opt/homebrew/bin/python3.12`), numpy, scipy, opencv-python-headless, onnxruntime (CPU EP; CoreML optional), vendored subset of rtmlib 0.0.16 (Apache-2.0: RTMPose pre/post-processing, NMS), YOLOX COCO ONNX or RF-DETR ONNX (decided by benchmark), supervision `ByteTrack` (MIT, pinned `<0.31` because it is removed in 0.31), DINOv2-small ONNX (Xenova export, Apache-2.0), imageio-ffmpeg (H.264 writing), matplotlib (server-side SVG/PNG charts), FastAPI + uvicorn, motmetrics, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-showroom-pulse-design.md` (approved; never edit it). Rules: `CLAUDE.md`. Sister spec for shared units: Showroom Coach §4.2/§4.3 (copied into this plan where needed).

## Global Constraints

- Local only: never add a git remote, push, deploy, or create accounts.
- Python venv from `/opt/homebrew/bin/python3.12`; run everything as `.venv/bin/python ...` / `.venv/bin/pulse ...`.
- Banned: ultralytics, boxmot, re-ID weights trained on Market-1501/DukeMTMC/MSMT17, demographic/emotion/face-recognition models, weights with no findable licence, non-commercial data in commits or shipped artefacts.
- Gitignored: `data/`, `models/`, `runs/`, `jobs/`, `.venv/`. Only `data/sources.csv`, `data/TODO_DOWNLOADS.md`, `models/MANIFEST.json` are committed from them. Never commit videos or weights.
- Disk: venv + models + data ≤ 15 GB combined; stop downloading if free disk < 20 GB.
- Privacy: no face recognition, no age/gender/emotion/ethnicity inference, no re-identification across visits. **Every rendered frame, thumbnail, keyframe image, heatmap background and contact sheet blurs faces.**
- Banner text (UI and report): "Anonymous, aggregated analytics; decision support only."
- Use imageio-ffmpeg in code (never the system `ffmpeg`), for portability to teammate A's Windows laptop.
- Frames are resized to max width 960 px; sampling default 5 fps.
- Inference on ONNX Runtime CPU; CPU speed numbers mandatory; CoreML numbers optional and separate.
- Shared unit names identical to Showroom Coach: `sources`, `perception.detect`, `perception.pose`, `tracking`, `identity`, `geometry`, `roles`, `render`; core dataclasses `Det`, `TrackObs`, `Person`, `RoleSuggestion`, `Event`.
- Presets (spec §4.6), exact values:

| Param | dealership | short |
|---|---|---|
| leave_after | 10 | 3 |
| zone_min / zone_gap | 1.5 / 2 | 1 / 1 |
| stop_speed / stop_time | 0.3 / 10 | 0.3 / 3 |
| interest_time | 60 | 10 |
| assist_dist / assist_time | 2.0 / 5 | 2.0 / 2 |
| alert_after | 45 | 8 |
| desk_time | 30 | 5 |
| min_sample | 5 | 2 (demo only) |

- Targets (spec §6, never edit): entry/exit count error ≤ 10%; occupancy MAE ≤ 0.5; zone dwell median error ≤ 2 s (short) or ≤ 10% (dealership); journey accuracy ≥ 80%; file mode ≥ 1× real time; live ≥ 5 fps. A miss is reported, never hidden.
- Never weaken a gate: no `pytest.mark.skip`, `xfail`, `pytest.skip()`, `.only`, deleting or mocking tests; tests that need data **fail** with a clear message when the data is missing.
- Default `pytest -q` excludes `slow` (`addopts = -m "not slow"`) and runs < 3 min; slow tests run via `pytest -q -m slow`.
- `pulse eval` output (`docs/EVAL.md`) must be **byte-identical** when regenerated from the same cached runs (no timestamps, no live re-measurement inside `eval`), because `verify.sh` requires a clean `git status` after it.
- Commits: end each message with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Empty showroom / nobody detected** — a clip with no people must produce valid outputs (zero KPIs, a "Nobody detected — camera too far or too dark?" warning), never a crash. Test: `tests/test_pipeline_edge.py::test_score_with_no_people` (Task 15) and `tests/test_api.py::test_job_with_no_people` (Task 16).
2. **Corrupt or non-video upload** (e.g. a `.txt` renamed `.mp4`) — the job ends in `failed` with a readable message; the API never returns 500. Test: `tests/test_api.py::test_upload_garbage_fails_cleanly` (Task 16).
3. **Over-long video** (> 5 min) — analysed up to the limit with a "trimmed" warning in the report. Test: `tests/test_sources.py::test_max_duration_trims_and_flags` (Task 12) and warning propagation in `tests/test_pipeline_edge.py::test_trim_warning_in_stats` (Task 15).
4. **User deletes every zone** — re-score still works, store-level stats only, per-car table empty, warning "No zones: draw zones to get per-car statistics". Test: `tests/test_pipeline_edge.py::test_score_with_no_zones` (Task 15).
5. **Portrait / odd-size phone video** (e.g. 1080×1920 at 29.97 fps) — resized so width ≤ 960 keeping aspect, timestamps = frame_index / fps. Test: `tests/test_sources.py::test_portrait_video_resized_and_timed` (Task 12).

---

## File Structure

```
pyproject.toml                      # package "showroom-pulse", import name "pulse", entry point "pulse"
run.sh                              # starts uvicorn on :8000 (Task 21)
scripts/verify.sh                   # verification chain (spec §9), grows each milestone
scripts/check_outputs.py            # schema check of every `pulse analyze` output file
scripts/smoke_api.py                # upload → zones → roles → results → zone edit → re-score (no re-detection)
scripts/smoke_live.py               # file-fed WebSocket; overlays, timers, ≥1 alert, prints fps
scripts/webcam_grab.py              # grabs one non-blank frame from the default camera
scripts/make_clip.py                # cuts/re-encodes excerpts with imageio-ffmpeg (fixtures)
src/pulse/__init__.py
src/pulse/datatypes.py              # all dataclasses (shared + pulse-specific)
src/pulse/config.py                 # Preset, PRESETS, Config, get_config
src/pulse/geometry.py               # foot point, pinhole distance, facing, homography (shared)
src/pulse/zones.py                  # auto car zones, point-in-polygon, zone_of, hysteresis, load/save
src/pulse/timeline.py               # TrackObs + persons → per-person Sample timeline (foot, pos_m, speed)
src/pulse/journey.py                # per-person zone visits, stops, sequences, edit distance
src/pulse/interest.py               # deep interest, assistance, unserved episodes, live alerts
src/pulse/stats.py                  # store / per-car / occupancy / funnel / groups / heatmap
src/pulse/insights.py               # rules R1–R7 with evidence + confidence
src/pulse/roles.py                  # staff features, suggestions with reasons, roles.json import (shared)
src/pulse/identity.py               # appearance embeddings + track merge + live re-attach (shared)
src/pulse/sources.py                # file frame iterator (OpenCV), JPEG decode for live (shared)
src/pulse/perception/__init__.py
src/pulse/perception/models.py      # model registry, download w/ retries + fallback, MANIFEST.json
src/pulse/perception/detect.py      # Detector (YOLOX / RF-DETR ONNX) → list[Det] (shared)
src/pulse/perception/pose.py        # PoseEstimator (RTMPose ONNX) → keypoints [N,17,3] (shared)
src/pulse/perception/embed.py       # torso HSV histogram + DINOv2-small embedding
src/pulse/perception/vendor_rtmlib/ # vendored rtmlib 0.0.16 subset + LICENSE (Apache-2.0)
src/pulse/tracking.py               # ByteTrack wrapper → list[TrackObs] (shared)
src/pulse/camera_motion.py          # stable/moving segments, same-view check
src/pulse/engine.py                 # PerceptionEngine.process(t, frame) used by file + live
src/pulse/cache.py                  # PerceptionCache save/load, video key, run counter
src/pulse/pipeline.py               # perceive (cached), build_people, score, write_outputs, analyze
src/pulse/render.py                 # face blur, annotated H.264 mp4, thumbnails, contact sheets (shared)
src/pulse/report.py                 # report.json, CSVs, heatmap PNG, SVG charts, journey flow
src/pulse/live.py                   # LiveSession: rolling timeline, overlays, timers, alerts
src/pulse/app/__init__.py
src/pulse/app/main.py               # FastAPI app: REST + /ws/live + static
src/pulse/app/jobs.py               # job store on disk, background worker, 24 h cleanup
src/pulse/app/static/{index.html,app.js,zones.js,dashboard.js,live.html,live.js,style.css}
src/pulse/eval/__init__.py
src/pulse/eval/meva.py              # KPF (geom/activities/types yml) parser → GT tracks + activities
src/pulse/eval/metrics.py           # MOTA/IDF1, count error, occupancy MAE, dwell error, journey acc
src/pulse/eval/run_eval.py          # `pulse eval` → docs/EVAL.md (deterministic)
src/pulse/bench.py                  # `pulse bench` detector / pose / live-fps benchmark
src/pulse/cli.py                    # `pulse analyze|eval|bench|zones`
labels/zones/<clip>.json            # evaluation zones drawn by the agent ("ai-agent, unreviewed")
labels/<clip>.json                  # stock-clip labels (visits, zone visits, stops)
labels/split.json                   # dev/test split by clip
tests/...                           # one test file per unit, `tests/integration/` for slow ones
docs/EVAL.md docs/PITCH_NOTES.md docs/INTEGRATION.md README.md PROGRESS.md
data/sources.csv data/TODO_DOWNLOADS.md models/MANIFEST.json
```

**Coordinate conventions (used everywhere):**
- `bbox` = `(x1, y1, x2, y2)` in **analysis-frame pixels** (after resizing to width ≤ 960).
- `Zone.polygon` and `Sample.foot` are **normalised** image coordinates in `[0, 1]` (x / frame_w, y / frame_h), so zone files work at any resolution.
- `pos_m` = `(X, Z)` metres on the ground plane (camera-centred pinhole estimate, or floor coordinates when a floor homography is calibrated).
- `t` = seconds from clip start; samples of the same frame share exactly the same `t` (`round(frame_idx / sample_fps, 3)`).

---

## M0 — Skeleton, models, benchmark

### Task 1: Repo skeleton, datatypes, config, verify.sh skeleton

**Files:**
- Create: `pyproject.toml`, `src/pulse/__init__.py`, `src/pulse/datatypes.py`, `src/pulse/config.py`, `tests/test_config.py`, `tests/test_datatypes.py`, `tests/conftest.py`, `scripts/verify.sh`, `PROGRESS.md`, `README.md` (stub), `data/sources.csv`, `data/TODO_DOWNLOADS.md`
- Modify: `.gitignore` (already exists; ensure `jobs/` and `runs/` present)

**Interfaces:**
- Produces: every dataclass below (exact field names), `Preset`, `PRESETS`, `Config`, `get_config(preset_name: str = "short", **overrides) -> Config`.

- [ ] **Step 1: pyproject**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "showroom-pulse"
version = "0.1.0"
description = "Customer-experience analytics for car showrooms (anonymous, aggregated)"
requires-python = ">=3.12,<3.13"
license = {text = "Apache-2.0"}
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "opencv-python-headless>=4.9",
  "onnxruntime>=1.17",
  "supervision>=0.26,<0.31",
  "imageio-ffmpeg>=0.5",
  "matplotlib>=3.8",
  "pyyaml>=6",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "python-multipart>=0.0.9",
  "motmetrics>=1.4",
  "tqdm>=4",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[project.scripts]
pulse = "pulse.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
pulse = ["app/static/*", "perception/vendor_rtmlib/LICENSE"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m \"not slow\""
markers = ["slow: needs real video/models; run with -m slow"]
filterwarnings = ["ignore::FutureWarning:supervision.*"]
```

`src/pulse/cli.py` does not exist yet; create a minimal one so the entry point installs:

```python
import argparse, sys

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pulse")
    p.add_subparsers(dest="cmd")
    args = p.parse_args(argv)
    p.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write failing tests**

`tests/test_config.py`:
```python
from pulse.config import PRESETS, get_config

def test_presets_match_spec_table():
    d, s = PRESETS["dealership"], PRESETS["short"]
    assert (d.leave_after, s.leave_after) == (10, 3)
    assert (d.zone_min, d.zone_gap, s.zone_min, s.zone_gap) == (1.5, 2, 1, 1)
    assert (d.stop_speed, d.stop_time, s.stop_speed, s.stop_time) == (0.3, 10, 0.3, 3)
    assert (d.interest_time, s.interest_time) == (60, 10)
    assert (d.assist_dist, d.assist_time, s.assist_dist, s.assist_time) == (2.0, 5, 2.0, 2)
    assert (d.alert_after, s.alert_after) == (45, 8)
    assert (d.desk_time, s.desk_time) == (30, 5)
    assert (d.min_sample, s.min_sample) == (5, 2)
    assert s.demo_only and not d.demo_only

def test_get_config_overrides_and_preset():
    c = get_config("dealership", sample_fps=10)
    assert c.preset.name == "dealership" and c.sample_fps == 10 and c.max_width == 960

def test_unknown_preset_raises():
    import pytest
    with pytest.raises(KeyError):
        get_config("nope")
```

`tests/test_datatypes.py`:
```python
import json
from pulse.datatypes import (Det, TrackObs, Person, RoleSuggestion, Event, Zone, Sample,
                             ZoneVisit, InterestEpisode, Insight, StableSegment, Stats)

def roundtrip(obj):
    d = json.loads(json.dumps(obj.to_dict()))
    return type(obj).from_dict(d)

def test_roundtrip_all_types():
    objs = [
        Det((1, 2, 3, 4), 0.9, "person"),
        TrackObs(0.2, 1, 7, (1, 2, 3, 4), 0.8, [[1.0, 2.0, 0.9]] * 17),
        Person(1, [7, 9], 0.0, 12.5),
        RoleSuggestion(1, "staff", 0.8, ["same outfit as 2 others"], {"contacts": 3.0}),
        Event("unserved_interest", 1.0, 5.0, [1], {"zone_id": "car-1"}),
        Zone("car-1", "Car 1", "car", [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)], "auto"),
        Sample(0.2, 1, (0.5, 0.9), (10, 20, 30, 40), (0.1, 3.0), 0.2, "full"),
        ZoneVisit(1, "car-1", 1.0, 5.0, True),
        InterestEpisode(1, "car-1", 1.0, 13.0, 12.0, 1, False, True, []),
        Insight("R1", "t", {"n": 1}, "a", ["car-1"], (0.0, 60.0), "low", 2, [{"t": 1.0}], True),
        StableSegment(0.0, 10.0, True, True),
    ]
    for o in objs:
        assert roundtrip(o).to_dict() == json.loads(json.dumps(o.to_dict()))

def test_zone_visit_dwell():
    assert ZoneVisit(1, "a", 2.0, 5.5).dwell == 3.5

def test_stats_empty_factory():
    s = Stats.empty()
    assert s.store["visitors"] == 0 and s.per_car == [] and s.occupancy == []
```

`tests/conftest.py` (shared helpers for the pure-unit tests):
```python
import pytest
from pulse.datatypes import Sample, Zone

def rect(zid, name, kind, x0, y0, x1, y1):
    return Zone(zid, name, kind, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "user")

def track(pid, points, t0=0.0, dt=0.2, speed=None, pos=None):
    """points: list of normalised (x, y) feet, one per sample. pos: list of (X, Z) or None."""
    out = []
    for i, (x, y) in enumerate(points):
        p = pos[i] if pos is not None else None
        out.append(Sample(round(t0 + i * dt, 3), pid, (x, y), (0, 0, 10, 30), p,
                          speed if speed is not None else 0.0, "full"))
    return out

def stay(pid, x, y, t0, seconds, dt=0.2, speed=0.0, pos=None):
    n = int(round(seconds / dt)) + 1
    return track(pid, [(x, y)] * n, t0=t0, dt=dt, speed=speed,
                 pos=[pos] * n if pos is not None else None)

@pytest.fixture
def zones4():
    return [rect("car-a", "Car A", "car", 0.05, 0.5, 0.3, 0.9),
            rect("car-b", "Car B", "car", 0.6, 0.5, 0.85, 0.9),
            rect("desk", "Desk", "desk", 0.4, 0.1, 0.5, 0.2),
            rect("entrance", "Entrance", "entrance", 0.9, 0.0, 1.0, 1.0)]
```

- [ ] **Step 3: Run tests, verify they fail** — `.venv/bin/python -m pytest -q tests/test_config.py tests/test_datatypes.py` → ImportError.

- [ ] **Step 4: Implement `src/pulse/datatypes.py`**

```python
"""Core dataclasses. Det/TrackObs/Person/RoleSuggestion/Event are shared with Showroom Coach
(spec §4.2) — keep names and fields identical. All serialise to JSON via to_dict/from_dict."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from typing import Any

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2, analysis-frame pixels


class JsonMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Det(JsonMixin):
    bbox: BBox
    score: float
    cls: str  # COCO class name: 'person', 'car', 'truck', 'motorcycle', 'bus'


@dataclass
class TrackObs(JsonMixin):
    t: float
    frame_idx: int                      # index of the sampled frame
    track_id: int
    bbox: BBox
    score: float
    keypoints: list[list[float]] | None = None  # 17 x [x, y, score], COCO order, pixels


@dataclass
class Person(JsonMixin):
    person_id: int
    track_ids: list[int]
    first_t: float
    last_t: float


@dataclass
class RoleSuggestion(JsonMixin):
    person_id: int
    role: str            # 'staff' | 'customer' | 'unsure'
    confidence: float    # 0..1
    reasons: list[str]
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Event(JsonMixin):
    kind: str
    t_start: float
    t_end: float | None
    person_ids: list[int]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Zone(JsonMixin):
    zone_id: str
    name: str
    kind: str                               # 'car' | 'desk' | 'entrance' | 'lounge' | 'other'
    polygon: list[tuple[float, float]]      # normalised [0,1] image coords
    source: str = "user"                    # 'auto' | 'user'


@dataclass
class Sample(JsonMixin):
    t: float
    person_id: int
    foot: tuple[float, float]               # normalised [0,1] image coords
    bbox: BBox
    pos_m: tuple[float, float] | None       # (X, Z) metres
    speed: float | None                     # smoothed m/s
    dist_method: str = "box"                # 'full' | 'torso' | 'box' | 'floor'


Timeline = dict[int, list[Sample]]


@dataclass
class ZoneVisit(JsonMixin):
    person_id: int
    zone_id: str
    t_in: float
    t_out: float
    stopped: bool = False

    @property
    def dwell(self) -> float:
        return self.t_out - self.t_in


@dataclass
class InterestEpisode(JsonMixin):
    person_id: int
    zone_id: str
    t_start: float
    t_end: float
    dwell: float
    visits: int
    assisted: bool
    unserved: bool
    staff_ids: list[int] = field(default_factory=list)


@dataclass
class Insight(JsonMixin):
    rule: str
    title: str
    evidence: dict[str, Any]
    action: str
    cars: list[str]
    window: tuple[float, float] | None
    confidence: str                         # 'insufficient' | 'low' | 'medium' | 'high'
    sample: int
    moments: list[dict[str, Any]]
    demo_only: bool = False


@dataclass
class StableSegment(JsonMixin):
    t0: float
    t1: float
    stable: bool
    same_view: bool = True                  # zones drawn on the keyframe apply here


STORE_KEYS_DEFAULTS: dict[str, Any] = {
    "visitors": 0, "staff": 0, "unsure": 0, "already_present": 0, "entries": 0, "exits": 0,
    "groups": 0, "n_solo": 0, "n_group_members": 0, "median_visit_s": None, "browse_rate": None,
    "assisted_share": None, "assisted_customers": 0, "median_time_to_assist": None,
    "unserved_episodes": 0, "deep_interest_visitors": 0, "reached_desk": 0,
    "deep_interest_reached_desk": 0, "has_desk": False, "peak_occupancy": 0, "peak_t": None,
    "funnel": {}, "group_assisted_share": None, "solo_assisted_share": None,
    "duration_s": 0.0, "analysed_share": 1.0, "distance_approx": False,
}


@dataclass
class Stats(JsonMixin):
    store: dict[str, Any]
    per_car: list[dict[str, Any]]
    per_zone: list[dict[str, Any]]
    occupancy: list[list[float]]            # [[t, customers present]] every 1 s
    waiting: list[list[float]]              # [[t, customers present & not yet assisted]]
    crowding: list[list[float]]             # [[t, customers per staff]] (-1 when no staff present)
    time_buckets: list[dict[str, Any]]
    heatmap: dict[str, Any]                 # {"w": 48, "h": 27, "counts": [[int]*48]*27}
    paths: list[dict[str, Any]]             # [{"person_id", "points": [[x, y], ...]}] normalised
    visits: list[dict[str, Any]]            # one anonymous record per person
    groups: list[list[int]]
    warnings: list[str]

    @classmethod
    def empty(cls) -> "Stats":
        return cls(dict(STORE_KEYS_DEFAULTS), [], [], [], [], [], [],
                   {"w": 48, "h": 27, "counts": [[0] * 48 for _ in range(27)]}, [], [], [], [])
```

- [ ] **Step 5: Implement `src/pulse/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Preset:
    name: str
    leave_after: float
    zone_min: float
    zone_gap: float
    stop_speed: float
    stop_time: float
    interest_time: float
    assist_dist: float
    assist_time: float
    alert_after: float
    desk_time: float
    min_sample: int
    demo_only: bool = False


PRESETS: dict[str, Preset] = {
    "dealership": Preset("dealership", 10, 1.5, 2, 0.3, 10, 60, 2.0, 5, 45, 30, 5, False),
    "short": Preset("short", 3, 1, 1, 0.3, 3, 10, 2.0, 2, 8, 5, 2, True),
}


@dataclass(frozen=True)
class Config:
    preset: Preset
    sample_fps: float = 5.0
    max_width: int = 960
    max_duration_s: float = 300.0          # longer videos are trimmed with a warning
    person_height_m: float = 1.70
    hfov_deg: float = 70.0
    kp_thr: float = 0.3                     # keypoint confidence threshold
    detector: str = "yolox_tiny"            # replaced by the M0 benchmark decision
    pose_model: str = "rtmpose_s"
    det_score_thr: float = 0.3
    det_nms_thr: float = 0.45
    track_activation_thr: float = 0.35
    track_buffer_s: float = 2.0
    track_match_thr: float = 0.8
    edge_margin: float = 0.02               # normalised distance to frame edge counted as "at edge"
    speed_window_s: float = 1.0             # foot-point smoothing window for speed
    group_arrival_s: float = 5.0            # party rule (Coach §4.5)
    group_dist_m: float = 1.5
    group_window_s: float = 30.0
    contact_dist_m: float = 2.0             # roles: contacts
    contact_time_s: float = 5.0
    outfit_sim_thr: float = 0.85            # roles: outfit_peers (cosine)
    presence_min_clip_s: float = 60.0
    role_weights: dict = field(default_factory=lambda: {
        "outfit_peers": 1.0, "contacts": 1.0, "presence": 1.0, "initiator": 0.5})
    role_staff_thr: float = 0.6
    role_customer_thr: float = 0.35
    merge_sim_thr: float = 0.75             # identity merge cosine threshold
    merge_max_speed: float = 0.25           # normalised frame widths per second
    crowding_ratio: float = 3.0             # R5
    crowding_time_s: float = 120.0          # R5
    heatmap_w: int = 48
    heatmap_h: int = 27


def get_config(preset_name: str = "short", **overrides) -> Config:
    preset = PRESETS[preset_name]
    return replace(Config(preset=preset), **overrides)
```

- [ ] **Step 6: `scripts/verify.sh` skeleton** (grows every milestone; each later task appends its step *before* the git step)

```bash
#!/usr/bin/env bash
# Showroom Pulse verification chain (spec §9). Runs each step in order, prints its output,
# exits non-zero at the first failure.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
step() {
  local name="$1"; shift
  echo; echo "=== [verify] $name ==="; echo "\$ $*"
  "$@"; local rc=$?
  echo "=== [verify] $name -> exit $rc ==="
  if [ $rc -ne 0 ]; then echo "VERIFY FAILED at: $name"; exit $rc; fi
}
step "pytest (default)" $PY -m pytest -q
step "pytest (slow)" $PY -m pytest -q -m slow
# --- later milestones insert steps here ---
step "git status clean" bash -c 'out=$(git status --short); echo "$out"; test -z "$out"'
step "no git remote" bash -c 'out=$(git remote -v); echo "$out"; test -z "$out"'
echo; echo "VERIFY OK"
```

Note: until the first slow test exists, `pytest -m slow` exits 5 ("no tests collected"). Add `tests/integration/test_smoke_marker.py` now so the step is real, not empty:

```python
import pytest, shutil, imageio_ffmpeg

@pytest.mark.slow
def test_imageio_ffmpeg_binary_available():
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    assert exe and shutil.which(exe) or exe.endswith("ffmpeg") or "ffmpeg" in exe
```

- [ ] **Step 7: PROGRESS.md, README stub, data records**

`PROGRESS.md` starts with sections `## Status`, `## Milestones` (append per milestone: what was done, measured numbers, decisions, open problems), `## Decisions` (dated bullets), `## Open problems`. Record at once these decisions already taken by the orchestrator:
- Pre-flight was not run by the owner; the agent created the repo, copied the spec/CLAUDE.md/prompt; `data/stock/` and `data/synthetic/share/` are empty.
- rtmlib is vendored (subset) instead of pip-installed, because its wheel requires `opencv-python` + `opencv-contrib-python`, which clash with `opencv-python-headless`, and its multiclass YOLOX path drops scores (ByteTrack needs them).
- `supervision` pinned `<0.31`: `ByteTrack` is deprecated since 0.28 and removed in 0.31.
- No `docs/journal.md` logging gate (optional pre-flight step not run); PROGRESS.md is the log.

`data/sources.csv` header: `source_id,kind,url,licence,attribution,local_path,committed`.
`data/TODO_DOWNLOADS.md`: list of manual downloads for the owner (Pexels clips, e.g. pexels.com/video/7154222, into `data/stock/pexels-<id>.mp4`).
`README.md`: title + one-paragraph description + "work in progress" (filled in Task 21).

- [ ] **Step 8: Install and run**

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # PASS
bash scripts/verify.sh                 # fails only at "git status clean" until committed
```

- [ ] **Step 9: Commit** — `git add -A && git commit -m "M0: skeleton, datatypes, presets, verify.sh"`; then `bash scripts/verify.sh` → `VERIFY OK`.

### Task 2: Model registry, downloads, MEVA KPF parser, detector benchmark

**Files:**
- Create: `src/pulse/perception/__init__.py`, `src/pulse/perception/models.py`, `src/pulse/perception/vendor_rtmlib/{__init__.py,LICENSE,NOTICE.md,base.py,rtmpose.py,pose_pre.py,pose_post.py,det_post.py}`, `src/pulse/perception/detect.py`, `src/pulse/perception/pose.py`, `src/pulse/eval/__init__.py`, `src/pulse/eval/meva.py`, `src/pulse/bench.py`, `tests/test_models.py`, `tests/test_meva.py`, `tests/fixtures/meva_mini/{mini.geom.yml,mini.activities.yml,mini.types.yml}`, `tests/integration/test_detect_pose.py`
- Modify: `src/pulse/cli.py` (add `bench`), `models/MANIFEST.json`, `README.md` (licence table rows), `PROGRESS.md`

**Interfaces:**
- Produces:
  - `models.REGISTRY: dict[str, ModelSpec]` with `ModelSpec(name, url, file, kind, licence, trained_on, input_size, fallback, sha256=None)`; `ensure_model(name: str, models_dir="models") -> Path` (3 retries, then `fallback`, logs to stderr; verifies/records sha256 in `models/MANIFEST.json`).
  - `detect.Detector(name: str, cfg: Config, providers=("CPUExecutionProvider",))` with `detect(frame_bgr, classes=("person",)) -> list[Det]`. `VEHICLE_CLASSES = ("car", "truck", "motorcycle", "bus")`.
  - `pose.PoseEstimator(name: str, providers=...)` with `pose(frame_bgr, boxes: list[BBox]) -> np.ndarray  # [N,17,3] (x, y, score)`.
  - `eval.meva.load_kpf(annotation_dir: Path, clip: str) -> MevaClip` where `MevaClip(clip, tracks: dict[int, dict[int, BBox]]  # id1 -> {frame: bbox (source px)}, track_types: dict[int, str], activities: list[MevaActivity])`, `MevaActivity(act_id, name, frame_start, frame_end, actors: list[int])`.
  - `bench.run_detector_bench(...) -> dict` and CLI `pulse bench detectors --clip <avi> --kpf <annotation_dir> --showroom <img> --out runs/bench/detectors.json`.

- [ ] **Step 1: Vendor rtmlib subset.** `pip download rtmlib==0.0.16 --no-deps -d /tmp/rtmlib_dl && unzip -o /tmp/rtmlib_dl/rtmlib-0.0.16-py3-none-any.whl -d /tmp/rtmlib_x`. Copy `rtmlib/tools/pose_estimation/{rtmpose.py,pre_processings.py→pose_pre.py,post_processings.py→pose_post.py}`, `rtmlib/tools/object_detection/post_processings.py→det_post.py`, the wheel's `LICENSE`. Replace `base.py` with a minimal onnxruntime-only `BaseTool` (no openvino/opencv-dnn backends, no auto-download: takes a local path; `inference(img)` does `img.transpose(2,0,1)[None].astype(np.float32)` after mean/std as the original does). Keep the original copyright header in each file plus a line `# Vendored from rtmlib 0.0.16 (Apache-2.0); modified: imports, onnxruntime-only base.` `NOTICE.md` lists what changed.

- [ ] **Step 2: Registry** (`models.py`). Entries (all URLs verified reachable 2026-10-11):

| name | url | kind | licence | trained_on | input |
|---|---|---|---|---|---|
| yolox_nano | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx | yolox | Apache-2.0 | COCO 2017 | 416 |
| yolox_tiny | …/0.1.1rc0/yolox_tiny.onnx | yolox | Apache-2.0 | COCO 2017 | 416 |
| yolox_s | …/0.1.1rc0/yolox_s.onnx | yolox | Apache-2.0 | COCO 2017 | 640 |
| yolox_m | …/0.1.1rc0/yolox_m.onnx | yolox | Apache-2.0 | COCO 2017 | 640 |
| rfdetr_n | https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_n_v142_384x384.onnx | rfdetr | Apache-2.0 | COCO 2017 (Objects365 pretrain) | 384 |
| rfdetr_s | …/rfdetr/rfdetr_s_v142_512x512.onnx | rfdetr | Apache-2.0 | COCO 2017 (Objects365 pretrain) | 512 |
| rtmpose_s | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip | rtmpose (zip → end2end.onnx) | Apache-2.0 | Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) | 192x256 |
| rtmpose_m | …/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip | rtmpose | Apache-2.0 | Body7 | 192x256 |
| dinov2_small | https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx | embed | Apache-2.0 (facebook/dinov2-small; ONNX export by Xenova) | LVD-142M | 224 |

Fallbacks: yolox_tiny→yolox_nano, yolox_s→yolox_tiny, rfdetr_*→yolox_tiny, rtmpose_m→rtmpose_s, dinov2_small→None (embedding falls back to HSV only, logged). Downloads go to `models/<name>/` via `urllib.request` with a User-Agent; zips are extracted with `zipfile` after checking member names contain no `..`/absolute paths. `MANIFEST.json`: `{"models": [{"name","url","file","sha256","bytes","licence","trained_on","downloaded"}]}`; `downloaded` is a date string `YYYY-MM-DD`.

Unit test (`tests/test_models.py`, no network): registry entries all have licence/trained_on/url; no banned names (`ultralytics`, `boxmot`, `market`, `duke`, `msmt`) in any url; `ensure_model` with a monkeypatched downloader that fails 3 times uses the fallback and writes the manifest entry (use `tmp_path` as models_dir; the fake downloader writes a dummy file). This is dependency injection of the network call, not mocking out the test subject.

- [ ] **Step 3: Detector** (`detect.py`). YOLOX: letterbox preprocess exactly as rtmlib (`padded 114`, ratio = min(in/h, in/w)), no mean/std, NCHW float32; decode grids (strides 8/16/32), `scores = obj * cls`, per-class NMS via vendored `multiclass_nms(boxes, scores, nms_thr, score_thr)` keeping scores; RF-DETR: rtmlib's preprocess/postprocess with scores kept. Return `Det(bbox=(x1,y1,x2,y2) clipped to frame, score, cls=COCO name)` filtered by `classes`. One `ort.InferenceSession` per Detector with `intra_op_num_threads = os.cpu_count()`.

- [ ] **Step 4: Pose** (`pose.py`): wraps vendored `RTMPose(onnx_path, model_input_size=(192,256))`; `pose(frame, boxes)` returns `np.concatenate([kps, scores[...,None]], -1)` shape `[N,17,3]`; empty boxes → `np.zeros((0,17,3))`.

- [ ] **Step 5: MEVA KPF parser** (`eval/meva.py`) — facts verified by the orchestrator (details in `data/meva/CLIPS.md`):
  - Files: `data/meva/annotations/<clip>/<clip>.{geom,activities,types}.yml`; videos `data/meva/videos/<clip>.r13.avi` (30 fps, 1920×1072 except G299 1920×1080, ~9000 frames).
  - **Two dialects**, both one entry per line:
    - flow: `- { geom: {id1: 718954285527769780, id0: 1, ts0: 674, ts1: 22.47, g0: 908 337 943 458, keyframe: true } }`
    - quoted python-dict: `- {'geom': {'g0': '1257 477 1411 815', 'id0': 6316, 'id1': 16, 'keyframe': True, 'ts0': 669}}`
    - activities: `- { act: { act2: {person_exits_scene_through_structure: 1.0}, id2: 813..., timespan: [{tsr0: [674, 723]}], src_status: janitor, actors: [{id1: 718954285527769780, timespan: [{tsr0: [674, 723]}]}] } }` (or the quoted equivalent)
    - types: `- { types: { id1: 718954285527769780, cset3: { person: 1.0 } } }`
    - lines `- { meta: ... }` are ignored.
  - `id1` = track id (links geom/types/actors); `id0` = global detection index (ignore); `ts0` = 0-based frame index matching the decoded frame; `g0` = `x1 y1 x2 y2` native pixels. Geom has one row per track per frame (non-keyframes interpolated). G299/G436 geom files are huge (100k–236k rows) and contain non-person tracks → filter by types.
  - **Parse geom with a regex** (fast; PyYAML on 236k lines is too slow): pull `id1`, `ts0`, `g0` with patterns tolerant to both dialects, e.g. `r"['\"]?id1['\"]?:\s*(\d+)"`, `r"['\"]?ts0['\"]?:\s*(\d+)"`, `r"['\"]?g0['\"]?:\s*['\"]?(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"`. Parse activities and types lines with `yaml.safe_load` (use `yaml.CSafeLoader` when available) — they are small; both dialects are valid YAML flow mappings.
  - Exact activity names: `person_enters_scene_through_structure`, `person_exits_scene_through_structure`; vehicle: `person_enters_vehicle`, `person_exits_vehicle` (others listed in CLIPS.md).
  - Fixtures `tests/fixtures/meva_mini/` contain ~10 lines of **each** dialect (two mini clips) copied in structure from the real files. Tests: track boxes parsed for both dialects, `meta` lines ignored, activity names and per-actor frame spans parsed, person-type filtering via the types file.

- [ ] **Step 6: Benchmark** (`bench.py`, `pulse bench detectors`). For each candidate detector: (a) **person recall** on 60 frames evenly sampled from the MEVA dev clip, against GT person boxes present at those frames (boxes scaled by analysis_w/source_w), match IoU ≥ 0.5, count GT boxes with height ≥ 20 px at analysis scale; (b) **cars on a showroom keyframe**: number of `car|truck|motorcycle` detections with score ≥ 0.3 on each image in `data/stock/keyframes/*.jpg` (if none exist, record "no showroom image" and use the MEVA outdoor annotation clip's first frame only if downloaded; otherwise mark car check "not run"); (c) **speed**: median ms per 960×540 frame over 30 runs after 3 warm-ups, CPU EP, plus pose ms for 4 boxes with rtmpose_s and rtmpose_m. Selection rule (spec §5): fastest detector whose recall is within 5 points of the best **and** which detects ≥ 1 car on every showroom image. Write `runs/bench/detectors.json` (all numbers + chosen name + rule text) and print a table.

- [ ] **Step 7: Slow integration test** `tests/integration/test_detect_pose.py` (`@pytest.mark.slow`): on frame 300 of the MEVA dev clip (resized to 960 wide) the chosen detector returns ≥ 1 person; pose returns `[N,17,3]` with finite values. The dev clip path comes from `tests/integration/conftest.py::meva_dev_clip` fixture which reads `labels/split.json` if present else `data/meva/videos/` sorted first file; if missing, `pytest.fail("MEVA dev clip missing: run data download (see data/TODO_DOWNLOADS.md)")`.

- [ ] **Step 8: Run bench, decide, record.** `.venv/bin/pulse bench detectors ...`; set `Config.detector` / `Config.pose_model` defaults to the chosen models; README licence table rows for every downloaded model; `models/MANIFEST.json` committed; PROGRESS.md gets the benchmark table and the decision.

- [ ] **Step 9: Commit** `M0: models, manifest, MEVA parser, detector benchmark (<chosen>)`; run `bash scripts/verify.sh` → `VERIFY OK`.

---

## M1 — Pure core (test-first, synthetic data only)

### Task 3: geometry (shared with Showroom Coach §4.3)

**Files:** Create `src/pulse/geometry.py`, `tests/test_geometry.py`.

**Interfaces:**
- Produces:
  - `NOSE_TO_ANKLE_FRAC = 0.88`, `TORSO_FRAC = 0.30` (shoulder-mid to hip-mid as fraction of height)
  - `focal_px(frame_w: int, hfov_deg: float) -> float`  (= W / (2·tan(hfov/2)))
  - `foot_point(bbox, keypoints|None, kp_thr=0.3) -> tuple[float, float]` (mean of confident ankles 15/16, else bottom-centre of box), pixels
  - `@dataclass PosEstimate(x: float, z: float, method: str)` ; `position_m(obs: TrackObs, frame_size: tuple[int,int], hfov_deg=70.0, person_h=1.70, kp_thr=0.3, floor_h: np.ndarray|None=None) -> PosEstimate`
  - `distance_m(a: tuple[float,float], b: tuple[float,float]) -> float`
  - `facing_dir(keypoints, kp_thr=0.3) -> tuple[float,float] | None` and `faces(a_obs, a_pos, b_pos, max_angle_deg=60) -> bool | None`
  - `fit_homography(img_pts: list[(u,v)], floor_pts: list[(x,y)]) -> np.ndarray(3,3)`, `to_floor(h, pt) -> (x, y)`

`position_m` rules: method `full` if nose (0) and at least one ankle (15/16) have score ≥ kp_thr and ankle_y > nose_y: `h_px = (ankle_y - nose_y) / NOSE_TO_ANKLE_FRAC`; else `torso` if both shoulders (5,6) and both hips (11,12) confident: `h_px = (hip_mid_y - sh_mid_y) / TORSO_FRAC`; else `box`: `h_px = y2 - y1`. `Z = f * person_h / h_px`, `X = (u_foot - W/2) * Z / f`. If `floor_h` is given: `method = "floor"`, `(x, z) = to_floor(floor_h, foot_point_px)`.

`facing_dir`: needs both shoulders; `sw = |r_sh.x - l_sh.x|`, torso height `th` from shoulder-mid to hip-mid (or box/3 when hips missing). If `sw < 0.25*th` (side view) and nose confident: `(sign(nose.x - sh_mid.x), 0)`. Else if `l_sh.x > r_sh.x` (person faces camera) → `(0, -1)`; else `(0, 1)`. `faces` = angle between facing dir and `(b - a)` ≤ max_angle; `None` when facing unknown.

- [ ] **Step 1: Failing tests** `tests/test_geometry.py`:

```python
import math
import numpy as np
import pytest
from pulse import geometry as g
from pulse.datatypes import TrackObs

W, H = 960, 540
F = g.focal_px(W, 70.0)
CX, CY = W / 2, H / 2
CAM_H = 1.5

def synth_person(X, Z, height=1.70, mode="full", tid=1):
    """Project a standing person at ground (X, Z) through a level pinhole camera at CAM_H."""
    u = CX + F * X / Z
    v_foot = CY + F * CAM_H / Z
    v_head = CY - F * (height - CAM_H) / Z
    h = v_foot - v_head
    def y(frac): return v_head + frac * h
    kps = [[u, y(0.07), 0.9]] + [[u, y(0.05), 0.9]] * 2 + [[u, y(0.06), 0.9]] * 2 \
        + [[u + 0.1 * h, y(0.18), 0.9], [u - 0.1 * h, y(0.18), 0.9]] \
        + [[u, y(0.33), 0.9]] * 4 + [[u + 0.05 * h, y(0.48), 0.9], [u - 0.05 * h, y(0.48), 0.9]] \
        + [[u, y(0.72), 0.9]] * 2 + [[u, y(0.95), 0.9]] * 2
    if mode == "torso":          # cut off below the hips (e.g. sitting / frame bottom)
        for i in (0, 13, 14, 15, 16):
            kps[i][2] = 0.05
    if mode == "box":
        kps = None
    bbox = (u - 0.15 * h, v_head, u + 0.15 * h, v_foot)
    return TrackObs(0.0, 0, tid, bbox, 0.9, kps)

@pytest.mark.parametrize("mode", ["full", "torso", "box"])
@pytest.mark.parametrize("sep", [1.0, 2.0, 3.0, 4.0])
def test_distance_within_15pct(mode, sep):
    a = synth_person(-sep / 2, 4.0, mode=mode)
    b = synth_person(sep / 2, 4.5, mode=mode, tid=2)
    pa = g.position_m(a, (W, H)); pb = g.position_m(b, (W, H))
    true = math.hypot(sep, 0.5)
    assert abs(g.distance_m((pa.x, pa.z), (pb.x, pb.z)) - true) / true <= 0.15
    assert pa.method == mode

def test_depth_full_body_close_to_truth():
    p = g.position_m(synth_person(0.5, 3.0), (W, H))
    assert p.z == pytest.approx(3.0, rel=0.05) and p.x == pytest.approx(0.5, rel=0.1, abs=0.05)

def test_foot_point_uses_ankles_else_box_bottom():
    o = synth_person(0, 3.0)
    u, v = g.foot_point(o.bbox, o.keypoints)
    assert v == pytest.approx(o.keypoints[15][1])
    u2, v2 = g.foot_point(o.bbox, None)
    assert (u2, v2) == pytest.approx(((o.bbox[0] + o.bbox[2]) / 2, o.bbox[3]))

def test_homography_roundtrip():
    img = [(100, 400), (800, 400), (900, 530), (50, 530)]
    floor = [(0, 10), (10, 10), (10, 0), (0, 0)]
    h = g.fit_homography(img, floor)
    for p, q in zip(img, floor):
        assert g.to_floor(h, p) == pytest.approx(q, abs=1e-6)

def test_position_uses_floor_homography_when_given():
    img = [(100, 400), (800, 400), (900, 530), (50, 530)]
    h = g.fit_homography(img, [(0, 10), (10, 10), (10, 0), (0, 0)])
    o = TrackObs(0, 0, 1, (440, 300, 460, 400), 0.9, None)  # foot (450, 400) on the far edge
    p = g.position_m(o, (W, H), floor_h=h)
    assert p.method == "floor" and p.z == pytest.approx(10, abs=1e-6)

def test_facing_front_and_unknown():
    o = synth_person(0, 3.0)                           # l_sh.x > r_sh.x → faces camera
    assert g.facing_dir(o.keypoints) == (0, -1)
    assert g.facing_dir(None) is None
    assert g.faces(o, (0, 3.0), (0, 1.5)) is True     # B between person and camera
    assert g.faces(o, (0, 3.0), (0, 5.0)) is False
```

- [ ] **Step 2:** run → fail. **Step 3:** implement per the rules above (homography: `cv2.findHomography(np.float32(img), np.float32(floor), 0)`; `to_floor` uses homogeneous division). **Step 4:** run → pass. **Step 5:** commit `M1: geometry (pinhole distance, facing, homography)`.

### Task 4: zones

**Files:** Create `src/pulse/zones.py`, `tests/test_zones.py`.

**Interfaces:**
- Consumes: `Det`, `Zone` (Task 1).
- Produces:
  - `auto_zones(car_dets: list[Det], frame_shape: tuple[int, int], apron: float = 0.25, foot_frac: float = 0.4, min_area_frac: float = 0.002, merge_iou: float = 0.5) -> list[Zone]` — `frame_shape` is `(h, w)`; vehicle classes only (`car`, `truck`, `motorcycle`, `bus`); duplicates (IoU > merge_iou) keep the higher score; boxes with area < `min_area_frac` of the frame dropped. Zone = lower `foot_frac` of the box (`y from y2 - foot_frac*h to y2`), expanded horizontally by `apron*w` each side and downward by `apron*h`, clipped to the frame, normalised; ids `car-1..n` and names `Car 1..n` (`Motorcycle 1` for motorcycles) numbered left→right by box centre x; `source="auto"`, `kind="car"`.
  - `point_in_polygon(pt, polygon) -> bool` (ray casting; points on the boundary count as inside)
  - `polygon_area(polygon) -> float`
  - `zone_of(point, zones) -> str | None` — smallest-area zone containing the point.
  - `presence_intervals(ts: list[float], inside: list[bool], zone_min: float, zone_gap: float) -> list[tuple[float, float]]` — **hysteresis**: take the times of inside samples; split them where `t_next_inside - t_prev_inside > zone_gap + 1e-6`; each cluster `[t_first, t_last]` is kept only if `t_last - t_first >= zone_min - 1e-6`.
  - `save_zones(path, zones, frame_size: tuple[int,int] | None = None)`, `load_zones(path) -> list[Zone]`; file format `{"version": 1, "frame_size": [w, h] | null, "labeller": optional str, "zones": [Zone.to_dict(), ...]}`.

- [ ] **Step 1: Failing tests** `tests/test_zones.py`:

```python
import pytest
from pulse.datatypes import Det, Zone
from pulse import zones as Z

FS = (540, 960)  # (h, w)

def n(x, y): return (x / 960, y / 540)

def test_auto_zones_two_cars_left_to_right():
    dets = [Det((500, 220, 800, 420), 0.8, "car"), Det((100, 200, 400, 400), 0.9, "car"),
            Det((0, 0, 10, 10), 0.9, "person")]
    zs = Z.auto_zones(dets, FS)
    assert [z.name for z in zs] == ["Car 1", "Car 2"]
    assert all(z.kind == "car" and z.source == "auto" for z in zs)
    assert Z.zone_of(n(250, 430), zs) == zs[0].zone_id     # just below the car: feet of a viewer
    assert Z.zone_of(n(250, 205), zs) is None               # roof is not floor
    assert Z.zone_of(n(90, 380), zs) == zs[0].zone_id       # beside the car (apron)
    for z in zs:
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in z.polygon)

def test_auto_zones_merges_duplicates_and_drops_tiny():
    dets = [Det((100, 200, 400, 400), 0.9, "car"), Det((110, 205, 405, 400), 0.6, "car"),
            Det((900, 10, 905, 14), 0.9, "car")]
    zs = Z.auto_zones(dets, FS)
    assert len(zs) == 1

def test_motorcycle_named():
    zs = Z.auto_zones([Det((100, 300, 200, 400), 0.9, "motorcycle")], FS)
    assert zs[0].name == "Motorcycle 1" and zs[0].kind == "car"

def test_point_in_polygon_concave_and_boundary():
    L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert Z.point_in_polygon((0.5, 1.5), L)
    assert not Z.point_in_polygon((1.5, 1.5), L)
    assert Z.point_in_polygon((2, 0.5), L)

def test_zone_of_prefers_smallest():
    big = Zone("big", "Floor", "other", [(0, 0), (1, 0), (1, 1), (0, 1)])
    small = Zone("s", "Desk", "desk", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    assert Z.zone_of((0.5, 0.5), [big, small]) == "s"
    assert Z.zone_of((0.1, 0.1), [big, small]) == "big"
    assert Z.zone_of((0.5, 0.5), []) is None

def test_hysteresis_bridges_short_exit():
    ts = [round(i * 0.2, 3) for i in range(50)]
    inside = [False] * 5 + [True] * 20 + [False] * 3 + [True] * 12 + [False] * 10
    assert Z.presence_intervals(ts, inside, zone_min=1.0, zone_gap=1.0) == [pytest.approx((1.0, 7.8))]

def test_hysteresis_ignores_blip_and_splits_long_gap():
    ts = [round(i * 0.2, 3) for i in range(60)]
    blip = [False] * 10 + [True] * 3 + [False] * 47              # 0.4 s inside
    assert Z.presence_intervals(ts, blip, 1.0, 1.0) == []
    two = [True] * 10 + [False] * 10 + [True] * 10 + [False] * 30  # gap 2.2 s > 1 s
    assert Z.presence_intervals(ts, two, 1.0, 1.0) == [pytest.approx((0.0, 1.8)), pytest.approx((4.0, 5.8))]

def test_hysteresis_missing_observations_count_as_gap():
    ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]   # unseen 1.0 → 4.0
    assert len(Z.presence_intervals(ts, [True] * 12, 1.0, 2.0)) == 2

def test_save_load_roundtrip(tmp_path):
    zs = [Zone("desk", "Desk", "desk", [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)], "user")]
    p = tmp_path / "z.json"
    Z.save_zones(p, zs, (960, 540))
    assert [z.to_dict() for z in Z.load_zones(p)] == [
        {**zs[0].to_dict(), "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}]
```

`load_zones` returns polygons as lists of lists (JSON) — keep that; consumers only iterate pairs. **Step 2–5:** fail → implement → pass → commit `M1: zones (auto car zones, hysteresis, zone files)`.

### Task 5: timeline (TrackObs + persons → Samples)

**Files:** Create `src/pulse/timeline.py`, `tests/test_timeline.py`.

**Interfaces:**
- Consumes: `geometry.position_m`, `geometry.foot_point`, `TrackObs`, `Person`, `StableSegment`.
- Produces:
  - `build_timeline(obs: list[TrackObs], track_to_person: dict[int, int], frame_size: tuple[int,int], cfg: Config, floor_h=None) -> Timeline` — one Sample per (person, frame); if two tracks of one person share a frame keep the higher score; foot normalised; `pos_m` from `position_m`; `speed` = smoothed speed (below); samples sorted by t.
  - `smooth_speed(ts, feet_px, depths_z, f_px, window_s) -> list[float | None]` — centred moving average of foot pixels over `window_s`, then `|Δfoot_px|/Δt × (Z/f)` between neighbouring smoothed points (central difference; ends use one-sided); `None` for a person with < 2 samples.
  - `filter_stable(timeline, segments) -> Timeline` — drops samples whose `t` falls in a segment with `stable=False` or `same_view=False`; returns a new dict (persons with no samples left are kept with `[]`).
  - `already_present(person: Person, leave_after: float) -> bool` (`first_t < leave_after`).
  - `at_edge(sample, margin) -> bool` (any bbox side within `margin` of the frame border, in normalised units; needs `frame_size` → signature `at_edge(sample, frame_size, margin)`).

- [ ] **Step 1: Failing tests**:

```python
import pytest
from pulse.config import get_config
from pulse.datatypes import TrackObs, StableSegment, Sample
from pulse import timeline as T

CFG = get_config("short")

def walker(tid, x0, vx_px, n, y=400, t0=0.0, h=200):
    return [TrackObs(round(t0 + i * 0.2, 3), i, tid, (x0 + vx_px * i - 30, y - h, x0 + vx_px * i + 30, y), 0.9, None)
            for i in range(n)]

def test_build_timeline_maps_tracks_to_persons_and_normalises():
    obs = walker(1, 100, 0, 5) + walker(2, 500, 0, 5, t0=1.0)
    tl = T.build_timeline(obs, {1: 10, 2: 10}, (960, 540), CFG)
    assert list(tl) == [10]
    s = tl[10]
    assert len(s) == 10 and s[0].foot == pytest.approx((100 / 960, 400 / 540))
    assert all(a.t < b.t for a, b in zip(s, s[1:]))

def test_speed_still_vs_walking():
    still = T.build_timeline(walker(1, 300, 0, 20), {1: 1}, (960, 540), CFG)[1]
    assert max(x.speed for x in still) < 0.05
    # 200 px tall person at 1.7 m → 0.0085 m/px; 20 px per 0.2 s = 100 px/s ≈ 0.85 m/s
    walk = T.build_timeline(walker(2, 100, 20, 20), {2: 2}, (960, 540), CFG)[2]
    mid = [x.speed for x in walk[5:15]]
    assert all(0.6 < v < 1.1 for v in mid)

def test_filter_stable_drops_moving_segments():
    tl = {1: [Sample(t, 1, (0.5, 0.5), (0, 0, 1, 1), None, 0.0) for t in (0.0, 1.0, 2.0, 3.0)]}
    segs = [StableSegment(0.0, 1.5, True, True), StableSegment(1.5, 2.5, False, True),
            StableSegment(2.5, 4.0, True, False)]
    assert [s.t for s in T.filter_stable(tl, segs)[1]] == [0.0, 1.0]

def test_already_present_and_edge():
    from pulse.datatypes import Person
    assert T.already_present(Person(1, [1], 2.0, 9.0), 3.0)
    assert not T.already_present(Person(1, [1], 3.5, 9.0), 3.0)
    s = Sample(0, 1, (0.01, 0.5), (2, 100, 40, 300), None, 0.0)
    assert T.at_edge(s, (960, 540), 0.02)
    s2 = Sample(0, 1, (0.5, 0.5), (400, 100, 440, 300), None, 0.0)
    assert not T.at_edge(s2, (960, 540), 0.02)
```

**Step 2–5:** fail → implement → pass → commit `M1: timeline (samples, smoothed speed, stable filter)`.

### Task 6: journey

**Files:** Create `src/pulse/journey.py`, `tests/test_journey.py`.

**Interfaces:**
- Consumes: `zones.zone_of`, `zones.presence_intervals`, `Sample`, `Zone`, `Preset`.
- Produces:
  - `build(timeline: Timeline, zones: list[Zone], preset: Preset) -> dict[int, list[ZoneVisit]]` — per person: label each sample with `zone_of(sample.foot, zones)`; for each zone, `presence_intervals(ts, [label == zid], zone_min, zone_gap)`; each interval → `ZoneVisit(pid, zid, t_in, t_out, stopped)`; visits sorted by `t_in`. **stopped** (car zones only): within `[t_in, t_out]` there is a run of consecutive samples with `speed is not None and speed < stop_speed` whose time span (`t_last - t_first`) ≥ `stop_time`. Persons with no visits map to `[]`.
  - `sequence(visits: list[ZoneVisit], zones_by_id: dict[str, Zone] | None = None) -> list[str]` — zone ids in `t_in` order, consecutive duplicates collapsed.
  - `edit_distance(a: list[str], b: list[str]) -> int` (Levenshtein).
  - `journey_accuracy(pred: list[str], gt: list[str]) -> float` = `1 - edit_distance / max(len(pred), len(gt))`; `1.0` when both empty.

- [ ] **Step 1: Failing tests** (use `conftest.track/stay` and `zones4`):

```python
import pytest
from pulse.config import PRESETS
from pulse import journey as J
from tests.conftest import stay, track

S = PRESETS["short"]

def test_walk_a_then_b(zones4):
    tl = {1: stay(1, 0.2, 0.7, 0, 3) + track(1, [(0.45, 0.7)] * 10, t0=3.2) + stay(1, 0.7, 0.7, 5.2, 4)}
    v = J.build(tl, zones4, S)[1]
    assert [x.zone_id for x in v] == ["car-a", "car-b"]
    assert v[0].t_in == pytest.approx(0.0) and v[0].t_out == pytest.approx(3.0)
    assert v[1].dwell == pytest.approx(4.0)

def test_short_exit_merged(zones4):
    tl = {1: stay(1, 0.2, 0.7, 0, 2) + track(1, [(0.45, 0.7)] * 3, t0=2.2) + stay(1, 0.2, 0.7, 2.8, 2)}
    v = J.build(tl, zones4, S)[1]
    assert len(v) == 1 and v[0].t_out == pytest.approx(4.8)

def test_flicker_below_zone_min(zones4):
    tl = {1: stay(1, 0.45, 0.7, 0, 2) + stay(1, 0.7, 0.7, 2.2, 0.4) + stay(1, 0.45, 0.7, 2.8, 2)}
    assert J.build(tl, zones4, S)[1] == []

def test_stop_flag_requires_slow_and_long(zones4):
    slow = {1: stay(1, 0.2, 0.7, 0, 4, speed=0.05)}
    fast = {2: stay(2, 0.2, 0.7, 0, 4, speed=1.0)}
    brief = {3: stay(3, 0.2, 0.7, 0, 2, speed=0.05)}
    assert J.build(slow, zones4, S)[1][0].stopped
    assert not J.build(fast, zones4, S)[2][0].stopped
    assert not J.build(brief, zones4, S)[3][0].stopped

def test_desk_visit_never_stopped(zones4):
    v = J.build({1: stay(1, 0.45, 0.15, 0, 5, speed=0.0)}, zones4, S)[1]
    assert v[0].zone_id == "desk" and not v[0].stopped

def test_sequence_and_accuracy():
    from pulse.datatypes import ZoneVisit
    vs = [ZoneVisit(1, "a", 0, 1), ZoneVisit(1, "b", 2, 3), ZoneVisit(1, "b", 5, 6), ZoneVisit(1, "a", 8, 9)]
    assert J.sequence(vs) == ["a", "b", "a"]
    assert J.edit_distance(["a", "b", "c"], ["a", "b", "d"]) == 1
    assert J.journey_accuracy(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(2 / 3)
    assert J.journey_accuracy([], []) == 1.0
    assert J.journey_accuracy(["a"], []) == 0.0

def test_no_zones_gives_empty_visits():
    assert J.build({1: stay(1, 0.2, 0.7, 0, 3)}, [], S) == {1: []}
```

**Step 2–5:** fail → implement → pass → commit `M1: journey (zone visits, stops, sequences)`.

### Task 7: interest

**Files:** Create `src/pulse/interest.py`, `tests/test_interest.py`.

**Interfaces:**
- Consumes: `ZoneVisit`, `Sample`, `Zone`, `geometry.distance_m`, `Preset`.
- Produces:
  - `@dataclass AssistInfo(assisted: bool, t_first: float | None, staff_ids: list[int])`
  - `staff_index(staff_tracks: Timeline) -> dict[float, list[tuple[int, tuple[float,float]]]]` (key `round(t, 2)`, only samples with `pos_m`).
  - `near_staff(sample: Sample, index, dist: float) -> list[int]` — staff ids within `dist` metres at the same `t` (requires `sample.pos_m`).
  - `assistance(timeline: Timeline, staff_tracks: Timeline, preset: Preset) -> dict[int, AssistInfo]` — for each customer and each staff member: times where within `assist_dist`; runs bridged when the gap between successive near-times ≤ 1.0 s; a run with span ≥ `assist_time` makes the customer assisted; `t_first` = start of the earliest qualifying run (any staff).
  - `episodes(journeys, staff_tracks, preset, timeline, zones) -> list[InterestEpisode]` — per (customer, car zone): `dwell = Σ visit.dwell`, `visits = len`; deep if `dwell ≥ interest_time` or `visits ≥ 2`; episode spans first `t_in` to last `t_out`; `staff_ids` = staff ever within `assist_dist` during the visits' `[t_in, t_out]` samples; `unserved = not staff_ids`; `assisted` = some staff within `assist_dist` for a bridged run ≥ `assist_time` inside the visits. Only `kind == "car"` zones. Sorted by `(t_start, person_id)`. `staff_tracks` keys are staff person ids; `journeys`/`timeline` may contain staff too — skip persons that are keys of `staff_tracks`.
  - `@dataclass LiveAlert(person_id: int, zone_id: str, dwell: float, t_in: float, message: str)`; `open_alerts(journeys, staff_tracks, preset, timeline, zones, now: float) -> list[LiveAlert]` — for each customer whose **last** visit to a car zone is still open (`now - t_out <= zone_gap`), `dwell_so_far = now - t_in`; alert when `dwell_so_far ≥ alert_after` and no staff within `assist_dist` at any sample of that visit; message `f"Customer at {zone.name} for {int(dwell)} s, no staff nearby"`.

- [ ] **Step 1: Failing tests**:

```python
import pytest
from pulse.config import PRESETS
from pulse import interest as I, journey as J
from tests.conftest import stay

S = PRESETS["short"]
POS_A = (0.0, 4.0)

def cust_at_a(pid, t0, secs):
    return stay(pid, 0.2, 0.7, t0, secs, speed=0.05, pos=POS_A)

def staff_at(pid, t0, secs, pos):
    return stay(pid, 0.95, 0.3, t0, secs, pos=pos)

def run(tl, staff, zones):
    journeys = J.build(tl, zones, S)
    return I.episodes(journeys, staff, S, tl, zones)

def test_deep_by_cumulative_dwell_unserved(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 12, (5.0, 4.0))}
    ep = run(tl, staff, zones4)
    assert len(ep) == 1 and ep[0].zone_id == "car-a" and ep[0].dwell == pytest.approx(12)
    assert ep[0].unserved and not ep[0].assisted

def test_deep_by_two_visits(zones4):
    tl = {1: cust_at_a(1, 0, 2) + stay(1, 0.45, 0.7, 2.2, 3) + cust_at_a(1, 5.4, 2)}
    ep = run(tl, {}, zones4)
    assert len(ep) == 1 and ep[0].visits == 2

def test_short_single_visit_not_deep(zones4):
    assert run({1: cust_at_a(1, 0, 5)}, {}, zones4) == []

def test_assisted_when_staff_close_long_enough(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 4, (5.0, 4.0)) + staff_at(9, 4.2, 3, (1.0, 4.0)) + staff_at(9, 7.4, 4.6, (5.0, 4.0))}
    ep = run(tl, staff, zones4)[0]
    assert ep.assisted and not ep.unserved and ep.staff_ids == [9]

def test_brief_staff_contact_not_unserved_not_assisted(zones4):
    tl = {1: cust_at_a(1, 0, 12)}
    staff = {9: staff_at(9, 0, 4, (5.0, 4.0)) + staff_at(9, 4.2, 1, (1.0, 4.0)) + staff_at(9, 5.4, 6.6, (5.0, 4.0))}
    ep = run(tl, staff, zones4)[0]
    assert not ep.unserved and not ep.assisted

def test_desk_dwell_is_not_interest(zones4):
    assert run({1: stay(1, 0.45, 0.15, 0, 20, pos=POS_A)}, {}, zones4) == []

def test_assistance_store_level(zones4):
    tl = {1: cust_at_a(1, 0, 12), 2: cust_at_a(2, 0, 12)}
    staff = {9: staff_at(9, 3, 4, (0.5, 4.0))}
    info = I.assistance({k: v for k, v in tl.items()}, staff, S)
    assert info[1].assisted and info[1].t_first == pytest.approx(3.0)

def test_live_alert_after_threshold(zones4):
    tl = {1: cust_at_a(1, 0, 9)}
    j = J.build(tl, zones4, S)
    alerts = I.open_alerts(j, {}, S, tl, zones4, now=9.0)
    assert len(alerts) == 1 and "Car A" in alerts[0].message and alerts[0].dwell == pytest.approx(9.0)
    tl7 = {1: cust_at_a(1, 0, 7)}
    assert I.open_alerts(J.build(tl7, zones4, S), {}, S, tl7, zones4, now=7.0) == []

def test_live_alert_suppressed_by_staff(zones4):
    tl = {1: cust_at_a(1, 0, 9)}
    staff = {9: staff_at(9, 3, 0.4, (0.5, 4.0))}
    assert I.open_alerts(J.build(tl, zones4, S), staff, S, tl, zones4, now=9.0) == []
```

**Step 2–5:** fail → implement → pass → commit `M1: interest (deep interest, assistance, unserved, live alerts)`.

### Task 8: stats

**Files:** Create `src/pulse/stats.py`, `tests/test_stats.py`.

**Interfaces:**
- Consumes: `Person`, `Timeline`, `ZoneVisit`, `InterestEpisode`, `AssistInfo`, `Zone`, `Config`, `timeline.already_present/at_edge`, `zones.zone_of`, `geometry.distance_m`.
- Produces: `compute(persons, journeys, roles: dict[int, str], cfg: Config, *, timeline, episodes, assist: dict[int, AssistInfo], zones, duration: float, frame_size: tuple[int,int], analysed_share: float = 1.0, extra_warnings: list[str] = ()) -> Stats`. Keys exactly as `STORE_KEYS_DEFAULTS` (Task 1). Definitions (spec §4.5, §4.8):
  - customers = persons whose role is not `"staff"` (`"unsure"` counts as customer; `store.unsure` counts them).
  - `already_present` via `timeline.already_present(p, preset.leave_after)`.
  - `entries`: non-already-present customers whose **first** sample is inside an `entrance`-kind zone if any entrance zone exists, else whose first sample is `at_edge`. `exits`: customers whose last sample is inside an entrance zone (if any) else `at_edge`, and with `last_t <= duration - leave_after`.
  - `median_visit_s` over non-already-present customers of `last_t - first_t`.
  - groups: union-find over customer pairs with `|first_t_i - first_t_j| ≤ group_arrival_s` and, over common sample times in `[max(first), max(first)+group_window_s]` that both have `pos_m`, distance ≤ `group_dist_m` for more than 50% of those times (and ≥ 3 common samples). `groups` = number of components of size ≥ 2; `n_group_members`, `n_solo`.
  - `browse_rate` = share of customers with any `stopped` car visit. `assisted_share`/`assisted_customers` from `assist`. `median_time_to_assist` over non-already-present assisted customers of `t_first - first_t`.
  - `unserved_episodes` = count of `episodes` with `unserved`. `deep_interest_visitors` = customers with ≥ 1 episode. `has_desk` = any zone kind `desk`; `reached_desk` = customers with a desk visit of dwell ≥ `desk_time`; `deep_interest_reached_desk` likewise among deep-interest customers.
  - occupancy: for `t = 0, 1, 2, … ≤ duration`: customers with `first_t ≤ t ≤ last_t`; `peak_occupancy`, `peak_t` (first max). waiting: customers present and (not assisted or `t < t_first`). crowding: `customers / staff_present` when ≥ 1 staff present, else `-1`.
  - funnel: `{"entered": 1.0, "browsed", "deep_interest", "assisted", "reached_desk"}` as shares of customers (all `0.0` when no customers).
  - per_car (car zones only, ordered by rank): `zone_id, name, visitors` (distinct customers with a visit), `stops` (distinct customers with a stopped visit), `stop_rate = stops / customers`, `hold_rate = stops / visitors`, `median_dwell`, `total_dwell` (over customer visits), `repeat_visits` (customers with ≥ 2 visits), `deep_interest`, `unserved`, `unassisted_share = unserved / deep_interest` (None if 0), `rank` (1 = highest total_dwell; ties by stops). per_zone: same counting for every zone (`zone_id, name, kind, visitors, median_dwell, total_dwell`).
  - group/solo assisted shares: share of assisted among group members / solo customers (None when the denominator is 0).
  - time_buckets: 60-s buckets when duration < 3600 else 3600-s; each `{"t0","t1","arrivals","mean_occupancy","unserved","stops"}`.
  - heatmap: `cfg.heatmap_w × cfg.heatmap_h` counts of customer foot samples (clip coordinates into the grid). paths: up to 10 customers with the most samples, foot points downsampled to ≤ 50 points.
  - visits: per customer `{"person_id","role","first_t","last_t","already_present","visit_s","zones": sequence,"stopped_cars","assisted","t_first_assist","group"}`.
  - warnings: `"Nobody detected — camera too far or too dark?"` when no persons; `"No zones: draw zones to get per-car statistics"` when `zones` is empty; `"Distance estimate approximate (box-height fallback for >50% of samples)"` when > 50% of samples have `dist_method == "box"` (sets `distance_approx`); `"Only {pct}% of the clip analysed (camera moving or cut)"` when `analysed_share < 0.99`; plus `extra_warnings`.

- [ ] **Step 1: Failing tests** (scenario builder inside the test file):

```python
import pytest
from pulse.config import get_config
from pulse.datatypes import Person
from pulse import stats as ST, journey as J, interest as I
from tests.conftest import stay, track

CFG = get_config("short")

def scenario(zones):
    tl = {
        # already present, browses car A 12 s (deep, unserved), no assist
        1: stay(1, 0.2, 0.7, 0, 12, speed=0.05, pos=(0.0, 4.0)),
        # enters via entrance at t=5, stops at car B 5 s, walks to desk 6 s
        2: stay(2, 0.95, 0.5, 5, 1, pos=(6.0, 4.0)) + stay(2, 0.7, 0.7, 6.2, 5, speed=0.05, pos=(3.0, 4.0))
           + stay(2, 0.45, 0.15, 11.4, 6, pos=(1.0, 8.0)),
        # group with 2 (arrives within 5 s, stays 1 m away)
        3: stay(3, 0.95, 0.55, 6, 1, pos=(6.0, 5.0)) + stay(3, 0.72, 0.72, 7.2, 4, speed=0.05, pos=(3.5, 4.5)),
        # staff near desk the whole time
        9: stay(9, 0.47, 0.15, 0, 20, pos=(1.5, 8.0)),
    }
    persons = [Person(pid, [pid], s[0].t, s[-1].t) for pid, s in tl.items()]
    roles = {1: "customer", 2: "customer", 3: "unsure", 9: "staff"}
    cust = {k: v for k, v in tl.items() if roles[k] != "staff"}
    staff = {9: tl[9]}
    journeys = J.build(tl, zones, CFG.preset)
    eps = I.episodes(journeys, staff, CFG.preset, tl, zones)
    assist = I.assistance(cust, staff, CFG.preset)
    return ST.compute(persons, journeys, roles, CFG, timeline=tl, episodes=eps, assist=assist,
                      zones=zones, duration=20.0, frame_size=(960, 540))

def test_store_kpis(zones4):
    s = scenario(zones4).store
    assert s["visitors"] == 3 and s["staff"] == 1 and s["unsure"] == 1
    assert s["already_present"] == 1 and s["entries"] == 2
    assert s["unserved_episodes"] == 1 and s["deep_interest_visitors"] == 1
    assert s["has_desk"] and s["reached_desk"] == 1
    assert s["assisted_customers"] == 1          # customer 2 at the desk next to staff
    assert s["browse_rate"] == pytest.approx(3 / 3)
    assert s["groups"] == 1 and s["n_group_members"] == 2 and s["n_solo"] == 1
    assert s["median_visit_s"] == pytest.approx((12.4 + 5.0) / 2, abs=0.3)

def test_per_car_table(zones4):
    cars = {c["zone_id"]: c for c in scenario(zones4).per_car}
    assert set(cars) == {"car-a", "car-b"}
    assert cars["car-b"]["visitors"] == 2 and cars["car-b"]["stops"] == 2
    assert cars["car-a"]["unserved"] == 1 and cars["car-a"]["unassisted_share"] == 1.0
    assert cars["car-a"]["stop_rate"] == pytest.approx(1 / 3)
    assert sorted(c["rank"] for c in cars.values()) == [1, 2]

def test_occupancy_and_peak(zones4):
    st = scenario(zones4)
    occ = dict((t, n) for t, n in st.occupancy)
    assert occ[0] == 1 and occ[8] == 3 and st.store["peak_occupancy"] == 3
    assert all(r == -1 or r >= 0 for _, r in st.crowding)

def test_funnel_shares(zones4):
    f = scenario(zones4).store["funnel"]
    assert f["entered"] == 1.0 and f["deep_interest"] == pytest.approx(1 / 3)
    assert f["reached_desk"] == pytest.approx(1 / 3)

def test_heatmap_counts_customer_samples_only(zones4):
    st = scenario(zones4)
    total = sum(sum(r) for r in st.heatmap["counts"])
    n_customer_samples = 61 + (6 + 26 + 31) + (6 + 21)
    assert total == n_customer_samples

def test_empty_inputs_no_crash():
    st = ST.compute([], {}, {}, CFG, timeline={}, episodes=[], assist={}, zones=[], duration=10.0,
                    frame_size=(960, 540))
    assert st.store["visitors"] == 0 and st.per_car == []
    assert any("Nobody detected" in w for w in st.warnings)
    assert any("No zones" in w for w in st.warnings)

def test_entries_by_frame_edge_without_entrance_zone(zones4):
    no_ent = [z for z in zones4 if z.kind != "entrance"]
    tl = {5: track(5, [(0.0, 0.6)] + [(0.1, 0.6)] * 30, t0=6.0)}
    tl[5][0].bbox = (0, 200, 30, 320)                   # touching the left edge
    persons = [Person(5, [5], 6.0, tl[5][-1].t)]
    j = J.build(tl, no_ent, CFG.preset)
    st = ST.compute(persons, j, {5: "customer"}, CFG, timeline=tl, episodes=[], assist={},
                    zones=no_ent, duration=20.0, frame_size=(960, 540))
    assert st.store["entries"] == 1
```

**Step 2–5:** fail → implement → pass → commit `M1: stats (store, per-car, occupancy, funnel, groups, heatmap)`.

### Task 9: insights (R1–R7)

**Files:** Create `src/pulse/insights.py`, `tests/test_insights.py`.

**Interfaces:**
- Consumes: `Stats`, `InterestEpisode`, `Config`, `Insight`.
- Produces: `recommend(stats: Stats, episodes: list[InterestEpisode], cfg: Config) -> list[Insight]`; `confidence(n: int, min_sample: int) -> str` (`n < m` → `"insufficient"`, `< 2m` → `"low"`, `< 4m` → `"medium"`, else `"high"`). Every insight: `demo_only = cfg.preset.demo_only`; `moments` = list of `{"t", "person_id", "zone_id"}` (≤ 5); titles/actions use the spec §4.7 templates with numbers filled in. Insights with `confidence == "insufficient"` are still returned (UI shows "insufficient data"). Rules:
  - **R1** per car: `u` = unserved episodes at the car, `m` = deep-interest episodes at the car; fires if `u ≥ 3` or (`m > 0` and `u / m ≥ 0.4`). sample = `m`. window = the 1-min bucket (or the stats bucket size) containing most unserved episode starts. Action: `"Station a salesperson near {car} during {window}; {u} of {m} interested customers got no contact."`
  - **R2** needs ≥ 4 cars: car with `stop_rate ≥ Q3(stop_rate)` and `median_dwell ≤ Q1(median_dwell)` (numpy `percentile`, linear) and stop_rate > 0. sample = `store.visitors`.
  - **R3** per car: `stop_rate < 0.10` while median of per-car stop rates `> 0.30`. sample = `store.visitors`.
  - **R4**: `median_time_to_assist` is not None and `> alert_after * 2`. sample = `assisted_customers`.
  - **R5**: longest contiguous run of `crowding` points with ratio `> crowding_ratio` (ratio −1 breaks the run) lasting ≥ `crowding_time_s`; window = that run; evidence `k` = max ratio. sample = `store.visitors`.
  - **R6**: `has_desk` and `deep_interest_visitors > 0` and `(deep - deep_reached_desk) / deep ≥ 0.5`. sample = `deep_interest_visitors`.
  - **R7**: both shares not None and `solo_assisted_share - group_assisted_share ≥ 0.20`. sample = `min(groups, n_solo)`.

- [ ] **Step 1: Failing tests** — one must-fire and one must-not-fire per rule:

```python
import pytest
from pulse.config import get_config
from pulse.datatypes import Stats, InterestEpisode
from pulse.insights import recommend, confidence

CFG = get_config("short")       # min_sample 2, alert_after 8

def car(zid, stop_rate=0.3, median_dwell=10.0, deep=0, unserved=0, stops=3):
    return {"zone_id": zid, "name": zid.upper(), "visitors": 10, "stops": stops, "stop_rate": stop_rate,
            "hold_rate": 0.5, "median_dwell": median_dwell, "total_dwell": 100.0, "repeat_visits": 0,
            "deep_interest": deep, "unserved": unserved, "unassisted_share": None, "rank": 1}

def stats(per_car=(), crowding=(), **store):
    s = Stats.empty()
    s.store.update({"visitors": 10, "assisted_customers": 4}, **store)
    s.per_car = list(per_car)
    s.crowding = [list(x) for x in crowding]
    return s

def eps(zid, n_unserved, n_served):
    return ([InterestEpisode(i, zid, 10.0 * i, 10.0 * i + 12, 12, 1, False, True) for i in range(n_unserved)]
            + [InterestEpisode(100 + i, zid, 5.0, 17.0, 12, 1, True, False, [9]) for i in range(n_served)])

def rules(ins): return [i.rule for i in ins]

def test_r1_fires_on_three_unserved():
    ins = recommend(stats([car("a", deep=3, unserved=3)]), eps("a", 3, 0), CFG)
    r1 = [i for i in ins if i.rule == "R1"][0]
    assert "3 of 3" in r1.action and r1.cars == ["a"] and r1.moments

def test_r1_quiet_when_mostly_served():
    assert "R1" not in rules(recommend(stats([car("a", deep=5, unserved=1)]), eps("a", 1, 4), CFG))

def test_r2_glance_and_go():
    cs = [car("a", 0.8, 2.0), car("b", 0.3, 20), car("c", 0.35, 25), car("d", 0.4, 30)]
    ins = [i for i in recommend(stats(cs), [], CFG) if i.rule == "R2"]
    assert [i.cars for i in ins] == [["a"]]

def test_r2_quiet_when_dwell_high():
    cs = [car("a", 0.8, 40.0), car("b", 0.3, 20), car("c", 0.35, 25), car("d", 0.4, 30)]
    assert "R2" not in rules(recommend(stats(cs), [], CFG))

def test_r3_dead_zone():
    cs = [car("a", 0.05), car("b", 0.4), car("c", 0.5)]
    assert [i.cars for i in recommend(stats(cs), [], CFG) if i.rule == "R3"] == [["a"]]

def test_r3_quiet_when_store_median_low():
    cs = [car("a", 0.05), car("b", 0.2), car("c", 0.25)]
    assert "R3" not in rules(recommend(stats(cs), [], CFG))

def test_r4_long_waits():
    assert "R4" in rules(recommend(stats(median_time_to_assist=20.0), [], CFG))
    assert "R4" not in rules(recommend(stats(median_time_to_assist=10.0), [], CFG))

def test_r5_crowding_needs_two_minutes():
    long = [(t, 4.0) for t in range(0, 150)]
    short = [(t, 4.0) for t in range(0, 60)] + [(t, 1.0) for t in range(60, 150)]
    assert "R5" in rules(recommend(stats(crowding=long), [], CFG))
    assert "R5" not in rules(recommend(stats(crowding=short), [], CFG))

def test_r6_desk_drop_off():
    fire = stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=2)
    quiet = stats(has_desk=True, deep_interest_visitors=6, deep_interest_reached_desk=5)
    nodesk = stats(has_desk=False, deep_interest_visitors=6, deep_interest_reached_desk=0)
    assert "R6" in rules(recommend(fire, [], CFG))
    assert "R6" not in rules(recommend(quiet, [], CFG))
    assert "R6" not in rules(recommend(nodesk, [], CFG))

def test_r7_groups_get_less_attention():
    fire = stats(groups=3, n_solo=5, group_assisted_share=0.2, solo_assisted_share=0.6)
    quiet = stats(groups=3, n_solo=5, group_assisted_share=0.5, solo_assisted_share=0.6)
    assert "R7" in rules(recommend(fire, [], CFG))
    assert "R7" not in rules(recommend(quiet, [], CFG))

def test_confidence_levels_and_demo_flag():
    assert [confidence(n, 5) for n in (2, 5, 12, 25)] == ["insufficient", "low", "medium", "high"]
    ins = recommend(stats(median_time_to_assist=20.0, assisted_customers=1), [], CFG)
    r4 = [i for i in ins if i.rule == "R4"][0]
    assert r4.confidence == "insufficient" and r4.demo_only
```

**Step 2–5:** fail → implement → pass → commit `M1: insights R1–R7 with evidence and confidence`.

### Task 10: roles (reduced feature set, shared names)

**Files:** Create `src/pulse/roles.py`, `tests/test_roles.py`.

**Interfaces:**
- Consumes: `Person`, `Timeline`, `RoleSuggestion`, `Config`, `geometry.distance_m`.
- Produces:
  - `interactions(persons, timeline, embeddings: dict[int, np.ndarray], duration: float, cfg) -> dict[int, dict[str, float]]` with keys `outfit_peers` (count of other persons with cosine(emb) ≥ `outfit_sim_thr`), `contacts` (distinct others within `contact_dist_m` for a bridged run ≥ `contact_time_s`, gap bridge 1.0 s), `initiator` (share of this person's contacts where, over the 3 s before the contact run started, this person's displacement toward the other was larger than the other's toward them; 0 when no contacts), `presence` (observed sample count / (duration × sample_fps), capped at 1; `None` when `duration < presence_min_clip_s`).
  - `suggest(persons, interactions, cfg=None) -> list[RoleSuggestion]` — score = Σ wᵢ·normᵢ / Σ active wᵢ with `norm(outfit_peers)=min(v,3)/3`, `norm(contacts)=min(v,4)/4`, `norm(presence)=v`, `norm(initiator)=v`; `presence=None` is inactive. `staff` if score ≥ `role_staff_thr`, `customer` if ≤ `role_customer_thr`, else `unsure`; `confidence` = score for staff, 1−score for customer, 0.5 for unsure. Reasons: top ≤ 3 features by wᵢ·normᵢ > 0, phrased: `"same outfit as {n} others"`, `"close to {n} people for 5 s or more"`, `"on screen {pct}% of the clip"`, `"approached others first in {pct}% of contacts"`.
  - `load_roles_json(path) -> dict[int, str]` accepting `{"persons": [{"person_id": 1, "role": "staff", ...}]}` (Showroom Coach export) or `{"1": "staff"}`; roles other than staff/customer raise `ValueError`.
  - `final_roles(suggestions, imported: dict[int,str] | None, overrides: dict[int,str] | None) -> dict[int, str]` — precedence: user override > imported > suggestion (unsure stays `"unsure"`, counted as customer downstream).
  - `export_roles_json(roles: dict[int,str], names: dict[int,str] | None = None) -> dict` in the Coach format above.

- [ ] **Step 1: Failing tests**:

```python
import numpy as np, json, pytest
from pulse.config import get_config
from pulse.datatypes import Person
from pulse import roles as R
from tests.conftest import stay

CFG = get_config("dealership")

def test_suggest_staff_customer_unsure_with_reasons():
    persons = [Person(i, [i], 0, 100) for i in (1, 2, 3)]
    feats = {1: {"outfit_peers": 3, "contacts": 5, "presence": 0.95, "initiator": 0.8},
             2: {"outfit_peers": 0, "contacts": 0, "presence": 0.1, "initiator": 0.0},
             3: {"outfit_peers": 1, "contacts": 2, "presence": 0.5, "initiator": 0.5}}
    sug = {s.person_id: s for s in R.suggest(persons, feats, CFG)}
    assert sug[1].role == "staff" and sug[2].role == "customer" and sug[3].role == "unsure"
    assert "same outfit as 3 others" in sug[1].reasons and len(sug[1].reasons) <= 3
    assert sug[2].reasons == []

def test_presence_ignored_for_short_clips():
    persons = [Person(1, [1], 0, 30), Person(2, [2], 0, 30)]
    tl = {1: stay(1, 0.5, 0.5, 0, 30, pos=(0, 4)), 2: stay(2, 0.1, 0.5, 0, 2, pos=(8, 4))}
    emb = {1: np.array([1.0, 0]), 2: np.array([0.0, 1])}
    f = R.interactions(persons, tl, emb, 30.0, get_config("short"))
    assert f[1]["presence"] is None

def test_contacts_and_initiator():
    # person 1 walks up to person 2 (standing) and stays 6 s within 1 m
    from tests.conftest import track
    approach = [(0.0 + 0.1 * i, 4.0) for i in range(30)]          # 0 → 2.9 m along X
    p1 = track(1, [(0.3, 0.5)] * 30, pos=approach) + stay(1, 0.3, 0.5, 6.0, 6, pos=(3.0, 4.0))
    p2 = stay(2, 0.35, 0.5, 0, 12, pos=(3.8, 4.0))
    persons = [Person(1, [1], 0, 12), Person(2, [2], 0, 12)]
    f = R.interactions(persons, {1: p1, 2: p2}, {1: np.ones(2), 2: -np.ones(2)}, 12.0, get_config("short"))
    assert f[1]["contacts"] == 1 and f[2]["contacts"] == 1
    assert f[1]["initiator"] == 1.0 and f[2]["initiator"] == 0.0

def test_outfit_peers_cosine():
    persons = [Person(i, [i], 0, 10) for i in (1, 2, 3)]
    emb = {1: np.array([1.0, 0.1]), 2: np.array([1.0, 0.12]), 3: np.array([0.0, 1.0])}
    tl = {i: stay(i, 0.1 * i, 0.5, 0, 1, pos=(10.0 * i, 4)) for i in (1, 2, 3)}
    f = R.interactions(persons, tl, emb, 10.0, get_config("short"))
    assert (f[1]["outfit_peers"], f[2]["outfit_peers"], f[3]["outfit_peers"]) == (1, 1, 0)

def test_roles_json_formats_and_precedence(tmp_path):
    p = tmp_path / "roles.json"
    p.write_text(json.dumps({"persons": [{"person_id": 1, "role": "staff", "name": "Aziz"}]}))
    assert R.load_roles_json(p) == {1: "staff"}
    p.write_text(json.dumps({"2": "customer"}))
    assert R.load_roles_json(p) == {2: "customer"}
    p.write_text(json.dumps({"2": "boss"}))
    with pytest.raises(ValueError):
        R.load_roles_json(p)
    from pulse.datatypes import RoleSuggestion
    sug = [RoleSuggestion(1, "customer", 0.9, []), RoleSuggestion(2, "unsure", 0.5, []),
           RoleSuggestion(3, "staff", 0.8, [])]
    assert R.final_roles(sug, {1: "staff"}, {3: "customer"}) == {1: "staff", 2: "unsure", 3: "customer"}
    assert R.export_roles_json({1: "staff"}, {1: "Aziz"})["persons"][0] == {"person_id": 1, "role": "staff", "name": "Aziz"}
```

**Step 2–5:** fail → implement → pass → commit `M1: roles (features, suggestions with reasons, roles.json)`.

### Task 11: identity merge (shared)

**Files:** Create `src/pulse/identity.py`, `tests/test_identity.py`.

**Interfaces:**
- Consumes: `TrackObs`, `Person`, `Config`.
- Produces:
  - `@dataclass TrackSummary(track_id: int, t_start: float, t_end: float, start_foot: tuple[float,float], end_foot: tuple[float,float], emb: np.ndarray | None, n_obs: int)`
  - `summarize(obs: list[TrackObs], embeddings: dict[int, np.ndarray], frame_size) -> dict[int, TrackSummary]` (feet normalised via `geometry.foot_point`).
  - `merge(tracks: dict[int, TrackSummary], cfg: Config, max_gap_s: float | None = None) -> list[Person]` — `max_gap_s` defaults to `cfg.preset.leave_after`. Candidate pairs (a ends before b starts): gap `b.t_start - a.t_end` in `(-0.2, max_gap_s]`, cosine(emb) ≥ `merge_sim_thr` (pairs with a missing emb are not merged), and spatial plausibility `|b.start_foot - a.end_foot| ≤ merge_max_speed × max(gap, 0.2) + 0.05`. Greedy by descending similarity: union two clusters only if no track of one overlaps in time (> 0.2 s) with any track of the other **and** the joining pair are each other's cluster end/start (chain order). Person ids are `1..n` ordered by `first_t`; `track_ids` sorted.
  - `reattach(lost: list[TrackSummary], new: TrackSummary, now: float, cfg) -> int | None` — live: best lost track with `now - lost.t_end ≤ leave_after`, cosine ≥ `merge_sim_thr`, spatial plausibility as above; returns its `track_id`.

- [ ] **Step 1: Failing tests**:

```python
import numpy as np, pytest
from pulse.config import get_config
from pulse.identity import TrackSummary, merge, reattach

CFG = get_config("dealership")       # leave_after 10

def ts(tid, t0, t1, a=(0.5, 0.8), b=(0.5, 0.8), emb=(1.0, 0.0)):
    return TrackSummary(tid, t0, t1, a, b, np.array(emb, float), int((t1 - t0) * 5) + 1)

def groups(persons): return sorted(tuple(p.track_ids) for p in persons)

def test_merges_fragments_of_one_person():
    tr = {1: ts(1, 0, 10, b=(0.5, 0.8)), 2: ts(2, 11, 20, a=(0.52, 0.8))}
    assert groups(merge(tr, CFG)) == [(1, 2)]

def test_never_merges_overlapping_tracks():
    tr = {1: ts(1, 0, 10), 2: ts(2, 5, 15)}
    assert groups(merge(tr, CFG)) == [(1,), (2,)]

def test_respects_max_gap_and_space():
    far_time = {1: ts(1, 0, 10), 2: ts(2, 25, 30)}
    far_space = {1: ts(1, 0, 10, b=(0.1, 0.8)), 2: ts(2, 10.5, 20, a=(0.9, 0.8))}
    assert groups(merge(far_time, CFG)) == [(1,), (2,)]
    assert groups(merge(far_space, CFG)) == [(1,), (2,)]

def test_dissimilar_kept_apart_and_transitive_constraint():
    tr = {1: ts(1, 0, 10), 2: ts(2, 11, 20, emb=(0, 1)), 3: ts(3, 12, 18, emb=(1, 0.05))}
    persons = merge(tr, CFG)
    assert groups(persons) == [(1, 3), (2,)]
    assert [p.person_id for p in sorted(persons, key=lambda p: p.first_t)] == [1, 2]

def test_reattach_recently_lost():
    lost = [ts(7, 0, 10, b=(0.5, 0.8)), ts(8, 0, 3, b=(0.5, 0.8), emb=(0, 1))]
    new = ts(20, 12, 12, a=(0.52, 0.8))
    assert reattach(lost, new, now=12, cfg=CFG) == 7
    assert reattach(lost, ts(21, 30, 30), now=30, cfg=CFG) is None
```

**Step 2–5:** fail → implement → pass → commit `M1: identity merge + live re-attach`. Then **milestone close (M1)**: run `bash scripts/verify.sh` (must print `VERIFY OK`), append M1 to PROGRESS.md (tests count, decisions), commit.

---

## M2 — Perception chain on real video, cache, camera motion

### Task 12: sources, tracking, engine (detect → track → pose)

**Files:** Create `src/pulse/sources.py`, `src/pulse/tracking.py`, `src/pulse/engine.py`, `scripts/make_clip.py`, `tests/test_sources.py`, `tests/test_tracking.py`, `tests/integration/conftest.py`, `tests/integration/test_engine.py`.

**Interfaces:**
- Produces:
  - `sources.probe(path) -> VideoInfo(width, height, fps, n_frames, duration)`; raises `VideoError("Unreadable or unsupported video: …")` when OpenCV cannot open/read a frame.
  - `sources.iter_frames(path, sample_fps=5.0, max_width=960, max_duration=None) -> Iterator[tuple[float, np.ndarray]]` — OpenCV `VideoCapture`; picks source frame `i` when `i == round(k * src_fps / sample_fps)` for sample `k`; `t = round(k / sample_fps, 3)`; resizes so width ≤ max_width keeping aspect (INTER_AREA); stops at `max_duration`. Also `sources.sampled_frame_size(info, max_width) -> (w, h)` and `sources.decode_jpeg(data: bytes, max_width=960) -> np.ndarray` (raises `ValueError` on garbage).
  - `scripts/make_clip.py SRC DST --start S --dur D [--width W] [--rotate-portrait]` — imageio-ffmpeg cut + H.264 re-encode (used for fixtures).
  - `tracking.Tracker(sample_fps: float, cfg: Config)` with `update(dets: list[Det], t: float, frame_idx: int) -> list[TrackObs]` (wraps `sv.ByteTrack(track_activation_threshold=cfg.track_activation_thr, lost_track_buffer=int(cfg.track_buffer_s*sample_fps), minimum_matching_threshold=cfg.track_match_thr, frame_rate=sample_fps)`; keypoints `None` here) and `reset()`.
  - `engine.PerceptionEngine(cfg: Config, providers=("CPUExecutionProvider",))` with `process(t, frame, frame_idx) -> list[TrackObs]` (detect persons → track → pose on tracked boxes → TrackObs with keypoints), `detect_vehicles(frame) -> list[Det]`, attributes `detector`, `pose_est`, `tracker`, and `timings` (rolling ms per stage).

- [ ] **Step 1: Failing unit tests** (`tests/test_sources.py`) — generate tiny synthetic videos with imageio-ffmpeg in `tmp_path` (moving white rectangle on grey, 30 fps):

```python
import numpy as np, imageio_ffmpeg, pytest
from pulse import sources

def write_video(path, w, h, fps, seconds):
    gen = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                      macro_block_size=1, output_params=["-loglevel", "error"])
    gen.send(None)
    for i in range(int(fps * seconds)):
        f = np.full((h, w, 3), 90, np.uint8); x = (i * 5) % (w - 40)
        f[h // 2 - 20:h // 2 + 20, x:x + 40] = 255
        gen.send(f)
    gen.close()

def test_iter_frames_samples_at_5fps(tmp_path):
    p = tmp_path / "a.mp4"; write_video(p, 640, 360, 30, 2)
    frames = list(sources.iter_frames(p, 5.0, 960))
    assert [t for t, _ in frames] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    assert frames[0][1].shape == (360, 640, 3)

def test_portrait_video_resized_and_timed(tmp_path):
    p = tmp_path / "p.mp4"; write_video(p, 1080, 1920, 29.97, 1.2)
    frames = list(sources.iter_frames(p, 5.0, 960))
    assert frames[0][1].shape == (1707, 960, 3)
    assert frames[1][0] == pytest.approx(0.2)

def test_max_duration_trims_and_flags(tmp_path):
    p = tmp_path / "l.mp4"; write_video(p, 320, 240, 10, 3)
    assert sources.probe(p).duration == pytest.approx(3.0, abs=0.15)
    frames = list(sources.iter_frames(p, 5.0, 960, max_duration=1.0))
    assert frames[-1][0] < 1.0 + 1e-9

def test_garbage_raises_video_error(tmp_path):
    p = tmp_path / "x.mp4"; p.write_text("not a video")
    with pytest.raises(sources.VideoError):
        sources.probe(p)

def test_decode_jpeg_rejects_garbage():
    with pytest.raises(ValueError):
        sources.decode_jpeg(b"nope")
```

`tests/test_tracking.py`: two synthetic `Det` boxes moving 5 px/frame for 20 frames keep 2 stable ids; a box missing for 3 frames (< buffer) keeps its id.

- [ ] **Step 2:** fail → implement `sources.py`, `tracking.py`, `engine.py` → pass.
- [ ] **Step 3: Slow integration** `tests/integration/conftest.py` provides session fixtures: `meva_dev_clip` (see Task 2 Step 7), `meva_10s(tmp_path_factory)` = `make_clip` of `meva_dev_clip` from the 60 s mark, 10 s, width 960, and `showroom_clip` (first of `data/stock/*.mp4|*.mov|*.webm`, else `data/synthetic/share/*.mp4`, else `pytest.fail` with a message naming both folders). `tests/integration/test_engine.py` (`@pytest.mark.slow`): processing the 10 s clip yields ≥ 1 person; a track id that appears in frame k appears again in ≥ 80% of the next 10 frames for at least one track (stable ids); keypoints shape 17×3; on the middle frame of `showroom_clip`, `detect_vehicles` returns ≥ 1 vehicle.
- [ ] **Step 4:** commit `M2: sources, tracking, perception engine`.

### Task 13: embeddings, camera motion, perception cache, offline pass

**Files:** Create `src/pulse/perception/embed.py`, `src/pulse/camera_motion.py`, `src/pulse/cache.py`, `src/pulse/pipeline.py` (perceive + build_people only), `tests/test_camera_motion.py`, `tests/test_cache.py`, `tests/test_embed.py`, `tests/integration/test_perceive.py`.

**Interfaces:**
- Produces:
  - `embed.hsv_hist(crop_bgr, bins=(16, 4)) -> np.ndarray` (L2-normalised, H×S histogram of the torso crop); `embed.torso_crop(frame, obs: TrackObs, kp_thr) -> np.ndarray | None` (shoulder→hip box from keypoints, else box rows 20%–60%); `embed.Embedder(use_dino: bool = True)` with `embed(crops: list[np.ndarray]) -> np.ndarray` = concat(0.5·mean HSV, 0.5·mean DINOv2 CLS) L2-normalised (HSV only when DINOv2 is unavailable; log once).
  - `camera_motion.motion_score(prev_gray, gray) -> tuple[float, bool]` (ORB 500 features + `cv2.findHomography` RANSAC on 320-px-wide grays; returns mean corner displacement as a fraction of width, and `cut=True` when < 15 good matches or HSV-histogram correlation < 0.5) and `segments(times: list[float], grays: list[np.ndarray], key_idx: int, move_thr: float = 0.01) -> list[StableSegment]` — a frame is "moving" when displacement from the previous frame > `move_thr` or a cut; consecutive frames with the same state form a segment; stable segments get `same_view` = displacement between the segment's middle frame and the keyframe < 0.02 and not a cut. `segments(video_path, cfg)` convenience wrapper reads frames itself.
  - `cache.video_key(path, cfg) -> str` (sha1 of file bytes + `sample_fps`, `max_width`, `max_duration_s`, detector, pose model); `cache.PerceptionCache` dataclass `(dir, meta, obs, cars, keyframe_t, segments, embeddings)`; `cache.save(pc)`, `cache.load(dir) -> PerceptionCache`, `cache.exists(dir) -> bool`; files: `meta.json`, `tracks.npz` (`t, frame_idx, track_id, bbox[N,4], score, kps[N,17,3]` with NaN for missing), `cars.json`, `segments.json`, `embeddings.npz`, `keyframe.jpg` (face-blurred via `render.blur_faces`, Task 14 — until then write the raw keyframe only into the cache dir, never into outputs).
  - `cache.PERCEPTION_RUNS` — module-level counter incremented each time models are run for a video; `cache.perception_runs() -> int`.
  - `pipeline.perceive(video, cache_root: Path, cfg, progress: Callable[[float], None] | None = None, engine: PerceptionEngine | None = None) -> PerceptionCache` — cache hit returns `cache.load(...)` **without** touching models; miss: probe (VideoError propagates), iterate frames, `engine.process`, collect per-track up to 5 torso crops spread over the track life with score ≥ 0.5, keyframe = middle frame of the longest stable segment (compute segments from 320-px grays stored during the pass), `detect_vehicles` on the keyframe, embeddings, `meta = {video, sha1, key, src_w, src_h, src_fps, duration, analysed_duration, trimmed: bool, frame_w, frame_h, sample_fps, n_samples, detector, pose_model, providers, perception_seconds, rtf}`; `rtf = analysed_duration / perception_seconds`.
  - `pipeline.build_people(pc: PerceptionCache, cfg, floor_h=None) -> tuple[list[Person], Timeline, dict[int, int]]` — `identity.summarize` + `identity.merge` + `timeline.build_timeline`.

- [ ] **Step 1: Failing tests**: `test_camera_motion.py` (shifted synthetic textured frames → moving; identical frames → stable; random noise frame → cut; a shifted-then-still sequence yields segments with the right `stable` flags); `test_embed.py` (red vs blue torso crops: same colour cosine > 0.95, different < 0.5; `torso_crop` uses keypoints when confident); `test_cache.py` (save→load round trip of a hand-built PerceptionCache in `tmp_path`, NaN keypoints → `None`, `video_key` changes when `sample_fps` changes).
- [ ] **Step 2:** fail → implement → pass.
- [ ] **Step 3: Slow integration** `tests/integration/test_perceive.py`: `perceive(meva_10s)` creates a cache; a second call returns equal `obs` length and `perception_runs()` did not increase; `build_people` returns ≥ 1 person with a non-empty timeline; `meta["rtf"] > 0`.
- [ ] **Step 4:** commit; **milestone close (M2):** verify.sh green, PROGRESS.md (measured RTF on the 10 s clip, decisions), commit.

---

## M3 — CLI `analyze` and every output file

### Task 14: render (face blur, annotated H.264, thumbnails, contact sheets)

**Files:** Create `src/pulse/render.py`, `tests/test_render.py`.

**Interfaces:**
- Produces:
  - `face_box(obs_bbox, keypoints, kp_thr=0.3) -> tuple[int,int,int,int]` — from confident nose/eyes/ears (indices 0–4): their bounding box padded by 0.6× its max side (min side = 0.12 × box width); fallback: top 20% of the person box.
  - `blur_faces(frame, observations: list[tuple[BBox, keypoints|None]]) -> np.ndarray` — strong pixelation + Gaussian of each face box (copy; input untouched).
  - `@dataclass FrameOverlay(t, persons: list[dict] # {person_id, bbox, keypoints, role, zone_id, dwell, alert}, zones: list[dict] # {zone_id, name, polygon, now, total})`
  - `draw_overlay(frame, overlay, frame_size) -> np.ndarray` — zones outlined with name + "now n · total m"; customers orange `(0,140,255)` with dwell timer text when in a zone; staff dimmed blue `(180,120,60)` at 50% alpha; alert persons red outline; always calls `blur_faces` first.
  - `render(video, overlays: dict[float, FrameOverlay], out_path, cfg) -> Path` — re-reads frames with `sources.iter_frames` (same sampling, so `t` matches), draws, writes H.264 `yuv420p` mp4 at `sample_fps` via `imageio_ffmpeg.write_frames(..., codec="libx264", macro_block_size=2)` (shared signature with Coach: `render(video, timeline, events, out)` — provide that as a thin wrapper building overlays from timeline + events).
  - `thumbnail(frame, bbox, keypoints, size=(128, 256)) -> np.ndarray` (blurred crop), `contact_sheet(video, moments: list[dict], overlays, out_path, cfg, cols=2) -> Path` (up to 4 tiles, each the frame nearest `moment.t`, annotated and blurred, caption `t=..s`).
- [ ] **Step 1: Failing tests**: `face_box` from keypoints and fallback; `blur_faces` changes pixels inside the face box (variance drops > 50% on a random-noise frame) and not outside; `render` on a synthetic 2 s video with one overlay per sample writes an mp4 whose `imageio_ffmpeg.count_frames_and_secs` gives 10 frames and whose codec is h264 (`ffprobe`-free check: read first bytes for `ftyp` + decode one frame with OpenCV); `contact_sheet` writes a JPEG of the expected grid size.
- [ ] **Step 2–4:** fail → implement → pass → commit `M3: render with face blur, H.264 output, contact sheets`.

### Task 15: report, scoring pipeline, CLI analyze, check_outputs

**Files:** Create `src/pulse/report.py`, extend `src/pulse/pipeline.py` (`score`, `write_outputs`, `analyze`), `src/pulse/cli.py` (`analyze`), `scripts/check_outputs.py`, `tests/test_report.py`, `tests/test_pipeline_edge.py`, `tests/integration/test_analyze_cli.py`; modify `scripts/verify.sh`.

**Interfaces:**
- Produces:
  - `pipeline.Analysis` dataclass `(persons, timeline, suggestions, roles, journeys, episodes, assist, stats, insights, zones, meta)`.
  - `pipeline.score(pc: PerceptionCache, cfg, zones: list[Zone] | None = None, overrides: dict[int,str] | None = None, imported_roles: dict[int,str] | None = None) -> Analysis` — `zones=None` → `zones.auto_zones(pc.cars, (frame_h, frame_w))`; pure (no model calls; asserts `perception_runs()` unchanged in tests); filters timeline with `filter_stable` for zone units; `analysed_share` = stable same-view seconds / duration; trimmed videos add `"Video trimmed to the first {max} s"`.
  - `pipeline.write_outputs(pc, analysis, out_dir, cfg, video=None, render_video=True) -> list[Path]`.
  - `pipeline.analyze(video, out_dir, preset="short", zones_path=None, roles_path=None, cache_root=Path("runs/cache"), render_video=True, cfg=None) -> Analysis`.
  - `report.build_report(analysis, pc, out_dir, cfg) -> dict` writes: `report.json` (keys: `banner, meta, store, per_car, per_zone, zones, people, insights, journey_flow, charts, files, warnings`), `stats.json`, `insights.json`, `zones.json`, `roles.json` (Coach format), `people.json` (suggestions + thumbnail paths), `visits.csv`, `per_car.csv`, `occupancy.csv`, `heatmap.png` (keyframe, face-blurred, with a JET heat overlay + up to 10 paths), `occupancy.svg`, `waiting.svg`, `journey_flow.json` (`{"nodes":[{"id","name"}], "links":[{"source","target","count"}], "top_paths":[{"path":[...], "count"}]}` with `Start`/`End` nodes), `thumbs/person_<id>.jpg`, `contact_sheets/insight_<k>.jpg` for each insight with moments, `keyframe.jpg` (blurred).
  - CLI: `pulse analyze VIDEO --out DIR [--preset short|dealership] [--zones FILE] [--roles FILE] [--no-video] [--cache DIR]`; prints a summary (visitors, insights, RTF) and exits 0; exits 2 with the VideoError message for unreadable input.
  - `scripts/check_outputs.py DIR [DIR ...]` — validates every file above: JSON parses and has required keys/types (`report.json` keys listed above; `stats.json` has all `STORE_KEYS_DEFAULTS` keys under `store`), CSV headers exact (`visits.csv`: `person_id,role,first_t,last_t,already_present,visit_s,zones,stopped_cars,assisted,t_first_assist,group`; `per_car.csv`: `zone_id,name,visitors,stops,stop_rate,hold_rate,median_dwell,total_dwell,repeat_visits,deep_interest,unserved,unassisted_share,rank`; `occupancy.csv`: `t,customers,waiting,customers_per_staff`), PNG/JPEG decode with OpenCV and are non-empty, SVG parses as XML with `<svg`, `annotated.mp4` decodes ≥ 1 frame and its sample-count ≈ duration × sample_fps (±10%), every path in `report.json["files"]` exists; prints `OK <file>` per file and exits 1 on the first failure with the reason.
- [ ] **Step 1: Failing tests** `tests/test_pipeline_edge.py` (fast; builds a `PerceptionCache` in memory from synthetic `TrackObs`, no models):

```python
from pulse import pipeline, cache
from pulse.config import get_config

def synthetic_cache(tmp_path, obs, duration=20.0, trimmed=False):
    meta = {"video": "synthetic", "sha1": "x", "key": "k", "src_w": 960, "src_h": 540, "src_fps": 25.0,
            "duration": duration, "analysed_duration": duration, "trimmed": trimmed, "frame_w": 960,
            "frame_h": 540, "sample_fps": 5.0, "n_samples": int(duration * 5), "detector": "test",
            "pose_model": "test", "providers": ["CPUExecutionProvider"], "perception_seconds": 1.0, "rtf": duration}
    from pulse.datatypes import StableSegment
    return cache.PerceptionCache(tmp_path, meta, obs, [], duration / 2, [StableSegment(0, duration, True, True)], {})

def test_score_with_no_people(tmp_path):
    a = pipeline.score(synthetic_cache(tmp_path, []), get_config("short"), zones=[])
    assert a.stats.store["visitors"] == 0 and a.insights == []
    assert any("Nobody detected" in w for w in a.stats.warnings)

def test_score_with_no_zones(tmp_path):
    from pulse.datatypes import TrackObs
    obs = [TrackObs(round(i * 0.2, 3), i, 1, (400, 200, 460, 400), 0.9, None) for i in range(50)]
    a = pipeline.score(synthetic_cache(tmp_path, obs), get_config("short"), zones=[])
    assert a.stats.store["visitors"] == 1 and a.stats.per_car == []
    assert any("No zones" in w for w in a.stats.warnings)

def test_trim_warning_in_stats(tmp_path):
    a = pipeline.score(synthetic_cache(tmp_path, [], trimmed=True), get_config("short"), zones=[])
    assert any("trimmed" in w for w in a.stats.warnings)

def test_rescore_does_not_run_models(tmp_path):
    before = cache.perception_runs()
    pipeline.score(synthetic_cache(tmp_path, []), get_config("dealership"), zones=[])
    assert cache.perception_runs() == before
```

`tests/test_report.py`: `build_report` on a synthetic analysis writes every file listed and `scripts/check_outputs.py` (imported as a module via `runpy`/`importlib`) passes on it, with `render_video=False` and check_outputs invoked with `--no-video`.
- [ ] **Step 2:** fail → implement → pass.
- [ ] **Step 3: Slow integration** `tests/integration/test_analyze_cli.py`: `subprocess.run([".venv/bin/pulse", "analyze", meva_10s, "--out", tmp/out])` exits 0 and `check_outputs.py tmp/out` exits 0; garbage file → exit 2 with "Unreadable".
- [ ] **Step 4: verify.sh** — insert before the git steps:

```bash
MEVA_DEV=$($PY -c "import json;print(json.load(open('labels/split.json'))['dev'][0]['video'])" 2>/dev/null || ls data/meva/videos/*.avi | head -1)
SHOW=$(ls data/stock/*.mp4 data/stock/*.mov data/stock/*.webm data/synthetic/share/*.mp4 2>/dev/null | head -1)
step "analyze MEVA dev clip" .venv/bin/pulse analyze "$MEVA_DEV" --out runs/verify/meva --preset dealership
step "analyze showroom clip" .venv/bin/pulse analyze "$SHOW" --out runs/verify/stock --preset short
step "check outputs" $PY scripts/check_outputs.py runs/verify/meva runs/verify/stock
```

If `SHOW` is empty the step must fail (do not silently drop it).
- [ ] **Step 5:** commit; **milestone close (M3):** verify.sh green, PROGRESS.md, commit.

---

## M4 — Web upload flow

### Task 16: FastAPI backend + jobs + smoke_api

**Files:** Create `src/pulse/app/__init__.py`, `src/pulse/app/main.py`, `src/pulse/app/jobs.py`, `scripts/smoke_api.py`, `tests/test_api.py`, `tests/integration/test_api_flow.py`; modify `scripts/verify.sh`.

**Interfaces:**
- Produces (`main.create_app(jobs_dir: Path = Path("jobs"), cache_root: Path | None = None, engine_factory=None) -> FastAPI`; module-level `app = create_app()` for uvicorn):
  - `GET /` → `static/index.html`; `GET /live` → `static/live.html`; `/static/*`.
  - `POST /api/jobs` (multipart `file`, form `preset` = `short|dealership`) → `{"job_id"}`; rejects extensions other than mp4/mov/avi/webm/mkv with 400 JSON `{"error"}`; saves `jobs/<id>/input.<ext>`; starts perception in a single background worker thread (`concurrent.futures.ThreadPoolExecutor(1)`); job states `queued → perceiving → ready_for_zones → scored | failed`.
  - `GET /api/jobs/{id}` → `{"job_id","state","progress" (0..1),"message","preset","warnings"}`; unknown id → 404 JSON.
  - `GET /api/jobs/{id}/keyframe.jpg` (face-blurred); `GET /api/jobs/{id}/zones` → zone file dict (auto zones on first call); `PUT /api/jobs/{id}/zones` (zone file dict; validates polygons ≥ 3 points in [0,1]; 422 otherwise).
  - `GET /api/jobs/{id}/people` → `[{person_id, role, suggested_role, confidence, reasons, thumb_url, first_t, last_t}]`; `PUT /api/jobs/{id}/roles` `{"overrides": {"<pid>": "staff|customer"}, "skip": bool}`; `POST /api/jobs/{id}/roles/import` (multipart roles.json).
  - `POST /api/jobs/{id}/score` (`{"preset"?: str, "render_video"?: bool}`) → runs `pipeline.score` + `write_outputs` (video render in the background worker; JSON outputs synchronously) → `report.json` content; sets state `scored`. `GET /api/jobs/{id}/results` → report.json (404 until scored). `GET /api/jobs/{id}/files/{path}` (only inside the job's out dir; no `..`). `GET /api/jobs/{id}/export.json|export.csv`.
  - `GET /api/health` → `{"ok": true, "perception_runs": cache.perception_runs()}`.
  - Startup: delete job dirs older than 24 h (`jobs.cleanup(jobs_dir, max_age_s=86400)`).
  - Failed perception (VideoError or any exception) → state `failed`, `message` = readable text; API stays 200 on status endpoints.
- `scripts/smoke_api.py` (in-process `fastapi.testclient.TestClient`, real models, real clip `runs/fixtures/meva_20s.mp4` made by `make_clip` from the MEVA dev clip if absent): upload → poll until `ready_for_zones` (timeout 300 s) → GET zones → PUT zones (one rectangle "Car: Demo" kind car, one "Desk") → GET people → PUT roles (flip the first person to staff) → POST score → GET results (assert KPIs present, visitors ≥ 1) → PUT zones (move the rectangle) → POST score → assert results changed in `per_zone`/`per_car` and `perception_runs` unchanged and `tracks.npz` mtime unchanged → prints each step and `SMOKE API OK`.
- [ ] **Step 1: Failing tests** `tests/test_api.py` (fast: `engine_factory` injects a stub engine object whose `process` returns scripted `TrackObs` for a synthetic video — this is dependency injection of the model layer for unit speed; the real-model path is covered by `smoke_api.py` and the slow test, so no gate is weakened):
  - `test_upload_garbage_fails_cleanly`: upload `b"not a video"` as `x.mp4` → state eventually `failed`, message contains "Unreadable", status endpoints return 200.
  - `test_bad_extension_400`.
  - `test_job_with_no_people`: synthetic video + stub engine returning `[]` → score → results warnings contain "Nobody detected".
  - `test_zone_validation_422`.
  - `test_full_flow_rescore_no_redetect`: stub engine with scripted people → zones → roles → score → edit zones → score: second results differ and the stub's call count did not increase.
  - `test_files_path_traversal_blocked` (`/files/../job.json` → 404/400).
- [ ] **Step 2:** fail → implement → pass. Slow test `tests/integration/test_api_flow.py` runs `scripts/smoke_api.py`'s `main()` and asserts it returns 0.
- [ ] **Step 3: verify.sh** — add `step "smoke API" $PY scripts/smoke_api.py` before the git steps. Commit `M4: FastAPI backend, jobs, re-score without re-detection`.

### Task 17: Frontend (upload, progress, zone editor, staff gallery, dashboard)

**Files:** Create `src/pulse/app/static/{index.html,app.js,zones.js,dashboard.js,style.css}`, `tests/test_static.py`.

**Interfaces:** consumes the REST API above. No build step, no CDN, no external fonts. Plain ES modules.

- `index.html`: banner "Anonymous, aggregated analytics; decision support only." + steps: (1) upload form (file + preset radio: Dealership / Short clip) with a progress bar polling `GET /api/jobs/{id}` every 1 s; (2) zone editor; (3) staff gallery; (4) dashboard. A link to `/live`.
- `zones.js`: canvas over `keyframe.jpg`; draws polygons; drag vertices; select + Delete key / button deletes; "Add rectangle" (drag) and "Add polygon" (click points, double-click to close); name + kind (`car|desk|entrance|lounge|other`) inputs for the selected zone; "Looks right" → `PUT zones` then go to staff step. Coordinates stored normalised.
- Staff gallery: one card per person (blurred thumbnail, suggested role badge, confidence, ≤ 3 reasons, a toggle button staff/customer), "Looks right" → `PUT roles` → `POST score`; "Skip" → `PUT roles {"skip": true}` → score; results then show the warning banner "Roles skipped: everyone counted as a customer".
- `dashboard.js`: KPI tiles (visitors, groups, median visit length, browse rate, assisted share, unserved-interest episodes, peak occupancy); warnings list; per-car table (stops & stop rate, median & total dwell, repeat visits, unassisted share, rank); heatmap image; journey flow drawn as SVG (columns = step index, boxes = zones, links with widths ∝ count, top 5 paths listed as text); charts (`occupancy.svg`, `waiting.svg`); insights cards (title, evidence numbers, action, confidence badge or "insufficient data", "demo only" tag, links that seek the annotated video to each moment, contact-sheet thumbnail); annotated `<video controls>`; export links (JSON, CSV); buttons "Edit zones" / "Edit roles" that return to those steps and re-score.
- `tests/test_static.py`: `GET /`, `/live`, `/static/app.js`, `/static/zones.js`, `/static/dashboard.js`, `/static/style.css` all 200; index contains the banner text; no `http://`/`https://` URLs referencing external hosts in any static file (regex check), i.e. no CDN.
- [ ] Steps: failing static test → write files → pass → manual check via `./run.sh`-equivalent `uvicorn pulse.app.main:app` + `curl -s localhost:8000/ | head` → commit `M4: web UI (upload, zone editor, staff gallery, dashboard)`. **Milestone close (M4)**: verify.sh green, PROGRESS.md, commit.

---

## M5 — Live mode

### Task 18: LiveSession, /ws/live, live page, smoke_live, webcam_grab

**Files:** Create `src/pulse/live.py`, `src/pulse/app/static/{live.html,live.js}`, `scripts/smoke_live.py`, `scripts/webcam_grab.py`, `tests/test_live.py`, `tests/integration/test_live_ws.py`; modify `src/pulse/app/main.py`, `scripts/verify.sh`.

**Interfaces:**
- Produces:
  - `live.LiveSession(cfg, engine: PerceptionEngine, zones: list[Zone] | None = None, window_s: float = 300.0)` with:
    - `step(t: float, frame) -> dict` — runs `engine.process`, re-attaches new tracks to recently lost ones (`identity.reattach` with HSV embeddings), appends samples (via `timeline.build_timeline` on the new observations with person ids = re-attached ids), trims samples older than `window_s`, recomputes `journey.build` + `interest.open_alerts` over the window, returns the overlay message below.
    - `set_zones(zones)`, `toggle_role_at(x_norm, y_norm) -> int | None` (person whose current box contains the point; flips staff/customer), `roles: dict[int, str]` (default customer).
  - Overlay message (JSON): `{"type": "overlay", "t", "frame_size": [w, h], "persons": [{"person_id", "bbox", "role", "zone_id", "zone_name", "dwell", "alert"}], "zones": [{"zone_id", "name", "now", "total"}], "occupancy": int, "alerts": [{"person_id", "zone_id", "dwell", "message"}], "fps": float, "dropped": int}`.
  - WebSocket `/ws/live`: client → binary JPEG frames, or text JSON `{"type": "zones", "zones": [...]}` / `{"type": "toggle", "x", "y"}` / `{"type": "preset", "name"}` / `{"type": "load_zones", "path"}` (only files under `labels/zones/` or `jobs/`); server → overlay messages and `{"type": "error", "message"}` for undecodable frames (connection stays open). Frame dropping: a reader task keeps only the newest unprocessed frame; a worker processes it in a thread (`asyncio.to_thread`) and sends the overlay; `dropped` counts discarded frames; `fps` = processed frames per second over the last 5 s.
  - `live.html`/`live.js`: getUserMedia (`{video: {width: 960}}`), draws to a hidden canvas, sends JPEG (`toBlob('image/jpeg', 0.7)`) at ~5 fps whenever no frame is in flight; draws overlays on a visible canvas (boxes, roles colours, dwell timers, zone outlines + counters, occupancy, fps, alert banner "Customer at Car A for 45 s, no staff nearby"); zone drawing on the first frame (rectangle tool) or "Load zone file"; click on a person toggles role; auto-reconnect with backoff when the socket closes; works on `http://localhost` (no HTTPS).
  - `scripts/smoke_live.py [--clip PATH] [--seconds 30]`: in-process `TestClient` websocket; sends zones (one full-frame rectangle `Car: Demo`, kind car, and preset `short` so `alert_after = 8 s`), then streams JPEG frames of the MEVA dev clip (from the 60 s mark, 5 fps, 30 s) as fast as replies arrive; asserts ≥ 1 overlay with ≥ 1 person, ≥ 1 person with `dwell > 0`, ≥ 1 alert; prints `processed fps: X.XX` and `SMOKE LIVE OK`; writes `runs/bench/live_smoke.json` (`fps`, `frames`, `people_max`, `alerts`).
  - `scripts/webcam_grab.py`: `cv2.VideoCapture(0)` (AVFoundation on macOS), read 15 frames (warm-up), keep the last; if OpenCV fails, fall back to imageio-ffmpeg `-f avfoundation -framerate 30 -i 0 -frames:v 1` (macOS) / `-f dshow` (Windows); assert not blank (`gray.std() > 5 and 5 < gray.mean() < 250`); save to `runs/verify/webcam.jpg` (gitignored, local only); print size/mean/std and `WEBCAM OK`; exit 1 otherwise.
- [ ] **Step 1: Failing tests** `tests/test_live.py` (fast; a stub engine returns scripted TrackObs per call — injected dependency, real logic under test): a customer standing in a car zone for 9 s at short preset produces an alert in the overlay at t ≥ 8 and not before; `toggle_role_at` on that person makes them staff and the alert disappears on the next step; samples older than `window_s` are trimmed; websocket via TestClient with the stub engine: garbage bytes → `{"type": "error"}` and the next valid JPEG still gets an overlay.
- [ ] **Step 2:** fail → implement → pass. Slow test `tests/integration/test_live_ws.py` runs `smoke_live.main(["--seconds", "20"])` → 0.
- [ ] **Step 3: verify.sh** — add `step "smoke live" $PY scripts/smoke_live.py` and `step "webcam grab" $PY scripts/webcam_grab.py`. Commit `M5: live mode (WebSocket engine, overlays, dwell timers, alerts)`. **Milestone close (M5)**: verify.sh green, PROGRESS.md (live fps measured), commit.

---

## M6 — Data and evaluation

### Task 19: MEVA evaluation, zones labels, split, speed, `pulse eval` → docs/EVAL.md

**Files:** Create `src/pulse/eval/metrics.py`, `src/pulse/eval/run_eval.py`, `labels/zones/<clip>.json` (one per MEVA eval clip), `labels/split.json`, `tests/test_metrics.py`, `tests/integration/test_eval.py`; modify `src/pulse/cli.py` (`eval`, `bench live`), `src/pulse/bench.py`, `scripts/verify.sh`, `docs/EVAL.md` (generated), `data/sources.csv`, `PROGRESS.md`.

**Interfaces:**
- Produces:
  - `metrics.mot(gt: dict[int, dict[int, BBox]], pred: dict[int, dict[int, BBox]], frames: list[int]) -> dict` (`mota, idf1, num_switches, precision, recall` via `motmetrics`, IoU distance, `max_iou=0.5`).
  - `metrics.count_error(pred: int, gt: int) -> float` (`|pred - gt| / gt` as %, `None` when gt = 0).
  - `metrics.occupancy_mae(pred: list[int], gt: list[int]) -> float`.
  - `metrics.match_persons(gt_tl: Timeline, pred_tl: Timeline) -> dict[int, int]` (GT person → pred person by max mean IoU of foot-point-aligned boxes over shared times, Hungarian on cost = 1 − overlap score, min overlap 0.3).
  - `metrics.dwell_errors(gt_j, pred_j, match) -> list[tuple[float, float]]` (per matched person × zone: (gt_dwell, pred_dwell), zones with gt or pred dwell > 0); `metrics.journey_accuracies(gt_j, pred_j, match) -> list[float]`.
  - `run_eval.main(out: Path, clips: list[str] | None = None) -> int`: for each clip in `labels/split.json` (`{"dev": [{"clip", "video", "kpf"}], "test": [...]}`), load the cached perception (`runs/cache`; if missing, run `perceive` once and cache), GT tracks from KPF (persons only, boxes scaled to analysis size, sampled at the same 5-fps frame indices `round(k * src_fps / 5)`), zones from `labels/zones/<clip>.json`, compute: MOTA/IDF1 (raw tracks and merged persons), entries/exits vs `person_enters_scene_through_structure` / `person_exits_scene_through_structure` counts (exact names from `data/meva/CLIPS.md`) with median timing error (each GT activity matched to the nearest predicted entry within 5 s), occupancy MAE (GT occupancy from GT tracks with the same `first_t ≤ t ≤ last_t` rule), zone dwell median abs error (s and %), journey accuracy (mean), with dev and test reported **separately**; reads `runs/bench/detectors.json`, `runs/bench/speed.json` and `runs/bench/live_smoke.json` for speed (file-mode RTF at 5 fps / 720p input, live fps with 1–4 people) and the insights scenarios from `tests/test_insights.py` names; writes `docs/EVAL.md` deterministically (sorted keys, fixed float formatting `.2f`/`.1f`, no timestamps), each number followed by the reproducing command in backticks, and a targets table `metric | target | measured (test) | met?`. Returns 0 even when targets are missed (a miss is a finding); returns 1 only when inputs are missing.
  - `pulse bench speed --clip <meva_dev> --out runs/bench/speed.json`: file-mode RTF on a 60 s 720p excerpt at 5 fps (CPU EP; CoreML separately if available, labelled), and engine fps on frames with 1–4 people (pick frames by GT person count).
- [ ] **Step 1: Zones labels (agent-drawn).** For each MEVA clip: extract a keyframe (`make_clip`/OpenCV), view it, draw `entrance` zones over every door/doorway, plus 2–4 floor-area zones (`kind: "other"`, names like "Hallway left", "Desk area" — use kind `desk` if a desk/counter is visible); save with `"labeller": "ai-agent, unreviewed"`. Overlay the zones on the keyframe (blurred) and view it to check placement before saving. Enter/exit activities' first/last GT boxes must fall inside an entrance zone for ≥ 70% of activities on **dev** clips (print this check); fix zones on dev clips only.
- [ ] **Step 2: Split** — `labels/split.json` (decided by the orchestrator at plan time, before any tuning; already recorded in PROGRESS.md): **dev** = `2018-03-11.16-35-01.16-40-00.school.G423`, `2018-03-11.12-00-00.12-05-00.school.G420`; **test** = `2018-03-12.11-05-01.11-10-01.school.G423`, `2018-03-13.17-10-00.17-15-00.school.G421`, `2018-03-09.10-40-00.10-45-00.school.G419`, `2018-03-09.10-15-00.10-20-00.school.G299`. The verify.sh "MEVA dev clip" is the first dev clip. (Task 2 creates `labels/split.json` with this content so the benchmark and fixtures use the dev clip from the start.)
- [ ] **Step 3: Failing tests** `tests/test_metrics.py`: perfect prediction → MOTA 1, IDF1 1; one ID switch counted; `count_error(9, 10) == 10.0`; `occupancy_mae([1,2],[1,3]) == 0.5`; journey accuracy average; `match_persons` pairs overlapping tracks and leaves unmatched ones out.
- [ ] **Step 4:** implement → pass. **Tune on dev only** (allowed knobs: `det_score_thr`, `track_activation_thr`, `track_buffer_s`, `merge_sim_thr`, `merge_max_speed`), log every trial and its dev numbers in PROGRESS.md, then freeze config and run test clips once.
- [ ] **Step 5:** `pulse bench speed`; `pulse eval --out docs/EVAL.md`; run it twice and `diff` the outputs (must be identical). Slow test `tests/integration/test_eval.py`: `run_eval.main(tmp/EVAL.md)` returns 0 and the file contains the headings `Entries and exits`, `Occupancy`, `Zone dwell`, `Journeys`, `Tracking`, `Speed`.
- [ ] **Step 6: verify.sh** — add `step "eval" .venv/bin/pulse eval --out docs/EVAL.md` before the git steps (it must leave `git status` clean, i.e. byte-identical output). Commit `M6: MEVA evaluation, agent-drawn zones, EVAL.md`. **Milestone close (M6).**

### Task 20: Showroom / synthetic clip labels and stop precision/recall

**Files:** Create `labels/<stock_clip>.json`, `src/pulse/eval/stops.py`, `tests/test_stops_eval.py`; modify `src/pulse/eval/run_eval.py`, `data/sources.csv`.

Only if a showroom (stock) or synthetic clip exists. Labels by viewing frames sampled at 1 s: `{"clip", "labeller": "ai-agent, unreviewed", "fps_sampled": 1, "zones_file", "visits": [{"id", "t_in", "t_out", "role"}], "zone_visits": [{"visit_id", "zone_id", "t_in", "t_out"}], "stops": [{"visit_id", "zone_id", "t_start", "t_end"}]}`. `stops.precision_recall(pred_visits: list[ZoneVisit], gt_stops, tol_s=2.0) -> dict` (a predicted stopped visit matches a GT stop at the same zone overlapping in time ± tol). If synthetic clips with `gt.json` exist (teammate C), also compute dwell/journey/walk-out errors from their GT. EVAL.md gets a "Stops" section; if no labelled clip exists the section says so explicitly. Commit `M6: stock-clip labels and stop P/R`.

---

## M7 — Docs and run script

### Task 21: README, PITCH_NOTES, INTEGRATION, run.sh

**Files:** Create/replace `README.md`, `docs/PITCH_NOTES.md`, `docs/INTEGRATION.md`, `run.sh`, `tests/test_docs.py`; modify `scripts/verify.sh` (none — run.sh is checked separately by the goal), `PROGRESS.md`.

- `run.sh`: `#!/usr/bin/env bash`, `cd "$(dirname "$0")"`, creates the venv with `/opt/homebrew/bin/python3.12` if `.venv` is missing (falls back to `python3.12` on PATH) and installs `-e .`, then `exec .venv/bin/uvicorn pulse.app.main:app --host 127.0.0.1 --port "${PORT:-8000}"`. Executable bit set.
- README: problem; what it measures; how it works (pipeline diagram in text); quickstart (`./run.sh`, `pulse analyze`); results (link to EVAL.md + headline numbers copied **from** EVAL.md); **licence table** `component | licence | trained on (data) | where used` covering every Python dependency in pyproject, the vendored rtmlib subset, every model in MANIFEST.json, the imageio-ffmpeg bundled ffmpeg binary (record its licence by running `imageio_ffmpeg.get_ffmpeg_version()` and reading the binary's `-L` output), MEVA (CC BY 4.0, Corona et al., WACV 2021 citation), stock clips (their licences); **privacy** (no face recognition, no age/gender/emotion/ethnicity inference, no cross-visit re-ID, faces blurred in every output, on-premise, 24 h deletion, customers must be informed by signage that analytics is in use, decision support only); limitations.
- `docs/PITCH_NOTES.md`: customer-side story; competitor table (V-Count, FootfallCam, Deepingsource, Palexy vs Showroom Pulse on the five differences in spec §1); ROI model with clearly marked assumptions (`missed-interest episodes/month × conversion uplift × margin per car`, each assumption labelled ASSUMPTION with a placeholder value range and the source to replace it); questions for Royal Motors.
- `docs/INTEGRATION.md`: merging with Showroom Coach (one perception engine and cache, shared `roles.json` format, one web app with two tabs Salespeople / Customers); table of **every interface that may differ** from Coach's spec (e.g. `datatypes.py` module name, `Det.cls` as COCO name string, `Sample`/`Zone` pulse-only types, normalised zone coordinates, vendored rtmlib vs pip rtmlib, `render.render` wrapper signature, `roles.suggest` reduced features + `cfg` arg, `identity.merge` `max_gap_s` = `leave_after`, `geometry.position_m` returning `PosEstimate`, cache file layout, `/ws/live` message schema).
- `tests/test_docs.py`: README contains "Licence" table header `| Component | Licence | Trained on | Where used |` and a row for every `models/MANIFEST.json` entry name and for MEVA; README has a "Privacy" heading; PITCH_NOTES mentions all four competitors and "ASSUMPTION"; INTEGRATION has the differing-interfaces table; `run.sh` is executable.
- [ ] Steps: failing doc tests → write docs → pass → `./run.sh &` then `curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/` prints 200 → stop server → commit `M7: README with licence table and privacy, pitch notes, integration guide, run.sh`. **Milestone close (M7)**: verify.sh green; PROGRESS.md Morning summary draft.

---

## M8 — Stretch (in this order; keep verify.sh green after each)

### Task 22: Sat-in-car detection
`src/pulse/sat_in_car.py`: `detect(timeline, obs_by_person, car_dets, cfg) -> list[Event(kind="sat_in_car")]` — (a) ≥ 60% of the person box inside a car box, only torso keypoints confident (ankles/knees low), and foot speed < 0.1 m/s for ≥ 5 s; or (b) the track ends inside a car zone and a track starts there again within 60 s (disappear/reappear). Tests with synthetic boxes; per-car `sat_in_car` column in stats/per_car.csv/dashboard; EVAL: recall against MEVA vehicle entry/exit activities on the outdoor clip (download it only if disk allows). Commit.

### Task 23: Floor calibration → top-down heatmap
Zone editor gets a "Calibrate floor" mode (click 4 points, enter metres); stored as `floor` in the zone file (`{"img": [[u,v]...], "world": [[x,y]...]}` normalised image coords); `build_people(..., floor_h=...)`; report adds `heatmap_topdown.png` in metres. Tests: homography from zone file applied; top-down heatmap written. Commit.

### Task 24: Printable one-page HTML report
`report.html` (inline CSS, print media query, KPIs, per-car table, top insights, heatmap) written by `build_report`; linked from the dashboard; check_outputs validates it. Commit.

### Task 25: Sales CSV join
Upload/CLI `--sales sales.csv` (`car_model,period,sold`); per-car `attention_to_sales` (stops per sale); ship `data/sample_sales_SYNTHETIC.csv` clearly labelled synthetic (committed as an exception? No — place it at `docs/sample_sales_SYNTHETIC.csv`). Tests. Commit.

### Task 26: Uzbek/Russian UI toggle
`static/i18n.js` with `en`, `uz`, `ru` dictionaries for all UI strings; toggle persisted in `localStorage`; test that every key exists in all three. Commit.

### Task 27: Dockerfile
`python:3.12-slim`, installs `.`, exposes 8000, models mounted as a volume (not baked). Not built if Docker is unavailable (record). Commit.

### Task 28: Deploy script (written, NOT run)
`scripts/deploy.sh` for an on-prem Linux box (systemd unit, reverse proxy notes), header comment "NOT RUN — local-only hackathon rule". Commit.

---

## Milestone close procedure (after every milestone)

1. `bash scripts/verify.sh 2>&1 | tee runs/verify/last.log` → must end with `VERIFY OK` (after committing, since the git step requires a clean tree; commit first, then run, then if anything is generated, commit again and re-run).
2. Append to `PROGRESS.md`: milestone, what was done, measured numbers, decisions taken alone, open problems.
3. `git add -A && git commit -m "Mx: progress log"`.
