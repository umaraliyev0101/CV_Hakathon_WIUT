# Showroom Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one showroom camera (uploaded video or live webcam) into per-salesperson coaching data — time to greet, waits while staff were free, walk-outs, engagement time — with a one-click staff/customer confirmation.

**Architecture:** One online engine (detect → track → pose → geometry → event state machine) serves both modes. File mode feeds it from a sampled video, caches perception per video, then runs an offline pass (identity merge, staff suggestion, event re-run with final roles) so roles and thresholds can change without re-detection. A FastAPI backend with a no-build static frontend exposes upload, confirm, results, export and a `/ws/live` WebSocket.

**Tech Stack:** Python 3.12, numpy, scipy, opencv-python, onnxruntime (CPU), supervision (ByteTrack, MIT), YOLOX ONNX (Apache-2.0), RTMPose ONNX (Apache-2.0, pre/post-processing vendored from rtmlib, Apache-2.0), DINOv2-small ONNX (Apache-2.0), FastAPI + uvicorn, imageio-ffmpeg, motmetrics, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-showroom-coach-design.md` (approved). Rules: `CLAUDE.md`. Running log: `PROGRESS.md`.

## Global Constraints

- Python 3.12 venv at `.venv` (`C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe`), never 3.14. Run tools as `.venv/Scripts/python -m ...`.
- Local only: never add a git remote, push, deploy, or create accounts.
- Commits: conventional style `feat(yak-44): ...`, **no `Co-Authored-By` trailer**.
- Licences permissive only. Banned: ultralytics, boxmot, re-ID weights trained on Market-1501/DukeMTMC/MSMT17. Every model/dataset gets a row in the README licence table and `models/MANIFEST.json` (and datasets in `data/sources.csv`).
- `data/`, `models/`, `runs/`, `.venv/`, `uploads/` are gitignored; never commit videos or weights.
- Disk: venv + models + data ≤ 4 GB combined; stop downloading if free disk < 3 GB.
- Privacy: no face recognition; every rendered frame and thumbnail blurs faces (nose/eye/ear keypoints, fallback top 20% of the box).
- Prefer onnxruntime over torch; torch is not installed unless a measured need is logged.
- Downloads are untrusted: each source in its own folder; Python that reads them runs with `-I`.
- Processing size: frames resized to max width 960 px; default sampling 5 fps; uploads longer than 300 s are trimmed with a warning.
- Presets (spec §4.6) — Dealership: greet 1.5 m / 5 s, engage 2.0 m / 120 s, gap_tol 3 s, wait_alert 60 s, leave_after 10 s. Short clip: 1.5 m / 2 s, 2.0 m / 8 s, 1.5 s, 10 s, 3 s.
- Honest numbers only. Tune only on dev clips, never on test clips.
- Full test suite < 3 minutes; slow tests carry `@pytest.mark.slow` and skip cleanly when their data file is missing.
- After every milestone: full test suite, commit, append to `PROGRESS.md` (done / numbers / decisions / open problems).

## Review Focus

1. **Video with nobody in it, or one person only** → results page with "nobody detected — camera too far or too dark?" or store-only metrics and a "mark staff" prompt; no crash, no division by zero in metrics (Task 1.4 metrics test `test_empty_events`, Task 3.2 analysis test `test_analyze_no_people`).
2. **Unreadable / non-video upload (e.g. a .txt renamed .mp4)** → job ends in `failed` with a clear message, server keeps running (Task 4.1 test `test_upload_garbage_fails_cleanly`).
3. **Roles changed after results exist** → re-score returns new scorecards in seconds and never re-runs detection; the perception cache is reused (Task 4.1 test `test_rescore_does_not_redetect`).
4. **Live client sends frames faster than the CPU can process / disconnects mid-stream** → server processes only the newest frame (stop-and-wait protocol), reports processed fps, survives disconnects (Task 5.1 test `test_ws_disconnect_and_reconnect`).
5. **A person cut off at the bottom of the frame, sitting, or with no keypoints** → distance falls back torso → box, flags "approximate", results show a warning badge when >50% of samples are box-fallback (Task 1.1 tests `test_cutoff_uses_torso`, `test_no_kpts_box_fallback`; Task 3.2 `test_approx_warning`).

---

## File Structure

```
pyproject.toml                       package metadata, deps, console script, pytest config
run.bat / run.sh                     start the app on http://localhost:8000
README.md                            problem, how it works, quickstart, results, licence table, privacy, declaration
PROGRESS.md                          milestone log + decisions + Morning summary
data/sources.csv                     every dataset/clip used (committed)
data/TODO_DOWNLOADS.md               what the owner should download by hand (committed)
models/MANIFEST.json                 each weight: file, url, sha256, licence, trained_on (committed)
labels/<clip>.json                   agent-made labels ("labeller": "ai-agent, unreviewed")
docs/EVAL.md                         measured numbers + reproduce commands
docs/PITCH_NOTES.md                  deck outline, SKAIVISION comparison, ROI model
src/showroom_coach/
  __init__.py                        version
  __main__.py                        python -m showroom_coach → cli.main
  types.py                           Det, TrackObs, Person, RoleSuggestion, Event, StaffCard, StoreCard, Scorecard, to_jsonable
  config.py                          Thresholds + PRESETS, CameraConfig, RoleWeights, paths
  geometry.py                        Camera, height selection, position_m, distance_m, facing_vector, faces
  events.py                          PersonState, EventEngine (pure state machine), run_events
  metrics.py                         scorecard(events, persons, ...) → Scorecard; optional sales CSV join
  roles.py                           interaction_features, outfit_peers, suggest_roles (+ reasons)
  identity.py                        TrackSummary, merge_tracks (constrained clustering), LiveIdentity (re-attach)
  models.py                          model registry, download with retry + sha256, MANIFEST writer
  sources.py                         VideoInfo, FileSource (sampling, resize, trim), decode_jpeg
  perception/
    __init__.py
    rtm_onnx.py                      vendored YOLOX + RTMPose ONNX pre/post-processing (Apache-2.0, from rtmlib 0.0.16)
    detect.py                        Detector (YOLOX / RF-DETR) → list[Det]
    pose.py                          PoseEstimator → (N,17,3)
    embed.py                         Embedder: DINOv2-small ONNX + HSV histograms
    tracking.py                      Tracker (supervision ByteTrack) → list[TrackObs]
    stack.py                         PerceptionStack bundle (detector+pose+tracker factory+embedder), Fake stack for tests
    pipeline.py                      run_perception(video) → PerceptionResult; cache save/load
  analysis.py                        offline pass: perception → persons → features → roles → events → metrics; rescore()
  live.py                            LiveSession: frames → overlay dict + events, toggles, auto-suggest after 30 s
  render.py                          blur_faces, annotated H.264 mp4, timeline png, contact sheets, thumbnails
  export.py                          results JSON / events CSV / scorecard CSV
  cli.py                             analyze, fetch-models, bench, eval, label-sheet
  evaluation/
    __init__.py
    kpf.py                           MEVA KPF (geom/activities/types) parsing
    meva_eval.py                     tracking (MOTA/IDF1), talk/engagement P/R, enter/exit recall + timing
    labels_eval.py                   staff accuracy, clicks, greet-time error vs labels/<clip>.json
  app/
    __init__.py
    main.py                          create_app(stack_factory=...) FastAPI app, REST + /ws/live
    jobs.py                          JobManager: background threads, status, progress, 24 h cleanup
    static/index.html, app.js, live.html, live.js, style.css, i18n.js (M8)
tests/
  conftest.py                        synthetic video fixture, fake stack, markers
  synth.py                           synthetic pinhole people + scripted tracks helpers
  test_types.py test_config.py test_geometry.py test_events.py test_metrics.py test_roles.py test_identity.py
  test_sources.py test_rtm_onnx.py test_perception_pipeline.py test_analysis.py test_render.py test_export.py
  test_cli.py test_app.py test_live.py test_kpf.py test_meva_eval.py test_models.py
```

---

# M0 — Skeleton, venv, models, data, detector benchmark

### Task 0.1: Package skeleton, pyproject, test harness

**Files:**
- Create: `pyproject.toml`, `src/showroom_coach/__init__.py`, `src/showroom_coach/__main__.py`, `src/showroom_coach/cli.py` (stub with `main()` and `--version`), `tests/conftest.py`, `tests/test_cli.py`, `PROGRESS.md`
- Modify: `.gitignore` (add `jobs/`, `*.egg-info/`, `.coverage`)

**Interfaces:**
- Produces: importable package `showroom_coach`; console script `showroom-coach` → `showroom_coach.cli:main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "showroom-coach"
version = "0.1.0"
description = "Camera-based coaching for car-showroom salespeople (WIUT Hackathon 2026, YAK-44)"
requires-python = ">=3.11,<3.14"
license = {text = "Apache-2.0"}
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "opencv-python>=4.9",
  "onnxruntime>=1.18",
  "supervision>=0.25",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "python-multipart>=0.0.9",
  "imageio-ffmpeg>=0.5",
  "pyyaml>=6",
  "tqdm>=4.66",
]

[project.optional-dependencies]
eval = ["motmetrics>=1.4"]
dev = ["pytest>=8", "httpx>=0.27", "motmetrics>=1.4"]

[project.scripts]
showroom-coach = "showroom_coach.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
showroom_coach = ["app/static/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs real models/video; skipped when files are missing"]
addopts = "-ra"
```

- [ ] **Step 2: Write the failing CLI test** (`tests/test_cli.py`)

```python
from showroom_coach import __version__
from showroom_coach.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
```

- [ ] **Step 3: Run it to see it fail** — `.venv/Scripts/python -m pytest tests/test_cli.py -q` → ImportError.
- [ ] **Step 4: Implement** `__init__.py` (`__version__ = "0.1.0"`), `__main__.py` (`from .cli import main; raise SystemExit(main())`), `cli.py` with argparse, a `--version` flag, and subparsers added in later tasks. `tests/conftest.py` starts with `import pytest` only (fixtures added later).
- [ ] **Step 5: Install and run** — `.venv/Scripts/python -m pip install -e ".[dev]"` then `.venv/Scripts/python -m pytest -q` → 1 passed.
- [ ] **Step 6: Create `PROGRESS.md`** with sections `## Log`, `## Decisions`, `## Open problems`, and log the environment (Python 3.12.6, free disk, venv size).
- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat(yak-44): package skeleton, pyproject and test harness"`

### Task 0.2: Model registry, downloads, manifest

**Files:**
- Create: `src/showroom_coach/models.py`, `tests/test_models.py`, `models/MANIFEST.json` (generated)
- Modify: `src/showroom_coach/cli.py` (add `fetch-models` subcommand)

**Interfaces:**
- Produces:
  - `MODELS: dict[str, ModelSpec]` where `ModelSpec(name, url, filename, licence, trained_on, source_page, archive_member: str | None, sha256: str | None)`.
  - `models_dir() -> Path` (env `SHOWROOM_MODELS` or `<repo>/models`).
  - `ensure_model(name: str, retries: int = 3) -> Path` — downloads into `models/<name>/` if missing (zip archives extracted, only the `.onnx` member kept), verifies/records sha256, returns the `.onnx` path. Raises `ModelUnavailable` after retries.
  - `write_manifest() -> Path` — writes `models/MANIFEST.json` with one entry per present file: `{name, file, url, sha256, bytes, licence, trained_on, source_page}`.
  - CLI: `showroom-coach fetch-models [names...]`.
- Registry entries (all verified reachable 2026-10-11):
  - `yolox_nano` — `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx`, Apache-2.0, COCO 2017, input 416.
  - `yolox_tiny` — `.../0.1.1rc0/yolox_tiny.onnx`, Apache-2.0, COCO 2017, input 416.
  - `yolox_s` — `.../0.1.1rc0/yolox_s.onnx`, Apache-2.0, COCO 2017, input 640.
  - `rtmpose_s` — `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip` (zip; keep `end2end.onnx`), Apache-2.0, Body7 (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18), input 192×256.
  - `dinov2_small` — `https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx`, Apache-2.0, LVD-142M (self-supervised, Meta).
  - `rfdetr_nano` (benchmark candidate only) — `https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_n_v142_384x384.onnx`, Apache-2.0 (RF-DETR by Roboflow; ONNX export by a third party), COCO 2017, input 384.

- [ ] **Step 1: Failing tests** (`tests/test_models.py`) — use a local HTTP server fixture so tests stay offline:

```python
import hashlib, http.server, json, threading, zipfile
from pathlib import Path
import pytest
from showroom_coach import models as M


@pytest.fixture
def http_dir(tmp_path):
    root = tmp_path / "srv"; root.mkdir()
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ensure_model_downloads_and_hashes(http_dir, tmp_path, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    p = M.ensure_model("fake")
    assert p.read_bytes() == b"fake-onnx"
    man = json.loads(M.write_manifest().read_text())
    entry = [e for e in man["models"] if e["name"] == "fake"][0]
    assert entry["sha256"] == hashlib.sha256(b"fake-onnx").hexdigest()


def test_ensure_model_extracts_zip_member(http_dir, tmp_path, monkeypatch):
    root, base = http_dir
    with zipfile.ZipFile(root / "a.zip", "w") as z:
        z.writestr("dir/end2end.onnx", b"inner")
        z.writestr("dir/readme.txt", b"x")
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "z", M.ModelSpec("z", f"{base}/a.zip", "z.onnx", "Apache-2.0", "none", "x", archive_member="end2end.onnx"))
    assert M.ensure_model("z").read_bytes() == b"inner"


def test_ensure_model_raises_after_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "bad", M.ModelSpec("bad", "http://127.0.0.1:9/nope.onnx", "n.onnx", "MIT", "none", "x"))
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("bad", retries=2, backoff=0)
```

- [ ] **Step 2: Run, see fail.** `.venv/Scripts/python -m pytest tests/test_models.py -q`
- [ ] **Step 3: Implement `models.py`** with `urllib.request` (no new deps), streaming to a `.part` file, `zipfile` member extraction by basename, sha256 via `hashlib`, a free-disk guard (`shutil.disk_usage`; raise `ModelUnavailable("free disk < 3 GB")` below 3 GB), and `write_manifest()` scanning `MODELS` for present files. `ensure_model(name, retries=3, backoff=2.0)`.
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Fetch real weights** — `.venv/Scripts/showroom-coach fetch-models yolox_nano yolox_tiny yolox_s rtmpose_s dinov2_small rfdetr_nano` then check `du -sh models` (expect ≈ 300 MB) and `cat models/MANIFEST.json`.
- [ ] **Step 6: Commit** — `git add src tests models/MANIFEST.json && git commit -m "feat(yak-44): model registry with retrying download and sha256 manifest"`

### Task 0.3: Vendored ONNX wrappers (YOLOX, RTMPose, RF-DETR) and detector benchmark

**Files:**
- Create: `src/showroom_coach/perception/__init__.py`, `src/showroom_coach/perception/rtm_onnx.py`, `src/showroom_coach/perception/detect.py`, `src/showroom_coach/perception/pose.py`, `tests/test_rtm_onnx.py`, `runs/bench/` outputs (not committed)
- Modify: `src/showroom_coach/cli.py` (add `bench-detectors`)
- Reference (read-only): `runs/ref/src/rtmlib/tools/object_detection/yolox.py`, `.../rfdetr.py`, `.../post_processings.py`, `.../pose_estimation/rtmpose.py`, `.../pre_processings.py`, `.../post_processings.py`

**Interfaces:**
- Produces:
  - `rtm_onnx.make_session(path: Path, threads: int | None = None) -> ort.InferenceSession` (CPUExecutionProvider, `intra_op_num_threads=threads or 4`).
  - `rtm_onnx.yolox_preprocess(img_bgr, input_hw) -> (blob float32 [1,3,H,W], ratio)`; `rtm_onnx.yolox_postprocess(out [1,N,85], input_hw, ratio, score_thr, nms_thr) -> (boxes (K,4) xyxy, scores (K,), cls (K,))`; `rtm_onnx.nms(boxes, scores, thr) -> keep idx`.
  - `rtm_onnx.rtmpose_preprocess(img_bgr, bbox_xyxy, input_wh=(192,256)) -> (blob, center, scale)`; `rtm_onnx.rtmpose_decode(simcc_x, simcc_y, center, scale, input_wh, split_ratio=2.0) -> (kpts (17,2), scores (17,))`.
  - `rtm_onnx.rfdetr_preprocess(img_bgr, input_hw) -> blob`; `rtm_onnx.rfdetr_postprocess(boxes, logits, orig_hw, score_thr) -> (boxes, scores, cls)`.
  - `detect.Detector(name: str = "yolox_tiny", score_thr: float = 0.35, nms_thr: float = 0.45)` with `__call__(frame_bgr) -> list[Det]` (persons cls 0 plus all other COCO classes; callers filter) and `persons(frame_bgr) -> list[Det]`.
  - `pose.PoseEstimator(name: str = "rtmpose_s")` with `__call__(frame_bgr, boxes: np.ndarray (N,4)) -> np.ndarray (N,17,3)` where `[...,2]` is confidence in [0,1].
  - CLI `showroom-coach bench-detectors --video <meva.avi> --geom <geom.yml> --types <types.yml> --frames 60` → prints a table: model, ms/frame (median), person recall @IoU0.5 vs MEVA geom boxes, precision; writes `runs/bench/detectors.json`.
- File header of `rtm_onnx.py` must say: `# Adapted from rtmlib 0.0.16 (Apache-2.0, https://github.com/Tau-J/rtmlib); YOLOX postprocess from Megvii YOLOX (Apache-2.0).`

- [ ] **Step 1: Failing tests** (`tests/test_rtm_onnx.py`) — pure-numpy tests (no weights):

```python
import numpy as np
from showroom_coach.perception import rtm_onnx as R


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = R.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45)
    assert list(keep) == [0, 2]


def test_yolox_preprocess_letterbox():
    img = np.full((480, 640, 3), 200, np.uint8)
    blob, ratio = R.yolox_preprocess(img, (416, 416))
    assert blob.shape == (1, 3, 416, 416) and blob.dtype == np.float32
    assert abs(ratio - 0.65) < 1e-6
    assert blob[0, 0, 415, 0] == 114  # padding row below the resized image


def test_yolox_postprocess_decodes_one_box():
    hw = (64, 64)  # strides 8,16,32 -> 64+16+4 = 84 anchors
    out = np.zeros((1, 84, 85), np.float32)
    # anchor 0 at grid (0,0) stride 8: centre offset 0.5 -> 4 px, w=h=exp(1)*8
    out[0, 0, :4] = [0.5, 0.5, 1.0, 1.0]
    out[0, 0, 4] = 0.9
    out[0, 0, 5] = 0.95  # class 0
    boxes, scores, cls = R.yolox_postprocess(out, hw, ratio=1.0, score_thr=0.3, nms_thr=0.45)
    assert len(boxes) == 1 and cls[0] == 0
    w = np.exp(1.0) * 8
    np.testing.assert_allclose(boxes[0], [4 - w / 2, 4 - w / 2, 4 + w / 2, 4 + w / 2], rtol=1e-4)


def test_rtmpose_roundtrip_decode():
    # A peak at the centre of the SimCC vectors decodes to the bbox centre.
    bbox = np.array([100, 50, 200, 350], float)
    img = np.zeros((480, 640, 3), np.uint8)
    blob, center, scale = R.rtmpose_preprocess(img, bbox, (192, 256))
    assert blob.shape == (1, 3, 256, 192)
    sx = np.zeros((1, 17, 192 * 2), np.float32); sy = np.zeros((1, 17, 256 * 2), np.float32)
    sx[..., 192] = 1.0; sy[..., 256] = 1.0
    kpts, scores = R.rtmpose_decode(sx, sy, center, scale, (192, 256))
    np.testing.assert_allclose(kpts[0], center, atol=1.0)
```

  Note for the implementer: Megvii YOLOX ONNX outputs (0.1.1rc0) already have sigmoid applied to objectness and class scores and raw `(dx, dy, log w, log h)` box terms; decode exactly as rtmlib `YOLOX.postprocess` does (grid + stride, `exp` for w/h, `scores = obj * cls`). Preprocess: letterbox top-left with value 114, BGR, float32 0–255, no mean/std (YOLOX ≥ 0.1.1 convention). If the real model's output disagrees (test it in Step 5 on a real frame), fix the implementation and the test together and log it.

- [ ] **Step 2: Run, see fail.**
- [ ] **Step 3: Implement `rtm_onnx.py`** by porting the rtmlib functions (YOLOX letterbox/decode/multiclass NMS; RTMPose `bbox_xyxy2cs` with padding 1.25, `top_down_affine` warp, mean `[123.675,116.28,103.53]` / std `[58.395,57.12,57.375]` on RGB, SimCC argmax decode with `split_ratio=2.0`, score = max of x/y peaks clipped to [0,1]; RF-DETR resize + ImageNet normalisation + sigmoid top-k decode with background index 0 removed). Then `detect.py` and `pose.py` that load models with `models.ensure_model`.
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Real-frame smoke check** (needs a MEVA clip in `data/meva/videos/`): a `@pytest.mark.slow` test in `tests/test_rtm_onnx.py` that loads frame 300 of the first MEVA clip, runs `Detector("yolox_tiny").persons` and `PoseEstimator()`; asserts ≥ 1 person and keypoint confidences in [0,1]; skips if no clip/model.
- [ ] **Step 6: Benchmark** — implement `bench-detectors` (frames sampled evenly from the clip at original resolution resized to max width 960, matched with geom boxes of `Person` actors at the same frame index scaled by the same factor; greedy IoU ≥ 0.5 matching; ignore GT boxes smaller than 20 px high). Run on the first MEVA clip, 60 frames, for yolox_nano, yolox_tiny, yolox_s, rfdetr_nano. **Choose the fastest model whose recall is within 5 points of the best.** Record the table + choice in `PROGRESS.md` and set `config.DEFAULT_DETECTOR`.
- [ ] **Step 7: Commit** — `git commit -m "feat(yak-44): vendored YOLOX/RTMPose/RF-DETR ONNX wrappers and detector benchmark"`

### Task 0.4: Data bookkeeping

**Files:**
- Create: `data/sources.csv`, `data/TODO_DOWNLOADS.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1:** `data/sources.csv` header `dataset,item,url,licence,attribution,use,committed` with one row per MEVA clip (from `data/meva/clips.csv` produced by the scout) and one row per `data/stock/pexels-<id>.mp4` (URL `https://www.pexels.com/video/<id>/`, licence "Pexels License (free commercial use)"). If `data/stock/` is empty write that in PROGRESS.md.
- [ ] **Step 2:** `data/TODO_DOWNLOADS.md` lists for the owner: Pexels 7154222 (car dealer and customer, Antoni Shkraba) and how to name files `data/stock/pexels-<id>.mp4`.
- [ ] **Step 3:** M0 PROGRESS entry (environment, disk, model sizes, benchmark table, detector decision, data state). Run full suite. Commit `chore(yak-44): data sources, download todo and M0 progress log`.

---

# M1 — Pure core (TDD, synthetic data only)

All M1 units are pure Python + numpy, no models, no I/O. Write each test file first, run it red, implement, run green, commit.

### Task 1.0: Core types and config

**Files:**
- Create: `src/showroom_coach/types.py`, `src/showroom_coach/config.py`, `tests/test_types.py`, `tests/test_config.py`

**Interfaces (Produces):**

```python
# types.py
BBox = tuple[float, float, float, float]          # x1, y1, x2, y2 in processed-frame pixels

@dataclass
class Det:
    bbox: BBox
    score: float
    cls: int = 0                                  # COCO class id; 0 = person, 2 = car

@dataclass
class TrackObs:
    track_id: int
    frame_idx: int                                # sampled-frame index
    t: float                                      # seconds from clip start
    bbox: BBox
    score: float
    kpts: list[list[float]] | None = None         # 17 x [x, y, conf] (COCO order)

@dataclass
class Person:
    pid: int
    track_ids: list[int]
    t_first: float
    t_last: float
    n_obs: int
    thumb: str | None = None                      # relative file name of the blurred thumbnail

@dataclass
class RoleSuggestion:
    pid: int
    role: str                                     # "staff" | "customer" | "unsure"
    confidence: float                             # P(staff) in [0, 1]
    reasons: list[str]
    features: dict[str, float]

@dataclass
class Event:
    type: str      # enter | leave | greet | engagement_start | engagement | wait_alert | walkout | party | handoff
    t: float
    customer: int | None = None
    staff: int | None = None
    t_end: float | None = None
    data: dict = field(default_factory=dict)

@dataclass
class StaffCard:
    staff: int
    name: str | None
    greets: int
    median_time_to_greet: float | None
    customers_engaged: int
    engaged_total_s: float
    engaged_median_s: float | None
    missed_while_free: int
    walkouts_while_free: int
    sold: int | None = None
    conversion: float | None = None

@dataclass
class StoreCard:
    customers: int
    customers_entry_known: int
    greeted: int
    greet_rate: float | None
    median_time_to_greet: float | None
    walkouts: int
    unattended_wait_s: float
    wait_alerts: int
    engagements: int

@dataclass
class Scorecard:
    staff: list[StaffCard]
    store: StoreCard

def to_jsonable(obj) -> Any:   # dataclasses/np scalars/np arrays/tuples/dict keys → plain JSON types
```

```python
# config.py
@dataclass(frozen=True)
class Thresholds:
    greet_dist: float = 1.5
    greet_time: float = 5.0
    engage_dist: float = 2.0
    engage_time: float = 120.0
    gap_tol: float = 3.0
    wait_alert: float = 60.0
    leave_after: float = 10.0
    party_arrive: float = 5.0
    party_dist: float = 1.5
    party_window: float = 30.0
    facing_max_deg: float = 60.0
    edge_leave: float = 1.0          # grace before "left" when the last box touched a frame edge

PRESETS: dict[str, Thresholds] = {
    "dealership": Thresholds(),
    "short": Thresholds(greet_time=2.0, engage_time=8.0, gap_tol=1.5, wait_alert=10.0,
                        leave_after=3.0, party_window=10.0),
}
PRESET_LABELS = {"dealership": "Dealership (SKAIVISION thresholds)", "short": "Short clip (10–60 s)"}
def get_preset(name: str) -> Thresholds      # unknown → ValueError("unknown preset ...")

@dataclass(frozen=True)
class CameraConfig:
    hfov_deg: float = 70.0
    person_height_m: float = 1.70
    floor_points: tuple[tuple[float, float, float, float], ...] | None = None   # (u, v, X, Z) x4

@dataclass(frozen=True)
class RoleWeights:
    bias: float = -1.5
    outfit_peers: float = 1.2
    contacts: float = 1.6
    presence: float = 1.0
    initiator: float = 1.0
    staff_thr: float = 0.6
    customer_thr: float = 0.4
    outfit_sim_thr: float = 0.85
    contact_dist: float = 2.0
    contact_min_s: float = 5.0

DEFAULT_DETECTOR = "yolox_tiny"   # set from the M0 benchmark
SAMPLE_FPS = 5.0
MAX_WIDTH = 960
MAX_SECONDS = 300.0
def repo_root() -> Path
def jobs_dir() -> Path    # env SHOWROOM_JOBS or <repo>/jobs
def cache_dir() -> Path   # env SHOWROOM_CACHE or <repo>/runs/cache
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_types.py
import json
import numpy as np
from showroom_coach.types import Event, Det, to_jsonable


def test_event_roundtrips_to_json():
    e = Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": np.float32(4.0)})
    d = json.loads(json.dumps(to_jsonable(e)))
    assert d == {"type": "greet", "t": 9.0, "customer": 2, "staff": 1, "t_end": None,
                 "data": {"time_to_greet": 4.0}}


def test_to_jsonable_handles_numpy_and_int_keys():
    d = to_jsonable({1: np.array([1.5, 2.0]), "b": (np.int64(3), Det((0, 0, 1, 1), 0.5))})
    assert d == {"1": [1.5, 2.0], "b": [3, {"bbox": [0, 0, 1, 1], "score": 0.5, "cls": 0}]}
```

```python
# tests/test_config.py
import pytest
from showroom_coach.config import PRESETS, get_preset


def test_presets_match_spec():
    d, s = PRESETS["dealership"], PRESETS["short"]
    assert (d.greet_dist, d.greet_time, d.engage_dist, d.engage_time) == (1.5, 5.0, 2.0, 120.0)
    assert (d.gap_tol, d.wait_alert, d.leave_after) == (3.0, 60.0, 10.0)
    assert (s.greet_dist, s.greet_time, s.engage_dist, s.engage_time) == (1.5, 2.0, 2.0, 8.0)
    assert (s.gap_tol, s.wait_alert, s.leave_after) == (1.5, 10.0, 3.0)


def test_unknown_preset():
    with pytest.raises(ValueError):
        get_preset("nope")
```

- [ ] **Step 2:** run red. **Step 3:** implement. **Step 4:** run green.
- [ ] **Step 5: Commit** `feat(yak-44): core dataclasses, JSON helper and presets`

### Task 1.1: Geometry (distance without calibration, facing)

**Files:**
- Create: `src/showroom_coach/geometry.py`, `tests/__init__.py` (empty, so `tests.synth` imports), `tests/synth.py`, `tests/test_geometry.py`

**Interfaces (Produces):**

```python
NOSE_ANKLE_RATIO = 0.89      # nose height 0.93 H − ankle height 0.04 H
SHOULDER_HIP_RATIO = 0.30    # shoulder 0.82 H − hip 0.52 H
SHOULDER_WIDTH_RATIO = 0.23  # biacromial width ≈ 0.39 m for H = 1.70 m
STANDING_MIN_RATIO = 2.4     # (ankle_y − nose_y) / torso_px above this ⇒ standing
KPT_THR = 0.3

@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov_deg: float = 70.0
    person_height_m: float = 1.70
    floor_h: np.ndarray | None = None            # 3x3 homography image(u,v,1) → floor (X,Z,1)
    # properties: f = width / (2·tan(hfov/2)); cx = width/2; cy = height/2
    @classmethod
    def from_config(cls, width, height, cfg: CameraConfig) -> "Camera": ...   # cv2.findHomography when cfg.floor_points has 4 points

@dataclass(frozen=True)
class Position:
    x: float
    z: float
    source: str      # "full" | "torso" | "box" | "floor"
    # properties: approx -> source == "box"; xz -> (x, z)

def body_height_px(bbox, kpts) -> tuple[float, str]        # equivalent full-body height in px and its source
def position_m(bbox, kpts, cam: Camera) -> Position
def distance_m(a, b) -> float                             # a, b: Position or (x, z)
def facing_vector(bbox, kpts, cam: Camera) -> tuple[float, float] | None   # unit (dx, dz) on the floor; None if unknown
def faces(a_xz, a_facing, b_xz, max_deg: float = 60.0) -> bool | None
def at_frame_edge(bbox, width, height, margin: float = 0.01) -> bool
```

Algorithm (spec §4.3): `kpts` is a `(17,3)` array-like or None.
- head point = nose if conf ≥ KPT_THR, else mean of confident eyes/ears; ankle_y = mean of confident ankles; torso_px = distance between mid-shoulders and mid-hips (both sides if confident, else the confident side's shoulder–hip distance).
- **full**: head and ≥1 ankle confident and (torso unknown or `(ankle_y − head_y)/torso_px ≥ STANDING_MIN_RATIO`) → `h = (ankle_y − head_y)/NOSE_ANKLE_RATIO`.
- **torso**: else if torso known → `h = torso_px / SHOULDER_HIP_RATIO`.
- **box**: else `h = y2 − y1`.
- `Z = f·H/h`, `u` = mid-hip x if confident else box centre x, `X = (u − cx)·Z/f`.
- **floor** (when `cam.floor_h` is set): foot point (mid-ankle if confident else box bottom centre) through the homography; source "floor".
- facing: `exp_dx = SHOULDER_WIDTH_RATIO·h`; `fz = clip(−(x_lsho − x_rsho)/exp_dx, −1, 1)`; `|fx| = sqrt(1−fz²)`; sign of fx = sign(nose_x − mid_shoulder_x) when the nose is confident; if the nose isn't confident and `|fz| < 0.9` → None; if either shoulder isn't confident → None.
- faces: None if `a_facing` is None; else angle between `a_facing` and `(b − a)` ≤ `max_deg`.

- [ ] **Step 1: Write `tests/synth.py`** (shared by later tests):

```python
"""Synthetic pinhole people and scripted timelines for tests."""
import math
import numpy as np
from showroom_coach.geometry import Camera

# (height fraction of H, lateral metres (+ = person's left), forward metres) in COCO order
STAND = [
    (0.930, 0.00, 0.09), (0.945, 0.03, 0.07), (0.945, -0.03, 0.07), (0.935, 0.075, 0.0), (0.935, -0.075, 0.0),
    (0.820, 0.195, 0.0), (0.820, -0.195, 0.0), (0.630, 0.22, 0.0), (0.630, -0.22, 0.0),
    (0.480, 0.22, 0.03), (0.480, -0.22, 0.03), (0.520, 0.10, 0.0), (0.520, -0.10, 0.0),
    (0.285, 0.10, 0.0), (0.285, -0.10, 0.0), (0.040, 0.10, 0.0), (0.040, -0.10, 0.0),
]
SIT = [
    (0.680, 0.00, 0.09), (0.695, 0.03, 0.07), (0.695, -0.03, 0.07), (0.685, 0.075, 0.0), (0.685, -0.075, 0.0),
    (0.570, 0.195, 0.0), (0.570, -0.195, 0.0), (0.420, 0.22, 0.05), (0.420, -0.22, 0.05),
    (0.330, 0.20, 0.25), (0.330, -0.20, 0.25), (0.270, 0.10, 0.0), (0.270, -0.10, 0.0),
    (0.300, 0.10, 0.45), (0.300, -0.10, 0.45), (0.040, 0.10, 0.45), (0.040, -0.10, 0.45),
]
CAM_HEIGHT = 1.6   # metres above the floor; optical axis horizontal


def project_person(cam: Camera, x: float, z: float, facing=(0.0, -1.0), H: float = 1.70, sitting: bool = False):
    """Return (bbox, kpts (17,3)) for a person standing/sitting at floor point (x, z)."""
    fx, fz = facing
    lx, lz = -fz, fx                      # person's left
    joints = SIT if sitting else STAND
    facing_camera = fz < 0.3
    k = np.zeros((17, 3))
    for i, (hf, lat, fwd) in enumerate(joints):
        X = x + lat * lx + fwd * fx
        Z = z + lat * lz + fwd * fz
        Y = CAM_HEIGHT - hf * H
        u, v = cam.cx + cam.f * X / Z, cam.cy + cam.f * Y / Z
        conf = 0.9
        if i <= 2 and not facing_camera:
            conf = 0.05
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            conf = 0.0
        k[i] = (u, v, conf)
    top_h = (0.75 if sitting else 1.0) * H
    us = [cam.cx + cam.f * (x + s * 0.25) / z for s in (-1, 1)]
    v_top = cam.cy + cam.f * (CAM_HEIGHT - top_h) / z
    v_bot = cam.cy + cam.f * CAM_HEIGHT / z
    bbox = (max(0.0, us[0]), max(0.0, v_top), min(cam.width - 1.0, us[1]), min(cam.height - 1.0, v_bot))
    return bbox, k


def unit(deg: float):
    """Floor direction from an angle in degrees: 0° = +X (image right), 90° = +Z (away from camera)."""
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
```

- [ ] **Step 2: Failing tests** (`tests/test_geometry.py`):

```python
import math
import numpy as np
import pytest
from showroom_coach.geometry import (Camera, position_m, distance_m, facing_vector, faces,
                                     body_height_px, at_frame_edge)
from tests.synth import project_person, unit

CAM = Camera(1280, 720, hfov_deg=70.0)


def test_focal_length():
    assert CAM.f == pytest.approx(1280 / (2 * math.tan(math.radians(35))))


@pytest.mark.parametrize("sep", [1.0, 1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("z0", [4.0, 6.0, 8.0])
def test_pair_distance_within_15pct_standing(sep, z0):
    a = position_m(*project_person(CAM, -sep / 2, z0), CAM)
    b = position_m(*project_person(CAM, sep / 2, z0 + 0.5), CAM)
    true = math.hypot(sep, 0.5)
    assert a.source == b.source == "full"
    assert distance_m(a, b) == pytest.approx(true, rel=0.15)


def test_depth_separation_within_15pct():
    a = position_m(*project_person(CAM, 0.0, 4.0), CAM)
    b = position_m(*project_person(CAM, 0.3, 7.0), CAM)
    assert distance_m(a, b) == pytest.approx(math.hypot(0.3, 3.0), rel=0.15)


def test_sitting_uses_torso_and_stays_accurate():
    bbox, k = project_person(CAM, 0.5, 5.0, sitting=True)
    p = position_m(bbox, k, CAM)
    assert p.source == "torso"
    assert p.z == pytest.approx(5.0, rel=0.15)


def test_cutoff_uses_torso():
    bbox, k = project_person(CAM, 0.0, 2.2)      # feet below the image at 2.2 m
    assert k[15, 2] == 0.0 and k[16, 2] == 0.0
    p = position_m(bbox, k, CAM)
    assert p.source == "torso"
    assert p.z == pytest.approx(2.2, rel=0.15)


def test_no_kpts_box_fallback():
    bbox, _ = project_person(CAM, 1.0, 5.0)
    p = position_m(bbox, None, CAM)
    assert p.source == "box" and p.approx
    assert p.z == pytest.approx(5.0, rel=0.25)


def test_floor_homography_overrides():
    from showroom_coach.config import CameraConfig
    pts = ((0, 720, -2.0, 2.0), (1280, 720, 2.0, 2.0), (1280, 400, 2.0, 8.0), (0, 400, -2.0, 8.0))
    cam = Camera.from_config(1280, 720, CameraConfig(floor_points=pts))
    p = position_m((600, 300, 680, 720), None, cam)
    assert p.source == "floor" and p.z == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("deg", [0, 45, 135, 180, 270, 300])
def test_facing_vector_recovers_direction(deg):
    want = unit(deg)
    bbox, k = project_person(CAM, 0.0, 5.0, facing=want)
    got = facing_vector(bbox, k, CAM)
    if got is None:                       # only allowed when facing away with the nose hidden
        assert want[1] >= 0.3
        return
    ang = math.degrees(math.acos(np.clip(got[0] * want[0] + got[1] * want[1], -1, 1)))
    assert ang < 30


def test_faces_true_false_none():
    assert faces((0, 5), (1, 0), (1.5, 5)) is True
    assert faces((0, 5), (1, 0), (-1.5, 5)) is False
    assert faces((0, 5), None, (1.5, 5)) is None


def test_body_height_sources():
    bbox, k = project_person(CAM, 0.0, 5.0)
    h, src = body_height_px(bbox, k)
    assert src == "full" and h == pytest.approx(CAM.f * 1.70 / 5.0, rel=0.1)


def test_at_frame_edge():
    assert at_frame_edge((0, 10, 50, 200), 1280, 720)
    assert not at_frame_edge((100, 10, 150, 200), 1280, 720)
```

- [ ] **Step 3:** run red → **Step 4:** implement `geometry.py` → **Step 5:** run green. If a synthetic case fails because of the synthetic helper itself (not the algorithm), fix the helper and say so in the report; never loosen the 15% bound.
- [ ] **Step 6: Commit** `feat(yak-44): calibration-free pinhole geometry with torso/box fallback and facing`

### Task 1.2: Event engine (pure state machine)

**Files:**
- Create: `src/showroom_coach/events.py`, `tests/test_events.py`
- Modify: `tests/synth.py` (add `script()` and `when()` helpers)

**Interfaces:**
- Consumes: `Thresholds`, `PRESETS` (Task 1.0), `distance_m`, `faces` (Task 1.1), `Event` (Task 1.0).
- Produces:

```python
@dataclass
class PersonState:
    pid: int
    xz: tuple[float, float]
    role: str = "customer"                       # "staff" | "customer" | "unsure" (unsure ⇒ customer)
    facing: tuple[float, float] | None = None
    at_edge: bool = False

class EventEngine:
    def __init__(self, th: Thresholds, t0: float = 0.0): ...
    def step(self, t: float, states: Sequence[PersonState]) -> list[Event]: ...
    def finish(self, t_end: float) -> list[Event]: ...      # closes open engagements (no walkouts for people still present)
    def status(self, t: float) -> dict[int, dict]: ...
        # customers: {"role": "customer", "state": "waiting"|"greeted"|"engaged", "wait_s": float|None,
        #             "entry_known": bool, "greeted_by": int|None}   (wait_s = t − enter_t while waiting and entry known)
        # staff:     {"role": "staff", "state": "busy"|"free"}
        # only people present in the last step are listed
    def active_pairs(self) -> list[tuple[int, int, str]]: ...   # (staff, customer, "engaged"|"close")

def run_events(timeline: Iterable[tuple[float, Sequence[PersonState]]], th: Thresholds,
               t_end: float | None = None, t0: float | None = None) -> list[Event]
    # t0 defaults to the first timeline time; t_end defaults to the last; calls finish(t_end); events sorted by (t, type)
```

Rules (spec §4.5), all thresholds from `th`:
1. **Visits.** A customer pid not currently in a visit starts one at its first-seen time `t`: `entry_known = (t − t0) ≥ th.leave_after`. Emit `enter` (`data={"entry_known": bool}`). Staff never get visits.
2. **Leaving.** A customer absent this step leaves when `t − last_seen ≥ th.leave_after`, or `t − last_seen ≥ min(th.edge_leave, th.leave_after)` if their last box was `at_edge`. Emit `leave` at `t = last_seen` (`data={"greeted", "entry_known", "detected_at": t}`). If never greeted and `entry_known` → also `walkout` at `last_seen` with `data={"free_staff": <free staff at last_seen>}`. A pid seen again after leaving starts a new visit.
3. **Pair streaks.** For every (staff, customer) pair, keep two streaks: *greet* (cond `d ≤ greet_dist and facing_ok`) and *engage* (cond `d ≤ engage_dist`). `facing_ok = not (faces(s→c) is False and faces(c→s) is False)`. Absent pairs count as cond False. A streak starts at the first True sample, `last_true` updates on True samples, and it closes when `t − last_true > gap_tol`.
4. **greet.** When an ungreeted customer's greet streak reaches `last_true − start ≥ greet_time`: emit `greet` at `t = start`, `staff`, `data={"time_to_greet": start − enter_t if entry_known else None, "confirmed_at": t, "via": "direct"}`.
5. **engagement.** When an engage streak reaches `≥ engage_time`, it becomes an active engagement (`engagement_start` event at `start`). If the customer is still ungreeted, emit `greet` at `start` with `via="engagement"`. When the streak closes (or at `finish`), emit `engagement` with `t=start, t_end=last_true`. If the customer had an earlier or active engagement with a *different* staff, emit `handoff` at the new start (`data={"from_staff", "to_staff"}`).
6. **busy / free.** Staff is *busy* at time t if it has an active engagement or an engage streak with duration ≥ `greet_time`; free otherwise. Only staff present in that step are listed as free.
7. **wait_alert.** Once per visit: customer with `entry_known`, not greeted, `t − enter_t > wait_alert` → emit at `t` with `data={"wait_s": t − enter_t, "free_staff": sorted free staff}`.
8. **party.** Two customers whose first-seen times differ by ≤ `party_arrive` are candidates. Over `[max(first), max(first) + party_window]`, count the samples where both are present and those with `d ≤ party_dist`. At window end (or when one leaves) confirm if `close/both ≥ 0.5` and `both ≥ 3`. Confirmed parties are unioned (union-find); emit `party` (`data={"members": sorted}`). When a party is confirmed, or later when any member is greeted, every ungreeted member still in its visit is greeted too (`greet` with `via="party"`, same staff, `t` = the greeting time of the member who was greeted, `time_to_greet` relative to each member's own entry if known).

- [ ] **Step 1: Add to `tests/synth.py`:**

```python
from showroom_coach.events import PersonState


def script(duration: float, dt: float, people: dict):
    """people: pid -> dict(role=..., pos=f(t)->(x,z)|None, facing=f(t)->vec|None, edge=f(t)->bool)."""
    tl = []
    for i in range(int(round(duration / dt)) + 1):
        t = round(i * dt, 6)
        states = []
        for pid, p in people.items():
            xz = p["pos"](t)
            if xz is None:
                continue
            fac = p.get("facing")
            edge = p.get("edge")
            states.append(PersonState(pid, tuple(xz), p.get("role", "customer"),
                                      fac(t) if fac else None, bool(edge(t)) if edge else False))
        tl.append((t, states))
    return tl


def when(t0, t1, xz):
    return lambda t: xz if t0 <= t <= t1 else None
```

- [ ] **Step 2: Failing tests** (`tests/test_events.py`):

```python
import pytest
from showroom_coach.config import PRESETS
from showroom_coach.events import EventEngine, PersonState, run_events
from tests.synth import script, when

SHORT, DEAL = PRESETS["short"], PRESETS["dealership"]


def of(events, kind):
    return [e for e in events if e.type == kind]


def approach(t_enter, t_move, xs=6.0, xe=1.0, z=5.0, t_out=40.0):
    """Customer enters at t_enter at x=xs, walks to xe during [t_move, t_move+1], stays until t_out."""
    def pos(t):
        if t < t_enter or t > t_out:
            return None
        if t < t_move:
            return (xs, z)
        if t < t_move + 1:
            return (xs + (xe - xs) * (t - t_move), z)
        return (xe, z)
    return pos


STAFF = {"role": "staff", "pos": when(0, 40, (0.0, 5.0))}


def test_greet_time_to_greet_short():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(5, 8)}}), SHORT)
    (g,) = of(ev, "greet")
    assert (g.staff, g.customer) == (1, 2)
    assert g.t == pytest.approx(9.0, abs=0.21)
    assert g.data["time_to_greet"] == pytest.approx(4.0, abs=0.21)
    (en,) = of(ev, "enter")
    assert en.data["entry_known"] is True


def test_engagement_short_closed_at_finish():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(5, 8)}}), SHORT, t_end=40.0)
    (e,) = of(ev, "engagement")
    assert e.t == pytest.approx(8.8, abs=0.21) and e.t_end == pytest.approx(40.0, abs=0.21)
    assert len(of(ev, "engagement_start")) == 1


def test_no_greet_when_both_face_away():
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0)), "facing": lambda t: (-1.0, 0.0)},
        2: {"pos": lambda t: (1.0, 5.0) if 5 <= t <= 10 else ((6.0, 5.0) if 10 < t <= 40 else None),
            "facing": lambda t: (1.0, 0.0)},
    }
    ev = run_events(script(40, 0.2, people), SHORT)
    assert of(ev, "greet") == []


def test_greet_when_one_faces_the_other():
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0)), "facing": lambda t: (1.0, 0.0)},
        2: {"pos": lambda t: (1.0, 5.0) if 5 <= t <= 10 else ((6.0, 5.0) if 10 < t <= 40 else None)},
    }
    assert len(of(run_events(script(40, 0.2, people), SHORT), "greet")) == 1


def _gap_customer(gap_end):
    def pos(t):
        if t < 5 or t > 40:
            return None
        close = (10 <= t <= 20) or (gap_end <= t <= 32)
        return (1.0, 5.0) if close else (5.0, 5.0)
    return pos


def test_short_gap_does_not_split_engagement():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": _gap_customer(21.0)}}), SHORT, t_end=40)
    (e,) = of(ev, "engagement")
    assert e.t == pytest.approx(10.0, abs=0.21) and e.t_end == pytest.approx(32.0, abs=0.21)


def test_long_gap_splits_engagement():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": _gap_customer(23.0)}}), SHORT, t_end=40)
    es = of(ev, "engagement")
    assert [(round(e.t), round(e.t_end)) for e in es] == [(10, 20), (23, 32)]


@pytest.mark.parametrize("close_until,expected", [(120.0, 0), (150.0, 1)])
def test_dealership_engagement_needs_120s(close_until, expected):
    people = {
        1: {"role": "staff", "pos": when(0, 300, (0.0, 5.0))},
        2: {"pos": lambda t: None if t < 15 else ((1.0, 5.0) if 20 <= t <= close_until else (6.0, 5.0))},
    }
    ev = run_events(script(300, 1.0, people), DEAL, t_end=300)
    assert len(of(ev, "engagement")) == expected
    assert len(of(ev, "greet")) == 1


def test_wait_alert_lists_free_staff():
    people = {1: STAFF, 2: {"pos": when(5, 40, (6.0, 5.0))}}
    (w,) = of(run_events(script(40, 0.2, people), SHORT), "wait_alert")
    assert w.customer == 2 and w.t == pytest.approx(15.2, abs=0.21)
    assert w.data["free_staff"] == [1] and w.data["wait_s"] > 10


def test_wait_alert_staff_busy_with_other_customer():
    people = {1: STAFF, 3: {"pos": when(0, 40, (0.5, 5.0))}, 2: {"pos": when(5, 40, (6.0, 5.0))}}
    w = [e for e in run_events(script(40, 0.2, people), SHORT) if e.type == "wait_alert" and e.customer == 2]
    assert len(w) == 1 and w[0].data["free_staff"] == []


def test_walkout_after_leave_after():
    people = {1: STAFF, 2: {"pos": when(5, 12, (6.0, 5.0))}}
    ev = run_events(script(40, 0.2, people), SHORT)
    (wo,) = of(ev, "walkout")
    assert wo.t == pytest.approx(12.0) and wo.data["free_staff"] == [1]
    (lv,) = [e for e in of(ev, "leave") if e.customer == 2]
    assert lv.data["detected_at"] == pytest.approx(15.0, abs=0.21)


def test_walkout_at_frame_edge_is_fast():
    people = {1: STAFF, 2: {"pos": when(5, 12, (6.0, 5.0)), "edge": lambda t: t >= 11.5}}
    ev = run_events(script(40, 0.2, people), SHORT)
    (lv,) = [e for e in of(ev, "leave") if e.customer == 2]
    assert lv.data["detected_at"] <= 13.01


def test_already_present_customer_excluded_from_wait_and_walkout():
    people = {1: STAFF, 2: {"pos": when(0, 20, (6.0, 5.0))}}
    ev = run_events(script(40, 0.2, people), SHORT)
    assert of(ev, "wait_alert") == [] and of(ev, "walkout") == []
    (en,) = of(ev, "enter")
    assert en.data["entry_known"] is False


def test_already_present_greet_has_no_time_to_greet():
    ev = run_events(script(40, 0.2, {1: STAFF, 2: {"pos": approach(0, 8)}}), SHORT)
    (g,) = of(ev, "greet")
    assert g.data["time_to_greet"] is None


def test_party_greet_propagates():
    def c2(t):
        if t < 5:
            return None
        return (6.0, 5.0) if t < 20 else (1.0, 5.0)
    people = {
        1: {"role": "staff", "pos": when(0, 60, (0.0, 5.0))},
        2: {"pos": c2},
        3: {"pos": when(7, 60, (6.0, 6.0))},
    }
    ev = run_events(script(60, 0.2, people), SHORT)
    (p,) = of(ev, "party")
    assert p.data["members"] == [2, 3]
    greets = {g.customer: g for g in of(ev, "greet")}
    assert set(greets) == {2, 3}
    assert greets[3].data["via"] == "party" and greets[3].staff == 1


def test_handoff_between_staff():
    def c(t):
        if t < 5 or t > 40:
            return None
        return (1.0, 5.0) if t <= 20 else (9.0, 5.0)
    people = {
        1: {"role": "staff", "pos": when(0, 40, (0.0, 5.0))},
        4: {"role": "staff", "pos": when(0, 40, (10.0, 5.0))},
        2: {"pos": c},
    }
    ev = run_events(script(40, 0.2, people), SHORT, t_end=40)
    assert sorted((e.staff, e.customer) for e in of(ev, "engagement")) == [(1, 2), (4, 2)]
    (h,) = of(ev, "handoff")
    assert h.data == {"from_staff": 1, "to_staff": 4} and h.t == pytest.approx(20.2, abs=0.21)


def test_unsure_is_treated_as_customer():
    ev = run_events(script(10, 0.2, {9: {"role": "unsure", "pos": when(4, 10, (3.0, 5.0))}}), SHORT)
    assert [e.customer for e in of(ev, "enter")] == [9]


def test_step_and_status_for_live_overlay():
    eng = EventEngine(SHORT)
    for t, states in script(16, 0.2, {1: STAFF, 2: {"pos": when(5, 40, (6.0, 5.0))}}):
        eng.step(t, states)
    st = eng.status(16.0)
    assert st[2]["state"] == "waiting" and st[2]["wait_s"] == pytest.approx(11.0, abs=0.21)
    assert st[1] == {"role": "staff", "state": "free"}


def test_role_flip_mid_stream_drops_customer_state():
    eng = EventEngine(SHORT)
    eng.step(4.0, [PersonState(5, (0, 5), "customer")])
    eng.step(4.2, [PersonState(5, (0, 5), "staff")])
    assert eng.status(4.2)[5]["role"] == "staff"
```

- [ ] **Step 3:** run red → **Step 4:** implement `events.py` (one `_Streak` helper class, a `_Visit` dataclass per customer, a small union-find for parties) → **Step 5:** run green.
- [ ] **Step 6: Commit** `feat(yak-44): event engine for greet, engagement, wait, walk-out, party and handoff`

### Task 1.3: Metrics / scorecard

**Files:**
- Create: `src/showroom_coach/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Event`, `StaffCard`, `StoreCard`, `Scorecard` (Task 1.0).
- Produces:

```python
def scorecard(events: list[Event], staff_ids: Iterable[int], customer_ids: Iterable[int], t_end: float,
              names: dict[int, str] | None = None) -> Scorecard
def unattended_wait(events: list[Event], t_end: float) -> float
```

Definitions (spec §4.7):
- staff `greets` = greet events with that staff (all `via` values); `median_time_to_greet` over non-None `time_to_greet`; `customers_engaged` = distinct customers in `engagement` events; `engaged_total_s` / `engaged_median_s` over `t_end − t`; `missed_while_free` = wait_alerts whose `free_staff` contains the staff; `walkouts_while_free` = walkouts whose `free_staff` contains the staff.
- store: `customers` = number of `enter` events (visits); `customers_entry_known`; `greeted` = number of visits with at least one greet (match a greet to the latest `enter` of that customer at or before it); `greet_rate = greeted/customers` (None if 0); `median_time_to_greet` over all non-None first greets; `walkouts`; `unattended_wait_s` = Σ over visits with known entry of `(first greet t, else leave t, else t_end) − enter t`; `wait_alerts`; `engagements`.
- staff list order: by staff id ascending; staff with no events still get a card with zeros/None. `customer_ids` is accepted for API symmetry and future use (visits come from `enter` events).

- [ ] **Step 1: Failing tests**

```python
import pytest
from showroom_coach.metrics import scorecard, unattended_wait
from showroom_coach.types import Event

EV = [
    Event("enter", 5.0, customer=2, data={"entry_known": True}),
    Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": 4.0, "via": "direct"}),
    Event("engagement", 9.0, customer=2, staff=1, t_end=40.0),
    Event("enter", 10.0, customer=3, data={"entry_known": True}),
    Event("wait_alert", 20.0, customer=3, data={"wait_s": 10.0, "free_staff": [4]}),
    Event("walkout", 25.0, customer=3, data={"free_staff": [4]}),
    Event("leave", 25.0, customer=3, data={"greeted": False, "entry_known": True}),
    Event("enter", 12.0, customer=5, data={"entry_known": True}),
    Event("greet", 18.0, customer=5, staff=4, data={"time_to_greet": 6.0, "via": "direct"}),
    Event("engagement", 18.0, customer=5, staff=4, t_end=30.0),
]


def test_staff_cards():
    sc = scorecard(EV, staff_ids=[4, 1], customer_ids=[2, 3, 5], t_end=60.0, names={1: "Aziz"})
    a, b = sc.staff
    assert (a.staff, a.name, a.greets, a.median_time_to_greet) == (1, "Aziz", 1, 4.0)
    assert (a.customers_engaged, a.engaged_total_s, a.engaged_median_s) == (1, 31.0, 31.0)
    assert (a.missed_while_free, a.walkouts_while_free) == (0, 0)
    assert (b.staff, b.greets, b.median_time_to_greet, b.engaged_total_s) == (4, 1, 6.0, 12.0)
    assert (b.missed_while_free, b.walkouts_while_free) == (1, 1)


def test_store_card():
    s = scorecard(EV, [1, 4], [2, 3, 5], 60.0).store
    assert (s.customers, s.greeted, s.walkouts, s.wait_alerts, s.engagements) == (3, 2, 1, 1, 2)
    assert s.greet_rate == pytest.approx(2 / 3)
    assert s.median_time_to_greet == pytest.approx(5.0)
    assert s.unattended_wait_s == pytest.approx(4 + 15 + 6)


def test_unattended_wait_open_visit_runs_to_end():
    ev = [Event("enter", 50.0, customer=7, data={"entry_known": True})]
    assert unattended_wait(ev, t_end=60.0) == pytest.approx(10.0)


def test_unknown_entry_not_counted_in_wait():
    ev = [Event("enter", 0.0, customer=7, data={"entry_known": False})]
    assert unattended_wait(ev, t_end=60.0) == 0.0


def test_empty_events():
    sc = scorecard([], [], [], 0.0)
    assert sc.staff == [] and sc.store.customers == 0
    assert sc.store.greet_rate is None and sc.store.median_time_to_greet is None


def test_staff_without_events_gets_zero_card():
    sc = scorecard([], [8], [], 10.0)
    assert sc.staff[0].greets == 0 and sc.staff[0].median_time_to_greet is None
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Commit** `feat(yak-44): per-salesperson and store scorecards`

### Task 1.4: Staff suggestion (roles)

**Files:**
- Create: `src/showroom_coach/roles.py`, `tests/test_roles.py`

**Interfaces:**
- Consumes: `RoleWeights`, `RoleSuggestion` (Task 1.0).
- Produces:

```python
def interaction_features(traj: dict[int, list[tuple[float, float, float]]], contact_dist: float = 2.0,
                         contact_min: float = 5.0, approach_window: float = 3.0,
                         min_approach_m: float = 0.5) -> dict[int, dict]
    # traj: pid -> time-sorted [(t, x, z)]; returns pid -> {"contacts": int, "initiated": int, "partners": [pid, ...]}
def presence(traj: dict[int, list[tuple[float, float, float]]], frame_times: Sequence[float]) -> dict[int, float]
def outfit_peers(descr: dict[int, np.ndarray], thr: float) -> dict[int, int]
def suggest_roles(features: dict[int, dict], clip_len: float, weights: RoleWeights = RoleWeights()) -> list[RoleSuggestion]
    # features[pid] keys: outfit_peers, contacts, presence, initiated (others ignored but copied into .features)
```

- Contacts: two people are close at a shared timestamp (|Δt| ≤ 1e-6 — all trajectories come from the same sampled frames) when `d ≤ contact_dist`; a partner counts once the cumulative close time ≥ `contact_min` (each close sample contributes the median sampling step of the trajectories).
- Initiator: at the first close sample `tc` of a counted pair, compare each person's displacement toward the other over `[tc − approach_window, tc]` (projection of own movement onto the unit vector toward the other's position at `tc`). The one who moved more, by at least `min_approach_m`, initiated.
- Score (pinned formula): `f_outfit = min(n,3)/3; f_contacts = min(n,4)/4; f_presence = (p − 0.5)·2 if clip_len ≥ 60 else 0; f_init = min(n,3)/3; z = bias + Σ w·f; P(staff) = 1/(1+e^−z)`. staff if P ≥ staff_thr, customer if P ≤ customer_thr, else unsure.
- Reasons (≤3): staff → features with the largest positive contribution (> 0.05); customer → features with the smallest contribution; unsure → largest |contribution|. Texts: `same outfit as {n} other(s)` / `outfit unlike anyone else`; `talked to {n} person|people`; `on screen {p}% of the clip`; `approached {n} person|people first` / `never approached anyone first`. Presence is never a reason when clip_len < 60.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
from showroom_coach.config import RoleWeights
from showroom_coach.roles import interaction_features, outfit_peers, presence, suggest_roles

W = RoleWeights(bias=-1.5, outfit_peers=1.2, contacts=1.6, presence=1.0, initiator=1.0,
                staff_thr=0.6, customer_thr=0.4)


def test_suggest_staff_and_customer():
    feats = {1: dict(outfit_peers=2, contacts=3, presence=0.95, initiated=2),
             2: dict(outfit_peers=0, contacts=1, presence=0.30, initiated=0)}
    out = {s.pid: s for s in suggest_roles(feats, clip_len=120.0, weights=W)}
    assert out[1].role == "staff" and out[1].confidence >= 0.6
    assert out[2].role == "customer" and out[2].confidence <= 0.4
    assert "talked to 3 people" in out[1].reasons
    assert 1 <= len(out[1].reasons) <= 3 and 1 <= len(out[2].reasons) <= 3
    assert out[1].features["contacts"] == 3


def test_presence_ignored_for_short_clips():
    a = suggest_roles({1: dict(outfit_peers=0, contacts=0, presence=1.0, initiated=0)}, 30.0, W)[0]
    b = suggest_roles({1: dict(outfit_peers=0, contacts=0, presence=0.0, initiated=0)}, 30.0, W)[0]
    assert a.confidence == pytest.approx(b.confidence)
    assert not any("on screen" in r for r in a.reasons)


def test_unsure_band():
    s = suggest_roles({1: dict(outfit_peers=0, contacts=2, presence=0.5, initiated=2)}, 120.0, W)[0]
    assert s.role == "unsure" and 0.4 < s.confidence < 0.6


def test_pinned_formula():
    s = suggest_roles({1: dict(outfit_peers=3, contacts=4, presence=1.0, initiated=3)}, 120.0, W)[0]
    z = -1.5 + 1.2 + 1.6 + 1.0 + 1.0
    assert s.confidence == pytest.approx(1 / (1 + np.exp(-z)))


def _traj():
    traj = {1: [], 2: [], 3: []}
    for i in range(101):                 # 20 s at 5 fps
        t = round(i * 0.2, 6)
        traj[1].append((t, max(1.0, 5.0 - t), 5.0))   # walks from x=5 to x=1, arrives t=4
        traj[2].append((t, 0.0, 5.0))
        traj[3].append((t, 10.0, 5.0))
    return traj


def test_interaction_features_contacts_and_initiator():
    f = interaction_features(_traj(), contact_dist=2.0, contact_min=5.0)
    assert (f[1]["contacts"], f[2]["contacts"], f[3]["contacts"]) == (1, 1, 0)
    assert (f[1]["initiated"], f[2]["initiated"]) == (1, 0)
    assert f[1]["partners"] == [2]


def test_short_contact_not_counted():
    f = interaction_features(_traj(), contact_dist=2.0, contact_min=30.0)
    assert f[1]["contacts"] == 0


def test_presence_fraction():
    times = [round(i * 0.2, 6) for i in range(101)]
    traj = {1: [(t, 0, 5) for t in times], 2: [(t, 0, 5) for t in times[:51]]}
    p = presence(traj, times)
    assert p[1] == pytest.approx(1.0) and p[2] == pytest.approx(51 / 101)


def test_outfit_peers_counts_similar_descriptors():
    a = np.array([1.0, 0, 0]); b = np.array([0.98, 0.2, 0]); c = np.array([0, 1.0, 0])
    assert outfit_peers({1: a, 2: b, 3: c}, thr=0.9) == {1: 1, 2: 1, 3: 0}
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Commit** `feat(yak-44): transparent staff suggestion with plain-language reasons`

### Task 1.5: Identity merge (offline) and re-attach (live)

**Files:**
- Create: `src/showroom_coach/identity.py`, `tests/test_identity.py`

**Interfaces:**
- Produces:

```python
@dataclass
class TrackSummary:
    track_id: int
    t0: float
    t1: float
    emb: np.ndarray            # L2-normalised appearance descriptor
    n_obs: int = 1

def merge_tracks(tracks: list[TrackSummary], thr: float = 0.30, max_overlap: float = 0.2) -> dict[int, int]
    # track_id -> pid; pids are 1..N ordered by each person's first appearance (t0, then track_id)

class LiveIdentity:
    def __init__(self, thr: float = 0.30, window: float = 10.0): ...
    def resolve(self, track_id: int, t: float, emb: np.ndarray | None) -> int: ...
    def end_step(self, t: float, active_track_ids: set[int]) -> None: ...
    def update_emb(self, pid: int, emb: np.ndarray, alpha: float = 0.2) -> None: ...   # running mean, re-normalised
```

- Offline: greedy constrained average-linkage. Start with one cluster per track. Repeatedly take the pair of clusters with the smallest cosine distance between their mean embeddings (mean of member embeddings weighted by `n_obs`, re-normalised); merge if distance < `thr` and no track in one cluster overlaps any track in the other by more than `max_overlap` seconds (overlap = `min(t1) − max(t0)`); stop when no allowed pair is below `thr`.
- Live: a known track keeps its pid. A new track with an embedding is compared against persons that are *not active* (no active track) and were last seen within `window` seconds; best cosine distance < `thr` → reuse that pid; else a new pid. A new track without an embedding gets a new pid.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from showroom_coach.identity import LiveIdentity, TrackSummary, merge_tracks


def emb(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_merge_fragments_of_same_person():
    m = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                      TrackSummary(2, 12.0, 20.0, emb([0.95, 0.1, 0])),
                      TrackSummary(3, 0.0, 20.0, emb([0, 1, 0]))], thr=0.3)
    assert m[1] == m[2] != m[3]
    assert sorted(set(m.values())) == [1, 2] and m[1] == 1


def test_time_overlap_forbids_merge_even_if_identical():
    m = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                      TrackSummary(2, 5.0, 15.0, emb([1, 0.01, 0]))], thr=0.3)
    assert m[1] != m[2]


def test_constraint_is_cluster_level():
    m = merge_tracks([TrackSummary(1, 0, 10, emb([1, 0, 0])),
                      TrackSummary(2, 11, 12, emb([1, 0.05, 0])),
                      TrackSummary(3, 5, 9, emb([1, 0.1, 0]))], thr=0.3)
    assert m[1] != m[3]
    assert m[2] in (m[1], m[3])


def test_dissimilar_tracks_stay_apart():
    m = merge_tracks([TrackSummary(1, 0, 5, emb([1, 0, 0])), TrackSummary(2, 6, 9, emb([0, 0, 1]))], thr=0.3)
    assert m[1] != m[2]


def test_live_reattach_recent_lost_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1})
    li.end_step(1.0, set())
    assert li.resolve(7, 3.0, emb([0.97, 0.1, 0])) == p1


def test_live_no_reattach_after_window():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    assert li.resolve(7, 30.0, emb([1, 0, 0])) != p1


def test_live_no_reattach_to_active_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1})
    assert li.resolve(2, 0.2, emb([1, 0, 0])) != p1


def test_known_track_keeps_pid_and_none_emb_gets_new_pid():
    li = LiveIdentity()
    p = li.resolve(1, 0.0, None)
    assert li.resolve(1, 0.2, emb([1, 0, 0])) == p
    assert li.resolve(2, 0.2, None) != p
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: M1 wrap-up** — full suite, PROGRESS entry (test counts), commit `feat(yak-44): constrained identity merge and live re-attach`.

---

# M2 — Perception chain on real video, with a per-video cache

### Task 2.1: Frame sources

**Files:**
- Create: `src/showroom_coach/sources.py`, `tests/test_sources.py`
- Modify: `tests/conftest.py` (add `synth_video` fixture)

**Interfaces (Produces):**

```python
class VideoError(Exception): ...

@dataclass
class VideoInfo:
    path: str
    src_width: int
    src_height: int
    src_fps: float
    src_frames: int
    duration: float          # seconds of the processed part (after trimming)
    src_duration: float
    width: int               # processed size (max width 960, even numbers)
    height: int
    scale: float             # processed / source
    sample_fps: float
    trimmed: bool
    max_seconds: float

def probe(path, max_width=960, sample_fps=5.0, max_seconds=300.0) -> VideoInfo        # VideoError if unreadable / 0 frames
def iter_frames(path, sample_fps=5.0, max_width=960, max_seconds=300.0) -> Iterator[tuple[int, float, np.ndarray]]
    # yields (src_frame_idx, t_sec, frame_bgr_resized); keeps frame i when i/src_fps ≥ next_t − 1e-6, next_t += 1/sample_fps;
    # uses cap.grab() for skipped frames and cap.retrieve() for kept ones; stops at max_seconds
def decode_jpeg(data: bytes, max_width=960) -> np.ndarray     # VideoError on bad bytes
def resize_max_width(frame, max_width) -> np.ndarray          # keeps aspect, even dims, INTER_AREA
```

Fixture (`tests/conftest.py`): `synth_video(tmp_path_factory)` writes `synth.avi` (MJPG, 30 fps, 1280×720, 6 s): grey background, a red rectangle (person A, 80×220 px) walking left→right, a blue rectangle (person B) standing still from 1 s to 6 s. Also a factory `make_video(path, seconds, fps, size, movers)` for other tests.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
from showroom_coach.sources import VideoError, decode_jpeg, iter_frames, probe


def test_probe_and_resize(synth_video):
    info = probe(synth_video)
    assert (info.src_width, info.src_height) == (1280, 720)
    assert info.width == 960 and info.height == 540 and info.scale == pytest.approx(0.75)
    assert info.src_fps == pytest.approx(30, abs=0.5) and not info.trimmed


def test_sampling_5fps(synth_video):
    frames = list(iter_frames(synth_video, sample_fps=5.0))
    ts = [t for _, t, _ in frames]
    assert len(frames) == 30
    assert np.allclose(np.diff(ts), 0.2, atol=0.034)
    assert frames[0][2].shape == (540, 960, 3)


def test_trimming(synth_video):
    info = probe(synth_video, max_seconds=2.0)
    assert info.trimmed and info.duration == pytest.approx(2.0, abs=0.05)
    assert len(list(iter_frames(synth_video, sample_fps=5.0, max_seconds=2.0))) == 10


def test_garbage_file_raises(tmp_path):
    p = tmp_path / "x.mp4"; p.write_text("not a video")
    with pytest.raises(VideoError):
        probe(p)


def test_decode_jpeg_roundtrip():
    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))
    assert decode_jpeg(buf.tobytes()).shape == (540, 960, 3)
    with pytest.raises(VideoError):
        decode_jpeg(b"nope")
```

- [ ] **Steps 2–4:** red → implement → green. **Step 5: Commit** `feat(yak-44): sampled, resized and trimmed video frame source`

### Task 2.2: Tracker, embedder and the perception stack (real + fake)

**Files:**
- Create: `src/showroom_coach/perception/tracking.py`, `src/showroom_coach/perception/embed.py`, `src/showroom_coach/perception/stack.py`, `tests/test_tracking.py`, `tests/test_stack.py`

**Interfaces:**
- Consumes: `Det` (1.0), `Detector`, `PoseEstimator`, `rtm_onnx.make_session` (0.3), `models.ensure_model` (0.2).
- Produces:

```python
# tracking.py
class Tracker:
    def __init__(self, fps: float = 5.0, lost_seconds: float = 2.0, activation: float = 0.30,
                 match_thr: float = 0.8): ...        # wraps supervision.ByteTrack(lost_track_buffer=int(lost_seconds*30), frame_rate=fps)
    def update(self, dets: list[Det]) -> list[tuple[int, Det]]: ...   # persons only (cls 0); returns (track_id, det)

# embed.py
def person_crop(frame, bbox, pad: float = 0.05) -> np.ndarray | None
def torso_crop(frame, bbox, kpts) -> np.ndarray | None   # shoulders→hips box (+15% pad); fallback rows 20%–55% of the box
def hsv_hist(crop_bgr, bins=(8, 4, 4)) -> np.ndarray      # 128-d, L2-normalised; zeros → returns zeros
class Embedder:
    def __init__(self, name: str = "dinov2_small", batch: int = 8): ...
    def __call__(self, crops: list[np.ndarray]) -> np.ndarray: ...  # (N,384) L2-normalised; RGB, 224×224, ImageNet mean/std;
                                                                     # use output "pooler_output" if present, else last_hidden_state[:,0]

# stack.py
@dataclass
class PerceptionStack:
    name: str                                                     # e.g. "yolox_tiny+rtmpose_s+dinov2" or "fake"
    detect: Callable[[np.ndarray], list[Det]]                     # all classes
    pose: Callable[[np.ndarray, np.ndarray], np.ndarray] | None   # (frame, boxes (N,4)) -> (N,17,3)
    embed: Callable[[list[np.ndarray]], np.ndarray] | None        # crops -> (N,D) L2-normalised
    def new_tracker(self, fps: float) -> Tracker
def default_stack(detector: str = DEFAULT_DETECTOR) -> PerceptionStack   # cached per detector name (module-level dict + lock)
    # spec §8: if the detector's weights can't be fetched (ModelUnavailable after 3 retries), fall back along
    # DETECTOR_FALLBACKS = {"yolox_tiny": "yolox_nano", "yolox_nano": "yolox_s", "yolox_s": "yolox_tiny", "rfdetr_nano": "yolox_tiny"}
    # and log a warning naming both models; if pose/embedder weights are unavailable the stack runs without them
    # (geometry falls back to box height, identity to HSV only)
def fake_stack() -> PerceptionStack
    # test double: persons = connected components of strongly saturated pixels (HSV S>120, V>60, area>400 px);
    # pose returns synthetic standing keypoints spread over each box (conf 0.9) so geometry runs "full";
    # embed = L2-normalised mean BGR + hsv_hist
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_tracking.py
from showroom_coach.perception.tracking import Tracker
from showroom_coach.types import Det


def test_two_movers_keep_ids():
    tr = Tracker(fps=5.0)
    ids = []
    for k in range(20):
        dets = [Det((100 + 5 * k, 100, 160 + 5 * k, 300), 0.9), Det((600, 100, 660, 300), 0.85),
                Det((10, 10, 40, 40), 0.9, cls=2)]           # a car is ignored
        out = tr.update(dets)
        assert len(out) == 2
        ids.append(tuple(sorted(i for i, _ in out)))
    assert len(set(ids[2:])) == 1


def test_short_occlusion_keeps_id():
    tr = Tracker(fps=5.0, lost_seconds=2.0)
    first = tr.update([Det((100, 100, 160, 300), 0.9)])[0][0]
    for _ in range(5):
        tr.update([])
    again = tr.update([Det((104, 100, 164, 300), 0.9)])
    assert again and again[0][0] == first
```

```python
# tests/test_stack.py
import numpy as np
from showroom_coach.perception.stack import fake_stack
from showroom_coach.perception.embed import hsv_hist, torso_crop


def test_fake_stack_finds_coloured_people():
    img = np.full((540, 960, 3), 128, np.uint8)
    img[100:320, 100:180] = (0, 0, 255)
    img[150:370, 600:680] = (255, 0, 0)
    st = fake_stack()
    dets = [d for d in st.detect(img) if d.cls == 0]
    assert len(dets) == 2
    k = st.pose(img, np.array([d.bbox for d in dets]))
    assert k.shape == (2, 17, 3)
    e = st.embed([img[100:320, 100:180], img[150:370, 600:680]])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1)
    assert float(e[0] @ e[1]) < 0.9


def test_hsv_hist_and_torso_crop_fallback():
    img = np.zeros((200, 100, 3), np.uint8); img[:, :] = (0, 200, 0)
    h = hsv_hist(img)
    assert h.shape == (128,) and abs(np.linalg.norm(h) - 1) < 1e-6
    crop = torso_crop(img, (0, 0, 100, 200), None)
    assert crop is not None and crop.shape[0] < 200
```

  Also a unit test `test_default_stack_falls_back_when_detector_missing`: monkeypatch `models.ensure_model` to raise
  `ModelUnavailable` for `yolox_tiny` only and stub the session constructors; assert the stack name contains `yolox_nano`.

  Slow test (in `tests/test_stack.py`, `@pytest.mark.slow`, skip if weights missing): `default_stack()` on a MEVA frame returns ≥1 person, pose `(N,17,3)`, embeddings `(N,384)` with unit norm.

- [ ] **Steps 2–4:** red → implement → green. **Step 5: Commit** `feat(yak-44): ByteTrack wrapper, DINOv2/HSV embedder and pluggable perception stack`

### Task 2.3: Perception pipeline with cache

**Files:**
- Create: `src/showroom_coach/perception/pipeline.py`, `tests/test_perception_pipeline.py`
- Modify: `src/showroom_coach/render.py` (create with `face_region` and `blur_faces` only — needed for blurred track crops; the rest comes in Task 3.1)

**Interfaces:**
- Consumes: `iter_frames`, `probe`, `VideoInfo` (2.1), `PerceptionStack`, `Tracker`, `person_crop`, `torso_crop`, `hsv_hist` (2.2).
- Produces:

```python
@dataclass
class PerceptionResult:
    video: VideoInfo
    stack_name: str
    frame_times: np.ndarray        # (F,) seconds of each sampled frame
    src_frame_idx: np.ndarray      # (F,) source frame index (for MEVA ground truth)
    obs_frame: np.ndarray          # (M,) sampled-frame index of each tracked observation
    obs_track: np.ndarray          # (M,) track id
    obs_bbox: np.ndarray           # (M,4) processed-frame pixels
    obs_score: np.ndarray          # (M,)
    obs_kpts: np.ndarray           # (M,17,3); conf 0 when pose is unavailable
    raw_frame: np.ndarray          # (R,) frame index of every person detection (for face blur)
    raw_bbox: np.ndarray           # (R,4)
    track_emb: dict[int, np.ndarray]     # identity descriptor: L2norm(concat(dino_mean, 0.5*hsv_full_mean))
    track_outfit: dict[int, np.ndarray]  # outfit descriptor: L2norm(concat(hsv_torso_mean, 0.5*dino_mean))
    crops_dir: str | None                # <dir>/<track_id>.jpg = face-blurred best crop of each track
    cars: list[list[float]]              # car boxes [x1,y1,x2,y2,score] from the first frame (used in M8)
    timings: dict[str, float]            # decode/detect/track/pose/embed/total seconds
    def track_ids(self) -> list[int]
    def obs_idx(self, track_id: int) -> np.ndarray

def run_perception(path, stack: PerceptionStack, sample_fps=5.0, max_width=960, max_seconds=300.0,
                   progress: Callable[[float, str], None] | None = None, crops_dir: str | Path | None = None,
                   crops_per_track: int = 4) -> PerceptionResult
def video_key(path, sample_fps, max_width, max_seconds, stack_name) -> str      # sha1(file bytes in 1 MB chunks + params)[:16]
def save_result(res: PerceptionResult, d: Path) -> None   # perception.npz (arrays) + perception.json (video, dict embeddings as lists, timings)
def load_result(d: Path) -> PerceptionResult
def cached_perception(path, stack, cache_root: Path, progress=None, **kw) -> tuple[PerceptionResult, bool]   # (result, was_cached)
# render.py (partial)
def face_region(bbox, kpts, kpt_thr: float = 0.3) -> tuple[int, int, int, int]
    # box around confident nose/eyes/ears expanded ×1.8 (min side 0.5·box width); fallback = top 20% of the box (full width)
def blur_faces(frame, bboxes, kpts_list) -> np.ndarray       # returns a copy; strong Gaussian blur (or pixelate) in each face region
```

Per frame: `dets = stack.detect(frame)`; persons → `raw_*`; first frame cars (cls 2, 5, 7) → `cars`; `tracker.update(persons)`; `kpts = stack.pose(frame, boxes)` on tracked boxes; keep per track the top-`crops_per_track` observations by `area × score` with their crops (person crop and torso crop, each downscaled to ≤ 160 px tall). After the pass: embed all kept crops in batches; write the best crop per track, face-blurred, to `crops_dir`. Progress callback every ~10 frames with fraction `t / duration`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from showroom_coach.perception.pipeline import cached_perception, load_result, run_perception, save_result
from showroom_coach.perception.stack import fake_stack
from showroom_coach.render import blur_faces, face_region


def test_pipeline_on_synthetic_video(synth_video, tmp_path):
    seen = []
    res = run_perception(synth_video, fake_stack(), progress=lambda f, m: seen.append(f), crops_dir=tmp_path / "crops")
    assert len(res.frame_times) == 30
    ids = res.track_ids()
    assert len(ids) == 2                                    # red mover + blue stander, stable ids
    assert all(res.obs_idx(i).size >= 20 for i in ids)
    assert set(res.track_emb) == set(ids) and set(res.track_outfit) == set(ids)
    assert all((tmp_path / "crops" / f"{i}.jpg").exists() for i in ids)
    assert seen and seen[-1] <= 1.0


def test_save_load_roundtrip(synth_video, tmp_path):
    res = run_perception(synth_video, fake_stack())
    save_result(res, tmp_path / "c")
    back = load_result(tmp_path / "c")
    assert np.array_equal(back.obs_bbox, res.obs_bbox) and back.video == res.video
    assert set(back.track_emb) == set(res.track_emb)


def test_cache_hit_skips_detection(synth_video, tmp_path):
    st = fake_stack()
    calls = {"n": 0}
    det = st.detect
    st.detect = lambda f: (calls.__setitem__("n", calls["n"] + 1), det(f))[1]
    _, cached1 = cached_perception(synth_video, st, tmp_path)
    n1 = calls["n"]
    _, cached2 = cached_perception(synth_video, st, tmp_path)
    assert (cached1, cached2) == (False, True) and calls["n"] == n1


def test_face_region_from_keypoints_and_fallback():
    k = np.zeros((17, 3)); k[0] = (50, 30, 0.9); k[1] = (45, 25, 0.9); k[2] = (55, 25, 0.9)
    x1, y1, x2, y2 = face_region((20, 10, 80, 200), k)
    assert x1 <= 45 and x2 >= 55 and y1 <= 25 and y2 >= 30
    assert face_region((20, 10, 80, 210), None) == (20, 10, 80, 50)


def test_blur_changes_only_face():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)
    out = blur_faces(img, [(20, 10, 80, 210)], [None])
    assert out[10:50, 20:80].std() < img[10:50, 20:80].std() * 0.6
    assert np.array_equal(out[100:, :], img[100:, :])
```

  Slow test: `run_perception` with `default_stack()` on the first 10 s of the first MEVA clip → ≥ 1 track whose observations span ≥ 5 s (stable IDs).

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Real run** — time `run_perception` on a full MEVA clip with the chosen detector; record per-stage timings and the real-time factor in PROGRESS.md.
- [ ] **Step 6: Commit** `feat(yak-44): perception pipeline (detect, track, pose, embed) with per-video cache`

---

# M3 — Offline analysis, rendering and the `analyze` CLI

### Task 3.1: Analysis pass (persons, geometry, roles, events, metrics, re-score)

**Files:**
- Create: `src/showroom_coach/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: everything from M1, `PerceptionResult` (2.3).
- Produces:

```python
@dataclass
class Analysis:
    preset: str
    thresholds: Thresholds
    duration: float
    persons: list[Person]
    suggestions: dict[int, RoleSuggestion]
    roles: dict[int, str]               # final: user override, else suggestion ("unsure" → "customer")
    overrides: dict[int, str]           # what the user set
    names: dict[int, str]
    events: list[Event]
    scorecard: Scorecard
    timelines: dict[int, dict]          # pid -> {"present": [[t0,t1],...], "waiting": [[...]], "engaged": [[...]], "greeted": t|None}
    warnings: list[str]
    approx_share: float                 # share of observations with box-fallback distance
    # per-observation arrays aligned with PerceptionResult.obs_*:
    obs_pid: np.ndarray                 # (M,) person id, 0 = dropped (noise track)
    obs_xz: np.ndarray                  # (M,2) metres
    obs_src: np.ndarray                 # (M,) source code 0 full,1 torso,2 box,3 floor
    obs_facing: np.ndarray              # (M,2), nan when unknown
    obs_edge: np.ndarray                # (M,) bool
    features: dict[int, dict]           # role features per person
    def to_dict(self) -> dict           # JSON-able (no per-observation arrays); includes persons, suggestions, roles, names,
                                        # events, scorecard, timelines, warnings, preset, thresholds, duration

def build_persons(res, merge_thr: float = 0.30, min_track_obs: int = 3, min_track_s: float = 1.0) -> tuple[list[Person], np.ndarray]
def compute_geometry(res, camera: CameraConfig = CameraConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def person_timeline(res, obs_pid, obs_xz, obs_facing, obs_edge, roles) -> list[tuple[float, list[PersonState]]]
def analyze(res, preset: str = "short", roles: dict[int, str] | None = None, names: dict[int, str] | None = None,
            camera: CameraConfig = CameraConfig(), weights: RoleWeights = RoleWeights()) -> Analysis
def rescore(prev: Analysis, res, roles: dict[int, str] | None = None, names: dict[int, str] | None = None,
            preset: str | None = None) -> Analysis     # reuses persons, geometry, features and suggestions; re-runs events + metrics only
```

Rules:
- `build_persons`: drop tracks with `< min_track_obs` observations or shorter than `min_track_s` (their `obs_pid` = 0); `merge_tracks` on `TrackSummary(track_id, t0, t1, res.track_emb[id], n_obs)`; Person.thumb = `"{best_track_id}.jpg"` of the person's longest track.
- One observation per person per frame: if merging produced two in the same frame, keep the higher score.
- Features: `interaction_features` on person trajectories (`weights.contact_dist`, `weights.contact_min_s`), `presence` over `res.frame_times`, `outfit_peers` on the mean `track_outfit` of each person with `weights.outfit_sim_thr`.
- Warnings (exact strings, tested): `"nobody detected — the camera may be too far or too dark"`, `"fewer than 2 people — store metrics only"`, `"nobody marked as staff — mark staff to get scorecards"`, `"distance is approximate for {p}% of samples"` (when `approx_share > 0.5`), `"video trimmed to the first {n} s"`.
- `timelines`: present intervals from observations (gaps > `leave_after` split intervals), waiting = `[enter, first greet or leave]` for visits with known entry, engaged = engagement intervals, greeted = first greet t.

- [ ] **Step 1: Failing tests** (use `fake_stack` + synthetic video with movers designed so that the red mover walks up to the blue stander and stays close; build with `make_video`)

```python
import numpy as np
import pytest
from showroom_coach.analysis import analyze, rescore
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack


@pytest.fixture(scope="module")
def meet_res(meet_video):
    # meet_video (conftest): 20 s, 30 fps, 1280x720; blue "staff" box (80x220) standing at x=600..680 the whole time;
    # red "customer" box (80x220) enters from the left edge at 4 s, walks until its box is x=470..550 at 8 s (a 50 px gap —
    # boxes must never touch or the fake detector merges them), stays until 18 s, then walks out left (gone by 20 s).
    return run_perception(meet_video, fake_stack())


def test_analyze_finds_two_people_and_greet(meet_res):
    an = analyze(meet_res, preset="short")
    assert len(an.persons) == 2
    blue = max(an.persons, key=lambda p: p.t_last - p.t_first).pid
    an2 = rescore(an, meet_res, roles={blue: "staff"})
    types = [e.type for e in an2.events]
    assert "greet" in types and "engagement" in types
    assert an2.scorecard.staff[0].staff == blue and an2.scorecard.staff[0].greets == 1


def test_rescore_is_fast_and_does_not_touch_perception(meet_res, monkeypatch):
    import showroom_coach.analysis as A
    an = analyze(meet_res, preset="short")
    monkeypatch.setattr(A, "merge_tracks", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-merged")))
    an2 = rescore(an, meet_res, roles={an.persons[0].pid: "staff"}, preset="dealership")
    assert an2.preset == "dealership" and an2.persons == an.persons


def test_analyze_no_people(tmp_path, make_video):
    v = make_video(tmp_path / "empty.avi", seconds=3, movers=[])
    an = analyze(run_perception(v, fake_stack()), preset="short")
    assert an.persons == [] and an.events == []
    assert "nobody detected — the camera may be too far or too dark" in an.warnings
    assert an.scorecard.store.customers == 0


def test_nobody_staff_warning(meet_res):
    an = analyze(meet_res, preset="short", roles={p: "customer" for p in [1, 2]})
    assert "nobody marked as staff — mark staff to get scorecards" in an.warnings


def test_approx_warning(meet_res):
    res = meet_res
    saved = res.obs_kpts.copy()
    try:
        res.obs_kpts[..., 2] = 0.0             # no keypoints → box fallback everywhere
        an = analyze(res, preset="short")
        assert an.approx_share == pytest.approx(1.0)
        assert any(w.startswith("distance is approximate") for w in an.warnings)
    finally:
        res.obs_kpts[:] = saved


def test_to_dict_is_json(meet_res):
    import json
    json.dumps(analyze(meet_res, preset="short").to_dict())
```

- [ ] **Steps 2–4:** red → implement (add `meet_video` + `make_video` fixtures to conftest) → green.
- [ ] **Step 5: Commit** `feat(yak-44): offline analysis pass with identity merge, staff suggestion and fast re-score`

### Task 3.2: Rendering and export

**Files:**
- Modify: `src/showroom_coach/render.py`
- Create: `src/showroom_coach/export.py`, `tests/test_render.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `Analysis` (3.1), `PerceptionResult` (2.3), `iter_frames` (2.1).
- Produces:

```python
# render.py
STAFF_BGR = (230, 120, 20)      # blue
CUSTOMER_BGR = (0, 140, 255)    # orange
class H264Writer:               # context manager around imageio_ffmpeg.write_frames(path, (w,h), pix_fmt_in="bgr24",
    def __init__(self, path, size, fps): ...   # codec="libx264", fps=fps, macro_block_size=2,
    def write(self, frame_bgr): ...            # output_params=["-preset","veryfast","-crf","26","-movflags","+faststart"])
def draw_overlay(frame, people: list[dict], pairs: list[tuple], hud: str, alert: str | None) -> np.ndarray
    # people dicts: {"bbox", "role", "label", "state", "wait_s"}; used by both render_video and the live page tests
def render_video(video_path, res, an, out_path, fps_out: float = 10.0, progress=None) -> Path
    # decodes the source at fps_out (same sampling rule as iter_frames, same trimming), linearly interpolates each
    # person's bbox/kpts between neighbouring sampled frames (≤ 0.5 s apart), blurs faces of EVERY person detection
    # (raw boxes of the nearest sampled frame + tracked boxes with keypoints) BEFORE drawing, draws boxes/labels/
    # engaged lines/waiting timers/greeted ticks, a HUD (time, preset) and a 3 s banner after each alert event
def timeline_png(an, out_path, width: int = 1200) -> Path      # one row per person: present (grey), waiting (orange),
                                                              # engaged (blue), greet tick, alert markers; matplotlib not required (cv2 drawing)
def contact_sheets(video_path, res, an, out_dir, max_events: int = 12) -> list[Path]   # one jpg per greet/engagement/wait_alert/walkout:
                                                              # 3 face-blurred frames (t−2 s, t, t+2 s) side by side with a caption
def save_thumbnails(res, an, out_dir) -> dict[int, str]       # copies the person's best blurred crop to <out_dir>/<pid>.jpg

# export.py
def results_dict(an, extra: dict | None = None) -> dict       # an.to_dict() + extra (urls, video info)
def write_events_csv(an, path) -> Path    # columns: type,t,t_end,customer,staff,staff_name,detail
def write_scorecard_csv(an, path) -> Path # one row per staff + a final "STORE" row
def write_all(an, out_dir, extra=None) -> dict[str, Path]     # results.json, events.csv, scorecard.csv
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_render.py
import cv2, numpy as np
from showroom_coach.analysis import analyze, rescore
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack
from showroom_coach.render import contact_sheets, draw_overlay, render_video, timeline_png


def test_render_video_is_h264_and_playable(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"})
    out = render_video(meet_video, res, an, tmp_path / "annotated.mp4", fps_out=10)
    head = out.read_bytes()[:4096]
    assert b"ftyp" in head and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 150
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore").lower()
    assert fourcc in ("avc1", "h264")


def test_timeline_and_contact_sheets(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"})
    assert cv2.imread(str(timeline_png(an, tmp_path / "timeline.png"))) is not None
    sheets = contact_sheets(meet_video, res, an, tmp_path / "sheets")
    assert sheets and all(p.exists() for p in sheets)


def test_draw_overlay_colours():
    img = np.zeros((200, 300, 3), np.uint8)
    out = draw_overlay(img, [{"bbox": (10, 10, 60, 150), "role": "staff", "label": "S1", "state": "free", "wait_s": None}],
                       [], "t=0", None)
    assert tuple(out[80, 10]) == (230, 120, 20)
```

```python
# tests/test_export.py
import csv, json
from showroom_coach.analysis import analyze, rescore
from showroom_coach.export import write_all
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack


def test_write_all(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"}, names={1: "Aziz"})
    files = write_all(an, tmp_path)
    data = json.loads(files["results.json"].read_text())
    assert data["names"]["1"] == "Aziz" and "scorecard" in data
    rows = list(csv.DictReader(open(files["scorecard.csv"], newline="")))
    assert rows[-1]["staff"] == "STORE" and rows[0]["name"] == "Aziz"
    ev = list(csv.DictReader(open(files["events.csv"], newline="")))
    assert {"type", "t", "customer", "staff"} <= set(ev[0])
```

- [ ] **Steps 2–4:** red → implement → green. **Step 5: Commit** `feat(yak-44): face-blurred H.264 rendering, timeline, contact sheets and CSV/JSON export`

### Task 3.3: `analyze` CLI end-to-end

**Files:**
- Modify: `src/showroom_coach/cli.py`
- Create: `tests/test_cli_analyze.py`

**Interfaces:**
- Produces: `showroom-coach analyze VIDEO --out DIR [--preset short|dealership] [--roles roles.json] [--fps 5] [--max-seconds 300] [--detector NAME] [--fake] [--no-video] [--cache DIR]`.
  - Writes `DIR/annotated.mp4` (unless `--no-video`), `timeline.png`, `results.json`, `events.csv`, `scorecard.csv`, `people/<pid>.jpg`, `contact_sheets/*.jpg`; prints a one-screen summary (persons, suggested roles, events per type, timings, real-time factor).
  - `roles.json`: `{"roles": {"3": "staff"}, "names": {"3": "Aziz"}}`; without it, the suggested roles are used.
  - `--fake` uses `fake_stack()` (tests only).
  - Exit code 2 with a clear message for an unreadable video.

- [ ] **Step 1: Failing test**

```python
import json
from showroom_coach.cli import main


def test_analyze_cli_writes_every_output(meet_video, tmp_path):
    roles = tmp_path / "roles.json"; roles.write_text(json.dumps({"roles": {"1": "staff"}, "names": {"1": "Aziz"}}))
    out = tmp_path / "out"
    assert main(["analyze", str(meet_video), "--out", str(out), "--fake", "--roles", str(roles),
                 "--cache", str(tmp_path / "cache")]) == 0
    for name in ["annotated.mp4", "timeline.png", "results.json", "events.csv", "scorecard.csv"]:
        assert (out / name).exists(), name
    assert list((out / "people").glob("*.jpg")) and list((out / "contact_sheets").glob("*.jpg"))


def test_analyze_cli_bad_video(tmp_path, capsys):
    bad = tmp_path / "bad.mp4"; bad.write_text("x")
    assert main(["analyze", str(bad), "--out", str(tmp_path / "o"), "--fake"]) == 2
    assert "cannot read video" in capsys.readouterr().err.lower()
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Real runs** — make two sample clips with imageio-ffmpeg: `data/samples/meva_a_60s.mp4` and `data/samples/meva_b_60s.mp4` (60 s cuts from two different MEVA clips with talk events, re-encoded H.264 960 px wide) and run `analyze` with the real stack on both; paste the summaries into PROGRESS.md; look at a few output frames (blur present, colours, labels).
- [ ] **Step 6: M3 wrap-up** — full suite, PROGRESS entry, commit `feat(yak-44): analyze CLI producing annotated video, scorecards and exports`.

---

# M4 — Web app: upload → progress → confirm staff → results → re-score

### Task 4.1: Job manager and REST API

**Files:**
- Create: `src/showroom_coach/app/__init__.py`, `src/showroom_coach/app/jobs.py`, `src/showroom_coach/app/main.py`, `tests/test_app.py`
- Modify: `src/showroom_coach/cli.py` (add `serve [--host 127.0.0.1] [--port 8000] [--fake]`)

**Interfaces:**
- Consumes: `cached_perception` (2.3), `analyze`/`rescore` (3.1), `render_video`, `timeline_png`, `contact_sheets`, `save_thumbnails` (3.2), `write_all` (3.2), `probe`/`VideoError` (2.1).
- Produces:

```python
# jobs.py
@dataclass
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
    def public(self) -> dict

class JobManager:
    def __init__(self, root: Path, stack_factory: Callable[[], PerceptionStack], max_age_h: float = 24.0): ...
    def cleanup(self) -> int                                   # deletes job dirs older than max_age_h; returns count
    def create(self, filename: str, data_path: Path, preset: str) -> Job   # starts a worker thread
    def get(self, job_id: str) -> Job                          # KeyError if unknown
    def analysis(self, job_id: str) -> Analysis                 # latest analysis (needs_confirm or later)
    def set_roles(self, job_id: str, roles: dict[int, str], names: dict[int, str], preset: str | None) -> Analysis
        # rescore (no perception), write exports + timeline, then render video in a background thread
    def results(self, job_id: str) -> dict
```

Worker: `probe` (fail fast on unreadable files) → `cached_perception` (cache in `<job>/cache`, progress mapped to 0–0.9) → `analyze` with suggested roles → thumbnails → status `needs_confirm`. Any exception → `failed` with `error = str(exc)` (never crash the server). Only one perception job runs at a time (a module-level semaphore) so the CPU isn't oversubscribed.

API (`create_app(stack_factory=default_stack, jobs_root=None) -> FastAPI`):

| Method | Path | Body / result |
|---|---|---|
| GET | `/` | `static/index.html` |
| GET | `/live` | `static/live.html` |
| GET | `/api/presets` | `{name: {label, thresholds}}` |
| POST | `/api/jobs` | multipart `file`, form `preset` → `{"job_id"}` (415 for extensions other than mp4/mov/avi/mkv/webm/m4v; 400 for an unknown preset) |
| GET | `/api/jobs/{id}` | `Job.public()` (404 if unknown) |
| GET | `/api/jobs/{id}/people` | `{"people": [{pid, thumb_url, suggestion: {role, confidence, reasons}, role, name, t_first, t_last}]}` (409 before `needs_confirm`) |
| POST | `/api/jobs/{id}/roles` | JSON `{"roles": {"3": "staff"}, "names": {"3": "Aziz"}, "preset": "short"}` → results dict |
| GET | `/api/jobs/{id}/results` | results dict: `results_dict(an)` + `video_status`, `video_url`, `timeline_url`, `export` urls |
| GET | `/api/jobs/{id}/files/{name:path}` | whitelisted: `annotated.mp4`, `timeline.png`, `results.json`, `events.csv`, `scorecard.csv`, `people/<int>.jpg`, `contact_sheets/<file>.jpg`; anything else 404 |
| WS | `/ws/live` | Task 5.1 |

- [ ] **Step 1: Failing tests** (`tests/test_app.py`, all with `fake_stack`, synthetic `meet_video`, and `tmp_path` jobs root)

```python
import time
import pytest
from fastapi.testclient import TestClient
from showroom_coach.app.main import create_app
from showroom_coach.perception.stack import fake_stack


@pytest.fixture
def client(tmp_path):
    app = create_app(stack_factory=fake_stack, jobs_root=tmp_path / "jobs")
    with TestClient(app) as c:
        yield c


def wait_status(c, jid, want, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get(f"/api/jobs/{jid}").json()
        if s["status"] in want:
            return s
        time.sleep(0.2)
    raise AssertionError(f"timeout, last={s}")


def upload(c, path, preset="short"):
    with open(path, "rb") as f:
        r = c.post("/api/jobs", files={"file": (path.name, f, "video/x-msvideo")}, data={"preset": preset})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_upload_confirm_results(client, meet_video):
    jid = upload(client, meet_video)
    s = wait_status(client, jid, {"needs_confirm", "failed"})
    assert s["status"] == "needs_confirm"
    people = client.get(f"/api/jobs/{jid}/people").json()["people"]
    assert len(people) == 2 and all(p["suggestion"]["reasons"] for p in people)
    assert client.get(people[0]["thumb_url"]).status_code == 200
    staff = max(people, key=lambda p: p["t_last"] - p["t_first"])["pid"]
    r = client.post(f"/api/jobs/{jid}/roles", json={"roles": {str(staff): "staff"}, "names": {str(staff): "Aziz"}})
    assert r.status_code == 200
    res = r.json()
    assert res["scorecard"]["staff"][0]["name"] == "Aziz"
    wait_status(client, jid, {"done"})
    assert client.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/events.csv").status_code == 200


def test_rescore_does_not_redetect(client, meet_video, monkeypatch):
    jid = upload(client, meet_video)
    wait_status(client, jid, {"needs_confirm"})
    import showroom_coach.perception.pipeline as P
    monkeypatch.setattr(P, "run_perception", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-detect")))
    t0 = time.time()
    r1 = client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff"}})
    r2 = client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "customer", "2": "staff"}, "preset": "dealership"})
    assert r1.status_code == r2.status_code == 200
    assert time.time() - t0 < 5
    assert r2.json()["roles"]["2"] == "staff" and r2.json()["preset"] == "dealership"


def test_upload_garbage_fails_cleanly(client, tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_text("definitely not a video")
    jid = upload(client, bad)
    s = wait_status(client, jid, {"failed", "needs_confirm"})
    assert s["status"] == "failed" and "video" in s["error"].lower()
    assert client.get("/api/presets").status_code == 200        # server still alive


def test_rejects_wrong_extension_and_unknown_preset(client, tmp_path):
    p = tmp_path / "x.txt"; p.write_text("x")
    with open(p, "rb") as f:
        assert client.post("/api/jobs", files={"file": ("x.txt", f)}, data={"preset": "short"}).status_code == 415
    with open(p, "rb") as f:
        assert client.post("/api/jobs", files={"file": ("x.mp4", f)}, data={"preset": "zzz"}).status_code == 400


def test_file_whitelist(client, meet_video):
    jid = upload(client, meet_video)
    wait_status(client, jid, {"needs_confirm"})
    assert client.get(f"/api/jobs/{jid}/files/../../pyproject.toml").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/cache/perception.npz").status_code == 404


def test_cleanup_removes_old_jobs(tmp_path):
    import os
    from showroom_coach.app.jobs import JobManager
    old = tmp_path / "jobs" / "old"; old.mkdir(parents=True)
    os.utime(old, (time.time() - 30 * 3600,) * 2)
    assert JobManager(tmp_path / "jobs", fake_stack).cleanup() == 1 and not old.exists()
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Commit** `feat(yak-44): FastAPI job API with background perception, confirm and instant re-score`

### Task 4.2: Static frontend (upload, progress, confirm gallery, results dashboard)

**Files:**
- Create: `src/showroom_coach/app/static/index.html`, `app.js`, `style.css`
- Modify: `tests/test_app.py` (add `test_index_served`)

Requirements (no build step, vanilla JS, works offline — no CDN):
- Header with product name and a permanent banner: "Coaching support only — not an automatic judgement of staff (Uzbekistan AI law ZRU-1115). Faces are blurred; processing stays on this computer. Staff must be informed that analytics is in use."
- **Upload view:** drag-and-drop/file input, preset selector (labels from `/api/presets`), "Analyse" button; shows a warning when the server reports trimming.
- **Progress view:** progress bar + message from polling `/api/jobs/{id}` every 700 ms.
- **Confirm view:** grid of person cards (blurred thumbnail, suggested role chip with confidence %, up to 3 reasons, a role toggle button "Staff ⇄ Customer" (one click), a name input for staff), "Looks right →" button posting roles.
- **Results view:** annotated video (shown when `video_status == ready`, otherwise "rendering…" with polling), a timeline drawn with divs from `timelines` (rows per person: present grey, waiting orange, engaged blue, greet tick), scorecard table per salesperson + store summary tiles (greet rate, median time to greet, walk-outs, unattended waiting), alert feed (missed greet / long wait while staff free / walk-out, each with time and free staff names), export links (JSON, events CSV, scorecard CSV), "Change roles" (back to the confirm view, re-score in place), warnings as badges (approximate distance, nobody detected, nobody staff).
- Times shown as m:ss. Staff colour blue, customers orange (same as the video).

- [ ] **Step 1: Failing test**

```python
def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ZRU-1115" in r.text and "app.js" in r.text
    assert client.get("/static/app.js").status_code == 200
```

- [ ] **Step 2–3:** implement → green.
- [ ] **Step 4: Manual smoke** — `showroom-coach serve` in the background; drive it with the Playwright MCP tools (upload `data/samples/meva_a_60s.mp4`, confirm, check results, screenshot to `runs/screens/`). Fix what looks broken.
- [ ] **Step 5: Commit** `feat(yak-44): upload, confirm-staff gallery and results dashboard frontend`

### Task 4.3: `run.bat` / `run.sh` and the M4 smoke test

**Files:**
- Create: `run.bat`, `run.sh`, `scripts/smoke_api.py`

- `run.bat`: creates `.venv` with the Python 3.12 path if missing (falls back to `py -3.12`), installs `-e .` if `showroom_coach` isn't importable, runs `showroom-coach fetch-models` for the default stack if weights are missing, then `.venv\Scripts\python -m showroom_coach serve --host 127.0.0.1 --port 8000`, and prints the URL.
- `run.sh`: same for bash.
- `scripts/smoke_api.py URL VIDEO`: real HTTP smoke test (upload → poll → people → roles with the suggested staff → poll until done → fetch video and CSV) using `httpx`; prints timings. Used in verification.

- [ ] **Step 1:** write the scripts. **Step 2:** start `run.bat` in the background (Bash tool `run_in_background`, `cmd //c run.bat`), wait until `curl -s localhost:8000/api/presets` answers, run `scripts/smoke_api.py http://127.0.0.1:8000 data/samples/meva_a_60s.mp4`, paste the output into PROGRESS.md. Stop the server.
- [ ] **Step 3: M4 wrap-up** — full suite, PROGRESS entry, commit `feat(yak-44): run scripts and API smoke test`.

---

# M5 — Live mode

### Task 5.1: LiveSession engine and `/ws/live`

**Files:**
- Create: `src/showroom_coach/live.py`, `tests/test_live.py`
- Modify: `src/showroom_coach/app/main.py`

**Interfaces:**
- Consumes: `PerceptionStack`, `Tracker` (2.2), `LiveIdentity` (1.5), geometry (1.1), `EventEngine` (1.2), roles (1.4), `decode_jpeg` (2.1), `blur_faces` (2.3).
- Produces:

```python
class LiveSession:
    def __init__(self, stack: PerceptionStack, preset: str = "short", camera: CameraConfig = CameraConfig(),
                 suggest_after: float = 30.0, suggest_every: float = 5.0, embed_every: int = 5): ...
    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> dict: ...
    def toggle(self, pid: int) -> str: ...            # flips staff ⇄ customer as a user override; returns new role
    def set_name(self, pid: int, name: str) -> None: ...
    def set_preset(self, name: str) -> None: ...       # rebuilds the EventEngine, keeps identities and roles
    def summary(self) -> dict: ...                     # live scorecard (metrics.scorecard over events so far)
```

`process` returns (all coordinates in the processed frame; the client scales):

```json
{"t": 12.4, "w": 960, "h": 540, "fps": 4.8,
 "people": [{"pid": 3, "bbox": [x1,y1,x2,y2], "face": [x1,y1,x2,y2], "role": "staff", "suggested": "staff",
             "confidence": 0.81, "override": false, "state": "free", "wait_s": null, "label": "S3 Aziz"}],
 "pairs": [[3, 5, "engaged"]],
 "events": [{"type": "greet", "t": 11.0, "customer": 5, "staff": 3, "data": {}}],
 "alerts": ["0:42 long wait: customer C7 (free: Aziz)"],
 "store": {"customers": 2, "greeted": 1, "walkouts": 0}}
```

- `t` defaults to `time.monotonic() − start`; tests pass explicit `t`.
- fps = exponential moving average of processing rate.
- Embedding: a new track gets an embedding on its first frame (for re-attach); each person's embedding is refreshed every `embed_every` frames (running mean).
- Auto-suggest: once `t ≥ suggest_after`, every `suggest_every` s recompute features over the last 5 min of trajectories and call `suggest_roles`; roles without a user override follow the suggestion (`unsure` → customer). Before that everyone is a customer with `suggested = "unsure"`.
- `face` is the face region to blur on the client canvas (the server never sends frames back).

WebSocket protocol `/ws/live?preset=short`:
- client → server **binary** = one JPEG frame; server replies with one JSON overlay (text). The client sends the next frame only after the reply (stop-and-wait ⇒ old frames are dropped on the client, never queued on the server).
- client → server **text** JSON control: `{"type":"toggle","pid":3}`, `{"type":"name","pid":3,"name":"Aziz"}`, `{"type":"preset","name":"dealership"}`, `{"type":"time","t":1.2}` (overrides the timestamp of the next frame — used by the file-fed test), `{"type":"summary"}` → server replies `{"type":"summary", ...}`.
- Bad JPEG → `{"type":"error","error":"bad frame"}` and the session continues.
- Each connection has its own `LiveSession`; models are shared (`stack_factory()` once per app).

- [ ] **Step 1: Failing tests**

```python
import cv2, json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from showroom_coach.app.main import create_app
from showroom_coach.live import LiveSession
from showroom_coach.perception.stack import fake_stack
from showroom_coach.sources import iter_frames


def test_session_tracks_and_toggles(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    out = None
    for _, t, frame in iter_frames(meet_video, sample_fps=5):
        out = s.process(frame, t)
    assert out["people"] and all(p["role"] in ("staff", "customer") for p in out["people"])
    pid = out["people"][0]["pid"]
    assert s.toggle(pid) == "staff"
    frame = next(iter_frames(meet_video))[2]
    again = s.process(frame, out["t"] + 0.2)
    assert [p for p in again["people"] if p["pid"] == pid][0]["override"] is True


def test_session_emits_events_with_staff(meet_video):
    s = LiveSession(fake_stack(), preset="short", suggest_after=1e9)
    frames = list(iter_frames(meet_video, sample_fps=5))
    first = s.process(frames[0][2], frames[0][1])
    s.toggle(first["people"][0]["pid"])           # the blue stander is the only person at t=0
    events = []
    for _, t, f in frames[1:]:
        events += s.process(f, t)["events"]
    assert any(e["type"] == "greet" for e in events)


def _jpeg(frame):
    return cv2.imencode(".jpg", frame)[1].tobytes()


def test_ws_file_fed(meet_video, tmp_path):
    app = create_app(stack_factory=fake_stack, jobs_root=tmp_path / "jobs")
    with TestClient(app) as c, c.websocket_connect("/ws/live?preset=short") as ws:
        n = 0
        for _, t, frame in iter_frames(meet_video, sample_fps=5):
            ws.send_text(json.dumps({"type": "time", "t": t}))
            ws.send_bytes(_jpeg(frame))
            msg = ws.receive_json()
            n += 1
            assert "people" in msg and "fps" in msg
        assert n == 100
        ws.send_bytes(b"garbage")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "summary"}))
        assert ws.receive_json()["type"] == "summary"


def test_ws_disconnect_and_reconnect(meet_video, tmp_path):
    app = create_app(stack_factory=fake_stack, jobs_root=tmp_path / "jobs")
    frame = next(iter_frames(meet_video))[2]
    with TestClient(app) as c:
        with c.websocket_connect("/ws/live") as ws:
            ws.send_bytes(_jpeg(frame)); ws.receive_json()
        with c.websocket_connect("/ws/live") as ws:          # new session after a drop
            ws.send_bytes(_jpeg(frame))
            assert "people" in ws.receive_json()
```

- [ ] **Steps 2–4:** red → implement → green.
- [ ] **Step 5: Commit** `feat(yak-44): live session engine and /ws/live WebSocket`

### Task 5.2: Live page and real-device checks

**Files:**
- Create: `src/showroom_coach/app/static/live.html`, `live.js`; `scripts/live_file_client.py`; `scripts/webcam_grab.py`
- Modify: `src/showroom_coach/cli.py` (add `webcam-check`), `tests/test_app.py` (add `test_live_page_served`)

Requirements:
- `live.html`: start/stop button, preset selector, `<video>` (hidden) + `<canvas>` showing the camera image with **face regions blurred on the canvas** (draw the frame, then blur the `face` rects via `ctx.filter = "blur(12px)"` on clipped regions), boxes coloured by role, labels, waiting timers, engaged lines, processed fps, a live event/alert feed and a mini scorecard (refresh `summary` every 5 s). Clicking inside a box sends `toggle` for that pid. Auto-reconnect with backoff when the socket closes. getUserMedia `{video: {width: 960}}`; frames are drawn to an offscreen canvas at ≤ 960 px wide and sent as JPEG quality 0.7, max 5 fps, stop-and-wait. Works on `http://localhost` (secure context) without HTTPS. Same ZRU-1115 banner.
- `scripts/live_file_client.py URL VIDEO [--fps 5]`: real WebSocket client (package `websockets`, already installed with uvicorn[standard]) streaming a video file in real time; prints processed fps, people per frame and events; used for verification against the real stack.
- `scripts/webcam_grab.py` / `showroom-coach webcam-check`: opens the "Web Camera" device with `cv2.VideoCapture(0, cv2.CAP_DSHOW)` (fallback MSMF, indices 0–2), grabs one frame, saves a **blurred, downscaled** copy to `runs/webcam_grab.jpg`, prints resolution and mean brightness; then runs the real `LiveSession` on that frame once and prints the overlay JSON.

- [ ] **Step 1: Failing test**

```python
def test_live_page_served(client):
    r = client.get("/live")
    assert r.status_code == 200 and "live.js" in r.text and "ZRU-1115" in r.text
```

- [ ] **Step 2:** implement page + scripts → test green.
- [ ] **Step 3: Real checks** — (a) server running with the real stack; `scripts/live_file_client.py ws://127.0.0.1:8000/ws/live data/samples/meva_a_60s.mp4` → record processed fps and number of people tracked; (b) `showroom-coach webcam-check` → record the output (nobody will be in front of the camera at night — that's fine, the check is that capture works); (c) open `/live` with Playwright to confirm the page loads without JS errors (the headless browser has no camera; note this).
- [ ] **Step 4: M5 wrap-up** — full suite, PROGRESS entry with live fps numbers, commit `feat(yak-44): live webcam page with click-to-toggle roles and live alerts`.

---

# M6 — Data and evaluation

### Task 6.1: MEVA KPF parsing

**Files:**
- Create: `src/showroom_coach/evaluation/__init__.py`, `src/showroom_coach/evaluation/kpf.py`, `tests/test_kpf.py`, `tests/data/mini.geom.yml`, `tests/data/mini.activities.yml`, `tests/data/mini.types.yml` (tiny hand-written samples in the real format — the format the scout reported, copied from real files, a handful of lines each, attributed to MEVA CC BY 4.0)

**Interfaces (Produces):**

```python
@dataclass
class GeomBox:
    actor: int
    frame: int
    bbox: tuple[float, float, float, float]    # x1, y1, x2, y2 in source pixels

@dataclass
class Activity:
    name: str
    act_id: int
    f0: int
    f1: int
    actors: list[int]

def load_types(path) -> dict[int, str]                   # actor id -> "Person" | "Vehicle" | ...
def load_geom(path, actors: set[int] | None = None) -> dict[int, list[GeomBox]]   # frame -> boxes (regex line parser, fast)
def load_activities(path) -> list[Activity]
def person_ids(types: dict[int, str]) -> set[int]
```

- [ ] **Steps:** failing tests on the mini files (counts, one exact box, one exact activity with its actors and frame span) → implement → green → commit `feat(yak-44): MEVA KPF annotation parsing`.

### Task 6.2: MEVA evaluation (tracking, engagement vs talk, enter/exit)

**Files:**
- Create: `src/showroom_coach/evaluation/meva_eval.py`, `tests/test_meva_eval.py`
- Modify: `src/showroom_coach/cli.py` (add `eval-meva`)

**Interfaces (Produces):**

```python
def match_ids(res, an_or_none, gt_boxes: dict[int, list[GeomBox]], use_persons: bool, iou_thr: float = 0.3) -> dict[int, int]
    # our id (track or person) -> GT actor by majority vote of per-frame IoU matches (boxes scaled by res.video.scale)
def tracking_metrics(res, an, gt_boxes, use_persons: bool) -> dict     # motmetrics: mota, idf1, num_switches, precision, recall
def pair_intervals(an, dist: float, min_dur: float, gap_tol: float) -> list[tuple[int, int, float, float]]
    # roles ignored: every person pair, same streak logic as the event engine (reuse events._Streak)
def talk_eval(intervals, talks: list[Activity], id_map, fps: float, tiou_thr: float = 0.3) -> dict
    # GT: person_talks_to_person with exactly 2 Person actors; a detection matches if its mapped pair == GT pair
    # and temporal IoU ≥ tiou_thr; returns precision, recall, n_gt, n_det, median_start_err_s
def enter_exit_eval(an, acts, id_map, fps, window_s: float = 5.0) -> dict
    # GT person_enters_scene_through_structure / person_exits_scene_through_structure: recall of our first-seen/last-seen
    # times of the mapped person within ±window_s of the GT interval start (enter) / end (exit); median |error|
def eval_clip(video, annotations_dir, stack, cache_root, preset: str = "short", dist: float = 2.0,
              min_dur: float = 8.0, gap_tol: float = 1.5) -> dict
```

- Unit tests use synthetic `GeomBox` lists and a hand-built tiny `PerceptionResult`/`Analysis` (or `SimpleNamespace` stand-ins) — e.g. two GT actors talking for 20 s and our pair interval offset by 2 s ⇒ recall 1, start error 2 s; a non-overlapping interval ⇒ recall 0.
- CLI: `showroom-coach eval-meva --clips data/meva/clips.csv --split dev|test|all --out runs/eval/meva_<split>.json` prints a table per clip and the totals (micro-averaged P/R, median errors).

- [ ] Steps: failing tests → implement → green → commit `feat(yak-44): MEVA evaluation for tracking, engagement and enter/exit`.

### Task 6.3: Split, tune on dev, measure on test, speed bench, labels, EVAL.md

**Files:**
- Create: `data/meva/split.json` (gitignored data — also copy the split into `docs/EVAL.md`), `labels/*.json` (only if `data/stock/` has clips), `src/showroom_coach/evaluation/labels_eval.py` (+ tests), `docs/EVAL.md`
- Modify: `src/showroom_coach/cli.py` (add `bench`, `eval-labels`)

Steps:
- [ ] **Split before tuning:** sort MEVA clips by name; assign roughly 40% to dev and 60% to test (at least 2 test clips from a camera that is not in dev, if possible). Write `split.json` and log it.
- [ ] **Tune on dev only:** grid over `engage_dist ∈ {1.5, 2.0, 2.5}`, `min_dur ∈ {4, 6, 8}`, identity `merge_thr ∈ {0.2, 0.3, 0.4}`, detector score threshold `{0.3, 0.4}` — pick by dev talk-recall F1 / IDF1; record the chosen values in config and the grid table in EVAL.md.
- [ ] **Measure on test once** with the chosen values: tracking (MOTA, IDF1 with/without merge), talk P/R + median start error, enter/exit recall + timing error. Paste exact numbers.
- [ ] **Speed:** `showroom-coach bench --video data/samples/meva_a_60s.mp4` → file mode real-time factor at 5 fps sampling on 720p input (make a 720p cut), per-stage ms; live mode processed fps with 1–4 people (from `live_file_client.py` runs on clips with different crowding; report people count per frame).
- [ ] **Showroom labels:** if `data/stock/*.mp4` exist: sample frames at 1 s, label persons/roles/greet times/engagements/walk-outs by viewing the frames (agent-made), write `labels/<clip>.json` with `"labeller": "ai-agent, unreviewed"`, split dev/test by clip, run `eval-labels` (staff accuracy before correction, clicks needed, greet-time median abs error). If empty: say so in EVAL.md ("no labelled staff/customer data exists; staff suggestion is unevaluated until the team films staged scenes").
- [ ] **EVAL.md:** sections Data (MEVA clips + licences + split), Method, Results tables (with the commands to reproduce), Targets vs measured (✅/❌, honest), Limitations (MEVA is not a showroom; talk ≠ engagement; agent labels unreviewed; CPU only).
- [ ] **M6 wrap-up** — full suite, PROGRESS entry, commit `docs(yak-44): MEVA evaluation results and speed benchmarks`.

---

# M7 — Docs and packaging

### Task 7.1: README, PITCH_NOTES, licence table, run scripts check

**Files:**
- Create/modify: `README.md`, `docs/PITCH_NOTES.md`, `models/MANIFEST.json` (regenerate), `data/sources.csv`

- [ ] **README.md** sections: Problem; What it does (upload + live); How it works (pipeline diagram in ASCII, units table); Quickstart (`run.bat`, then http://localhost:8000; CLI examples); Results (link + headline numbers from EVAL.md, honest); **Licence table** with columns `component | licence | trained on (data) | where used` covering: our code (Apache-2.0), numpy/scipy/opencv-python/onnxruntime/supervision/FastAPI/uvicorn/python-multipart/imageio-ffmpeg + its bundled FFmpeg binary (check the imageio-ffmpeg docs for the binary's licence — GPL build with libx264 — used as a separate process), motmetrics, pyyaml, tqdm, rtmlib (vendored code, Apache-2.0), YOLOX weights (COCO), RTMPose-s (Body7 datasets — list them), DINOv2-small (LVD-142M), RF-DETR-nano if kept, MEVA (CC BY 4.0, eval only, with the Corona et al. WACV 2021 citation), Pexels clips if any; **Privacy** (no face recognition, blur everywhere, on-prem, 24 h deletion, staff must be informed, decision support under ZRU-1115); **Pre-final declaration** ("Built for the WIUT Hackathon 2026 CV final (YAK-44). Code written during the hackathon with AI assistance (Claude Code); all third-party components are listed above; evaluation labels for showroom clips were made by an AI agent and are unreviewed; no partner footage was used."); Limitations & roadmap; Repo layout.
- [ ] **docs/PITCH_NOTES.md**: deck outline (problem → demo → how it works → innovation: open staff-vs-customer suggestion + one click → results → ROI → privacy → roadmap), SKAIVISION comparison table (their pipeline vs ours, cost, privacy, staff detection openness, edge hardware), ROI model with every assumption marked `ASSUMPTION — replace with Royal Motors numbers` (daily walk-ins, current walk-out share, close rate, average gross per car, % of walk-outs recoverable with faster greet), worked example + formula, and risks.
- [ ] Verify `run.bat` from a fresh shell once more (server starts, `/api/presets` answers); regenerate `models/MANIFEST.json`; check `data/sources.csv` rows.
- [ ] **M7 wrap-up** — full suite, PROGRESS entry, commit `docs(yak-44): README with licence table, privacy and declaration; pitch notes and ROI model`.

---

# Final verification (before claiming done)

- [ ] Use `superpowers:verification-before-completion`. Fresh evidence, pasted into PROGRESS.md:
  1. `.venv/Scripts/python -m pytest -q` (fresh run, `-p no:cacheprovider`) — full output tail.
  2. `showroom-coach analyze` on two clips (real stack) — summaries.
  3. `run.bat` + `scripts/smoke_api.py` upload → confirm → results — output.
  4. `scripts/live_file_client.py` against the real server — output.
  5. `showroom-coach webcam-check` — output.
  6. `git status` clean, `git remote -v` empty, `du -sh .venv models data` within 4 GB.
- [ ] Write the **Morning summary** at the end of PROGRESS.md: what works; what doesn't; measured numbers; every decision taken alone; the team's to-do list (review the agent-made labels, film staged showroom scenes, put real Royal Motors numbers into the ROI model, deploy).
- [ ] Commit `docs(yak-44): morning summary`.

---

# M8 — Stretch (in this order, each its own commit, only after everything above)

1. **Sales CSV conversion join** — `metrics.sales_conversion(card, rows, names)`: rows `salesperson,timestamp,sold`; match by staff name; `sold` count and `conversion = sold / customers_engaged`. Upload field on the results page; ship `data/sample_sales.csv` (committed via a `docs/sample_sales.csv` path since `data/` is ignored) labelled SYNTHETIC. TDD.
2. **Auto car zones** — cars detected on the first frame (`res.cars`) become zones; per person dwell time near each car (foot point inside the car box expanded 30%); shown in results as "dwell per car". TDD for the dwell computation.
3. **Uzbek/Russian UI toggle** — `static/i18n.js` with en/uz/ru strings for all UI text; a language selector stored in localStorage.
4. **Dockerfile** (python:3.12-slim, CPU onnxruntime, models fetched at build) — written, not built (Docker isn't running).
5. **Deploy script** `scripts/deploy_example.sh` — written but NOT run, with a header saying so.

