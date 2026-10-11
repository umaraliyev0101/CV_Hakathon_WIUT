# Showroom Coach

Camera-based coaching for car-showroom salespeople. One camera, an uploaded video or a live webcam, in; for each
salesperson out: how fast customers get greeted, who waited while staff were free, who walked out unattended, and how
long each conversation lasted.

Built for the WIUT Hackathon 2026 Computer Vision final (Jira YAK-44) with Toyota Royal Motors, BYD Royal Motors
(Royal Motors Group) and Ducati Tashkent in mind. It runs on a laptop CPU, keeps all video on the machine, blurs every
face it shows, and never recognises anyone.

## Problem

A showroom manager cannot watch every customer who walks in. The questions that matter are simple, and nobody can
answer them from memory:

- How long did the customer wait before someone said hello?
- Was a salesperson free while that customer waited?
- Did the customer leave without anyone speaking to them?
- Who spends time with customers, and for how long?

The reference commercial product, SKAIVISION (US, 150+ dealerships), answers these with a GPU edge box (an 8 GB RTX
3050/4060 handling 15 SD cameras) and a rule of "employee within 2 metres of a customer for over 2 minutes". It does
not disclose how it tells staff from customers. Showroom Coach reproduces the same measurements on a CPU laptop and
solves the staff-versus-customer question in the open: the system proposes who is staff, shows its reasons in plain
words, and a person confirms or flips each role with one click.

It is decision support for a sales manager. It is not an automatic judge of staff (see [Privacy](#privacy)).

## What it does

**Upload a video (main path)**

1. Upload an mp4, mov or avi (about 5 minutes at most; longer videos are trimmed with a warning) and pick a preset:
   "Dealership" (SKAIVISION's 2 m / 2 min engagement rule) or "Short clip" (tighter thresholds for 10 to 60 s clips).
2. A progress bar runs while the models work. The result of this step is cached per video.
3. **Confirm staff.** One card per person: a face-blurred thumbnail, the suggested role with its confidence, and up to
   three reasons ("same outfit as 2 others", "talked to 4 people", "on screen 95% of the clip"). One click flips a role;
   staff can be named. "Looks right" continues.
4. **Results.** An annotated video (staff boxes blue, customers orange, lines between engaged pairs, a timer above each
   waiting customer, faces blurred), a timeline with one row per person, a scorecard per salesperson and for the store,
   an alert feed (missed greet, long wait while staff were free, walk-out), and JSON/CSV export.
5. Changing a role re-scores in a fraction of a second and re-renders the video. Nothing is detected again.

**Live webcam** (`/live`)

The browser captures the camera, sends JPEG frames over a WebSocket at about 5 fps to the same engine, and draws boxes,
timers and alerts on a canvas (faces blurred on the canvas). Clicking a person toggles staff and customer; the
automatic suggestion starts after about 30 s of observation. It works on `localhost` without HTTPS.

**What is measured**

| Per salesperson | Store |
|---|---|
| greets, median time to greet | greet rate, median time to greet |
| customers engaged, total engagement time | walk-outs (customers who left never greeted) |
| customers who waited while this person was free | total unattended waiting time |
| walk-outs during this person's free time | wait alerts |

Events follow one state machine per customer: entered, waiting, greeted, engaged or browsing, left. A greet needs a
staff-customer pair within `greet_dist` (and facing, when known) for `greet_time`; an engagement needs `engage_dist`
for `engage_time`. All thresholds are in `src/showroom_coach/config.py` and the two presets are:

| Parameter | Dealership | Short clip |
|---|---|---|
| greet distance / time | 1.5 m / 5 s | 1.5 m / 2 s |
| engagement distance / time | 2.0 m / 120 s | 2.0 m / 8 s |
| gap tolerated inside an engagement | 3 s | 1.5 s |
| wait alert after | 60 s | 10 s |
| "left" after not seen for | 10 s | 3 s |

Customers who arrive together are treated as a party (greeting one greets all), a second salesperson taking over is a
handoff, and a customer already on screen when a clip starts has an unknown entry time, so they are left out of the
time-to-greet and waiting statistics.

## How it works

One engine serves both modes. Live mode feeds frames one at a time. File mode feeds the same engine from a file and
then runs an offline pass that merges broken tracks into persons, suggests staff, and re-runs the events with the final
roles.

```
  uploaded video (mp4/mov/avi)                    browser webcam (getUserMedia)
          |                                                  |  JPEG frames over WebSocket /ws/live, ~5 fps
          v                                                  v
   +-------------+                                   +-------------+
   |   sources   |  sample 5 fps, <= 960 px wide     | live frames |
   +------+------+                                   +------+------+
          |                                                  |
          v                                                  v
   +-------------------------------------------------------------------+
   | perception   YOLOX-nano on 2 overlapping tiles  -> person boxes   |
   |              ByteTrack (supervision)            -> tracks          |
   |              RTMPose-s                          -> 17 keypoints    |
   |              DINOv2-small + HSV histograms      -> body appearance |
   +---------------------------------+---------------------------------+
                                     |  file mode saves this per video (cache)
              +----------------------+-----------------------+
              v                                              v
   FILE MODE (offline pass)                       LIVE MODE (online, frame by frame)
   identity : merge track fragments               identity : re-attach a recently lost person
   geometry : floor metres, facing                geometry : same code
   roles    : suggest staff + reasons             roles    : suggest after ~30 s, click to override
   events   : greet, wait, walk-out, ...          events   : same state machine
   metrics  : scorecards                          metrics  : running scorecard
              |                                              |
              v                                              v
   render   : face-blurred H.264, timeline,       canvas overlay, timers, alerts
              contact sheets, CSV/JSON            (faces blurred on the canvas)
   app      : upload -> confirm -> results
```

| Unit | Module(s) | What it does |
|---|---|---|
| sources | `sources.py` | Reads a file with OpenCV, samples to 5 fps, resizes to at most 960 px wide, trims at 300 s, and fails cleanly on damaged files. |
| perception | `perception/` | YOLOX-nano person detector run on two overlapping tiles (picked by a CPU benchmark, see Results), ByteTrack from `supervision`, RTMPose-s keypoints (every 2nd frame per track, carried in between), and DINOv2-small plus HSV descriptors for real tracks only. ONNX Runtime on CPU; no PyTorch. A per-video cache keyed by file hash and every content-changing parameter lets roles and thresholds change without re-running any model. |
| identity | `identity.py` | Merges the track fragments of one person by body appearance. Offline: constrained clustering where two tracks that overlap in time can never be the same person. Live: re-attach to a person lost a moment ago. |
| geometry | `geometry.py` | Calibration-free pinhole model: assume a 1.70 m person and a 70 degree horizontal field of view, depth Z = f*H/h_px, using full body height, else shoulder-to-hip torso length (seated or cut-off people), else box height (flagged "distance approximate"). Facing comes from the shoulder line. An optional 4-point floor homography (`CameraConfig.floor_points`) replaces the model. |
| roles | `roles.py` | Transparent staff-versus-customer suggestion: four hand-set features (`outfit_peers`: others in a similar outfit; `contacts`: distinct people within 2 m for 5 s or more; `presence`: share of the clip on screen, ignored on clips shorter than 60 s; `initiator`: who walked toward whom) into one logistic score. Above 0.6 is staff, below 0.4 is customer, between is "unsure". Each suggestion carries its top reasons in plain words. A user's click always wins. |
| events | `events.py` | A pure state machine (no I/O, no clock): greet, engagement, wait alert, walk-out, party, handoff, already-present. |
| metrics | `metrics.py` | Scorecards for each salesperson and the store from the event list. |
| render | `render.py`, `export.py` | Face blur first, then drawing; annotated H.264 video via `imageio-ffmpeg`, timeline PNG, event contact sheets, `results.json`, `events.csv`, `scorecard.csv`. |
| app | `app/` | FastAPI backend and a static HTML/JS frontend with no build step: upload, job status, people and roles, results, files with HTTP range support. Background jobs; job directories older than 24 h are deleted when the app starts. |
| live | `live.py`, `live_ws.py`, `app/static/live.*` | The live session engine, its WebSocket protocol, and the webcam page. Frames are dropped, not queued, if processing falls behind; the page shows the processed fps and reconnects by itself. |

## Quickstart

### Windows, one click

```
run.bat
```

The first run creates `.venv` (Python 3.12), installs the package, downloads the three models the app needs
(`yolox_nano`, `rtmpose_s`, `dinov2_small`, about 50 MB) and starts the server. It needs internet only that first time.
Then open <http://localhost:8000>. Extra arguments go to the server, e.g. `run.bat --port 8080`. `run.sh` is the bash
equivalent (written for Linux/macOS/Git Bash, not tested on Linux or macOS).

### Manual setup

Python 3.11 to 3.13 is required (`requires-python = ">=3.11,<3.14"`); 3.12 is what this was built and tested with. If your
default `python` is 3.14, point at a 3.12 interpreter explicitly.

```
python -m venv .venv                                  :: a 3.12 interpreter
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m showroom_coach fetch-models   :: all six models (209 MB), writes models\MANIFEST.json
.venv\Scripts\python -m showroom_coach serve          :: http://localhost:8000  (upload)   /live  (webcam)
```

`fetch-models yolox_nano rtmpose_s dinov2_small` fetches only what the app uses. The other three (`yolox_tiny`,
`yolox_s`, `rfdetr_nano`) are detector-benchmark candidates. Downloads are retried, and each file is checked against a
pinned sha256.

### Command line

The console script is `showroom-coach` (`.venv\Scripts\showroom-coach`); `python -m showroom_coach` is the same.

```
:: analyse a clip: annotated.mp4, timeline.png, results.json, events.csv, scorecard.csv, people/, contact_sheets/
showroom-coach analyze data\samples\meva_a_60s.mp4 --out runs\demo --preset short

:: re-score with known roles instead of the suggestions (roles.json: {"roles": {"3": "staff"}, "names": {"3": "Aziz"}})
showroom-coach analyze clip.mp4 --out runs\demo --preset dealership --roles roles.json

showroom-coach serve --port 8000                      :: the web app
showroom-coach webcam-check                           :: grab one real camera frame, run the live engine once
showroom-coach fetch-models                           :: download weights, refresh models\MANIFEST.json
showroom-coach bench-detectors --video <clip.avi> --geom <clip.geom.yml> --types <clip.types.yml>

:: MEVA evaluation (needs the MEVA clips in data\meva, see data\TODO_DOWNLOADS.md and data\sources.csv)
python -m showroom_coach.evaluation.meva_eval --clips data\meva\clips.csv --split dev --out runs\eval\meva_dev.json
```

`scripts\live_file_client.py` streams a video file to `/ws/live` the way the browser does (a test of live mode that
needs no camera), and `scripts\webcam_grab.py` is the same as `webcam-check`. The MEVA-derived sample clips in
`data\samples\` exist on the build machine only; videos and weights are never committed.

### Tests

```
.venv\Scripts\python -m pytest -q          # default run: excludes tests marked slow
.venv\Scripts\python -m pytest -m slow     # real models and real video; skipped when the files are missing
```

## Results

Everything below was measured on one Intel i5-10210U laptop (4 cores, 16 GB, no GPU), Windows 11. Other processes were
sometimes loading the CPU during runs, so treat speeds as honest rather than best-case. The full log, with every
decision, is in [PROGRESS.md](PROGRESS.md).

**MEVA evaluation numbers: see [docs/EVAL.md](docs/EVAL.md)** (tracking MOTA/IDF1, engagement against annotated
conversations, arrival and leaving, with the commands to reproduce them). They are deliberately not repeated here.

Measured and already logged:

| What | Measured | Note |
|---|---|---|
| Detector choice | YOLOX-nano on 2 tiles: 36 ms per frame, 26.8% recall. YOLOX-s on 2 tiles: 361 ms, 30.9% recall. | 60 frames of a MEVA school clip, score 0.3, IoU 0.5. MEVA only boxes annotated actors, so recall is understated for every model alike. The rule "fastest within 5 points of the best recall" picked YOLOX-nano on 2 tiles. |
| File-mode speed, 60 s clip (about 11 people per frame, 960 px) | real-time factor 0.976 (61.5 s for 60 s) | The 1x target was **not** reached. |
| File-mode speed, 5 min clip (1920x1072, about 12 people per frame) | real-time factor 0.713 (421 s for 300 s) | The 1x target was **not** reached. Pose and DINOv2 dominate the time. |
| Rendering | the 60 s clip renders in 8.1 s (7.4x real time) | Face-blurred H.264 at 10 fps. |
| Re-scoring after a role change | 0.07 to 0.1 s | No detection is repeated. |
| Live mode | 5.2 to 6.3 fps with about 4 people; about 3.2 fps with 11 to 15 people | Meets the 4 fps goal for small scenes only. |
| Distance model | pair distances within 0.1% on synthetic pinhole scenes at 1 to 4 m | This only proves the maths. Accuracy on real footage is **not measured**. |
| Face blur | face-region standard deviation at most 9.2 with the blur against about 45 without it | Automated privacy guard test. |

Not measured, because no suitable data exists: staff-suggestion accuracy on real showroom footage, and greet-time error
against a showroom ground truth. See [Limitations](#limitations).

## Licences

Rules: permissive licences only. No ultralytics, no boxmot (AGPL). No re-ID weights trained on Market-1501, DukeMTMC or
MSMT17 are used; the only appearance model is DINOv2-small (self-supervised on LVD-142M) plus colour histograms.
Non-commercial data would live only in `data/eval_only/`, never committed or shipped. Licences below were read from the
installed package metadata on 2026-10-11 (versions in brackets) and from `models/MANIFEST.json`.

| component | licence | trained on (data) | where used |
|---|---|---|---|
| This repository's code | Apache-2.0 (declared in `pyproject.toml`) | not applicable | everything under `src/`, `tests/`, `scripts/` |
| numpy (2.5.4) | BSD-3-Clause (bundles 0BSD, MIT, Zlib, CC0-1.0 parts) | not applicable | array maths throughout |
| scipy (1.18.1) | BSD-3-Clause | not applicable | declared dependency, also required by `supervision`; not imported by our own code |
| opencv-python (5.0.0.93) | Apache-2.0; the wheel redistributes FFmpeg libraries under LGPL-2.1 (per its `LICENSE-3RD-PARTY.txt`) | not applicable | video decoding, resizing, drawing, face blur |
| onnxruntime (1.31.0) | MIT | not applicable | runs every ONNX model on CPU |
| supervision (0.30.9; pinned `>=0.28,<0.31`) | MIT | not applicable | ByteTrack tracker (removed in 0.31, hence the pin) |
| fastapi (0.143.0) | MIT | not applicable | web API and static pages |
| uvicorn[standard] (0.54.0) | BSD-3-Clause | not applicable | web server |
| python-multipart (0.0.32) | Apache-2.0 | not applicable | video upload parsing |
| imageio-ffmpeg (0.6.0), the Python wrapper | BSD-2-Clause | not applicable | writes the annotated H.264 video |
| FFmpeg binary bundled in imageio-ffmpeg (`ffmpeg-win-x86_64-v7.1.exe`) | GPL-3.0-or-later as built (see note 1) | not applicable | H.264 (libx264) encoding of the annotated video, started as a separate process, not linked into our code |
| pyyaml (6.0.3) | MIT | not applicable | declared dependency, also required by `supervision`; not imported by our own code (MEVA annotations are read with plain parsers, never a YAML loader) |
| tqdm (4.70.1) | MPL-2.0 AND MIT | not applicable | declared dependency, also required by `supervision`; not imported by our own code |
| motmetrics (1.4.0) | MIT | not applicable | MOTA/IDF1 in the evaluation (`eval` extra, also in `dev`) |
| httpx (0.28.1), pytest (9.1.1) | BSD-3-Clause, MIT | not applicable | development and tests only |
| rtmlib 0.0.16 (code adapted, not a dependency) | Apache-2.0 | not applicable | `perception/rtm_onnx.py` ports the YOLOX and RTMPose pre/post-processing; rtmlib itself is not installed |
| YOLOX nano, tiny, s (ONNX weights, Megvii) | Apache-2.0 | COCO 2017 | person detection (nano on 2 tiles is the default; tiny and s are benchmark candidates) |
| RTMPose-s (ONNX, OpenMMLab, Body7 release) | Apache-2.0 (see note 2) | Body7: COCO, AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18 | 17-keypoint pose: distance, facing, face-blur region |
| DINOv2-small (int8 ONNX export, `Xenova/dinov2-small`) | Apache-2.0 | LVD-142M (Meta, self-supervised, no labels) | body-appearance embedding to merge track fragments and compare outfits |
| RF-DETR-nano (third-party ONNX export, `saifkhichi96/opendetect`) | Apache-2.0 | COCO 2017 | detector benchmark only, not used by the app |
| MEVA video and annotations (Kitware/IARPA) | CC BY 4.0 | not applicable (this is data) | evaluation only. Kept in `data/meva/`, never committed or shipped; derived demo cuts stay local too (see note 3) |
| Pexels stock clips | Pexels License (free for commercial use), if any are added | not applicable | none present: `data/stock/` was empty. Clips would be listed in `data/sources.csv` and never committed |

Not in the table because they are installed only as dependencies of the rows above and are not imported by our code:
`supervision` pulls in `av` (BSD-3-Clause; its wheel bundles FFmpeg libraries whose licence was not checked here),
`matplotlib` (Matplotlib licence), `pillow` (MIT-CMU), `requests` and `pydeprecate` (Apache-2.0) and `defusedxml`
(PSF); `fastapi` and `uvicorn[standard]` pull in `starlette` (BSD-3-Clause), `pydantic` (MIT), `websockets`,
`click`, `python-dotenv` (BSD-3-Clause), `httptools`, `watchfiles`, `h11`, `anyio` (MIT) and `onnxruntime` pulls in
`flatbuffers` (Apache-2.0) and `protobuf` (BSD-3-Clause). Regenerate the full list with `pip-licenses` before shipping
a bundle.

1. **FFmpeg binary.** The `imageio-ffmpeg` package is BSD-2-Clause, but the executable it ships is a separate work. The
   package's `binaries/README.md` says only that the executables are dropped there by its release script and gives no
   licence. The binary's own build configuration, read from the file, contains `--enable-gpl --enable-version3
   --enable-libx264 --enable-libx265 --enable-libxvid` and it embeds the text "GNU General Public License ... either
   version 3 of the License, or (at your option) any later version". So it is a GPL-3.0-or-later build. It is
   installed by `pip` into the local `.venv` and run through `subprocess` with command-line arguments; no part of it is
   linked into this repository's code and it is never committed. If a future product bundle ships this executable,
   the GPL source-offer terms apply to that executable, and H.264 has its own patent licensing to review. Alternatives:
   let the customer `pip install` it, or use an LGPL build of FFmpeg.
2. **RTMPose-s and its training data.** The weights are published by OpenMMLab under Apache-2.0, but Body7 combines seven
   datasets, and some of them carry research-only or non-commercial terms. We have not audited each one. The team
   should review whether those terms reach the released weights before any commercial deployment; the fallback is a
   different pose model with clean provenance.
3. **MEVA.** CC BY 4.0 allows commercial use with attribution. Cite: K. Corona, K. Osterdahl, R. Collins and A. Hoogs,
   "MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection", WACV 2021. Clips came from
   `mevadata-public-01.s3.amazonaws.com` and annotations from `gitlab.kitware.com/meva/meva-data-repo`; every item is
   listed in `data/sources.csv`.
4. The Apache-2.0 licence is declared in `pyproject.toml`; the full `LICENSE` text file has not been added to the
   repository yet and should be before the repo is shared.

## Privacy

- **No face recognition, ever.** There is no face detector, face embedding or face matching anywhere in the code. A
  person's face is never stored as an identity.
- **Faces are blurred in everything we show.** Every rendered video frame, contact-sheet panel and saved thumbnail is
  blurred before anything is drawn on it, and so is the live canvas. The blur region comes from the nose, eye and ear
  keypoints, with the top 20% of the person box as a fallback, including people the tracker discarded as noise. An
  automated test measures the face region before and after (see Results).
- **Body-appearance descriptors are used only to merge track fragments inside one video** (so one person who left and
  re-entered the frame is counted once) and to compare outfits for the staff suggestion. They are never matched to a
  database and never stored as an identity. They are computed from whole-body crops, which include the head, so treat
  the per-video cache with the same care as the video itself.
- **All processing is on-premise.** The app runs on the local machine; nothing is uploaded anywhere and there are no
  accounts. Uploads and job data (cache, thumbnails, results, rendered video) are deleted after 24 hours, by a cleanup that
  runs when the app starts. The command-line tool keeps its cache in `runs/cache/` (not committed) until you delete it.
- **Decision support only.** The UI carries a banner saying this is coaching support under Uzbekistan's 2026 AI law
  (ZRU-1115). Scores never trigger an action on their own: no automatic warnings, penalties or pay decisions. A person
  reads the scorecard, looks at the contact sheets, and decides.
- **Staff must be informed** that analytics is in use before any camera is pointed at them, and the partner's legal
  counsel should review the local personal-data rules for the camera placement before a pilot.

## Limitations

- **No real showroom footage.** The partners did not supply any and `data/stock/` was empty, so the system was built and
  evaluated on MEVA, a public CCTV dataset of schools, a bus station and similar places. No partner footage was used.
- **Staff suggestion is unevaluated on real staff and customers.** Its four features and weights are hand-set and
  were never tuned on showroom data. Expect to correct some roles; that is why the one-click confirmation exists.
- **MEVA is not a showroom.** Only the actors of annotated activities are boxed, so false-positive counts and
  tracking precision are pessimistic, and scenes are crowded in ways a showroom is not.
- **Speed is CPU-only and measured on one laptop.** File mode ran below real time on both clips we timed (0.976 and
  0.713), and live mode drops to about 3.2 fps with 11 to 15 people on screen.
- **Distance is approximate.** It assumes a 1.70 m person and a 70 degree field of view, is flagged when it falls back to
  box height, and was only validated on synthetic scenes. Real accuracy is unmeasured.
- **Detection misses are possible.** The small, fast detector misses some small, distant or heavily occluded people. A
  person cut off at the frame edge may not be detected, and a person who is not detected is not blurred. Use a camera
  angle that shows full bodies, keep people reasonably large in the frame, and where possible fill in the optional
  manual floor calibration (four floor points, a configuration option, not yet in the UI).
- **One camera per job.** No tracking across cameras. Videos are limited to about 5 minutes and 1 GB.
- **Windows only tested.** `run.sh` has not been run on Linux or macOS.

## Pre-final declaration

Built for the WIUT Hackathon 2026 Computer Vision final (Jira YAK-44). The code was written during the hackathon with AI assistance (Claude Code). Every third-party component and dataset is listed in the licence table above. Evaluation labels made for showroom clips, if any, were produced by an AI agent and are unreviewed. No partner footage was used.

## Roadmap

1. **Pilot on real footage**: film staged showroom scenes with consent, have a person label them, and measure
   staff-suggestion accuracy and greet-time error for real. Replace the hand-set role weights only if data supports it.
2. **Real numbers into the ROI model** in [docs/PITCH_NOTES.md](docs/PITCH_NOTES.md): every input there is an
   assumption until Royal Motors supplies its own.
3. **Join sales data**: an uploaded sales CSV (salesperson, timestamp, sold) to show engaged-to-sold conversion per
   salesperson.
4. **Zones**: cars detected on a keyframe become zones, with dwell time per car; desk and counter zones as a staff signal.
5. **Calibration in the UI**: click four floor points to replace the pinhole assumption.
6. **Uzbek and Russian interface.**
7. **More speed**: batching and quantisation first; a small GPU later if a showroom has one.
8. **Multi-camera** tracking, and CRM/DMS integration, both out of scope for the hackathon.
9. **Packaging**: container image, signed installer, and the FFmpeg licence decision from note 1 above.

## Repo layout

```
README.md               this file
PROGRESS.md             progress log: measured numbers and every decision
CLAUDE.md               project rules for the AI assistant (local only, permissive licences, privacy)
pyproject.toml          package metadata and dependencies; pytest settings
run.bat, run.sh         one-click launchers (create venv, install, fetch models, serve)
src/showroom_coach/
  sources.py            video reading and sampling
  perception/           detector, tracker, pose, appearance embedder, the per-video cache, benchmark
  identity.py           merge track fragments / re-attach lost people
  geometry.py           calibration-free floor distance and facing
  roles.py              staff-versus-customer suggestion with reasons
  events.py             greet / engagement / wait / walk-out state machine
  metrics.py            scorecards
  analysis.py           the offline pass that ties the units together, and fast re-scoring
  render.py, export.py  face-blurred video, thumbnails, timeline, contact sheets, JSON and CSV
  live.py, live_ws.py   live session engine and its WebSocket protocol
  webcam.py             real-camera check
  app/                  FastAPI app, background jobs, static frontend (upload flow and /live)
  evaluation/           MEVA annotation parser and the MEVA evaluation
  models.py, config.py  model registry and downloads; thresholds, presets, paths
  cli.py                the showroom-coach command
tests/                  unit and integration tests (slow ones are marked)
scripts/                live_file_client.py (file-fed live test), webcam_grab.py
docs/
  PITCH_NOTES.md        deck outline, SKAIVISION comparison, ROI model
  EVAL.md               evaluation report (MEVA numbers and how to reproduce them)
  superpowers/          design spec and plan
models/MANIFEST.json    every weight file with URL, sha256, licence (the weights themselves are git-ignored)
data/sources.csv        every dataset item with licence and attribution (the data itself is git-ignored)
data/TODO_DOWNLOADS.md  clips the owner has to download by hand
```

`data/`, `models/`, `runs/`, `jobs/` and `.venv/` are git-ignored. Only `data/sources.csv`, `data/TODO_DOWNLOADS.md` and
`models/MANIFEST.json` are committed from them. Videos and weights are never committed. There is no git remote.
