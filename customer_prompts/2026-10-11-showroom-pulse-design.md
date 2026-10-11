# Showroom Pulse: design spec (customer experience analytics)

Status: draft for the owner's review, 2026-10-11. Launching `docs/OVERNIGHT_PROMPT.md` in a new
session counts as approval. "Showroom Pulse" is a working name.

Sister project: **Showroom Coach** (teammate A, salesperson coaching), spec
`2026-10-11-showroom-coach-design.md`. The two are built in parallel overnight on different machines
and merged into one product in the morning. Section 4.2 keeps the shared units identical so the merge
is cheap.

## 1. Goal

Turn one showroom camera (an uploaded video or a live webcam) into **customer-experience statistics
and concrete recommendations** for the dealer's general manager and marketing lead:
- which cars attract attention, and which get ignored;
- how customers move through the showroom;
- how long visits last;
- where interest goes unserved, meaning a customer studies a car and no salesperson comes;
- when the floor gets crowded.

This is decision support built on aggregated, anonymous statistics. Scoring individual salespeople
is Showroom Coach's job, not this product's.

Context:
- This is the WIUT Hackathon 2026 Computer Vision final, held 2026-10-12.
- The partners are Toyota Royal Motors, BYD Royal Motors (Royal Motors Group) and Ducati.
- Jury scoring: tech complexity 25%, innovation 25%, impact 25%, presentation 15%, ROI 10%.
- Deliverables:
  - a web demo that works on a video the jury brings;
  - a deck;
  - a repo whose README lists every data and model licence.
- The partners will NOT supply footage, so everything is built and evaluated on open data.

### Existing products, and how we differ

Camera retail analytics already exist:
- **V-Count** and **FootfallCam** sell dwell per car, heatmaps and staff exclusion to showrooms.
- **Deepingsource** runs counts, paths, dwell, funnels and family detection in a Korean car showroom.
- **Palexy** sells time-to-greet and staff-interaction metrics.

The jury may know these, so the pitch must say plainly what is new:
1. **Car-specific meaning.** Attention per car model, stop rate, repeat visits, reaching the sales
   desk, and (stretch) sitting in the car. This goes beyond generic zones.
2. **Unserved interest.** Joining customer interest to where staff are, in one engine. This is the
   bridge to Showroom Coach and the most actionable output: "Car X: 5 of 7 interested customers got
   no salesperson".
3. **Recommendations with evidence.** Every insight cites its numbers and links to the clip moment.
4. **No new hardware.** Works with existing CCTV or a phone video. Competitors sell sensors.
5. **Open evaluation.** Measured accuracy on public data, published in `docs/EVAL.md`.

## 2. Decisions (owner's defaults; edit before launch if needed)

| Topic | Decision |
|---|---|
| Staff vs customer | Same **auto-suggest + one click** as Showroom Coach, with a reduced feature set (§4.4). Also accepts a `roles.json` exported by Showroom Coach. Staff are excluded from customer statistics. |
| Zones | **Auto + editable.** Cars are detected on a keyframe to make car zones. In the UI the user can move or delete them, draw extra zones (desk, entrance, lounge), and name them. A video without cars, such as a classroom with chairs, works with hand-drawn zones only. |
| Data | Free stock video (commercial-use licences) for the demo. MEVA (CC BY 4.0) for evaluation. Synthetic clips from teammate C if present. Non-commercial data is for evaluation only and is never committed. |
| Reach | **Local only.** Commit locally. No git remote, no push, no deploy, no accounts. |
| Live mode | **Required, after the upload flow.** Live occupancy, zone dwell timers, and an unserved-interest alert. |
| Sensitive inference | **None.** No face recognition, no age, gender or emotion, no re-identification across visits. |

## 3. What the user sees

### Upload flow (main path)

1. **Upload.** A video (mp4/mov/avi, up to about 5 minutes; longer videos are trimmed with a
   warning) and a preset: **Dealership** for long footage, or **Short clip** for 10–60 s clips (§4.6).
2. **Progress.** A progress bar while perception runs. Perception is cached per video.
3. **Zones.** A keyframe with auto-detected car zones drawn as editable polygons. The user can:
   - drag a vertex;
   - delete a zone;
   - add a rectangle or polygon;
   - name it (`Car: BYD Song Plus`, `Desk`, `Entrance`, `Lounge`).

   "Looks right" continues. With no zones, only store-level stats are computed.
4. **Confirm staff.** The same gallery as Showroom Coach: face-blurred thumbnails, the suggested
   role with up to three plain reasons, and one click to flip. "Skip" is allowed; everyone is then
   treated as a customer and a warning banner appears.
5. **Results dashboard:**
   - **KPI tiles:** visitors, groups, median visit length, browse rate, assisted share,
     unserved-interest episodes, peak occupancy.
   - **Per-car table:**
     - visitors who stopped and the stop rate;
     - median and total dwell;
     - repeat visits;
     - unassisted share of deep-interest dwell;
     - attention rank;
     - sat-in-car count (stretch).
   - **Heatmap** of foot positions over a keyframe (a top-down map if calibrated), plus a few
     sampled paths.
   - **Journey flow:** common zone sequences, e.g. Entrance → Car A → Car C → Desk → Exit, drawn as
     a simple flow diagram with counts.
   - **Charts:** occupancy over time; unattended waiting over time.
   - **Recommendations** (§4.7). Each one has its evidence numbers, a confidence based on sample
     size, and links to the clip moments and contact sheets.
   - **Annotated video:**
     - customers in orange, with a dwell timer while inside a zone;
     - staff in dimmed blue;
     - zone outlines, a counter on each zone, faces blurred.
   - **Export:** JSON, CSV, and a printable one-page HTML report (stretch).
6. **Re-scoring.** Editing zones or roles re-computes everything in seconds; there is no
   re-detection.

### Live flow (required, after upload)

1. The browser captures the webcam with getUserMedia.
2. JPEG frames go over a WebSocket at about 5 fps to the same engine.
3. The page shows on a canvas:
   - live occupancy;
   - a dwell timer per person in a zone;
   - zone counters;
   - an alert: "customer at Car A for 45 s, no staff nearby".
4. Zones are drawn on the first live frame or loaded from a saved zone file. Clicking a person
   toggles staff/customer.
5. This works on `localhost` without HTTPS.

## 4. How it works

### 4.1 Principle

This is the same as Showroom Coach.
- **Live mode** feeds frames one at a time.
- **File mode** feeds the same engine from a file, then runs an offline pass on top. That pass merges
  broken tracks into persons, suggests roles, and recomputes statistics.
- **Cache:** perception output per video is cached (`.npz`/`.json`), so zones, roles and thresholds
  can change without re-running models.

### 4.2 Units

**Shared with Showroom Coach.** Keep these names, interfaces and dataclasses identical to its spec
§4.2: `sources`, `perception.detect`, `perception.pose`, `tracking`, `identity`, `geometry`,
`roles`, `render`. Core types are plain dataclasses that serialise to JSON:
`Det`, `TrackObs`, `Person`, `RoleSuggestion`, `Event`.

**Differences in the shared units:**
- `perception.detect` must also return the **`car`** class (COCO), plus `truck`/`motorcycle` for the
  Ducati case, on one keyframe per scene (and on demand). Car zones are built from these.
- `roles` gets a reduced feature set (§4.4) and can import a `roles.json` from Showroom Coach.

**New units (pure Python, test-first):**

| Unit | Does | Interface (sketch) |
|---|---|---|
| `zones` | Builds car zones from car boxes: the footprint polygon from the box's lower part, expanded by an apron. Also polygon operations, point-in-polygon on the foot point, and zone-file load/save. | `auto_zones(car_dets, frame_shape) -> list[Zone]`; `zone_of(point) -> zone_id or None` |
| `journey` | Per-person zone sequence with enter/exit hysteresis, merging short exits. | `build(timeline, zones) -> dict[person_id, list[ZoneVisit]]` |
| `stats` | Footfall, occupancy, visit length, dwell per zone, stop rate, repeat visits, funnel, groups, heatmap accumulation. | `compute(persons, journeys, roles, cfg) -> Stats` |
| `interest` | Browsing, deep interest, assisted, and unserved-interest episodes (§4.5). | `episodes(journeys, staff_tracks, cfg) -> list[InterestEpisode]` |
| `insights` | Rule-based recommendations with evidence and confidence (§4.7). | `recommend(stats, episodes, cfg) -> list[Insight]` |
| `report` | Dashboard JSON, charts (SVG/PNG), CSV, heatmap image, journey-flow data. | `build_report(stats, insights, out_dir)` |
| `camera_motion` | Detects a moving or cutting camera, which is common in stock footage, using a homography between keyframes. Zone statistics are disabled or flagged on moving segments. | `segments(video) -> list[StableSegment]` |

### 4.3 Position and distance

- **Foot point:** the mean of the ankle keypoints when they are confident; otherwise the
  bottom-centre of the box. Zone membership uses the foot point in image space. This needs no
  calibration and works on any jury clip.
- **Distances** between people, needed for "staff nearby": use Showroom Coach's pinhole
  back-projection, copied from its spec §4.3 with the same tests.
- **Optional 4-point floor calibration (homography):** gives a top-down heatmap in metres and
  metric dwell positions. The UI lets the user click 4 floor points and enter their real
  coordinates.
- **Speed** (for "stopped"): smoothed foot-point speed in metres per second, converted from pixels
  per second with the local scale from the pinhole estimate.

### 4.4 Staff exclusion

Use a subset of Showroom Coach's rules:
- `outfit_peers`: shared uniform;
- `contacts`: distinct persons within 2 m for ≥ 5 s;
- `presence`: share of the clip visible, only for clips of 60 s or more;
- `initiator`: approached the other person.

Persons scored "unsure" count as customers, but they are flagged and the dashboard shows how many.
The user's correction always wins. An imported `roles.json` overrides suggestions. Weights live in
config and are tuned only on dev clips.

### 4.5 Definitions

All thresholds are in config, and the values for each preset are in §4.6.

- **Visit:** a merged customer person from first to last seen.
  - Persons visible within the first `leave_after` seconds are "already present". They are excluded
    from visit-length and entry statistics, the same as in Showroom Coach.
- **Entry/exit:** appearing or disappearing via the `Entrance` zone if one is defined, otherwise via
  a frame edge, or not seen for `leave_after` seconds.
- **Zone presence:** a person enters a zone once their foot point has been inside for `zone_min`.
  They leave after `zone_gap` seconds outside. This hysteresis stops jitter at the edges.
- **Stop (browse):** inside a car zone with speed below `stop_speed` for at least `stop_time`.
- **Deep interest:** cumulative dwell at one car of at least `interest_time`, or two or more separate
  visits to the same car.
- **Assisted:** a staff member within `assist_dist` for at least `assist_time` during the visit. If
  Showroom Coach's greet events are available (imported), use them instead.
- **Unserved interest:** a deep-interest dwell with no staff member within `assist_dist` at any point
  during it. Live mode raises an alert once the dwell passes `alert_after`.
- **Reached desk:** in the `Desk` zone for at least `desk_time`. This is a proxy for negotiation or
  booking a test drive.
- **Group/party:** Showroom Coach's party rule: customers who arrive within 5 s of each other and
  stay ≤ 1.5 m apart for most of their first 30 s. Statistics are reported per group and per person.
- **Occupancy:** the number of customers present, sampled every second.
- **Funnel:** the share of visits that reach each stage: entered → browsed → deep interest →
  assisted → reached desk. The stages are not strictly ordered; report each as a share of all
  visits.
- **Sat in car (stretch M8):** most of the person's box lies inside the car box, only the torso is
  visible, and there is no translation for 5 s or more; or the person disappears inside a car zone
  and reappears there. Evaluate it on MEVA vehicle-entry activities (§6).

### 4.6 Presets

| Param | Dealership | Short clip |
|---|---|---|
| `leave_after` | 10 s | 3 s |
| `zone_min` / `zone_gap` | 1.5 s / 2 s | 1 s / 1 s |
| `stop_speed` / `stop_time` | 0.3 m/s / 10 s | 0.3 m/s / 3 s |
| `interest_time` | 60 s | 10 s |
| `assist_dist` / `assist_time` | 2.0 m / 5 s | 2.0 m / 2 s |
| `alert_after` | 45 s | 8 s |
| `desk_time` | 30 s | 5 s |
| `min_sample` (visits before an insight is shown) | 5 | 2 (marked "demo only") |

### 4.7 Insight rules

Each insight has:
- a title;
- the evidence numbers;
- the suggested action;
- the cars and time window it affects;
- a confidence: low, medium or high, from sample size against `min_sample`;
- links to the clip moments.

Below `min_sample`, the insight shows as "insufficient data". Insights are suggestions; nothing acts
on its own.

| ID | Trigger | Suggested action (template) |
|---|---|---|
| R1 Unserved hotspot | A car with ≥ 3 unserved-interest episodes, or ≥ 40% of its deep-interest visits unassisted | "Station a salesperson near {car} during {window}; {n} of {m} interested customers got no contact." |
| R2 Glance-and-go | Stop rate in the top quartile but median dwell in the bottom quartile | "{car} draws people in but doesn't hold them. Check the price board or info display." |
| R3 Dead zone | Stop rate < 10% while the store median is > 30% | "{car} is rarely visited. Consider moving it, lighting or signage." |
| R4 Long unattended waits | Median time to first assistance > `alert_after` × 2 | "Customers wait {t} s on average before contact. Review floor staffing. Showroom Coach has per-salesperson detail." |
| R5 Crowding | Customers per staff member > 3 for ≥ 2 min | "Peak at {window}: {k} customers per staff member." |
| R6 Desk drop-off | ≥ 50% of deep-interest visits never reach the desk | "Interest isn't turning into desk conversations. Review the follow-up process." |
| R7 Groups | Groups have lower assisted share than solo visitors (difference ≥ 20 points) | "Groups get less attention. Consider a family-friendly approach." |

### 4.8 Metrics and report

- **Store:** visitors, groups, median visit length, browse rate, assisted share, unserved episodes,
  peak occupancy, funnel.
- **Per car:** stops, stop rate, median and total dwell, repeat visits, deep-interest count,
  unassisted share, rank.
- **Per time bucket:** 1-minute buckets for clips, hourly for long footage.
- **Optional sales CSV join:** upload `car_model,period,sold`; the report adds attention-to-sales
  per car. Ship a clearly labelled synthetic sample CSV.

## 5. Stack and licences

This is the same as Showroom Coach §5, so the repos can merge.
- Target: a permissive (Apache-2.0) repo.
- Inference: ONNX Runtime on CPU, so no torch unless a measured need appears; log the reason if so.
  The CoreML execution provider is an optional speed-up, but CPU numbers are mandatory because A's
  laptop is CPU-only.
- **Allowed:**
  - onnxruntime, opencv-python-headless, numpy, scipy;
  - `rtmlib` with YOLOX and RTMPose ONNX weights (Apache-2.0); RF-DETR Nano/Small (Apache-2.0) as
    the alternative detector;
  - `supervision` (MIT, ByteTrack);
  - DINOv2-small ONNX (Apache-2.0);
  - FastAPI and uvicorn;
  - imageio-ffmpeg (use it even though ffmpeg is on PATH here, for portability to A's Windows
    machine);
  - motmetrics;
  - charts: matplotlib (server-side SVG/PNG), or a vendored MIT JS library (e.g. Chart.js) served
    locally with no build step and no CDN at runtime.
- **Banned:**
  - ultralytics and boxmot (AGPL);
  - re-ID weights trained on Market-1501, DukeMTMC or MSMT17;
  - any non-commercial dataset inside a shipped artifact or a commit;
  - weights with no findable licence;
  - demographic, emotion or face-recognition models.
- **Detector choice:** pick by CPU benchmark at M0. Choose the fastest detector whose person recall
  on a MEVA sample is within 5 points of the best, **and** which detects cars on showroom keyframes.
- **Records:** README licence table (component | licence | trained on | where used) and
  `models/MANIFEST.json` (URL, sha256, licence for each weight file).

## 6. Data and evaluation

| Set | Source | Licence handling | Use |
|---|---|---|---|
| **MEVA** (Kitware/IARPA) | Anonymous HTTPS from `https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/<date>/<hour>/<clip>.r13.avi`. KPF annotations (geom, activities, types YAML) in `gitlab.kitware.com/meva/meva-data-repo`, folder `annotation/DIVA-phase-2/MEVA/kitware/<date>/<hour>/`. The list of annotated clips is in `list-of-annotated-meva-clips.txt`. | **CC BY 4.0**, so commercial use is OK with attribution. Cite Corona et al., WACV 2021. Clips stay in `data/meva/`, uncommitted. | **Main evaluation set.** Pick 4–6 indoor clips with the most entries and exits (`person_enters_scene_through_structure` / `person_exits_scene_through_structure`). If disk and time allow, add 1–2 outdoor parking-lot clips with person–vehicle activities for the sat-in-car stretch. Check the exact activity names in the types YAML first; never guess them. |
| **Showroom stock clips** | The owner downloads them by hand into `data/stock/`, named `pexels-<id>.mp4`. Pexels blocks scripts. Lead: pexels.com/video/7154222. If the folder is empty, continue and note it in PROGRESS.md. | Pexels licence; don't portray the people in a negative light. Build `data/sources.csv` from the IDs. Never commit the clips. | Demo footage and agent-labelled visits. |
| **Synthetic showroom** (teammate C) | `data/synthetic/share/` if present. Each clip has a `gt.json` with true positions, zones, roles and events. | Own generation, plus asset licences listed by C. | Exact ground truth for dwell, journeys, walk-outs and groups. If absent, skip and say so. |

**Ground truth for zone statistics on MEVA:**
- Draw zones on each evaluation clip yourself (doors, desks, open floor areas) and save them in
  `labels/zones/<clip>.json`.
- Compute "true" dwell, journeys and occupancy from the MEVA geom tracks with the **same** `zones`
  and `journey` code.
- Compare against our pipeline's output. This isolates errors from perception and tracking.

**Labels you make by hand** go in `labels/<clip>.json` and carry `"labeller": "ai-agent, unreviewed"`.
They cover visits, zone visits and stops on stock clips, sampled at 1 s intervals. Split dev vs test
by clip before tuning anything.

**Report** in `docs/EVAL.md`, with honest numbers and the command to reproduce each:
- **Tracking on MEVA:** MOTA and IDF1, with and without the identity merge. The same numbers as
  Showroom Coach are fine.
- **Entries and exits:** count error (%) and median timing error against MEVA enter/exit
  activities.
- **Occupancy:** mean absolute error in persons per second, against GT tracks.
- **Zone dwell:** median absolute error in seconds and %, against GT-track-derived dwell.
- **Journeys:** accuracy as 1 − normalised edit distance between our zone sequences and the GT
  sequences.
- **Stops:** precision and recall on the labelled stock clips (and synthetic clips if present).
- **Sat-in-car (stretch):** recall against MEVA vehicle-entry activities.
- **Speed on this Mac's CPU:** file-mode real-time factor at 5 fps sampling and 720p; live fps with
  1–4 people. Report CoreML numbers separately if measured.
- **Insights:** the unit-test scenarios and the recommendations each one produced.

**Targets.** A missed target is a finding to report, not a reason to fake numbers.
- entry/exit count error ≤ 10%;
- occupancy MAE ≤ 0.5;
- zone dwell median error ≤ 2 s (Short clip) or ≤ 10% (Dealership);
- journey accuracy ≥ 80%;
- file mode ≥ 1× real time;
- live mode ≥ 5 fps.

## 7. Privacy

- **Faces:** never recognised or stored as identity. Every rendered output and thumbnail blurs the
  face region, using Showroom Coach's method.
- **No sensitive inference:** no age, gender, emotion or ethnicity inference. Visits are anonymous.
  The same person on another day is a new visitor.
- **Aggregated output:** the dashboard shows aggregates and per-visit records with anonymous IDs.
  Staff appear only as the "assisted" signal; per-salesperson scores are out of scope.
- **On-premise:** processing is local. Uploads and job data are deleted after 24 h by a cleanup on
  app start.
- **Banner:** "anonymous, aggregated analytics; decision support only". Use the same wording style
  as Showroom Coach. The README says customers must be informed by signage that analytics is in use.

## 8. Error handling

- **Unreadable or unsupported video:** a clear message; the job fails cleanly.
- **Nobody detected:** a results page saying so, with a tip ("camera too far or too dark").
- **No cars detected and no zones drawn:** only store-level statistics, plus a prompt to draw zones.
- **Fewer than `min_sample` visits:** insights show as "insufficient data". KPIs are still shown.
- **Moving camera or cuts:** zone statistics are computed only on stable segments, with a banner
  showing the share of the clip analysed.
- **Distance estimate unreliable** (box-height fallback for more than 50% of samples): a warning
  badge, and `assist_dist` checks are marked approximate.
- **Live mode:** the client reconnects automatically if the WebSocket drops; if processing falls
  behind, frames are dropped and the processed fps is shown.
- **Model download fails:** retry 3 times, then try the alternative model, and log it.

## 9. Testing and the verification chain

**TDD for all pure units:**
- `zones`: auto zones from synthetic car boxes, point-in-polygon, hysteresis;
- `journey`;
- `stats`;
- `interest`;
- `insights`: one scripted scenario per rule that must fire, plus one per rule that must not;
- `geometry` (shared tests);
- `roles`: synthetic feature vectors;
- identity merge.

**Integration:**
- Perception on a 10 s real clip gives at least 1 person, stable track IDs, and at least 1 car on a
  showroom keyframe (if a stock clip exists).
- CLI `analyze` end to end produces every output file.
- FastAPI TestClient: upload → zones → roles → results → edit zone → re-score with no re-detection.
  Assert the perception cache is not rebuilt.
- The live WebSocket path, fed from a video file, returns overlays, timers and an alert.

**Real webcam check:** grab one frame from the default camera. macOS needs camera permission for
the terminal app; the owner grants it before launch.

**Speed:** the full pytest suite runs in under 3 minutes. Slow tests use a `slow` marker and run in
the verification chain, not in the default `pytest -q`.

**Verification chain** (`scripts/verify.sh`): create it at M0 and grow it each milestone. It runs
each step in order, prints its output, and exits non-zero at the first failure.
1. `.venv/bin/python -m pytest -q`, then `.venv/bin/python -m pytest -q -m slow`
2. `.venv/bin/pulse analyze <meva_dev_clip> --out runs/verify/meva` and
   `.venv/bin/pulse analyze <stock_or_synthetic_clip> --out runs/verify/stock`, then a schema check
   on every output file (`scripts/check_outputs.py`)
3. `.venv/bin/python scripts/smoke_api.py`: upload → zones → roles → results → zone edit →
   re-score; asserts no re-detection
4. `.venv/bin/python scripts/smoke_live.py`: file-fed WebSocket; asserts overlays, timers, at least
   one alert, and processed fps
5. `.venv/bin/python scripts/webcam_grab.py`: saves one frame and asserts it isn't blank
6. `.venv/bin/pulse eval --out docs/EVAL.md`: regenerates every number in EVAL.md from the cached
   runs and exits 0
7. `git status --short` (must be empty) and `git remote -v` (must be empty)

## 10. Milestones (each one ends demo-able and committed)

| # | Milestone |
|---|---|
| M0 | Repo skeleton, Python 3.12 venv, dependencies, model downloads plus manifest, `scripts/verify.sh` skeleton. Download 1 MEVA clip and its annotations, and check `data/stock/` and `data/synthetic/share/`. Run the CPU benchmark (person recall, car detection, speed), choose the detector, and record it in PROGRESS.md. |
| M1 | Pure core, test-first on synthetic data: zones, journey, stats, interest, insights, plus the shared geometry, roles and identity merge |
| M2 | Perception chain on real video (detect persons and cars, pose, track, embed, merge) with a per-video cache; camera-motion segments |
| M3 | CLI `analyze`: annotated H.264 mp4 with face blur, stats JSON/CSV, heatmap PNG, journey-flow data, insights JSON, contact sheet per insight |
| M4 | Web upload flow: progress, zone editor, staff confirm, results dashboard (KPIs, per-car table, heatmap, journey flow, charts, insights), re-score without re-detection |
| M5 | Live mode: webcam → WebSocket → engine → overlay, zone dwell timers, live occupancy, unserved-interest alert, click-to-toggle role |
| M6 | Data and evaluation: fetch the remaining MEVA clips, draw evaluation zones, label stock clips, split dev/test, tune on dev only, write `docs/EVAL.md` |
| M7 | Docs: README (problem, how it works, quickstart, results, licence table, privacy), `docs/PITCH_NOTES.md`, `docs/INTEGRATION.md`, `run.sh` |
| M8 (stretch, in this order) | (1) sat-in-car detection; (2) floor calibration → top-down heatmap; (3) printable one-page HTML report; (4) sales CSV join; (5) Uzbek/Russian UI toggle; (6) Dockerfile; (7) a deploy script, written but NOT run |

**`docs/PITCH_NOTES.md` contents:**
- the customer-side story;
- a competitor table (V-Count, FootfallCam, Deepingsource, Palexy) with our differences from §1;
- an ROI model with clearly marked assumptions: missed-interest episodes × conversion × margin;
- the questions the team must ask Royal Motors.

**`docs/INTEGRATION.md` contents:**
- how to merge with Showroom Coach: one perception engine and cache, a shared `roles.json`, one web
  app with two tabs (Salespeople, Customers);
- the list of every interface that differs from Showroom Coach.

## 11. Machine constraints

- **Hardware:** MacBook Pro, Apple M3 Pro, 18 GB RAM, macOS, about 156 GB free disk.
- **Python:** the default `python3` is 3.14. Use `/opt/homebrew/bin/python3.12` for the venv.
- **Tools:** ffmpeg is at `/opt/homebrew/bin/ffmpeg`, but the code uses imageio-ffmpeg for
  portability.
- **Disk:** keep the venv, models and data at ≤ 15 GB combined. Stop downloading if free disk drops
  below 20 GB.
- **Sleep:** the owner launches the session under `caffeinate` so the Mac does not sleep.

## 12. Done by morning

1. `./run.sh` starts the app at http://localhost:8000.
   - Uploading a sample clip gives the full dashboard in under 3 minutes.
   - Editing a zone or flipping a role re-scores without re-detection.
2. The live page runs: tested via a file-fed WebSocket and a real webcam frame grab.
3. `bash scripts/verify.sh` exits 0, and its full output is pasted in PROGRESS.md.
4. `docs/EVAL.md` has real measured numbers, and the README licence table is complete.
5. `docs/PITCH_NOTES.md` and `docs/INTEGRATION.md` exist.
6. PROGRESS.md ends with a "Morning summary": what works, what doesn't, measured numbers, decisions
   taken, and the team's to-do list:
   - review the agent-made labels and zones;
   - film staged scenes;
   - merge with Showroom Coach;
   - fill in the ROI numbers;
   - deploy.
7. All work is committed locally. No remote exists.

## 13. Out of scope

- Face recognition; age, gender or emotion estimation.
- Re-identifying customers across visits or days.
- Tracking one person across multiple cameras (one camera per job; future work).
- Per-salesperson scoring (that is Showroom Coach).
- Live CRM or DMS integration.
- Training or fine-tuning models (teammate C owns that).
- Cloud deploy.
- Accounts and authentication.
