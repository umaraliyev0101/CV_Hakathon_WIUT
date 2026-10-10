# Showroom Coach: design spec (YAK-44)

Status: draft for the owner's review, 2026-10-11. Launching `docs/OVERNIGHT_PROMPT.md` in a new
session counts as approval. "Showroom Coach" is a working name.

## 1. Goal

Turn one showroom camera (an uploaded video or a live webcam) into coaching data for each
salesperson:
- how fast customers are greeted;
- which customers waited while staff were free;
- which customers left without being greeted;
- how long each engagement lasted.

It is decision support for a sales manager, not an automatic judge of staff.

Context:
- This is the WIUT Hackathon 2026 Computer Vision FINAL (Jira YAK-44).
- The partners are Toyota Royal Motors and BYD Royal Motors (Royal Motors Group), plus Ducati Tashkent.
- Jury scoring: tech complexity 25%, innovation 25%, impact 25%, presentation 15%, ROI 10%.
- Deliverables:
  - a web demo that works on a video the jury brings;
  - a deck;
  - a repo whose README lists every data and model licence.
- The partners will NOT supply footage, so everything is built and evaluated on open data.

### Reference product: SKAIVISION (US, 150+ dealerships)

This is from their CTO's slides at Embedded Vision 2025:
https://www.edge-ai-vision.com/wp-content/uploads/2025/09/BW04_Fayling_SKAIVISION_2025.pdf

- **Pipeline:**
  1. Detect people on every frame.
  2. Track them across cameras.
  3. Calculate "relationships and distances between tracked entities".
  4. Turn events into alerts.
- **Engagement rule:** "Employee within 2 meters of a customer for over 2 minutes."
- **Waiting customers:** each customer gets an ID on entry, and a timer runs until an employee interacts with them.
- **Why public trackers aren't enough:** ByteTrack and BoT-SORT score 90% on MOT but only 65% in dealerships.
- **Hardware:** an edge box with an 8 GB RTX 3050/4060 handles 15 SD cameras.
- **Privacy:** silhouettes only, no faces.
- **Not disclosed:** how they tell staff from customers. We solve this openly (section 4.4), and that is our main innovation claim.

## 2. Decisions already made by the owner

| Topic | Decision |
|---|---|
| Staff vs customer | **Auto-suggest + one click.** The system proposes roles with reasons, and a person confirms or flips each one with a single click. |
| Data | **Free stock video** (commercial-use licences), plus **research datasets for evaluation only** (non-commercial, never shipped or committed). |
| Reach | **Local only.** Commit locally. No git remote, no push, no deploy, no accounts. |
| Live mode | **Required.** Both uploaded video and a live webcam must work by morning. |

## 3. What the user sees

### Upload flow (main path)

1. **Upload.** The user uploads a video (mp4/mov/avi, up to about 5 minutes; longer videos are trimmed with a warning) and picks a preset:
   - "Dealership", which uses the SKAIVISION thresholds;
   - "Short clip", which uses tighter thresholds for 10–60 s clips (section 4.6).
2. **Progress.** A progress bar runs while perception works. Perception is cached per video.
3. **Confirm staff.** A gallery shows one card per person:
   - a face-blurred thumbnail;
   - the suggested role and its confidence;
   - up to three plain reasons, e.g. "same outfit as 2 others", "talked to 4 people", "on screen 95% of the clip".

   One click flips a role. Staff can be named (e.g. "Aziz"). "Looks right" goes on to the results.
4. **Results.**
   - An annotated video. Staff boxes are blue and customers orange. Lines join engaged pairs, and a timer above each waiting customer counts up. Faces are blurred.
   - A timeline with one row per person: present, waiting, greeted, engaged.
   - A scorecard for each salesperson and a summary for the store.
   - An alert feed: missed greet, long wait while staff were free, walk-out.
   - JSON and CSV export.
5. **Re-scoring.** Changing roles re-scores in seconds; there is no re-detection.

### Live flow (required)

1. The browser captures the webcam with getUserMedia.
2. JPEG frames go over a WebSocket at about 5 fps to the same engine.
3. The page draws overlays and timers on a canvas and shows events live.
4. Clicking a person on the live view toggles staff/customer. Auto-suggestion kicks in after about 30 s of observation.
5. This works on `localhost` without HTTPS.

## 4. How it works

### 4.1 Principle

One online engine serves both modes.
- **Live mode** feeds frames one at a time.
- **File mode** feeds the same engine from a file, then runs an offline pass on top. That pass merges broken tracks into persons, suggests staff, and re-runs the event engine with the final roles.
- **Cache:** perception output per video is cached (`.npz`/`.json`), so roles and thresholds can change without re-running models.

### 4.2 Units

Each unit has one job and a typed interface, and each is unit-tested.

| Unit | Does | Interface (sketch) |
|---|---|---|
| `sources` | File reader (OpenCV, with sampling to a target fps; default 5) and webcam/WebSocket frame source. Resizes to a max width of 960 px. | `iter_frames() -> (t_sec, frame_bgr)` |
| `perception.detect` | Person detector (plus car detection on one keyframe, for zones). Pluggable. | `detect(frame) -> list[Det(bbox, score, cls)]` |
| `perception.pose` | 17 COCO keypoints per person box. | `pose(frame, boxes) -> keypoints[N,17,3]` |
| `tracking` | ByteTrack from `supervision` (MIT). | `update(dets) -> list[TrackObs]` |
| `identity` | Appearance embedding per track: torso HSV histogram plus DINOv2-small, on a few crops. Merges track fragments into persons. Offline: clustering with a constraint that tracks overlapping in time cannot be the same person. Live: re-attach to a recently lost person. | `embed(crops)`, `merge(tracks) -> persons` |
| `geometry` | Distances in metres and facing direction, with no calibration (section 4.3). | `position_m(obs)`, `distance_m(a,b)`, `faces(a,b) -> bool or None` |
| `roles` | Staff suggestion with reasons (section 4.4). | `suggest(persons, interactions) -> list[RoleSuggestion]` |
| `events` | A pure state machine over per-frame person states. Emits greet, engagement, wait, walk-out and party events (section 4.5). | `step(t, states) -> list[Event]`; `run(timeline) -> list[Event]` |
| `metrics` | Scorecard for each salesperson and the store. Optional join to a sales CSV. | `scorecard(events, persons) -> Scorecard` |
| `render` | Annotated H.264 mp4 (via `imageio-ffmpeg`, because browsers can't play OpenCV's mp4v), face blur, timeline image, a contact sheet per event. | `render(video, timeline, events, out)` |
| `app` | FastAPI backend plus a static HTML/JS frontend with no build step. Endpoints: upload, job status, people/roles, results, export, `/ws/live`. | REST + WebSocket |
| `cli` | `analyze <video> --out <dir> [--preset] [--roles roles.json]`, `eval ...`, `bench`. | console entry point |

The core types (`Det`, `TrackObs`, `Person`, `RoleSuggestion`, `Event`, `Scorecard`) are plain dataclasses that serialise to JSON.

### 4.3 Distance without calibration

- **Per-frame pinhole back-projection.** It works with a moving camera too, and stock footage often moves.
  - Assume person height H = 1.70 m and a horizontal field of view (default 70°, configurable). That gives focal length f = W / (2·tan(FOV/2)).
  - For each person: depth Z = f·H / h_px, X = (u_foot − cx)·Z / f.
  - Distance between two people = ‖(X, Z)_i − (X, Z)_j‖.
- **Choosing h_px.** Use the most reliable body segment visible:
  - full height when standing and the head-to-ankle keypoints are confident;
  - otherwise shoulder-to-hip torso length (about 0.30·H). This covers sitting staff and people cut off at the bottom of the frame;
  - otherwise box height, with a "distance approximate" flag.
- **Optional manual calibration** in config: 4 floor points with real coordinates, giving a homography. This replaces the above when present.
- **Facing:** the normal of the shoulder line, with left/right decided by where the nose sits relative to the shoulders. Person A "faces" B if the angle is ≤ 60°. When keypoints are missing, facing is unknown and isn't required.
- **Tests:** project synthetic 3D people through a known pinhole camera. Estimated distances must be within 15% at 1–4 m.

### 4.4 Telling staff from customers (auto-suggest + one click)

No labelled data exists, so this uses transparent hand-set rules, evaluated on our labels:

| Feature | Staff signal |
|---|---|
| `outfit_peers` | Number of other persons whose torso appearance (DINOv2 + HSV) is within a threshold, i.e. a shared uniform or suit style |
| `contacts` | Number of distinct persons this person was within 2 m of for ≥ 5 s |
| `presence` | Share of the clip (or of the live window) the person is visible. Not used when the clip is shorter than 60 s. |
| `initiator` | Moved toward the other person before a contact started. This shows who approached whom. |
| `zone_dwell` (optional) | Time near a desk or behind-counter zone, if zones are defined |
| `looks_staff` (optional, only if it helps on labels) | Zero-shot image–text score from an MIT/Apache CLIP or SigLIP ONNX model: "a car salesperson in a uniform or suit" vs "a customer in casual clothes" |

- **Score:** a weighted sum mapped to a confidence. Persons above a threshold are suggested as staff, those below as customers, and those in between are marked "unsure".
- **Reasons:** the top contributing features, in plain words.
- **Corrections:** the user's correction always wins. Weights live in config and are tuned only on the labelled dev clips, never on the test clips.

### 4.5 Events and rules

All thresholds are in config. For each customer the states are:
**entered → waiting → greeted → engaged ⇄ browsing → left**

- **greet:** a staff–customer pair stays ≤ `greet_dist` with facing OK (or unknown) for ≥ `greet_time`. Logs the staff member and the time to greet (from entry).
- **engagement:** a pair stays ≤ `engage_dist` for ≥ `engage_time`. Ends after `gap_tol` seconds apart. Each engagement records its staff, customer, start and end.
- **wait_alert:** a customer who hasn't been greeted waits more than `wait_alert`. The alert lists the staff who were free at that moment (staff not in any engagement).
- **walkout:** a customer leaves without ever being greeted. "Left" means not seen for `leave_after` seconds, or the track ended at a frame edge.
- **party:** customers who arrive within 5 s of each other and stay ≤ 1.5 m apart for most of their first 30 s. Greeting any member greets the whole party.
- **handoff:** a customer is engaged by a second staff member after the first.
- **already present:** a customer visible within the first `leave_after` seconds of a clip has an unknown entry
  time. They can still be greeted and engaged, but they are left out of time-to-greet and wait statistics. This
  matters for short stock clips.

### 4.6 Presets

| Param | Dealership | Short clip |
|---|---|---|
| `greet_dist` / `greet_time` | 1.5 m / 5 s | 1.5 m / 2 s |
| `engage_dist` / `engage_time` | 2.0 m / 120 s | 2.0 m / 8 s |
| `gap_tol` | 3 s | 1.5 s |
| `wait_alert` | 60 s | 10 s |
| `leave_after` | 10 s | 3 s |

The Dealership preset's 2.0 m / 120 s engagement threshold is SKAIVISION's own rule.

### 4.7 Metrics

**For each salesperson:**
- greets;
- median time to greet;
- customers engaged;
- total and median engagement time;
- missed-while-free (customers who triggered a wait_alert while this person was free);
- walk-outs during their free time.

**For the store:**
- greet rate;
- median time to greet;
- walk-outs;
- total unattended waiting time.

**Optional:** upload a sales CSV (`salesperson,timestamp,sold`) to get engaged → sold conversion for each salesperson. Ship a clearly labelled synthetic sample CSV.

## 5. Stack and licences

The goal is a permissive (Apache-2.0) repo. ONNX Runtime on CPU is preferred, so torch isn't needed.
Add torch only if a measured need appears, and log the reason.

**Allowed:**
- Inference: onnxruntime (MIT), opencv-python-headless (Apache-2.0), numpy, scipy.
- Pose and detection: `rtmlib` (Apache-2.0), with YOLOX (Apache-2.0) and RTMPose (Apache-2.0) ONNX weights. RF-DETR Nano/Small (Apache-2.0) is the alternative detector.
- Tracking: `supervision` (MIT, ByteTrack).
- Appearance: DINOv2-small (Apache-2.0), via an ONNX export such as `Xenova/dinov2-small` on Hugging Face.
- Web and video: FastAPI and uvicorn (MIT/BSD), imageio-ffmpeg (BSD wrapper; record the bundled ffmpeg binary's licence).
- Evaluation: motmetrics (MIT).

**Banned:**
- ultralytics and boxmot (AGPL).
- Re-ID weights trained on Market-1501, DukeMTMC or MSMT17 (research-only data).
- Any non-commercial dataset inside a shipped artifact or a commit.
- Weights whose licence can't be found. Pick another model and log the choice.

**Detector choice:** pick by a CPU benchmark at M0 (i5-10210U). The candidates are YOLOX-tiny/s via rtmlib and RF-DETR-Nano ONNX. Choose the fastest one whose person recall on a MEVA sample (its geom boxes) is within 5 points of the best.

**Records:** the README gets a licence table with columns component | licence | trained on (data) | where used. `models/MANIFEST.json` lists each weight file with its URL, sha256 and licence.

## 6. Data and evaluation

All of this was checked on 2026-10-11. Free showroom footage is scarce, so the plan doesn't depend on it.

| Set | Source (verified) | Licence handling | Use |
|---|---|---|---|
| **MEVA** (Kitware/IARPA), 6–8 indoor clips (school, hospital, admin), about 140 MB per 5-min clip | Anonymous HTTPS from `https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/<date>/<hour>/<clip>.r13.avi`. KPF annotations (geom, activities, types YAML) in `gitlab.kitware.com/meva/meva-data-repo`, folder `annotation/DIVA-phase-2/MEVA/kitware/<date>/<hour>/`. 769 clips are annotated. The list is in `list-of-annotated-meva-clips.txt`. | **CC BY 4.0**, so commercial use is OK with attribution. Cite Corona et al., WACV 2021. Clips stay in `data/meva/`, uncommitted. Annotations may be copied with attribution. | **Main evaluation set.** It is fixed CCTV-style footage with 3,397 `person_talks_to_person` intervals (ground truth for engagement), 2,993 `person_enters_scene_through_structure` and 2,912 exits (ground truth for arrival and leaving), and 906 `hand_interacts_with_person`. Pick indoor clips with the most talk events. Also used for tracking (geom boxes) and as demo footage. |
| **Showroom stock clips** | The owner downloads them by hand into `data/stock/`, named `pexels-<id>.mp4`. Pexels blocks scripts. Verified lead: pexels.com/video/7154222 (car dealer and customer, by Antoni Shkraba). If the folder is empty, carry on without it and note that in PROGRESS.md. | Pexels licence: free for commercial use. Build `data/sources.csv` from the IDs. Never commit the clips. | Showroom look for the demo and deck. Agent-made labels for staff suggestion and greet timing. |
| Coverr / Mixkit | Reachable by script, but the "car dealership" search returned only driving and scrapyard clips (checked 2026-10-11). | Free licences | Only if you find real staff–customer scenes (retail is fine) |
| CAVIAR shopping-centre sequences | homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/ | Research use, eval only, in `data/eval_only/` | Optional extra tracking check |

- **What has no free data:** labelled staff vs customer roles. Staff suggestion is evaluated on the agent-labelled showroom clips (if any) and on the team's staged footage later. Say this plainly in EVAL.md.
- **Labels:** made by you (the agent) from frames sampled at 1 s intervals, stored in `labels/<clip>.json`:
  - persons with roles;
  - greet times;
  - engagement intervals;
  - walk-outs.

  Each file says `"labeller": "ai-agent, unreviewed"`. Split dev vs test by clip before tuning anything.
- **Report** (in `docs/EVAL.md`, with honest numbers and the commands to reproduce them):
  - Tracking on MEVA: MOTA and IDF1, with and without the identity merge.
  - Engagement detection on MEVA vs `person_talks_to_person`, roles ignored:
    - precision and recall of detected pair intervals (IoU ≥ 0.3);
    - median error in start time.
  - Arrival and leaving on MEVA vs the enter/exit-through-structure events: recall, and timing error.
  - Staff suggestion on the labelled showroom clips: accuracy before correction, and clicks needed per clip.
  - Greet-time error: median absolute error.
  - Speed:
    - file mode: real-time factor at 5 fps sampling, 720p;
    - live mode: processed fps with 1–4 people.
- **Targets:** a target missed is a finding to report, not a reason to fake numbers.
  - staff suggestion ≥ 80% correct;
  - greet-time median error ≤ 2 s;
  - talk/engagement recall ≥ 70% on MEVA;
  - file mode ≥ 1× real time;
  - live mode ≥ 4 fps.

## 7. Privacy and legal (Uzbekistan)

- **Faces:** never recognised or stored as identity. Every rendered output and thumbnail blurs the face region, found from the nose, eye and ear keypoints, or the top 20% of the box as a fallback.
- **On-premise:** processing is local (this runs on-premise). Uploads and job data are deleted after 24 h by a cleanup on app start.
- **Decision support:** the UI has a banner saying it is coaching support under the 2026 AI law (ZRU-1115). Scores never trigger actions on their own. The README notes that staff must be informed that analytics is in use.

## 8. Error handling

- **Unreadable or unsupported video:** a clear message. The job fails cleanly.
- **Nobody detected:** a results page saying so, with a tip ("camera too far or too dark").
- **Fewer than 2 persons, or nobody suggested as staff:** store metrics only, plus a prompt to mark staff.
- **Distance estimate unreliable** (box-height fallback for more than 50% of samples): a warning badge on the results.
- **Live mode:** if the WebSocket drops, the client reconnects automatically; if processing falls behind, frames are dropped and the processed fps is shown.
- **Model download fails:** retry 3 times, then try the alternative model listed in section 5, and log it.

## 9. Testing

- **TDD for all pure units:**
  - `geometry`, using synthetic pinhole scenes;
  - `events`, using scripted synthetic tracks, with one test per rule and preset (including party and handoff);
  - `metrics`;
  - `roles`, using synthetic feature vectors;
  - identity merge, using synthetic embeddings and the time-overlap constraint.
- **Integration:**
  - perception on a 10 s real clip gives at least 1 person and stable track IDs;
  - CLI `analyze` end-to-end gives every output file;
  - FastAPI TestClient covers upload → roles → results;
  - the live WebSocket path is fed from a video file and returns overlays and events.
- **Real webcam check:** grab one frame from the "Web Camera" device to prove capture works. Nobody will be in front of it at night.
- **Speed:** the full suite runs in under 3 minutes. Slow tests use a `slow` marker.

## 10. Milestones (each one ends demo-able and committed)

| # | Milestone |
|---|---|
| M0 | Repo skeleton, Python 3.12 venv, deps, model downloads plus manifest. Download 1 MEVA clip and its annotations, and check `data/stock/` for owner-supplied showroom clips. Run the CPU benchmark, choose the detector, record it in PROGRESS.md. |
| M1 | Pure core: geometry, events, metrics, roles, identity merge, all test-first on synthetic data |
| M2 | Perception chain on real video (detect, pose, track, embed, merge) with a per-video cache |
| M3 | CLI `analyze`: annotated H.264 mp4 with face blur, timeline image, scorecard JSON/CSV, event contact sheets |
| M4 | Web app upload flow: progress, staff confirm gallery, results dashboard, re-score without re-detection |
| M5 | Live mode: webcam → WebSocket → engine → overlay, click-to-toggle role, live alerts |
| M6 | Data and evaluation: fetch the remaining clips, label, split dev/test, tune on dev only, write `docs/EVAL.md` |
| M7 | Docs: README (problem, how it works, quickstart, results, licence table, privacy, pre-final declaration), `docs/PITCH_NOTES.md` (deck outline, SKAIVISION comparison, ROI model with clearly marked assumptions for the team to replace with Royal Motors numbers), `run.bat` / `run.sh` |
| M8 (stretch, in this order) | (1) sales CSV conversion join; (2) auto car zones (car detection on a keyframe) → dwell per car; (3) Uzbek/Russian UI toggle; (4) Dockerfile; (5) a deploy script, written but NOT run |

## 11. Machine constraints

- **Hardware:** Intel i5-10210U (4C/8T), 16 GB RAM, **no GPU**, Windows 11, Git Bash and PowerShell.
- **Disk:** about 9 GB free. The budget for the venv, models and data combined is ≤ 4 GB. Stop any download if free disk drops below 3 GB.
- **Python:** the default `python` is 3.14, which lacks wheels for some libraries. Use 3.12 at `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe` for the venv.
- **Other tools:** no ffmpeg on PATH (use imageio-ffmpeg). The Docker daemon isn't running, so don't depend on it.

## 12. Done by morning

1. `run.bat` starts the app at http://localhost:8000. Uploading a sample clip gives results in under 3 minutes on this laptop, and confirming staff re-scores it.
2. The live webcam page runs. It was tested via a file-fed WebSocket and a real webcam frame grab.
3. `pytest` passes from a clean run, and the output is pasted in PROGRESS.md.
4. `docs/EVAL.md` has real measured numbers. The README licence table is complete. The pre-final declaration is present.
5. PROGRESS.md ends with a "Morning summary": what works, what doesn't, measured numbers, decisions taken, and the team's to-do list:
   - review the agent-made labels;
   - film staged scenes;
   - fill in the ROI numbers;
   - deploy.
6. All work is committed locally. No remote exists.

## 13. Out of scope

- Face recognition.
- Live CRM or DMS integration.
- Tracking one person across multiple cameras (one camera per job; future work).
- Training or fine-tuning models (no GPU).
- Cloud deploy.
- Accounts and authentication.
