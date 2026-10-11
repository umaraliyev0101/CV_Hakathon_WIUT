# Showroom Coach: progress log (Jira YAK-44)

Design: `docs/superpowers/specs/2026-10-11-showroom-coach-design.md`. Plan: `docs/superpowers/plans/2026-10-11-showroom-coach.md`.

## Log

- 2026-10-11 M0 / Task 0.1: package skeleton, `pyproject.toml` and pytest harness. Environment: Python 3.12.6 venv at `.venv` (not the default 3.14), Windows 11, i5-10210U, 16 GB, no GPU. Free disk on C: 21G; `.venv` is 708M. `pip install -e ".[dev]"` succeeded and `pytest -q` passes.
- 2026-10-11 M0 / Task 1.0: core dataclasses (`types.py`), `to_jsonable`, `Thresholds`/presets, `CameraConfig`, `RoleWeights`, `repo_root()`/`jobs_dir()`/`cache_dir()` (`config.py`). Presets match spec section 4.6.
- 2026-10-11 M0 / Task 0.2: model registry, retrying download with sha256 check, `showroom-coach fetch-models`, `models/MANIFEST.json`. Fetched 6 models (yolox_nano 3.7 MB, yolox_tiny 20.2 MB, yolox_s 35.9 MB, rtmpose_s 21.9 MB, dinov2_small 24.5 MB, rfdetr_nano 112.6 MB): `models/` is 209M, `.venv` is 708M, free disk on C: 21G. All six load in onnxruntime; input shapes match the registry notes.


### M0 summary (2026-10-11, ~03:00): skeleton, models, data, detector benchmark

- **Tests:** 87 passed in 15.7 s (`.venv/Scripts/python -m pytest -q`, after Task 0.3's addendum).
- **Data (MEVA, CC BY 4.0, eval only, gitignored):** 7 indoor clips from 5 cameras (school G299 x2, G330 x2, G421, G423; bus G331), 1.14 GB of video plus 126 MB of KPF annotations. They contain 297 `person_talks_to_person`, 56 enter and 68 exit events. Notes:
  - Talks are often groups of 3–16 people.
  - Both G330 clips have no enter/exit events.
  - `types.yml` also labels `other` and `bag` tracks, so filter to `person`.
  - Three derived sample cuts are in `data/samples/`: `meva_a_60s.mp4` (G299, 20–80 s, 960 px), `meva_a_60s_720p.mp4` and `meva_b_60s.mp4` (G421, 180–240 s).
- **`data/stock/` is empty:** the owner hasn't added Pexels showroom clips. M6 carries on with MEVA alone, and staff suggestion can't be evaluated on showroom footage.
- **Models (209 MB):** yolox_nano/tiny/s (Megvii COCO ONNX), rtmpose_s (Body7), dinov2_small (int8 ONNX), rfdetr_nano (benchmark only). Disk use is venv 708 MB + models 209 MB + data 1.27 GB ≈ 2.2 GB, under the 4 GB budget.
- **Detector benchmark:** school G299 clip, 60 frames at width 960, 1303 GT person boxes (median 47 px tall), score 0.3, IoU 0.5. Precision is a lower bound because MEVA only boxes annotated actors, and 13% of GT boxes are duplicate ids of the same person, so recall is understated for every model alike.

| detector | ms/frame (median of 3 noisy runs) | recall | recall, people >= 80 px |
|---|---|---|---|
| yolox_nano | 21 | 10.1% | 53.4% |
| yolox_tiny | 50 | 12.6% | 63.1% |
| yolox_s | 176 | 18.0% | 68.9% |
| rfdetr_nano | 241 | 21.9% | 68.0% |
| **yolox_nano@2** (2 overlapping tiles) | **36** | **26.8%** | 67.0% |
| yolox_tiny@2 | 91 | 27.7% | 70.9% |
| yolox_s@2 | 361 | 30.9% | 69.9% |

  - **Retained log** `runs/bench/detectors.json`, re-run after the tile-merge seam fix, supersedes the medians above: yolox_nano 10.1% recall / 16.7 ms, yolox_tiny 12.6% / 36.3 ms, yolox_s 18.0% / 138 ms, rfdetr_nano 22.0% / 205 ms, **yolox_nano@2 26.9% / 31.6 ms**, yolox_tiny@2 27.8% / 75.7 ms, yolox_s@2 31.0% / 322 ms (median ms per frame). The choice is unchanged: cut-off 26.0%, so yolox_nano@2.
  - **File mode:** spec rule "fastest within 5 points of the best recall": best is 30.9%, so the cut-off is 25.9%, and the pick is **yolox_nano@2** (`config.DEFAULT_DETECTOR`).
  - **Live mode:** the same rule over people >= 80 px also picks **yolox_nano@2** (`config.LIVE_DETECTOR`).

### M1 summary (2026-10-11, ~04:00): pure core, test-first on synthetic data

- Units: geometry (pinhole, full → torso → box fallback, facing), events (greet, engagement, wait, walk-out, party, handoff, already-present, browsing), metrics (staff and store scorecards), roles (outfit_peers, contacts, presence, initiator, with plain reasons), identity (constrained merge plus live re-attach).
- Each unit had an independent review, and the fix rounds caught real bugs:
  - A wait alert could fire for customers who had already left, or contradict a greet that was backdated.
  - Parties of 3 or more emitted duplicate party events.
  - A customer later marked as staff left a phantom visit in the store metrics.
  - A same-frame tracker ID switch was never re-attached in live mode.
  - The canonical seated pose was misread as standing.
- Geometry accuracy on synthetic pinhole people: pair distances within 0.1% (target 15%) at 1–4 m. Synthetic data only proves the maths; real accuracy is unmeasured.

### M2 summary (~05:00): perception on real video, with a per-video cache

- Chain: sampled frame source (5 fps, ≤ 960 px, ≤ 300 s; unreadable files fail cleanly, files that end early give partial results with a warning) → yolox_nano@2 → supervision ByteTrack → RTMPose-s (batched, every 2nd frame per track, keypoints carried in between) → DINOv2-small + HSV descriptors on real tracks only. The cache (`runs/cache/<key>/perception.npz/json`) is keyed by file hash plus every content-changing parameter, and loads without pickle.
- **File-mode speed** (wall clock, i5-10210U, honest; other agents were loading the CPU during some runs): `meva_a_60s` (960 px, about 11 people per frame) gives **RTF 0.976** (61.5 s for 60 s). The full G421 clip (1920×1072, 5 min, about 12 people per frame) gives **RTF 0.713** (421 s for 300 s). Before batching and the pose stride the figures were 0.706 and 0.506. **The ≥ 1× target is met only on uncrowded clips.** Pose and DINOv2 dominate the time.

### M3 summary (~05:30): analysis, rendering, export

- Offline pass: noise-track filter → constrained identity merge (112 tracks become 31 persons on `meva_a_60s`) → geometry → role features and suggestions → events (all sampled frames, t0 = 0) → scorecards. Re-scoring takes 0.07–0.1 s and never re-detects.
- Rendering: face-blurred H.264 at 10 fps. `meva_a_60s` renders in 8.1 s (7.4× real time). The privacy guard test measures face-region std ≤ 9.2 with blur against about 45 without it. Exports are timeline PNG, contact sheets, results.json and events/scorecard CSV.

### M4 summary (~06:00): web app, upload flow

- FastAPI app (`showroom-coach serve`, `run.bat`): upload (411/413/415/400 guards) → background perception (one job at a time; unreadable video fails cleanly; a video that ends early is analysed up to its end, with a warning) → confirm-staff gallery (blurred thumbnails, suggestion with confidence and ≤ 3 reasons, one-click toggle, names) → results (annotated video with HTTP Range, timeline, scorecards, alerts, JSON/CSV export). Re-scoring takes about 0.14 s and never re-detects. Each re-render is versioned. Job data older than 24 h is deleted at startup.
- **Real smoke via `run.bat` + `scripts/smoke_api.py`** (`meva_b_60s.mp4`, CPU shared with other agents): `SMOKE OK`, 331 requests. Upload 0.08 s, needs_confirm after 253 s, roles round-trip 0.46 s, video ready 67 s later. An earlier run on `meva_a_60s` reached needs_confirm in 122 s. **The < 3 min target was missed under load**; it is re-measured idle in the Morning summary.
- Frontend: vanilla JS, no CDN, works offline. Verified with Playwright at 1280/1920 px, including the nobody-detected and connection-lost states, with no console errors.

### M5 summary (~06:30): live mode

- `/live`: getUserMedia → JPEG over `/ws/live` (stop-and-wait, ≤ 5 fps, auto-reconnect) → the same engine → canvas overlay. Faces are blurred on the canvas for tracked and untracked detections. Clicking a box toggles staff/customer, and auto-suggestion starts after 30 s.
- Live people are confirmed (≥ 3 observations over ≥ 1 s) before they count, so no phantom customers. Alerts follow role changes. One person ID is never shown on two boxes (the duplicate-ID bug was fixed: 66 → 0 affected frames on `meva_b`).
- **Real webcam frame grab:** `showroom-coach webcam-check` captured "Web Camera" at index 0 (DSHOW), 640×480. Mean brightness was 8/255 (dark, empty room at night), 0 people, and a blurred copy was saved. The live page was also tested with the real camera in headed Chrome, reaching "Live" in 1.4 s.
- **Live fps (corrected after review):** an earlier note claimed 5.2–6.3 fps for about 4 people, but no retained log supports it, so it is withdrawn. The retained contended logs show 2.1–3.3 fps for the G423 window. Idle-machine verification numbers are in the Morning summary: 4.45 fps end to end over the real WebSocket and 3.61 fps in the in-process bench, both at 3.8 people per frame.

### M6 summary (~07:15): MEVA evaluation (details in `docs/EVAL.md`)

- 7 clips × first 150 s. The split was written before tuning: dev = 3 clips, test = 4 clips, 2 of the test cameras unseen in dev. Tuning was done on dev only (64 configs, flat). Test exposure: 5 post-split test runs (2 system, 3 oracle). Test clip G421 was also used before the split for a smoke evaluation, live and dedupe debugging and speed runs. All of this is listed in EVAL.md.
- **TEST (system):**
  - talk recall 0.5% strict / 1.1% twin-aware (target ≥ 70% ❌);
  - talk precision on mapped detections 14.3% / 10.3%;
  - enter/exit recall 2.9% / 20.6%, with median timing error 1.1 / 0.8 s;
  - persons IDF1 13.5%, MOTA −27.4%.
- Root cause: the detector matches only 17.6% of annotated MEVA person boxes (people median 47 px tall), so most talk pairs involve someone we never track. An oracle-tracks ablation (perfect detection) separates this from the engagement logic; see EVAL.md.
- Staff suggestion and greet-time error are **not measurable**: no labelled showroom footage exists (`data/stock/` empty).

## Decisions

- Python 3.12 venv, because the machine default (3.14) is outside the `requires-python` range `>=3.11,<3.14`.
- Registry sha256 values are pinned to the files fetched on 2026-10-11, so later fetches fail loudly if an upstream file changes (for rtmpose_s the hash is of the extracted `end2end.onnx`, not the zip).
- Every model is saved as `models/<name>/<name>.onnx`. From the rtmpose zip only the `end2end.onnx` member is kept (copied out by basename, never extracted by path).
- `rfdetr_nano` is a benchmark candidate only; it is a third-party ONNX export of Roboflow RF-DETR (Apache-2.0), recorded in the manifest note.
- `to_jsonable` maps NaN and infinity to `null`, because they are not valid JSON and `JSON.parse` rejects them.

- **Tiled detection (ruling, M0):** the first benchmark picked yolox_s (18.0% at 140–176 ms/frame), which uses 70–90% of the 200 ms/frame file-mode budget. Recall was limited by small people, so I added a 2-tile mode and benchmarked it as extra candidates under the same rule. yolox_nano@2 wins: higher recall, 4–5× faster.
- **Pose is skipped for boxes under 40 px tall:** keypoints there are noise and cost about 11 ms each. Geometry falls back to box height for those people.
- **Branch:** work happens on `yak-44/showroom-coach` and `main` is fast-forwarded at the end (local only).
- **`data/stock/` is empty:** I continued with MEVA alone, as the spec allows.

- **Wait alerts:** a `wait_alert` is held while a greet streak for that customer is still open, so file-mode metrics never contradict the backdated greet timestamps.
- **Standing test:** standing vs seated is decided by (ankle_y − hip_y) / torso_px ≥ 1.35, replacing the plan's nose-ankle ≥ 2.4 rule, which misread seated people at about 4–5 m.
- **Live identity call order:** `end_step(ids_now)` runs first, then `resolve`, so a same-frame ID switch re-attaches.
- **Store metrics:** they skip visits by people whose final role is staff.
- **Tiled detection edge rule:** in the 2-tile overlap a box is dropped only if it touches its own tile's inner cut. This keeps nested people (a child in front of a parent).
- **Tracker floor:** the ByteTrack new-track floor equals the detector threshold (0.35). supervision is pinned to `>=0.28,<0.31`, because ByteTrack is removed in 0.31.
- **DINOv2 batch size:** DINOv2 runs at batch 1, because the int8 dynamic quantisation makes embeddings depend on what else is in the batch.
- **Pose:** RTMPose is batched per frame and runs every 2nd sampled frame per track, with keypoints carried along with the box in between. Only real tracks (≥ 3 observations spanning ≥ 1 s) are embedded.
- **Damaged or short files (revised at the end of the night):** a container reporting more than 240 fps, or a video with no readable frames, fails cleanly with a `VideoError`. A video whose frames stop before 90% of the container's count was first rejected. The final review showed valid mkv/webm files with a longer audio track trip that check, and the duration skew caused false walk-outs. Such a video is now analysed up to its last frame (effective duration = last frame + one sample interval) with the warning "video ended at X s of an expected Y s — the file may be damaged or have a longer audio track; results cover the first X s".
- **Re-rendered videos:** each re-render is written to `annotated-<rev>.mp4`, because Windows browsers lock the mp4 that is playing.
- **Default test run:** it excludes `@pytest.mark.slow`; run those with `pytest -m slow`.

- **MEVA scope:** evaluation covers the first 150 s of each of the 7 clips, because full clips need more than 1 h of CPU. The split was fixed in `data/meva/split.json` before tuning.
- **Twin-aware mapping:** MEVA often annotates one person as several actors, so the evaluator reports the strict (as planned) and twin-aware id mapping side by side.
- **Concurrent commits:** commit `37aa229` contains both the MEVA evaluator (Task 6.2) and the analyze/serve CLI (Task 3.3), because concurrent agents shared the git index. History was kept, not rewritten.
- **Upload limits:** uploads without Content-Length (chunked) get 411, so the 1 GB cap is enforced before Starlette spools the body.
- **Launchers:** `run.bat`/`run.sh` refuse a venv outside Python 3.11–3.13. `.gitattributes` pins eol for the scripts.

## Open problems

- (resolved in M7) The README licence table lists every model and dataset; `models/MANIFEST.json` keeps the hashes.
- (fixed in 746394c) `write_manifest()` now keeps a row for every registry entry, so a partial fetch can no longer shrink the licence record.

## Morning summary (2026-10-11, written at the end of the overnight run)

All work is on branch `yak-44/showroom-coach`, committed locally (no remote exists, nothing pushed). `main` is still at `07a01f6`: merging is the owner's decision. `git checkout main && git merge --ff-only yak-44/showroom-coach` fast-forwards it. Every number below comes from a command whose output is in `runs/verify/` or `docs/EVAL.md`.

### What works (verified tonight)
- **Upload flow:** `run.bat` starts the app at http://localhost:8000. You can upload a clip, confirm staff in a gallery (face-blurred thumbnails, suggested role with confidence and up to 3 plain-language reasons, one-click toggle, names), and get results:
  - a face-blurred annotated H.264 video (staff blue, customers orange, engaged-pair lines, wait timers, alert banners);
  - a per-person timeline;
  - a scorecard per salesperson and a store summary;
  - an alert feed and contact sheets, shown in the results page (blurred; three frames around each event);
  - JSON/CSV export and an optional sales-CSV conversion.
  - Flipping a role re-scores in about 0.14 s and never re-runs detection. The video re-renders in the background.
- **Live mode:** the `/live` page runs the webcam over a WebSocket into the same engine, drawing overlays and timers on a canvas.
  - Faces are blurred on the canvas, including people who are not yet tracked.
  - Clicking a box toggles staff/customer, and auto-suggestion starts after 30 s.
  - It reconnects on its own after a drop.
- **CLI:** `showroom-coach analyze | serve | eval-meva | bench | webcam-check | fetch-models`.
- **Privacy and licences:**
  - No face recognition. Body-appearance embeddings are used only to merge track fragments within one video.
  - Faces are blurred in every rendered output.
  - Processing stays on this computer, and job data is deleted after 24 h.
  - The README has the licence table, the privacy notes and the pre-final declaration. The `LICENSE` file is Apache-2.0.

### Verification evidence (fresh runs, near-idle machine)
- `pytest` (fresh, `-p no:cacheprovider`): **835 passed, 11 deselected** (`runs/verify/pytest_final.txt`). Run time was 289 s on the final run with desktop apps active and 344 s while a reviewer ran tests in parallel, against 173–175 s on two quiet runs earlier. **The < 3 min target is met only on a quiet machine.** The 11 deselected tests are `@pytest.mark.slow` (real models and MEVA files; run them with `pytest -m slow`). They passed in their task runs but were not re-run together at the end.
- `showroom-coach analyze` on two clips (fresh cache):
  - `meva_a_60s`: RTF 1.07×, 31 people, all outputs written;
  - `meva_b_60s`: RTF 0.56× (possibly contended), 33 people, all outputs written.
  - Outputs: `runs/verify/analyze_{a,b}/`.
- `run.bat` + `scripts/smoke_api.py --budget 180` on `meva_a_60s`:
  - upload → staff confirmation 60.3 s (BUDGET MET);
  - roles round-trip 0.14 s;
  - video ready 12.2 s later;
  - SMOKE OK, 85 HTTP requests.
- File-fed live WebSocket (`scripts/live_file_client.py`, `meva_g423_20s`, 100 frames): 4.45 fps end to end, 5.21 fps round trip, 3.8 people per frame.
- Real webcam frame grab (`showroom-coach webcam-check`): "Web Camera" at index 0 (DSHOW), 640×480 captured, exit 0. The room was dark and empty, so 0 people.
- `showroom-coach bench`:
  - file mode, 720p, 12 people per frame: RTF 0.87×; render 4.3× real time;
  - live in-process, 3.8 people per frame: 3.61 fps mean.

### Measured numbers vs the spec targets (honest)
| Target (spec §6/§12) | Measured | Status |
|---|---|---|
| Upload → results < 3 min | 60.3 s for a 60 s clip, idle; 122–253 s under heavy load | ✅ idle |
| File mode ≥ 1× real time | 0.87× on the 720p benchmark (12 people per frame); 1.07× on the same footage at 960 px; 0.56× on a 14-person clip | ❌ on crowded scenes (borderline) |
| Live ≥ 4 fps with 1–4 people | 4.45 fps end to end over the WebSocket; 3.61 fps in the in-process bench (3.8 people per frame) | borderline |
| Talk/engagement recall ≥ 70% (MEVA) | test: 1.1% twin-aware / 0.5% strict; with oracle tracks (perfect detection) 44.1% / 32.3% | ❌ |
| Enter/exit (MEVA) | test recall 20.6% twin-aware / 2.9% strict, median timing error 0.8 / 1.1 s | — (no target) |
| Tracking (MEVA) | persons IDF1 13.5%, MOTA −27.4% (raw tracks 15.5% / −28.1%) | — |
| Staff suggestion ≥ 80% | not measurable: no labelled showroom footage (`data/stock/` was empty) | n/a |
| Greet-time median error ≤ 2 s | not measurable, for the same reason | n/a |
| Detector (M0 benchmark, MEVA) | yolox_nano@2 (2 tiles): recall 26.9% at 31.6 ms/frame (retained log), chosen by the spec rule | — |

### What doesn't work / known weaknesses
- **MEVA engagement recall is far below target.** The detector finds only 17.6% of annotated MEVA people, who are tiny (median 47 px tall), so most talks involve someone we never track. Even with perfect tracks, the 2.5 m / 8 s rule reaches 44% recall. Showroom cameras with larger people should do much better, but this is **unmeasured**.
- **Staff suggestion has never been checked on real staff/customer footage.** On crowded school clips the heuristics break down: 16 of 33 people were suggested as staff in one 60 s clip. Real showroom footage is needed.
- **Speed:** file mode is under 1× real time on crowded scenes (pose and DINOv2 dominate), and live is borderline at about 4 fps. The two biggest levers are fewer crops per track and a lighter appearance embedder.
- **Detector misses:** people the detector misses entirely, for example cut off at a frame edge, are not blurred. This is documented in the README. Mitigation: camera placement that shows full bodies.
- **Distance is calibration-free:** a pinhole model with H = 1.70 m and a 70° field of view. Optional 4-point floor calibration exists in config but has no UI.
- **Stretch goals not done:** auto car zones and the Uzbek/Russian UI. The Dockerfile is written but not built (Docker wasn't running). The deploy script is an example only and was never run.

### Every decision taken alone (rulings; full list with costs is under "Decisions" above)
1. **Branch, no worktree:** worked on branch `yak-44/showroom-coach` in the main checkout, not a worktree. `main` is left untouched for the owner to merge.
2. **No `Co-Authored-By` trailers:** CLAUDE.md overrides the tool's default.
3. **Detector:** chosen by the spec rule after adding a **2-tile mode**. yolox_nano@2 beat yolox_s (26.8% vs 18.0% recall, 4–5× faster).
4. **Tile-merge seam rule:** in the overlap, a box is dropped only if it touches its own tile's cut (keeps nested people).
5. **Standing vs seated:** decided by (ankle − hip) / torso ≥ 1.35 instead of the plan's 2.4 nose-ankle ratio, which misread seated people.
6. **Event-engine rules:**
   - a wait alert is held while a greet streak is open;
   - "browsing" state added;
   - visits by people finally marked staff are excluded from store metrics;
   - live counts only confirmed people (≥ 3 observations over ≥ 1 s).
7. **Live identity:** call order `end_step` then `resolve`, and one active track per person ID.
8. **Perception speed:** batched pose, pose every 2nd frame per track (keypoints carried), embeddings only for real tracks. crops_per_track kept at 4 for quality.
9. **Dependencies and tracker:** supervision pinned `>=0.28,<0.31`; tracker new-track floor = detector threshold (0.35).
10. **Re-renders:** written to `annotated-<rev>.mp4` (Windows file locks).
11. **Uploads:** no Content-Length / chunked → 411, so the 1 GB cap holds.
12. **Videos that end early:** at first such videos were rejected as damaged. The final review showed valid mkv/webm files with longer audio trip that check, so they are now analysed up to the last frame, with a warning, and events use the effective duration.
13. **Default pytest run:** excludes `@pytest.mark.slow`; run those with `pytest -m slow`.
14. **MEVA evaluation scope:** first 150 s of 7 clips. The dev/test split was written before tuning, and twin-aware mapping is reported next to strict. Test exposure is disclosed in EVAL.md: 5 post-split runs, plus test clip G421 used before the split.
15. **Oracle ablation:** added to separate detection failures from the engagement rule.
16. **History left as is:** commit `37aa229` mixes two tasks (a shared git index between concurrent agents). History was not rewritten.
17. **Thumbnails and live canvas:** they also blur the top 20% of every tracked box, not only the keypoint face region (final review).
18. **Stretch built in spec order 1, 4, 5:** sales-CSV join, Dockerfile, guarded deploy example. Items 2 (car zones) and 3 (uz/ru UI) were skipped for time.

### Team to-do list
- **Review the agent-made labels:** there are none yet, because there was no showroom footage. Download the Pexels clips listed in `data/TODO_DOWNLOADS.md`, label staff/greets/engagements, and re-run the evaluation.
- **Film staged showroom scenes** (staff + customers, a known greet time) to measure staff suggestion, greet-time error and real engagement recall. These are the numbers the jury will ask for.
- **Put real Royal Motors numbers into the ROI model** in `docs/PITCH_NOTES.md`. Every input there is marked ASSUMPTION.
- **Deploy (owner's decision):** the Dockerfile and `scripts/deploy_example.sh` are written but were never built or run. Decide on hosting, HTTPS for remote live mode, and the GPL FFmpeg / Body7 licence questions noted in the README.
- **Integrate the teammate's feature** (repo `umaraliyev0101/CV_Hakathon_WIUT`) in its own Jira task. This session declined it as outside YAK-44. A GitHub token was pasted into the chat and should be revoked.
- **API reference for integrators:** `docs/API.md` (REST + `/ws/live` JSON shapes).
- **Demo-day notes:**
  - pre-run the jury clip, and do not restart the server afterwards;
  - don't run an upload while `/live` is open;
  - keep `runs/verify/analyze_a/` as an offline fallback.
