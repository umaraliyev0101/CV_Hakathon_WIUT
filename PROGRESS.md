# Progress

## Status

Tasks 1-2 done (skeleton; models, MEVA parser, detector benchmark). M1 (pure core, Tasks 3-11) done. M3 (Task 15: report, score, `pulse analyze`, check_outputs) done. See Milestones.

## Milestones

### M3 / Task 15: report, scoring pipeline, CLI analyze, check_outputs
- `pipeline.score` (cache only, no models), `write_outputs`, `analyze`; `report.build_report` writes every result file; `pulse analyze VIDEO --out DIR`; `scripts/check_outputs.py`; verify.sh runs MEVA dev clip (dealership) and `data/stock/wm-bangalore-04.mp4` (short) then checks outputs.
- Privacy: all outputs blurred with every detection at that sampled t; the raw keyframe stays in the cache dir.

### M0 / Task 1: skeleton
- Added package `pulse` (src layout), datatypes, presets/config, minimal CLI, `scripts/verify.sh`,
  data records. Tests: config, datatypes, slow ffmpeg smoke marker.

### M0 / Task 2: models, MEVA parser, detector benchmark
- Models downloaded to `models/<name>/` (gitignored); `models/MANIFEST.json` committed (sha256, bytes, licence). rtmlib subset vendored in `src/pulse/perception/vendor_rtmlib`; `rtmlib` pip package removed from the venv.
- `pulse bench detectors` (CPUExecutionProvider only), dev clip `2018-03-11.16-35-01.16-40-00.school.G423`: 60 evenly sampled frames, 35 GT person boxes (height >= 20 px at 960 wide), IoU >= 0.5, score thr 0.3. Showroom images: `data/stock/keyframes/*.jpg` (3).

| detector | person recall | cars per image (BLR-07 / BYD-t07 / BYD-t13) | ms/frame 960x540 (CPU) |
|---|---|---|---|
| yolox_nano | 0.343 | 4 / 1 / 2 | 9.9 |
| yolox_tiny | 0.371 | 7 / 0 / 2 | 25.0 |
| yolox_s | 0.629 | 6 / 1 / 1 | 82.2 |
| yolox_m | 0.629 | 5 / 1 / 3 | 196.3 |
| rfdetr_n | 0.514 | 5 / 1 / 5 | 88.8 |
| rfdetr_s | 0.571 | 6 / 1 / 4 | 158.0 |

Pose, 4 boxes on a 960x540 frame: rtmpose_s 32.5 ms, rtmpose_m 68.8 ms.

Rule: Fastest detector whose person recall (IoU>=0.5, GT height>=20px) is within 5 points of the best AND which detects >=1 car|truck|motorcycle (score>=0.3) on every showroom image. If none qualifies: among detectors within 5 points of best recall, the one with the most vehicle detections (ties: fastest).

**Chosen detector: `yolox_s` (rule met); pose `rtmpose_s`.** Raw numbers: `runs/bench/detectors.json` (gitignored). Caveat: only 35 GT boxes (sparse hallway clip), so recall differences of a few points are noise; the nano/tiny vs s/m gap (about 0.26) is large enough to trust. M0 only covers one scene; no showroom person GT exists.

### M1: pure core (Tasks 3-11)
- Geometry, timeline, zones, journey, roles, interest, stats, insights R1-R7, identity merge. Default suite: 140 passed, 2 deselected (slow). Decisions are logged below.

### M2 / Task 13: embeddings, camera motion, perception cache
- `pulse.pipeline.perceive` caches per video in `runs/cache/<key>/` (CPU, yolox_s + rtmpose_s + DINOv2-small, `sample_fps` 5, width 960).
- Measured RTF (analysed seconds / perception seconds, includes embeddings and keyframe car detection):
  - MEVA 10 s clip (60 s mark): RTF 1.49, perception 6.69 s, embed 0.43 s (6%).
  - Full 300 s MEVA dev clip (G423): RTF 1.65, perception 181.6 s, embed 5.3 s (2.9%); 2197 obs, 69 embedded tracks, 51 persons, one stable segment, keyframe t=149.8 s, 0 vehicles (hallway). Cache kept at `runs/cache/` for later tasks.

## Decisions

- 2026-10-11: Pre-flight was not run by the owner; the agent created the repo, copied the spec/CLAUDE.md/prompt; `data/stock/` and `data/synthetic/share/` are empty.
- 2026-10-11: rtmlib is vendored (subset) instead of pip-installed, because its wheel requires `opencv-python` + `opencv-contrib-python`, which clash with `opencv-python-headless`, and its multiclass YOLOX path drops scores (ByteTrack needs them).
- 2026-10-11: `supervision` pinned `<0.31`: `ByteTrack` is deprecated since 0.28 and removed in 0.31.
- 2026-10-11: No `docs/journal.md` logging gate (optional pre-flight step not run); PROGRESS.md is the log.
- Owner instruction 2026-10-11 02:45: never ask questions; pick the recommended/spec-supported option and log it here.
- Dev/test split fixed at plan time: dev = 2018-03-11.16-35-01.16-40-00.school.G423, 2018-03-11.12-00-00.12-05-00.school.G420; test = 2018-03-12.11-05-01.11-10-01.school.G423, 2018-03-13.17-10-00.17-15-00.school.G421, 2018-03-09.10-40-00.10-45-00.school.G419, 2018-03-09.10-15-00.10-20-00.school.G299.
- MEVA: 6 indoor clips downloaded to data/meva/ (CC BY 4.0, Corona et al., WACV 2021); see data/meva/CLIPS.md.
- data/stock/ and data/synthetic/share/ empty at start; agent will look for a commercially licensed showroom clip by script (not Pexels).
- 2026-10-11: roles reasons (controller ruling R3): a feature yields a reason only when its normalised value >= 0.25; top 3 such features by w*norm. Needed so a customer with presence 0.1 has `reasons == []`.

## Open problems

- None yet.
- 2026-10-11 (Task 2): detector = `yolox_s` per the spec section 5 rule (numbers under M0). `Config.detector` default changed from yolox_tiny to yolox_s.
- 2026-10-11 (Task 2): slow test uses MEVA dev frame 700, not 300: frame 300 of the dev clip has no GT person (first appears ~621); frame 700 has a GT person (CLIPS.md).
- 2026-10-11 (Task 2): `labels/split.json` created now (dev = G423 03-11 + G420; test = other four) so `meva_dev_clip` can read it.
- 2026-10-11 (Task 2): fixtures named `mini.*` (flow dialect) and `minidict.*` (quoted dict dialect); `pulse bench detectors --showroom` accepts several images.
- 2026-10-11 (Task 3): `position_m` full mode uses the mean of confident ankles (same as the foot point) for height; falls back to box if hips/shoulders are inverted. `faces` returns None when facing is unknown or the two positions coincide. `facing_dir` torso height with missing hips = vertical extent of confident keypoints / 3 (the brief's "box/3" has no box argument).
- 2026-10-11 (Task 5): build_timeline dedups per (person, timestamp) rather than (person, frame_idx): the brief's test reuses frame_idx 0..4 for two tracks at different times and expects 10 samples. Invalid positions -> pos_m=None, dist_method="invalid"; speed uses own Z, else the person's median valid Z, else None; filter_stable uses t0 <= t < t1 (last segment includes t1), samples outside all segments are kept.
- 2026-10-11: Task 7 interest: assistance/episode runs use per-(customer,staff) near-times bridged at gap <= 1.0 s; open_alerts uses the customer's last visit per car zone and only samples within that visit.
- 2026-10-11 (Task 8): stats per_car 'unserved'/'deep_interest' count distinct customers per car; peak_t is None when there are no customers; heatmap/paths use customers from `persons` (missing role = customer); hold_rate None when a car has no visitors.
- 2026-10-11 (Task 9): insights: R1 m = all episodes at the car (episodes are deep-interest), window = 60-s bucket of unserved t_start; R5 run length = t_last - t_first; R7 gap compared with 1e-9 tolerance so 0.20 fires despite float error; moments only filled for R1/R2 (episode-based), other rules have none.
- 2026-10-11 (Task 11): identity merge chain rule: clusters A and B join via candidate (a, b) only if a is A's latest-ending track, b is B's earliest-starting track, and no track of A overlaps any track of B by > 0.2 s, so each person stays a time-ordered chain. Cosine on L2-normalised embeddings; None/zero embeddings never merge. Singleton tracks become Persons; ids 1..n by first_t (ties by smallest track id). `summarize` skips non-finite foot points. `reattach` uses gap = now - lost.t_end for both the leave_after limit and spatial plausibility.
- 2026-10-11 (Task 12): iter_frames stops at t >= max_duration (exclusive); sources slower than sample_fps skip duplicate targets; VideoInfo has an extra `fps_assumed` flag (fps 0/NaN -> 25). probe() decodes one frame. meva_10s fixture loads scripts/make_clip.py via importlib (scripts is not a package). Engine timings are rolling means over the last 50 frames.
- 2026-10-11 (Task 12): Tracker passes `frame_rate=30` (not sample_fps as the brief says) to sv.ByteTrack: supervision computes max_time_lost = int(frame_rate/30*lost_track_buffer), so frame_rate=5 turned the 10-frame buffer into 1 frame and a 3-frame gap lost the id (brief test failed). With 30, the buffer is exactly int(track_buffer_s*sample_fps) sampled frames.
- 2026-10-11 (Task 13): crops per track = 3 (controller ruling, brief said 5), score >= 0.5, one candidate per second (thinned to 12), picked evenly; tracks < 1.0 s get HSV-only embeddings padded with a zero DINO part so all vectors have equal length (such a vector can never reach merge_sim_thr against a full one: cosine <= 0.71). Crops stored 112x112, not full frames.
- 2026-10-11 (Task 13): camera-motion cut test uses a 64-bin grayscale histogram correlation (< 0.5), since only grays are kept (brief says HSV histogram). Cut displacement is reported as 1.0. Segment t1 = start of the next segment (contiguous cover). Keyframe = middle frame of the longest stable run; re-read afterwards with iter_frames. The raw keyframe is only written to the cache dir.
- 2026-10-11 (Task 13): `PerceptionCache` has an extra `keyframe` field (raw BGR, saved but not loaded; use `keyframe_path`). `cache.save` writes to a temp dir and renames; `exists` requires all five data files. `video_key` = sha1(file sha1 + sample_fps, max_width, max_duration_s, detector, pose_model).
- 2026-10-11 (Task 13 fix round 1, controller ruling): the zero-padded HSV-only scheme is removed. Every track with >= 1 crop gets the full HSV+DINOv2 embedding (<= 3 crops; if no crop reaches score 0.5 its best-score crop is used); a track with no crop has no embedding (never merged). DINOv2 unavailable -> HSV-only for all tracks. `PerceptionEngine.reset()` is called at the start of each perceive; missing/corrupt video raises VideoError before any hashing or cache dir; cache temp dir via tempfile.mkdtemp; Embedder uses the repo-relative models dir and the engine's providers; non-stable segments have same_view=False.
- 2026-10-11 (Task 14): Coach-compatible wrapper is named `render_timeline(video, timeline, events, out, cfg)` (Coach's `render(video, timeline, events, out)`); note for INTEGRATION.md. `render(..., observations_by_t=)` blurs faces on frames lacking overlays; frames with neither get zone outlines only (zones from the first overlay) and cannot be blurred. Output frames cropped to even dimensions (macro_block_size=2). Face min side = 0.12 x person-box width applied to the keypoint box before 0.6x padding. Contact-sheet tiles are 320 px wide; zone label uses "|" instead of the middle dot (cv2 Hershey fonts are ASCII only).
- 2026-10-11 (Task 14 fix round 1): `render` and `contact_sheet` take `observations_by_t` as a REQUIRED argument (privacy fail-closed; caller passes every sampled frame's detections; blur covers detected persons only). `render_timeline` derives it from timeline samples. render writes `<out>.part.mp4` then os.replace.
- 2026-10-11 (Task 15): `build_report` takes extra kwargs `video=None, render_video=False` (thumbnails, contact sheets and the mp4 need frames). Without a video those are skipped and `check_outputs` only checks what `report.json["files"]` lists. Alert overlay = customer in a car-zone visit inside an unserved episode with dwell >= alert_after. Visit/zone stats use the stable-filtered timeline; overlays use the full timeline. `analysed_share` = stable same-view seconds / analysed duration. Reports store the video basename only.
- 2026-10-11 (Task 15 fix 1): keyframe re-read uses cfg.max_width and any keyframe image is resized to the analysis frame (aspect mismatch -> grey, never unblurred); keyframe blur uses the exact sampled-t detections (nearest within half a period, with a warning); store stats use the UNFILTERED timeline (journeys/assist/episodes stay stable-only); build_report cleans only files it owns before writing.
- 2026-10-11 (Task 15 fix 2): cache v2 stores every raw person detection (score >= min(0.2, det_score_thr)) in raw_dets.npz; the tracker still only gets detections above det_score_thr. All blurring uses the union of tracked obs at t and t +- 1 sample plus raw detections. RTF after rebuild unchanged: MEVA dev 1.66 (perception 180.9 s), stock clip 0.75 (82.1 s).
- 2026-10-11 (Task 16): `perceive` takes an optional `embedder` (default unchanged); the app passes `engine.embedder` when the engine has one (test stubs use an HSV-only Embedder). `engine_factory` is called with no args (or with the cfg if it declares a parameter). After perception the worker writes preview outputs (keyframe, blurred thumbnails, people.json) with automatic zones; `results` stay 404 until the first `score`. `score` writes JSON/CSV/thumbnails synchronously; the annotated video is rendered in the worker (`video_state` in job status) and discarded if zones changed meanwhile. `roles` `skip: true` clears overrides (suggested roles only); `people[].role` is computed live as override > imported > suggested. `export.csv` = per_car.csv. User zones/roles live outside `out/` so report regeneration never deletes them. pytest ignores starlette's TestClient/httpx deprecation warning via pyproject filterwarnings. smoke clip = MEVA dev clip 40 s to 60 s (the 15 s window has only ~1 s of people).
