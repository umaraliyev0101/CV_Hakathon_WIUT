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

- Chain: sampled frame source (5 fps, ≤ 960 px, ≤ 300 s, fails cleanly on damaged files) → yolox_nano@2 → supervision ByteTrack → RTMPose-s (batched, every 2nd frame per track, keypoints carried in between) → DINOv2-small + HSV descriptors on real tracks only. The cache (`runs/cache/<key>/perception.npz/json`) is keyed by file hash plus every content-changing parameter, and loads without pickle.
- **File-mode speed** (wall clock, i5-10210U, honest; other agents were loading the CPU during some runs): `meva_a_60s` (960 px, about 11 people per frame) gives **RTF 0.976** (61.5 s for 60 s). The full G421 clip (1920×1072, 5 min, about 12 people per frame) gives **RTF 0.713** (421 s for 300 s). Before batching and the pose stride the figures were 0.706 and 0.506. **The ≥ 1× target is met only on uncrowded clips.** Pose and DINOv2 dominate the time.

### M3 summary (~05:30): analysis, rendering, export

- Offline pass: noise-track filter → constrained identity merge (112 tracks become 31 persons on `meva_a_60s`) → geometry → role features and suggestions → events (all sampled frames, t0 = 0) → scorecards. Re-scoring takes 0.07–0.1 s and never re-detects.
- Rendering: face-blurred H.264 at 10 fps. `meva_a_60s` renders in 8.1 s (7.4× real time). The privacy guard test measures face-region std ≤ 9.2 with blur against about 45 without it. Exports are timeline PNG, contact sheets, results.json and events/scorecard CSV.

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
- **Damaged files:** a damaged video (frames stop before 90% of the expected count) or a container reporting more than 240 fps fails cleanly with a `VideoError`.
- **Re-rendered videos:** each re-render is written to `annotated-<rev>.mp4`, because Windows browsers lock the mp4 that is playing.
- **Default test run:** it excludes `@pytest.mark.slow`; run those with `pytest -m slow`.

## Open problems

- README licence table (CLAUDE.md rule) does not exist yet: rows for yolox_nano/tiny/s, rtmpose_s, dinov2_small and rfdetr_nano are owed in M7. The data is in `models/MANIFEST.json`.
- (fixed in 746394c) `write_manifest()` now keeps a row for every registry entry, so a partial fetch can no longer shrink the licence record.
