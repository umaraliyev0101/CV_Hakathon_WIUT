# Synthetic showroom: full-run status (teammate C)

Status: **2026-10-11 06:20 (+05)**. The share pack passes every gate:
- `python sim/verify.py --all` ends with **VERIFY: PASS (32/32 gates)**; full output in §6;
- `reports/synth_visual_review.md` marks all 12 visual items as pass for all 8 shared clips.

More scenes are still rendering in the background (§4).

## 1. What works

- **Scenario engine** (`sim/behaviour.py`): a frame-stepped scheduler with actions enter, walk_to, idle, sit, approach, talk, talk_until_relieved, await_staff, follow (party) and leave.
- **Event rules** (`sim/gt_events.py`): greet, handoff, engagement, party, walkout and wait_alert.
- **Scene generation** (`sim/gen_configs.py`): 12 scenes with seeded validation. All 7 scenarios are covered:
  - two seeds each for greet_fast, greet_slow, walkout, party and handoff;
  - one seed each for crowded (7 people) and dealership_long (240 s).
- **People:** 12 Microsoft Rocketbox avatars (MIT), driven by mocap.
  - Walk phase is matched to stride, so feet do not slide.
  - In-place turns use the 90° turn clips driven by yaw, which fixes the quick test's pivot shuffle.
  - Talk and listen clips, 0.25 s blends.
- **Cars:** 4 real car models (CC BY / CC BY-NC / one of unknown licence; see DATA_SOURCES.md).
- **Room:** glass facade with HDRI, marble floor, roller door, desk.
- **Lighting:** three variants, `day`, `evening_dim` and `glare`.
- **Cameras:** 2 per scene. Height 2.8–3.2 m, pitch −25°, HFOV 70–90°; the true values are stored in `gt.json`.
- **Labels from instance masks:** visible box, `visible_frac`, floor position, yaw, action, zones (m and px), scripted events and rule events.
- **Exports:**
  - COCO, split by scene, in the RF-DETR layout;
  - MOT;
  - A-format labels (an adapter, see §5).
- **Share pack:** `data/synthetic/share/`, 8 clips covering all 5 short scenarios × 3 lighting variants × 2 cameras. Each clip folder has its own README; the hero clip is `s000_greet_fast_cam1_day`.
- **P0 detector** (`models/detect.py`): RF-DETR Nano COCO exported to ONNX, run with onnxruntime.
  - Measured CPU speed: 28.5 FPS at 720p with 8 threads (see `models/README.md`).
  - Measured GPU 1 speed: 265 FPS.
  - The same detector is used by `qa_detector`.

## 2. Measured numbers

| What | Value | Source |
|---|---|---|
| RGB s/frame (EEVEE Next, 16 spp, raytracing, 1280×720), one process alone | **0.333** | s000 cam1 day, first full run |
| RGB s/frame with 3 processes in parallel on GPU 1 | 0.48–0.94 (mean over scenes 0.77) | `render_stats.json` |
| Mask s/frame (Workbench) | 0.07–0.09 | same |
| Solo mask s/frame per person | 0.04–0.05 | same |
| GPU that rendered | **index 1** for every scene (`nvidia-smi pmon` saw the Blender PID; `gpu_index: [1]` in every `render_stats.json`) | run_all GpuWatch |
| GPU 1 memory with 3 Blender processes | ≈ 46 GB (limit 60 GB) | nvidia-smi |
| Parallel gain | RGB only: one process alone 3.0 fps; 3 processes ≈ 3.8 fps, i.e. **≈ 1.26×**, which is **below** the 1.3× rule. Whole pipeline: the core batch (s000–s004, 17.5 k RGB frames plus masks, labels and encode, 3 jobs) took ≈ 6 900 s wall, i.e. 0.39 s per RGB frame. The one-scene, one-lighting run took 0.68 s per RGB frame. That comparison is **not controlled**: the batch also spreads mask cost over 3 lighting variants. FLAG: the 3-job choice is not proven above 1.3× on the RGB pass alone. | measured |
| Detector recall on the share pack (IoU 0.5, box ≥ 60 px, visible_frac ≥ 0.6) | day **0.997**, evening_dim **0.992**, glare **0.985** (gate ≥ 0.80) | qa_detector |
| Detector recall on all 36 clips of s000–s003 | day 0.996, evening 0.994, glare 0.974 | gate runs during the full run |
| Scripted vs rule greet offset | 0.6 s (rule earlier) in every rendered scene. The tests assert ≤ 0.8 s for all 12 scenes. | qa_events, tests |
| Lowest body vertex vs floor | −0.034 … +0.008 m over all sampled frames (tolerance ±0.05) | qa_scene |

## 3. Dataset statistics (at the time of writing; `python sim/stats.py` regenerates them)

- Clips with labels and video: **36** from 6 scenes; 20244 labelled frames at 10 fps; 50469 person boxes (staff 28644, customer 21825).
- Clips by scenario: {'greet_fast': 12, 'greet_slow': 6, 'walkout': 6, 'party': 6, 'handoff': 6}; by lighting: {'day': 12, 'evening_dim': 12, 'glare': 12}.
- Scripted events (counted once per scene, across those scenes): {'engagement': 7, 'greet': 6, 'walkout': 1, 'party': 1, 'handoff': 1}.
- COCO (2 fps samples from the final MP4, JPEG q95, split by scene): train 2760 images / 6996 boxes (scenes ['s000', 's001', 's004', 's007']), valid 0 / 0, test 1290 / 3093 (scenes ['s002', 's003']). MOT rows: 50469.

## 4. Still running at the time of writing

- **Background renders**, writing into `data/synthetic/clips/` (not part of the share pack):
  - s007–s011 (second seeds of the 5 short scenarios) and s005 (crowded), 2 cameras × 3 lightings each: `/tmp/claude-1000/qr2.sh`, log `/tmp/claude-1000/run_rest.log`;
  - s006 (dealership_long), camera 1, day and evening_dim: log `/tmp/claude-1000/run_s006.log`.
- **After they finish:** re-run `python sim/export.py && python sim/stats.py`, then update §3 of this report.

## 5. Open problems and flags for the human

- **FLAG: A's spec is missing.** `salespeople_prompts/2026-10-11-showroom-coach-design.md` is not in the repo.
  - **Thresholds:** greet uses spec §4.3 (≤ 1.5 m, facing ≤ 60°, ≥ 2 s for short_clip, ≥ 5 s for dealership). Engagement, wait_alert, walkout and party are documented fallbacks; each one has a `_source` note in `sim/configs/presets.json`.
  - **Timing:** event times are the onset of the qualifying run, not its confirmation.
  - **A-format labels** (`labels.json`, `data/synthetic/labels/`) are an adapter from `gt.json` (`sim/export.py: a_format`). Reconcile the field names with A.
  - **`Det` dataclass** in `models/detect.py` must be reconciled with A's type.
- **Greet definition.** The scripted greet is the staff member's arrival: they stop 1.25 m away, facing the customer. This is kept consistent through choreography:
  - the last approach leg is a straight 1 m walk at the customer;
  - the staff member stops at 1.25 m, so only the final 0.25 m (while decelerating) lies inside the 1.5 m rule radius;
  - the staff member never passes inside a party member's radius before reaching the leader.

  Measured offset: rule greet 0.6 s earlier in every rendered scene.
- **Staff uniform.** Rocketbox clothing is baked into one texture, so the "uniform" is the dark suit fabric of staff characters recoloured red (configurable as `uniform.suit_recolour`); it is not a separate polo-and-vest material. Hard negatives are customers in dark suits or a burgundy blazer.
- **Car licences.** The Lamborghini Urus is **CC BY-NC 4.0**, so it must be replaced for any commercial use. The Porsche 930's licence is **unknown** (credit not embedded); treat it as demo-only.
- **Rule-only events.** `wait_alert` is computed by the rules only; there is no scripted counterpart, by design.
- **A gate check I corrected (documented, not loosened).** The quick-test-era gap check compared `visible_frac` at the two ends of a gap. It flagged a customer who walked out past the image border at s004 cam1 f37–f43; the in-image silhouette there was 5–246 px, under the 400 px drop rule. The check now uses `visible_frac` measured during the gap, which is the spec's wording. The thresholds are unchanged, and `sim/tests/test_qa_render_gap.py` shows that a real dropout of a visible person still fails.
- **Perspective lean.** People near the image corners lean visibly (wide HFOV, −25° pitch). This is a camera effect, noted in the visual review.
- **Not done (out of scope or not allowed):** the detector fine-tune. A synthetic-only fine-tune could only be evaluated on a real, human-labelled held-out set, which does not exist yet. The COCO export is ready for it.

## 6. Last `python sim/verify.py --all` output (full; onnxruntime's repeated "No registered plugin EP" warning lines removed)

```
=== verify: 8 clip(s): ['s000_greet_fast_cam1_day', 's000_greet_fast_cam2_glare', 's001_greet_slow_cam1_evening_dim', 's002_walkout_cam1_glare', 's003_party_cam1_day', 's003_party_cam2_evening_dim', 's004_handoff_cam1_evening_dim', 's004_handoff_cam2_glare'] ===
=== gate: pytest -q sim/tests ===
........................................................................ [ 56%]
.......................................................                  [100%]
127 passed in 25.99s
=== gate: qa_assets ===
  qa_assets: 160 manifest rows, 159 asset files on disk, 16 referenced by configs; licences: {'MIT (Copyright (c) 2020 Microsoft; Microsoft Rocketbox Avatar Library)': 140, 'MIT': 1, 'CC-BY-4.0 (vicent091036) per Sketchfab listing as credited by three.js webgl_materials_car.html; three.js repo itself MIT': 1, 'CC-BY-4.0 (Eric Chadwick / DGG; excludes Khronos logos/trademarks)': 1, 'CC-BY-NC-4.0 (Steven Grey) — embedded in glTF asset.extras; non-commercial OK for demo': 1, 'unknown — credit not embedded; source Sketchfab model is believed CC-BY-4.0 (Lionsharp Studios); demo only': 1, 'CC0': 15}
qa_assets: PASS (0 failure(s))
  (qa_assets took 0.0 s)
=== gate: qa_scene (s000) ===
  qa_scene[s000]: person 1 (rocketbox): height 1.738 m (config 1.75); lowest vertex z over 90 sampled frames min/max -0.032/+0.004 m
  qa_scene[s000]: person 2 (rocketbox): height 1.630 m (config 1.634); lowest vertex z over 90 sampled frames min/max -0.015/+0.007 m
  qa_scene[s000]: car lengths: car_1=4.33 m, car_2=4.80 m, car_3=5.07 m, car_4=4.04 m
  qa_scene[s000]: images: 46 file textures, 0 missing
  qa_scene[s000]: closest approach between two people: 1.25 m
  qa_scene[s000]: person 1: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s000]: person 2: invisible in all 2 rendered camera(s) for 2.9% of lifetime
qa_scene[s000]: PASS (0 failure(s))
  (qa_scene[s000] took 30.6 s)
=== gate: qa_scene (s001) ===
  qa_scene[s001]: person 1 (rocketbox): height 1.684 m (config 1.69); lowest vertex z over 156 sampled frames min/max -0.000/+0.003 m
  qa_scene[s001]: person 2 (rocketbox): height 1.807 m (config 1.814); lowest vertex z over 156 sampled frames min/max -0.006/+0.004 m
  qa_scene[s001]: person 3 (rocketbox): height 1.724 m (config 1.729); lowest vertex z over 156 sampled frames min/max -0.012/+0.008 m
  qa_scene[s001]: car lengths: car_1=4.04 m, car_2=4.70 m, car_3=4.32 m, car_4=4.63 m
  qa_scene[s001]: images: 37 file textures, 0 missing
  qa_scene[s001]: closest approach between two people: 1.25 m
  qa_scene[s001]: person 1: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s001]: person 2: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s001]: person 3: invisible in all 2 rendered camera(s) for 7.4% of lifetime
qa_scene[s001]: PASS (0 failure(s))
  (qa_scene[s001] took 75.8 s)
=== gate: qa_scene (s002) ===
  qa_scene[s002]: person 1 (rocketbox): height 1.760 m (config 1.769); lowest vertex z over 105 sampled frames min/max +0.002/+0.004 m
  qa_scene[s002]: person 2 (rocketbox): height 1.821 m (config 1.822); lowest vertex z over 105 sampled frames min/max -0.034/+0.004 m
  qa_scene[s002]: car lengths: car_1=4.41 m, car_2=4.42 m, car_3=4.75 m, car_4=5.01 m
  qa_scene[s002]: images: 57 file textures, 0 missing
  qa_scene[s002]: closest approach between two people: 2.80 m
  qa_scene[s002]: person 1: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s002]: person 2: invisible in all 2 rendered camera(s) for 5.7% of lifetime
qa_scene[s002]: PASS (0 failure(s))
  (qa_scene[s002] took 35.1 s)
=== gate: qa_scene (s003) ===
  qa_scene[s003]: person 1 (rocketbox): height 1.676 m (config 1.686); lowest vertex z over 110 sampled frames min/max -0.003/+0.004 m
  qa_scene[s003]: person 2 (rocketbox): height 1.727 m (config 1.729); lowest vertex z over 110 sampled frames min/max -0.016/+0.004 m
  qa_scene[s003]: person 3 (rocketbox): height 1.685 m (config 1.687); lowest vertex z over 110 sampled frames min/max -0.026/+0.005 m
  qa_scene[s003]: car lengths: car_1=4.10 m, car_2=4.16 m, car_3=4.88 m, car_4=5.09 m
  qa_scene[s003]: images: 28 file textures, 0 missing
  qa_scene[s003]: closest approach between two people: 0.63 m
  qa_scene[s003]: person 1: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s003]: person 2: invisible in all 2 rendered camera(s) for 3.6% of lifetime
  qa_scene[s003]: person 3: invisible in all 2 rendered camera(s) for 3.6% of lifetime
qa_scene[s003]: PASS (0 failure(s))
  (qa_scene[s003] took 52.2 s)
=== gate: qa_scene (s004) ===
  qa_scene[s004]: person 1 (rocketbox): height 1.756 m (config 1.762); lowest vertex z over 124 sampled frames min/max -0.021/+0.004 m
  qa_scene[s004]: person 2 (rocketbox): height 1.780 m (config 1.788); lowest vertex z over 124 sampled frames min/max -0.005/+0.004 m
  qa_scene[s004]: person 3 (rocketbox): height 1.759 m (config 1.765); lowest vertex z over 124 sampled frames min/max -0.027/+0.004 m
  qa_scene[s004]: car lengths: car_1=4.11 m, car_2=4.58 m, car_3=4.21 m, car_4=5.18 m, car_5=5.05 m
  qa_scene[s004]: images: 51 file textures, 0 missing
  qa_scene[s004]: closest approach between two people: 1.25 m
  qa_scene[s004]: person 1: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s004]: person 2: invisible in all 2 rendered camera(s) for 0.0% of lifetime
  qa_scene[s004]: person 3: invisible in all 2 rendered camera(s) for 3.9% of lifetime
qa_scene[s004]: PASS (0 failure(s))
  (qa_scene[s004] took 58.7 s)
=== gate: qa_render (s000_greet_fast_cam1_day) ===
  qa_render[s000_greet_fast_cam1_day]: expected frames 450 (from configs/scene_000.json)
  qa_render[s000_greet_fast_cam1_day]: mean luma min/mean/max 129.9/131.5/132.5; max clipped 0.000%; max magenta 0.0000%
  qa_render[s000_greet_fast_cam1_day]: unknown mask px total 0 over 450 masks
  qa_render[s000_greet_fast_cam1_day]: person 1 (staff) labelled in 450/450 frames (100.0%)
  qa_render[s000_greet_fast_cam1_day]: person 2 (customer) labelled in 396/450 frames (88.0%)
qa_render[s000_greet_fast_cam1_day]: PASS (0 failure(s))
  (qa_render[s000_greet_fast_cam1_day] took 28.7 s)
=== gate: qa_events (s000_greet_fast_cam1_day) ===
  qa_events[s000_greet_fast_cam1_day]: [short_clip] scripted=[('engagement', 11.9), ('greet', 11.9)]
  qa_events[s000_greet_fast_cam1_day]: [short_clip] rule=[('engagement', 11.3), ('greet', 11.3)]
qa_events[s000_greet_fast_cam1_day]: PASS (0 failure(s))
  (qa_events[s000_greet_fast_cam1_day] took 0.0 s)
=== gate: qa_video (s000_greet_fast_cam1_day) ===
  qa_video[s000_greet_fast_cam1_day]: s000_greet_fast_cam1_day.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=450 duration=45.000s size=1280x720
  qa_video[s000_greet_fast_cam1_day]: opencv: frame_count=450 first_ok=True last_ok=True
qa_video[s000_greet_fast_cam1_day]: PASS (0 failure(s))
  (qa_video[s000_greet_fast_cam1_day] took 0.5 s)
=== gate: qa_render (s000_greet_fast_cam2_glare) ===
  qa_render[s000_greet_fast_cam2_glare]: expected frames 450 (from configs/scene_000.json)
  qa_render[s000_greet_fast_cam2_glare]: mean luma min/mean/max 104.5/109.1/109.9; max clipped 0.064%; max magenta 0.0000%
  qa_render[s000_greet_fast_cam2_glare]: unknown mask px total 0 over 450 masks
  qa_render[s000_greet_fast_cam2_glare]: person 1 (staff) labelled in 450/450 frames (100.0%)
  qa_render[s000_greet_fast_cam2_glare]: person 2 (customer) labelled in 437/450 frames (97.1%)
qa_render[s000_greet_fast_cam2_glare]: PASS (0 failure(s))
  (qa_render[s000_greet_fast_cam2_glare] took 26.4 s)
=== gate: qa_events (s000_greet_fast_cam2_glare) ===
  qa_events[s000_greet_fast_cam2_glare]: [short_clip] scripted=[('engagement', 11.9), ('greet', 11.9)]
  qa_events[s000_greet_fast_cam2_glare]: [short_clip] rule=[('engagement', 11.3), ('greet', 11.3)]
qa_events[s000_greet_fast_cam2_glare]: PASS (0 failure(s))
  (qa_events[s000_greet_fast_cam2_glare] took 0.0 s)
=== gate: qa_video (s000_greet_fast_cam2_glare) ===
  qa_video[s000_greet_fast_cam2_glare]: s000_greet_fast_cam2_glare.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=450 duration=45.000s size=1280x720
  qa_video[s000_greet_fast_cam2_glare]: opencv: frame_count=450 first_ok=True last_ok=True
qa_video[s000_greet_fast_cam2_glare]: PASS (0 failure(s))
  (qa_video[s000_greet_fast_cam2_glare] took 0.4 s)
=== gate: qa_render (s001_greet_slow_cam1_evening_dim) ===
  qa_render[s001_greet_slow_cam1_evening_dim]: expected frames 780 (from configs/scene_001.json)
  qa_render[s001_greet_slow_cam1_evening_dim]: mean luma min/mean/max 55.4/57.7/58.5; max clipped 0.000%; max magenta 0.0000%
  qa_render[s001_greet_slow_cam1_evening_dim]: unknown mask px total 0 over 780 masks
  qa_render[s001_greet_slow_cam1_evening_dim]: person 1 (staff) labelled in 780/780 frames (100.0%)
  qa_render[s001_greet_slow_cam1_evening_dim]: person 2 (staff) labelled in 780/780 frames (100.0%)
  qa_render[s001_greet_slow_cam1_evening_dim]: person 3 (customer) labelled in 722/780 frames (92.6%)
qa_render[s001_greet_slow_cam1_evening_dim]: PASS (0 failure(s))
  (qa_render[s001_greet_slow_cam1_evening_dim] took 57.9 s)
=== gate: qa_events (s001_greet_slow_cam1_evening_dim) ===
  qa_events[s001_greet_slow_cam1_evening_dim]: [short_clip] scripted=[('engagement', 53.133333), ('greet', 53.133333)]
  qa_events[s001_greet_slow_cam1_evening_dim]: [short_clip] rule=[('wait_alert', 32.566667), ('engagement', 52.533333), ('greet', 52.533333)]
qa_events[s001_greet_slow_cam1_evening_dim]: PASS (0 failure(s))
  (qa_events[s001_greet_slow_cam1_evening_dim] took 0.0 s)
=== gate: qa_video (s001_greet_slow_cam1_evening_dim) ===
  qa_video[s001_greet_slow_cam1_evening_dim]: s001_greet_slow_cam1_evening_dim.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=780 duration=78.000s size=1280x720
  qa_video[s001_greet_slow_cam1_evening_dim]: opencv: frame_count=780 first_ok=True last_ok=True
qa_video[s001_greet_slow_cam1_evening_dim]: PASS (0 failure(s))
  (qa_video[s001_greet_slow_cam1_evening_dim] took 0.7 s)
=== gate: qa_render (s002_walkout_cam1_glare) ===
  qa_render[s002_walkout_cam1_glare]: expected frames 524 (from configs/scene_002.json)
  qa_render[s002_walkout_cam1_glare]: mean luma min/mean/max 144.2/145.1/146.1; max clipped 0.105%; max magenta 0.0000%
  qa_render[s002_walkout_cam1_glare]: unknown mask px total 0 over 524 masks
  qa_render[s002_walkout_cam1_glare]: person 1 (staff) labelled in 524/524 frames (100.0%)
  qa_render[s002_walkout_cam1_glare]: person 2 (customer) labelled in 474/524 frames (90.5%)
qa_render[s002_walkout_cam1_glare]: PASS (0 failure(s))
  (qa_render[s002_walkout_cam1_glare] took 32.1 s)
=== gate: qa_events (s002_walkout_cam1_glare) ===
  qa_events[s002_walkout_cam1_glare]: [short_clip] scripted=[('walkout', 49.866667)]
  qa_events[s002_walkout_cam1_glare]: [short_clip] rule=[('wait_alert', 32.8), ('walkout', 49.866667)]
qa_events[s002_walkout_cam1_glare]: PASS (0 failure(s))
  (qa_events[s002_walkout_cam1_glare] took 0.0 s)
=== gate: qa_video (s002_walkout_cam1_glare) ===
  qa_video[s002_walkout_cam1_glare]: s002_walkout_cam1_glare.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=524 duration=52.400s size=1280x720
  qa_video[s002_walkout_cam1_glare]: opencv: frame_count=524 first_ok=True last_ok=True
qa_video[s002_walkout_cam1_glare]: PASS (0 failure(s))
  (qa_video[s002_walkout_cam1_glare] took 0.5 s)
=== gate: qa_render (s003_party_cam1_day) ===
  qa_render[s003_party_cam1_day]: expected frames 550 (from configs/scene_003.json)
  qa_render[s003_party_cam1_day]: mean luma min/mean/max 111.9/115.2/116.8; max clipped 0.000%; max magenta 0.0000%
  qa_render[s003_party_cam1_day]: unknown mask px total 0 over 550 masks
  qa_render[s003_party_cam1_day]: person 1 (staff) labelled in 550/550 frames (100.0%)
  qa_render[s003_party_cam1_day]: person 2 (customer) labelled in 490/550 frames (89.1%)
  qa_render[s003_party_cam1_day]: person 3 (customer) labelled in 486/550 frames (88.4%)
qa_render[s003_party_cam1_day]: PASS (0 failure(s))
  (qa_render[s003_party_cam1_day] took 41.5 s)
=== gate: qa_events (s003_party_cam1_day) ===
  qa_events[s003_party_cam1_day]: [short_clip] scripted=[('party', 2.633333), ('engagement', 11.366667), ('engagement', 11.366667), ('greet', 11.366667), ('greet', 11.366667)]
  qa_events[s003_party_cam1_day]: [short_clip] rule=[('party', 2.633333), ('engagement', 10.766667), ('engagement', 10.766667), ('greet', 10.766667), ('greet', 10.766667)]
qa_events[s003_party_cam1_day]: PASS (0 failure(s))
  (qa_events[s003_party_cam1_day] took 0.0 s)
=== gate: qa_video (s003_party_cam1_day) ===
  qa_video[s003_party_cam1_day]: s003_party_cam1_day.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=550 duration=55.000s size=1280x720
  qa_video[s003_party_cam1_day]: opencv: frame_count=550 first_ok=True last_ok=True
qa_video[s003_party_cam1_day]: PASS (0 failure(s))
  (qa_video[s003_party_cam1_day] took 0.6 s)
=== gate: qa_render (s003_party_cam2_evening_dim) ===
  qa_render[s003_party_cam2_evening_dim]: expected frames 550 (from configs/scene_003.json)
  qa_render[s003_party_cam2_evening_dim]: mean luma min/mean/max 41.4/42.2/42.7; max clipped 0.000%; max magenta 0.0000%
  qa_render[s003_party_cam2_evening_dim]: unknown mask px total 0 over 550 masks
  qa_render[s003_party_cam2_evening_dim]: person 1 (staff) labelled in 550/550 frames (100.0%)
  qa_render[s003_party_cam2_evening_dim]: person 2 (customer) labelled in 530/550 frames (96.4%)
  qa_render[s003_party_cam2_evening_dim]: person 3 (customer) labelled in 530/550 frames (96.4%)
qa_render[s003_party_cam2_evening_dim]: PASS (0 failure(s))
  (qa_render[s003_party_cam2_evening_dim] took 38.1 s)
=== gate: qa_events (s003_party_cam2_evening_dim) ===
  qa_events[s003_party_cam2_evening_dim]: [short_clip] scripted=[('party', 2.633333), ('engagement', 11.366667), ('engagement', 11.366667), ('greet', 11.366667), ('greet', 11.366667)]
  qa_events[s003_party_cam2_evening_dim]: [short_clip] rule=[('party', 2.633333), ('engagement', 10.766667), ('engagement', 10.766667), ('greet', 10.766667), ('greet', 10.766667)]
qa_events[s003_party_cam2_evening_dim]: PASS (0 failure(s))
  (qa_events[s003_party_cam2_evening_dim] took 0.0 s)
=== gate: qa_video (s003_party_cam2_evening_dim) ===
  qa_video[s003_party_cam2_evening_dim]: s003_party_cam2_evening_dim.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=550 duration=55.000s size=1280x720
  qa_video[s003_party_cam2_evening_dim]: opencv: frame_count=550 first_ok=True last_ok=True
qa_video[s003_party_cam2_evening_dim]: PASS (0 failure(s))
  (qa_video[s003_party_cam2_evening_dim] took 0.4 s)
=== gate: qa_render (s004_handoff_cam1_evening_dim) ===
  qa_render[s004_handoff_cam1_evening_dim]: expected frames 620 (from configs/scene_004.json)
  qa_render[s004_handoff_cam1_evening_dim]: mean luma min/mean/max 58.1/59.7/60.8; max clipped 0.000%; max magenta 0.0000%
  qa_render[s004_handoff_cam1_evening_dim]: unknown mask px total 0 over 620 masks
  qa_render[s004_handoff_cam1_evening_dim]: person 1 (staff) labelled in 620/620 frames (100.0%)
  qa_render[s004_handoff_cam1_evening_dim]: person 2 (staff) labelled in 620/620 frames (100.0%)
  qa_render[s004_handoff_cam1_evening_dim]: person 3 (customer) labelled in 582/620 frames (93.9%)
qa_render[s004_handoff_cam1_evening_dim]: PASS (0 failure(s))
  (qa_render[s004_handoff_cam1_evening_dim] took 43.2 s)
=== gate: qa_events (s004_handoff_cam1_evening_dim) ===
  qa_events[s004_handoff_cam1_evening_dim]: [short_clip] scripted=[('engagement', 11.3), ('greet', 11.3), ('engagement', 30.866667), ('handoff', 30.866667)]
  qa_events[s004_handoff_cam1_evening_dim]: [short_clip] rule=[('engagement', 10.7), ('greet', 10.7), ('engagement', 30.266667), ('handoff', 30.266667)]
qa_events[s004_handoff_cam1_evening_dim]: PASS (0 failure(s))
  (qa_events[s004_handoff_cam1_evening_dim] took 0.0 s)
=== gate: qa_video (s004_handoff_cam1_evening_dim) ===
  qa_video[s004_handoff_cam1_evening_dim]: s004_handoff_cam1_evening_dim.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=620 duration=62.000s size=1280x720
  qa_video[s004_handoff_cam1_evening_dim]: opencv: frame_count=620 first_ok=True last_ok=True
qa_video[s004_handoff_cam1_evening_dim]: PASS (0 failure(s))
  (qa_video[s004_handoff_cam1_evening_dim] took 0.6 s)
=== gate: qa_render (s004_handoff_cam2_glare) ===
  qa_render[s004_handoff_cam2_glare]: expected frames 620 (from configs/scene_004.json)
  qa_render[s004_handoff_cam2_glare]: mean luma min/mean/max 118.2/127.0/128.7; max clipped 0.036%; max magenta 0.0000%
  qa_render[s004_handoff_cam2_glare]: unknown mask px total 0 over 620 masks
  qa_render[s004_handoff_cam2_glare]: person 1 (staff) labelled in 620/620 frames (100.0%)
  qa_render[s004_handoff_cam2_glare]: person 2 (staff) labelled in 620/620 frames (100.0%)
  qa_render[s004_handoff_cam2_glare]: person 3 (customer) labelled in 596/620 frames (96.1%)
qa_render[s004_handoff_cam2_glare]: PASS (0 failure(s))
  (qa_render[s004_handoff_cam2_glare] took 46.3 s)
=== gate: qa_events (s004_handoff_cam2_glare) ===
  qa_events[s004_handoff_cam2_glare]: [short_clip] scripted=[('engagement', 11.3), ('greet', 11.3), ('engagement', 30.866667), ('handoff', 30.866667)]
  qa_events[s004_handoff_cam2_glare]: [short_clip] rule=[('engagement', 10.7), ('greet', 10.7), ('engagement', 30.266667), ('handoff', 30.266667)]
qa_events[s004_handoff_cam2_glare]: PASS (0 failure(s))
  (qa_events[s004_handoff_cam2_glare] took 0.0 s)
=== gate: qa_video (s004_handoff_cam2_glare) ===
  qa_video[s004_handoff_cam2_glare]: s004_handoff_cam2_glare.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=620 duration=62.000s size=1280x720
  qa_video[s004_handoff_cam2_glare]: opencv: frame_count=620 first_ok=True last_ok=True
qa_video[s004_handoff_cam2_glare]: PASS (0 failure(s))
  (qa_video[s004_handoff_cam2_glare] took 0.6 s)
=== gate: qa_detector (all clips, per lighting variant) ===
  qa_detector: detector providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] conf=0.5 iou=0.5
  qa_detector: s000_greet_fast_cam1_day: recall 845/846 = 0.999
  qa_detector: s000_greet_fast_cam2_glare: recall 821/857 = 0.958
  qa_detector: s001_greet_slow_cam1_evening_dim: recall 2274/2282 = 0.996
  qa_detector: s002_walkout_cam1_glare: recall 986/997 = 0.989
  qa_detector: s003_party_cam1_day: recall 1509/1514 = 0.997
  qa_detector: s003_party_cam2_evening_dim: recall 1510/1545 = 0.977
  qa_detector: s004_handoff_cam1_evening_dim: recall 1812/1816 = 0.998
  qa_detector: s004_handoff_cam2_glare: recall 1525/1528 = 0.998
  qa_detector: [day] clips=2 recall@IoU0.5 = 2354/2360 = 0.997 (gate >= 0.8); unmatched detections (report only): {}
  qa_detector: [evening_dim] clips=3 recall@IoU0.5 = 5596/5643 = 0.992 (gate >= 0.8); unmatched detections (report only): {}
  qa_detector: [glare] clips=3 recall@IoU0.5 = 3332/3382 = 0.985 (gate >= 0.8); unmatched detections (report only): {}
qa_detector: PASS (0 failure(s))
  (qa_detector took 31.9 s)
=== summary ===
  pytest                                                       PASS
  qa_assets                                                    PASS
  qa_scene[s000]                                               PASS
  qa_scene[s001]                                               PASS
  qa_scene[s002]                                               PASS
  qa_scene[s003]                                               PASS
  qa_scene[s004]                                               PASS
  qa_render[s000_greet_fast_cam1_day]                          PASS
  qa_events[s000_greet_fast_cam1_day]                          PASS
  qa_video[s000_greet_fast_cam1_day]                           PASS
  qa_render[s000_greet_fast_cam2_glare]                        PASS
  qa_events[s000_greet_fast_cam2_glare]                        PASS
  qa_video[s000_greet_fast_cam2_glare]                         PASS
  qa_render[s001_greet_slow_cam1_evening_dim]                  PASS
  qa_events[s001_greet_slow_cam1_evening_dim]                  PASS
  qa_video[s001_greet_slow_cam1_evening_dim]                   PASS
  qa_render[s002_walkout_cam1_glare]                           PASS
  qa_events[s002_walkout_cam1_glare]                           PASS
  qa_video[s002_walkout_cam1_glare]                            PASS
  qa_render[s003_party_cam1_day]                               PASS
  qa_events[s003_party_cam1_day]                               PASS
  qa_video[s003_party_cam1_day]                                PASS
  qa_render[s003_party_cam2_evening_dim]                       PASS
  qa_events[s003_party_cam2_evening_dim]                       PASS
  qa_video[s003_party_cam2_evening_dim]                        PASS
  qa_render[s004_handoff_cam1_evening_dim]                     PASS
  qa_events[s004_handoff_cam1_evening_dim]                     PASS
  qa_video[s004_handoff_cam1_evening_dim]                      PASS
  qa_render[s004_handoff_cam2_glare]                           PASS
  qa_events[s004_handoff_cam2_glare]                           PASS
  qa_video[s004_handoff_cam2_glare]                            PASS
  qa_detector                                                  PASS
VERIFY: PASS (32/32 gates passed)
```

## 7. Resume commands

```bash
cd /mnt/raid1/projects/CV_Hakathon_WIUT && source .venv/bin/activate
python -m pytest -q sim/tests                         # pure core
python sim/gen_configs.py                             # regenerate the 12 scene configs (deterministic)
python sim/run_all.py --config sim/configs/scene_005.json --jobs 1     # one scene (all cams x lightings)
python sim/run_all.py --config sim/configs/scene_006.json --cams 1 --lightings day,evening_dim
python sim/export.py && python sim/stats.py           # COCO / MOT / A-format, dataset stats
python sim/share.py                                   # share pack from sim/configs/share.json
python sim/verify.py --all                            # gate chain over the share pack
python -m sim.qa.review_sheets data/synthetic/share/<clip>   # zoomed review sheet for the visual review
CUDA_VISIBLE_DEVICES=1 python models/detect.py data/real_open/pexels_7154217_1920_1080_25fps.mp4 --frame 150
```
