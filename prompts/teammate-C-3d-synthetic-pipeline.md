# Prompt for teammate C: synthetic 3D showroom pipeline (build, render, check, fix)

**Before launching, a human must:**
1. Fill in every `[...]` placeholder.
2. Complete the pre-flight checklist in §2.
3. Start the session in the repo where the pipeline should live.

Then paste everything below the line into a fresh Claude Code session.

---

## 1. Context and mission

We are a 3-person team in the WIUT Hackathon 2026 Computer Vision final. The partners are Toyota
Royal Motors, BYD Royal Motors and Ducati. The demo is on **2026-10-12**.

Our product turns showroom camera video into coaching for salespeople and analytics about customers.

- **Teammate A** builds "Showroom Coach", the salesperson analysis. Their spec:
  `salespeople_prompts/2026-10-11-showroom-coach-design.md` (or ask the human for a copy).
  It covers:
  - one camera per job;
  - person detection, then pose, then ByteTrack;
  - distance from pinhole back-projection, assuming people are 1.70 m tall and the horizontal FOV is 70°;
  - staff/customer roles from rules plus one click;
  - events: greet, engagement, wait_alert, walkout, party, handoff;
  - presets "Dealership" and "Short clip".
- **Teammate B** builds the customer analysis: dwell per car, paths, funnel, walk-outs.
- **Their gap:** no free footage exists with labelled staff/customer roles and exact greet and
  engagement times. They also cannot film inside a real showroom.

**Your mission:** build a reproducible pipeline that renders synthetic car-showroom CCTV video in
Blender, with exact ground truth. Then check its output and fix it until every gate in §6 passes.

The ground truth must include:
- each person's role;
- their 2D box in every frame;
- their position in metres and which way they face;
- car and desk zones;
- greet, engagement, walk-out, party and handoff events with exact timestamps.

A and B use these clips to evaluate and tune their pipelines. The renders also become training data
for a detector fine-tune later; that step is out of scope here. Design for visual plausibility,
correctness of labels, and reproducibility, in that order. A plausible, correctly labelled clip
beats a beautiful one with wrong labels.

## 2. Pre-flight checklist (the human does this before launch; you verify it in S0)

- [ ] Blender 4.2 or newer installed. Record its path: `[BLENDER_PATH]`.
- [ ] Mixamo assets, from a free Adobe ID, in `assets/mixamo/`. Mixamo has no API.
  - `characters/`: about 10 characters, diverse in gender, skin tone and body type, downloaded
    as FBX "With Skin" in T-pose. Include 2–3 in suits or business wear.
  - `anims/`: FBX "Without Skin", with "In Place" ticked for locomotion:
    Walking, Slow Walking, Idle (×2 variants), Talking (×2), Looking Around, Pointing, Waving,
    Turn Left 90, Turn Right 90.
- [ ] 4–6 car models in `assets/cars/` as `.glb`, from Sketchfab (CC BY, marked downloadable).
  Record the URL and author of each in `assets/MANIFEST.csv`.
- [ ] Optional: a sales-desk and chair model in `assets/props/`.
- [ ] Machine details for the run: `[OS / GPU / RAM / free disk]`.
- [ ] Deadline: `[DEADLINE, e.g. 2026-10-12 07:00]`. Render budget: `[RENDER_BUDGET_HOURS]` hours.
  Disk budget: `[DISK_BUDGET_GB]` GB.

You may download CC0 HDRIs and floor and wall textures from Poly Haven yourself, using its public
API, and record them in the manifest. Do not create accounts. Do not spend money. Do not upload
anything to external services.

## 3. Architecture

Keep every piece of logic that does not need Blender in plain Python (numpy only), so it can be
tested fast. Blender only builds the scene, poses it from a precomputed timeline, and renders.

```
sim/
  configs/            scene_*.json, presets.json (copied from A's spec §4.6), defaults.json
  schema.py           config + output dataclasses, JSON (de)serialisation, validation
  layout.py           room, car spots, desk, entrance/exit, zone polygons (floor metres)  [pure]
  navigation.py       waypoint graph around car footprints, shortest paths, speed profiles [pure]
  behaviour.py        scenario -> per-agent timeline: pos_m, yaw, action, at scene fps    [pure]
  gt_events.py        A's event rules (spec §4.5/4.6) run on true 3D states -> events      [pure]
  projection.py       K, R, t from camera params; world->pixel; floor polygons -> image    [pure]
  labels.py           mask PNGs -> visible boxes; tracks; export gt.json, COCO, MOT,
                      A-format labels                                                       [pure]
  post.py             CCTV degradation + H.264 encode (ffmpeg or imageio-ffmpeg)            [pure]
  blender/
    build_scene.py    room, materials, lights, HDRI, cars, props (reads layout + config)
    people.py         import Mixamo characters, normalise scale/bones, outfit materials
    animate.py        drive armatures from the timeline: NLA strips + root translation/rotation
    cameras.py        create cameras from config; export intrinsics/extrinsics
    render.py         RGB pass (EEVEE) + instance-mask pass (Workbench) per camera
  qa/                 gate scripts (§6), each exits non-zero on failure
  verify.py           runs every gate in order; THE verification chain
  run_all.py          configs -> timelines -> blender build/render -> labels -> post -> QA
  tests/              pytest for all [pure] modules
reports/              synth_status.md, synth_render_plan.md, synth_visual_review.md, contact sheets
data/synthetic/       outputs (gitignored)
```

## 4. Specification

### 4.1 Scenes
- **Room:** randomised per config, 15–30 m × 10–20 m, 4–6 m ceiling.
  - Floor: glossy tiles (Poly Haven textures).
  - Walls: one glass facade lit by an HDRI.
  - Ceiling: lights, a sales desk with chairs, an entrance, and a test-drive exit.
- **Cars:** 4–6 on spots with 2.5 m or more of clearance.
  - Normalise each car's length to 4.0–5.2 m.
  - Randomise paint colour where the paint material can be identified.
- **Zones:** `car_<k>` (footprint + 1.2 m apron), `desk`, `entrance`, `exit`, as floor polygons in
  metres. Also project them into each camera's image.

### 4.2 People
- **Characters:** from `assets/mixamo/characters/`.
  - Normalise bone prefixes (`mixamorig:`, `mixamorig1:`, … → `mixamorig:`) so any action plays
    on any character.
  - Normalise scale. Mixamo FBX often imports at 0.01 or 100×. Then randomise height to 1.55–1.95 m
    by uniform scale. A assumes 1.70 m, so the spread tests that assumption.
- **Staff:** share one uniform per scene, applied by material override on the clothing materials.
  The default is a dark polo with a red vest; make it configurable.
- **Customers:** random clothing colours. Every scene must include at least one **hard negative**:
  a customer whose outfit resembles the uniform (similar colour or a suit). That way A's
  `outfit_peers` rule cannot win on colour alone.

### 4.3 Behaviour (pure Python; `behaviour.py`)
- **Scenario JSON:** a list of agents, each with role, character, spawn time and actions:
  `walk_to(target)`, `idle(s)`, `face(agent)`, `talk(agent, s)`, `follow(agent)`, `sit(desk)`,
  `leave()`.
  - `sit` may be faked with idle at the desk if a sitting animation is unavailable.
- **Walking:**
  - Route along `navigation.py` paths that never enter a car footprint.
  - Speed 1.1–1.5 m/s, with acceleration ramps.
  - Turn the yaw smoothly at about 180°/s.
- **Greeting choreography:**
  - Staff stop 0.9–1.3 m from the customer and face them within 20°.
  - Talking lasts at least the scenario's greet duration plus a 1 s margin.
  - This keeps ground-truth events unambiguous under A's rules (greet ≤ 1.5 m, facing ≤ 60°,
    ≥ 2 s or 5 s).
- **Required scenarios.** The first five are each about 45–90 s, for the Short-clip preset:
  1. `greet_fast`: greeted within 5–15 s.
  2. `greet_slow`: greeted after 40–70 s while another staff member idles at the desk. This is
     "missed while free".
  3. `walkout`: never greeted, leaves after 30–60 s.
  4. `party`: two customers arrive together; greeting one greets both.
  5. `handoff`: engaged by staff A, then by staff B.
  6. `crowded`: 4–5 customers and 2 staff.
  7. `dealership_long`: 4 minutes with an engagement longer than 120 s, for the Dealership preset.
     Use one or two cameras only to save render time.
- **Timeline output** (`timeline.json`): per agent, per frame at scene fps: `pos_m [x, y]`,
  `yaw_deg`, `action`, `anim_clip`, `clip_time`. Blender reads this and makes no behavioural
  decisions.

### 4.4 Cameras
- 2–4 per scene in ceiling corners or on walls.
  - Height 2.6–3.5 m, pitch −25° to −50°, horizontal FOV 60–100°.
  - Record the true HFOV for every camera.
- Every agent must be in at least one camera's view for ≥ 80% of its lifetime.
- Every camera must see at least one person in ≥ 50% of frames.
- Each camera is exported as an **independent single-camera clip**, because A's scope is one camera
  per job.

### 4.5 Rendering
- **RGB pass:** EEVEE at 1280×720, 10 fps output, AgX/Filmic view transform, reflections on.
  - Reflections of people in car paint and glass are **unlabelled by design**: they are free hard
    negatives.
- **Animation timing:** scene at 30 fps with `frame_step = 3`, encoded at 10 fps, so motion plays
  at real speed. Mixamo clips are 30 fps.
- **Lighting variants per scene:** `day`, `evening_dim`, `glare` (low sun through the facade).
- **Instance-mask pass:** Workbench.
  - Flat lighting, `color_type='OBJECT'`, a unique `obj.color` per person applied to all that
    person's meshes, everything else black.
  - Anti-aliasing off, `Standard` view transform, dither 0, 8-bit PNG.
  - Hide glass objects in this pass, so people seen through glass are labelled as visible, as they
    are in RGB. Document this choice.
- **Measure seconds per frame** (both passes) in S2 and S5. Write `reports/synth_render_plan.md`
  before rendering at scale. The plan must fit `[RENDER_BUDGET_HOURS]` and `[DISK_BUDGET_GB]`.
  - Delete PNG sequences after encoding and QA.
  - Keep only the frames sampled for training (1–2 fps, JPEG q95) and the masks needed for QA.

### 4.6 Labels and exports (`labels.py`)
- **Visible box:** the bounding box of the person's mask pixels. Drop a person from a frame's
  labels if the mask has fewer than 400 px or the box is under 20 px tall.
  - Also record `visible_frac` = mask pixels ÷ pixels of the person's unoccluded projected
    silhouette (from a per-person solo mask, or an estimate from projected vertices; document which).
- **`gt.json` per clip:**

```json
{
  "clip_id": "s003_greet_slow_cam2_evening_dim",
  "labeller": "synthetic-ground-truth",
  "generator": {"git_sha": "...", "config": "configs/scene_003.json", "seed": 3},
  "fps": 10, "width": 1280, "height": 720, "duration_s": 72.0,
  "camera": {"hfov_deg": 74.0, "K": [[...]], "R": [[...]], "t": [...], "height_m": 3.1, "pitch_deg": -38},
  "zones": {"car_1": {"floor_m": [[0,0]], "image_px": [[0,0]]}},
  "persons": [{"id": 2, "role": "staff", "height_m": 1.78, "outfit": "uniform", "hard_negative": false,
               "enter_t": 0.0, "exit_t": null, "present_at_start": true}],
  "frames": [{"t": 0.0, "people": [{"id": 2, "box_xyxy": [0,0,0,0], "visible_frac": 0.83,
               "pos_m": [4.2, 7.9], "yaw_deg": 92.0, "action": "idle"}]}],
  "events_scripted": [{"type": "greet", "staff": 2, "customer": 5, "t": 12.4}],
  "events_by_rule": {"short_clip": [], "dealership": []}
}
```

- **`events_by_rule`:** `gt_events.py` computes these by applying A's rules and presets to the
  true 3D positions and yaws. Treat A's spec as the source of truth and copy its thresholds into
  `configs/presets.json`. If A's rules change, only that file changes.
- **Other exports:**
  - COCO detection set: category `person`, with `attributes.role` and `attributes.visible_frac`
    on each annotation. Sampled frames at 1–2 fps. **Split train/valid/test by scene, never by
    frame.** Folders `train/`, `valid/`, `test/` with `_annotations.coco.json` (RF-DETR layout).
  - MOT txt per clip.
  - **A-format labels** (`labels/<clip>.json`, A's spec §6: persons with roles, greet times,
    engagement intervals, walk-outs). Get an example file or the schema from A. If neither is
    available, write a documented adapter from `gt.json` and flag it in the status file.

### 4.7 CCTV look (`post.py`)
- Gaussian sensor noise, slight blur, mild barrel distortion (optional per clip), a burned-in
  timestamp, and a low-bitrate re-encode.
- Final encode: `libx264`, `yuv420p`, `+faststart`, 10 fps.
- Ground truth must stay aligned. Apply distortion to the boxes too, or skip distortion on clips
  used for box evaluation; document which.

## 5. Stages

Work through these in order. Each stage ends with `python sim/verify.py --upto <stage>` passing and
a local commit.

- **S0 Pre-flight.**
  - Detect OS, GPU, Blender version and free disk.
  - Verify the §2 assets and fill `assets/MANIFEST.csv` (file, source URL, author, licence).
  - Create a venv for the pure modules and QA: numpy, opencv-python-headless, pytest, plus a COCO
    person detector for `qa_detector`. Use RF-DETR Nano (Apache-2.0) or YOLOX via rtmlib
    (Apache-2.0). **No ultralytics (AGPL).**
  - If Mixamo or car assets are missing: continue with procedural stand-ins (capsule mannequins
    with a procedural walk; box cars), mark realism gates as "blocked by assets" in the status
    file, and go on.
- **S1 Pure core, test-first.** `schema`, `layout`, `navigation`, `behaviour`, `gt_events`,
  `projection`, `labels`. Tests cover:
  - paths never enter car footprints;
  - timelines are deterministic for a given seed (hash equality);
  - each scripted scenario produces its intended events under `gt_events` (both presets);
  - projection matches stored reference values from Blender's `world_to_camera_view`, where pixel
    v = (1 − y) · H;
  - label conversion round-trips.
- **S2 Walking skeleton.** One room, stand-in cars, one mannequin, one camera, 3 s. Both passes
  render, labels export, the video encodes, and every gate runs end to end. Measure s/frame.
  Get the whole chain working before making anything look good.
- **S3 Real assets.**
  - Mixamo characters and animations, cars, HDRI, textures.
  - Scale and bone normalisation.
  - NLA sequencing with 0.2–0.3 s blends.
  - Root motion driven from the timeline, with stride matched by strip scale.
  - `qa_scene` passes.
- **S4 Scenarios.** All seven from §4.3. `qa_events` passes: scripted and rule-based events agree.
- **S5 Look and realism.** Lighting variants, CCTV look, camera placement. `qa_render` and
  `qa_detector` pass. Do the visual review (§6.2). Re-measure s/frame and write the render plan.
- **S6 Scale.**
  - Render the dataset within budget, prioritising scenario coverage over volume.
  - Export COCO, MOT and A-format labels.
  - Build `data/synthetic/share/`: the best 5–8 clips, with `gt.json`, A-format labels and a
    README per clip describing what happens and the expected events. Include one "hero" clip
    suitable for the pitch deck.
- **S7 Docs.**
  - `sim/README.md`: one command per stage, how it works, measured s/frame, dataset stats, known
    limitations.
  - `DATA_SOURCES.md`: every asset with its licence; Mixamo raw files are never committed;
    Sketchfab CC BY attributions; Poly Haven CC0; how the data was generated.
  - Final `reports/synth_status.md`.

## 6. Checking and fixing

### 6.1 Automated gates (`sim/qa/`, all run by `python sim/verify.py --all`)

| Gate | Fails when |
|---|---|
| `pytest -q sim/tests` | any test fails |
| `qa_assets` | an asset is missing from `MANIFEST.csv`, or a row has no licence |
| `qa_scene` (in Blender) | person height outside 1.55–1.95 m; car length outside 3.8–5.5 m; a person's root inside a car footprint (+0.2 m); two people closer than 0.35 m for > 0.5 s; feet off the floor by > 5 cm at sampled contact frames; an armature unchanged over 2 s while its action says walk/talk (T-pose / animation not applied); a missing image texture; an agent visible in no camera for > 20% of its lifetime |
| `qa_render` | frame count ≠ expected; any frame with mean luminance < 15 or > 5% clipped pixels (glare variant: < 20%); magenta "missing texture" pixels > 0.1%; an unknown mask colour; a box outside the image; a track with gaps > 5 frames while `visible_frac` > 0.5 |
| `qa_events` | a scripted event not matched by a rule-based event of the same type within ±1.0 s; a rule-based greet, walkout or handoff with no scripted counterpart |
| `qa_detector` | off-the-shelf person detector recall at IoU 0.5 < **0.80** on GT people with box height ≥ 60 px and `visible_frac` ≥ 0.6, per lighting variant. Also report (not a gate) the unmatched detections, split into reflections (hit a reflective surface) and other |
| `qa_video` | ffprobe shows a codec other than h264/yuv420p, wrong fps, duration off by more than 1 frame, or OpenCV cannot decode the first and last frames |

All thresholds live in `sim/configs/qa.json`.

### 6.2 Visual review (you must look)

`sim/qa/contact_sheets.py` writes, for every clip, a grid of 12 evenly spaced frames with:
- boxes coloured by role, with track IDs;
- zone outlines;
- event markers.

Open the sheets with your image-reading tool and actually look at them. Write
`reports/synth_visual_review.md` with a pass/fail per clip for each item below, citing frame numbers:

1. T-pose or frozen characters
2. Floating or sunk feet
3. Visible moonwalking or sliding
4. People walking through cars, walls or each other
5. Staff not facing the customer during a greet
6. Missing or pink textures
7. Lighting too dark or blown out
8. People smaller than about 40 px, or cut off for most of the clip
9. Misaligned boxes
10. Wrong role colours
11. Labelled reflections
12. Wrong scale (giants or children where adults were meant)

### 6.3 Fix loop
- On any failure:
  1. Find the root cause.
  2. Fix it in code or config, never by editing outputs or labels by hand.
  3. Re-run the failing gate.
  4. Re-run the full `verify.py` before moving on.
- If the same gate fails with the same error **3 attempts in a row**, stop working on that item.
  - Record what you tried in `reports/synth_status.md`.
  - Continue with independent work if there is any; otherwise stop.

## 7. Rules

- **Do not weaken gates.**
  - No skipping, `xfail`, deleting or mocking out tests or QA checks.
  - No lowering thresholds in `qa.json`, and no shrinking the evaluation set, to make a gate pass.
  - If a threshold looks wrong, leave it, explain why in the status file, and flag it for the human.
- Report only measured numbers. Label every estimate as an estimate.
- **Reproducible:** `python sim/run_all.py --config sim/configs/scene_003.json` regenerates a scene
  bit-for-bit in its ground truth. Renders may differ by GPU noise.
- **Git:** commit locally at the end of each stage. Do not add remotes, push or deploy. Keep
  `assets/`, `data/`, renders and weights out of git (`.gitignore`).
- **Hard stop:**
  - At `[DEADLINE]`, or after `[MAX_HOURS]` hours of wall-clock, whichever comes first.
  - Stop rendering if free disk falls below 10 GB.
  - Before stopping for any reason, write `reports/synth_status.md` with:
    - what works and what doesn't;
    - gate results (last `verify.py` output);
    - measured s/frame;
    - dataset stats (clips, frames, persons, events by type);
    - open problems;
    - exact commands to resume.
- **Done** means:
  - `python sim/verify.py --all` exits 0, with its full output pasted in the status file;
  - `reports/synth_visual_review.md` covers every shared clip with all 12 items passing;
  - the share pack and `DATA_SOURCES.md` exist.

---

## Optional: `/goal` condition for an unattended run

> Run `python sim/verify.py --all` end to end in one go and show its full output in the
> conversation; it exits 0. That chain runs pytest on `sim/tests`, then `qa_assets`, `qa_scene`,
> `qa_render`, `qa_events`, `qa_detector` and `qa_video` over every clip in `data/synthetic/share/`.
>
> `data/synthetic/share/` holds at least 5 clips covering the scenarios `greet_fast`, `greet_slow`,
> `walkout`, `party` and `handoff`, each with `gt.json` and A-format labels.
>
> `reports/synth_visual_review.md` lists every shared clip with all 12 checklist items marked pass
> and frame numbers cited, and `cat reports/synth_visual_review.md` is shown.
>
> `reports/synth_status.md` and `DATA_SOURCES.md` exist and are shown. All work is committed
> locally (`git status --short` shows no tracked changes).
>
> No test or QA check was skipped, marked xfail, deleted or mocked, and no threshold in
> `sim/configs/qa.json` was lowered: `git diff <start-sha> -- sim/configs/qa.json` is empty and is
> shown.
>
> If the project has `docs/journal.md`, every commit has an entry in it capturing why the approach
> was chosen and anything now fragile (not a restatement of the diff), `docs/STATE.md` reflects
> the state at the end of the run, and `npm run log:check` (or `bash scripts/log-check.sh` where
> there is no package.json) exits 0.
>
> Hard stop: at `[DEADLINE]`, or after `[MAX_TURNS]` turns, or when the same gate has failed with
> the same error 3 turns running, write `reports/synth_status.md` (what works, last `verify.py`
> output, open problems, resume commands) and stop. That counts as finished-with-status, not
> success.
