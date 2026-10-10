# Quick test: synthetic 3D showroom pipeline (teammate C), one clip, then stop for review

Cut-down pilot of `prompts/teammate_c_synthetic_showroom.md`. Run this first so a human can review
the result before the full run. Start the session in the repo where the pipeline should live.
Assets are optional for this test.

---

## 1. Goal

We are a 3-person team in the WIUT Hackathon 2026 CV final. Teammate A builds "Showroom Coach":
one camera per job, person detection → pose → ByteTrack, distance by pinhole back-projection
(people assumed 1.70 m tall, HFOV 70°), staff/customer roles, and events (greet, engagement,
wait_alert, walkout, party, handoff). Spec: `salespeople_prompts/2026-10-11-showroom-coach-design.md`
(if it's missing, use the thresholds in §3 and flag it).

The full task is a Blender pipeline that renders synthetic showroom CCTV with exact ground truth.
**This run is only a quick test.** It shows the chain works end to end on **one short clip**, so a
human can review the result before we commit to the full run. Don't optimise for looks. Optimise
for correct labels and a working chain, and finish fast.

## 2. Scope (do only this)

Build only these parts of the full layout:

```
sim/
  configs/   scene_000.json, presets.json, qa.json
  schema.py, layout.py, navigation.py, behaviour.py, gt_events.py, projection.py, labels.py, post.py   [pure, numpy]
  blender/   build_scene.py, people.py, animate.py, cameras.py, render.py
  qa/        qa_render.py, qa_events.py, qa_video.py, contact_sheets.py
  verify.py  runs pytest + the gates above, exits non-zero on failure
  run_all.py config -> timeline -> blender render -> labels -> post -> verify
  tests/
reports/     synth_quicktest.md, contact sheet PNG
data/synthetic/quicktest/   outputs (gitignored)
```

**Scene:** one room, 20 m × 14 m with a 5 m ceiling, a plain floor and walls, 3 cars. Use box
stand-ins if `assets/cars/` is empty. Add one desk. Zones `car_<k>` (footprint + 1.2 m apron),
`desk`, `entrance`.

**People:** 1 staff member (uniform: dark top, red vest) and 1 customer. Use Mixamo characters and
animations from `assets/mixamo/` if they're present. Otherwise use capsule mannequins with a
procedural walk, and note "assets missing" in the report.

**Scenario `greet_fast`, 20 s:** the customer enters at t = 0 and walks to `car_1`. The staff member
walks over from the desk, stops 0.9–1.3 m away facing the customer (within 20°), and they talk for
at least 6 s. Walking paths must avoid car footprints. Speed is 1.1–1.5 m/s and yaw turns are smooth.

**Camera:** 1 camera in a ceiling corner, height 3.0 m, pitch −35°, HFOV 74°. Both people must be
in view for at least 80% of the clip.

**Render:**
- RGB pass in EEVEE at 1280×720, "day" lighting only.
- Scene runs at 30 fps with `frame_step = 3`, so the output is 10 fps and 200 frames.
- Instance-mask pass in Workbench: object colour per person, AA off, Standard view transform, 8-bit PNG.
- Measure seconds per frame for both passes.

**Labels:** a `gt.json` in the full schema (below). Visible box = bounding box of the mask pixels.
Drop a person if the mask is under 400 px or the box is under 20 px tall. Skip the COCO, MOT and
A-format exports in this test.

**Post-processing:** light noise and blur, a burned-in timestamp, then encode with `libx264`,
`yuv420p`, `+faststart`, 10 fps. No barrel distortion.

`gt.json` schema (keep it, so the full run reuses it):

```json
{"clip_id": "s000_greet_fast_cam1_day", "labeller": "synthetic-ground-truth",
 "generator": {"git_sha": "...", "config": "configs/scene_000.json", "seed": 0},
 "fps": 10, "width": 1280, "height": 720, "duration_s": 20.0,
 "camera": {"hfov_deg": 74.0, "K": [[...]], "R": [[...]], "t": [...], "height_m": 3.0, "pitch_deg": -35},
 "zones": {"car_1": {"floor_m": [[0,0]], "image_px": [[0,0]]}},
 "persons": [{"id": 1, "role": "staff", "height_m": 1.78, "outfit": "uniform", "hard_negative": false,
              "enter_t": 0.0, "exit_t": null, "present_at_start": true}],
 "frames": [{"t": 0.0, "people": [{"id": 1, "box_xyxy": [0,0,0,0], "visible_frac": 0.83,
              "pos_m": [4.2, 7.9], "yaw_deg": 92.0, "action": "idle"}]}],
 "events_scripted": [{"type": "greet", "staff": 1, "customer": 2, "t": 6.4}],
 "events_by_rule": {"short_clip": []}}
```

## 3. Gates (the quick-test subset; thresholds in `sim/configs/qa.json`)

| Gate | Fails when |
|---|---|
| `pytest -q sim/tests` | any test fails. Tests must cover: paths never enter car footprints; timeline is deterministic for a seed (hash equality); `greet_fast` produces a greet under `gt_events`; projection matches Blender's `world_to_camera_view` (v = (1 − y) · H) |
| `qa_render` | frame count ≠ 200; mean luminance < 15 or > 5% clipped pixels; magenta pixels > 0.1%; unknown mask colour; box outside the image |
| `qa_events` | the scripted greet has no rule-based greet within ±1.0 s, or a rule-based greet has no scripted counterpart. Rule: distance ≤ 1.5 m, facing ≤ 60°, sustained ≥ 2 s (take the exact values from A's spec if available) |
| `qa_video` | ffprobe shows a codec other than h264/yuv420p, fps ≠ 10, duration off by more than 1 frame, or OpenCV can't decode the first and last frames |

Not in this test: `qa_assets`, `qa_scene` in Blender, `qa_detector`, extra lighting variants, the
other scenarios, the share pack, docs.

## 4. Steps

1. **Pre-flight.** Detect the Blender path and version (4.2+), OS, GPU and free disk. Create a venv
   with numpy, opencv-python-headless and pytest. **No ultralytics.**
2. **Pure core, test-first.** Write the tests in §3, then the modules.
3. **Render.** Build the scene, render both passes, export labels, post-process and encode.
4. **Check.** Run `python sim/verify.py`. Generate the contact sheet: 12 evenly spaced frames,
   boxes coloured by role with track IDs, zone outlines, event markers. **Open it with your
   image tool and look at it.**
5. **Fix loop.** Find the root cause and fix it in code or config, never by editing outputs.
   Re-run the failing gate, then the full `verify.py`. There is no cap on attempts: keep fixing
   until every gate passes and the visual check is clean. If a fix keeps failing, change approach
   rather than repeating it. Stop early only if you are blocked on something only a human can
   provide (e.g. Blender not installed); record that in the report.
6. **Commit locally.** No remotes, no push. Keep `assets/` and `data/` gitignored.

## 5. Rules

- Don't weaken gates: no skip, xfail, mocks, or lowered thresholds. If a threshold looks wrong,
  leave it and flag it.
- Report measured numbers only. Label every estimate as an estimate.
- Don't create accounts, spend money, or upload anything.
- No deadline, but keep it quick: this is a pilot. **Do not continue into the full pipeline.**
  Stop once the report below is written and wait for human review.

## 6. Deliverable: `reports/synth_quicktest.md`

- Last `verify.py` output, in full.
- Contact sheet path, plus your visual check with frame numbers for: T-pose or frozen characters;
  floating feet; sliding; walking through cars; staff not facing the customer during the greet;
  pink textures; too dark or blown out; misaligned boxes; wrong role colours; wrong scale.
- Measured s/frame (RGB, mask) and a **labelled estimate** of render hours for the full run
  (7 scenarios × 2–4 cameras × 3 lighting variants).
- Scripted vs. rule-based events, with timestamps.
- Paths to the MP4, `gt.json` and contact sheet.
- Whether real assets or stand-ins were used, open problems, and the exact command to regenerate:
  `python sim/run_all.py --config sim/configs/scene_000.json`.
