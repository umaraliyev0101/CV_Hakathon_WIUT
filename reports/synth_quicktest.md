# Synthetic showroom quick test: `greet_fast`, one camera

**Status: all gates pass and the visual check is clean. Stopped here for human review; the full pipeline was not started.**

- Clip: `s000_greet_fast_cam1_day`. 20.0 s, 200 frames at 10 fps (scene runs at 30 fps, `frame_step = 3`), 1280×720.
- Generator: git `725aec3`, config `sim/configs/scene_000.json`, seed 0. The sha is recorded in `gt.json`.
- Machine: Ubuntu 25.10 (Linux 6.17), NVIDIA RTX PRO 6000 Blackwell (driver 595.71.05; Blender uses OpenGL on it), 128 cores, 122 GB RAM, 666 GB free on `/mnt/raid1`.
- Blender **4.2.23 LTS**, `~/opt/blender-4.2.23-linux-x64/blender`. It was not installed. I downloaded the official portable tarball from download.blender.org and checked its sha256. No account, no sudo.
- Python venv `.venv/` (gitignored): numpy 2.4.6, opencv-python-headless 5.0.0, pytest 9.1.1. **No ultralytics.**

## Regenerate

```bash
source .venv/bin/activate          # or: python = .venv/bin/python
python sim/run_all.py --config sim/configs/scene_000.json   # timeline -> render -> labels -> post -> contact sheet -> verify (~70 s)
python sim/verify.py                                         # gates only
```

Blender is found from `$BLENDER`, then `PATH`, then `~/opt/blender-*/blender`.

## Outputs

| What | Path |
|---|---|
| MP4 (h264, yuv420p, +faststart, 10 fps, 200 frames) | `data/synthetic/quicktest/s000_greet_fast_cam1_day/s000_greet_fast_cam1_day.mp4` |
| Ground truth (full schema) | `data/synthetic/quicktest/s000_greet_fast_cam1_day/gt.json` |
| Contact sheet | `reports/synth_quicktest_contact.png` (copy in `data/synthetic/quicktest/s000_greet_fast_cam1_day/contact_sheet.png`) |
| Timeline, render stats, .blend, RGB/mask/solo PNGs | same clip folder: `timeline.json`, `render_stats.json`, `scene.blend`, `rgb/`, `mask/`, `solo/p<id>/` |

## Last `python sim/verify.py` output (full)

```
=== gate: pytest -q sim/tests ===
........................                                                 [100%]
24 passed in 6.70s
=== gate: qa_render (/mnt/raid1/projects/CV_Hakathon_WIUT/data/synthetic/quicktest/s000_greet_fast_cam1_day) ===
  qa_render: mean luma min/mean/max 175.0/177.5/178.1; max clipped 0.061%; max magenta 0.0000%
  qa_render: unknown mask px total 0 over 200 masks
  qa_render: person 1 (staff) labelled in 200/200 frames (100.0%)
  qa_render: person 2 (customer) labelled in 192/200 frames (96.0%)
qa_render: PASS (0 failure(s))
=== gate: qa_events (/mnt/raid1/projects/CV_Hakathon_WIUT/data/synthetic/quicktest/s000_greet_fast_cam1_day) ===
  qa_events: [short_clip] scripted=[('greet', 6.533333)] rule=[('greet', 5.7)]
qa_events: PASS (0 failure(s))
=== gate: qa_video (/mnt/raid1/projects/CV_Hakathon_WIUT/data/synthetic/quicktest/s000_greet_fast_cam1_day) ===
  qa_video: s000_greet_fast_cam1_day.mp4: codec=h264 pix_fmt=yuv420p fps=10/1 avg=10/1 frames=200 duration=20.000s size=1280x720
  qa_video: opencv: frame_count=200 first_ok=True last_ok=True
qa_video: PASS (0 failure(s))
=== summary ===
  pytest     PASS
  qa_render  PASS
  qa_events  PASS
  qa_video   PASS
VERIFY: PASS
exit code: 0
```

`sim/configs/qa.json` is unchanged since commit `3f5afca`, the commit that added it. No test or gate is skipped, xfail'd or mocked.

## Events: scripted vs rule-based

| Event | Scripted t | Rule-based t (`short_clip`) | Δ |
|---|---|---|---|
| greet, staff #1 → customer #2 | **6.53 s** (staff stops 1.10 m away, facing the customer; talk starts) | **5.70 s** | −0.83 s (gate tolerance ±1.0 s) |

- Rule: distance ≤ 1.5 m, staff facing ≤ 60°, sustained ≥ 2 s. The event time is the start of the run.
- The rule always fires earlier. The staff member enters the 1.5 m radius, already walking toward the customer, while still braking toward the 1.1 m stop. This is deterministic, not noise. Deceleration is 1.5 m/s² (acceleration 1.0 m/s²), which keeps the gap at 0.83 s. At 1.0 m/s² it was 0.97 s, which would have been fragile.
- Talk lasts 13.47 s (minimum 6 s). Distance during the talk is 1.10 m every frame (allowed 0.9–1.3 m). Staff facing error during the talk is at most 0.0° (allowed ≤ 20°). Peak walking speed is 1.28 m/s (allowed 1.1–1.5 m/s). Yaw is rate-limited to 180°/s. The tests check all of these.

## Measured render speed

All numbers are from `render_stats.json` for this clip: 1280×720, mean per frame, excluding the first frame.

| Pass | s/frame | Note |
|---|---|---|
| RGB, EEVEE Next (16 TAA samples, raytracing on, AgX) | **0.112** | First frame 4.3 s cold (shader compile), 0.34 s warm |
| Instance mask, Workbench (flat, object colour, AA off, Standard) | **0.037** | |
| Solo masks (each person rendered alone, for `visible_frac`) | **0.031 per person** | 2 people here, so 0.062 per frame |
| Whole pipeline wall time for the clip | 70.5 s | About 28 s of that is non-render overhead: Blender start, scene build and keyframing, labels, encode, QA |

### Full-run render estimate (**ESTIMATE**, not measured)

Assumptions:
- 7 scenarios × 2–4 cameras × 3 lighting variants.
- Six scenarios at 45–90 s each; `dealership_long` at 240 s with 1–2 cameras.
- 2.5–3.5 people per frame on average (solo-mask cost scales with this).
- About 28 s of overhead per clip, as measured here.
- **The same stand-in scene complexity as this test.**

| | Frames | Clips | s/frame | **Render hours (estimate)** | PNG disk before cleanup (estimate) |
|---|---|---|---|---|---|
| Low (45 s clips, 2 cameras) | 23,400 | 39 | 0.227 | **≈ 1.8 h** | ≈ 21 GB |
| High (90 s clips, 4 cameras) | 79,200 | 78 | 0.258 | **≈ 6.3 h** | ≈ 73 GB |

Real Mixamo characters, car meshes, HDRI and glossy textures will make EEVEE slower. I have not measured that. Re-measure s/frame after S3 before committing to a budget. A guess of 3–5× on the RGB pass would put the full run at roughly 4–25 h (**unmeasured guess**).

## Visual check

I opened `reports/synth_quicktest_contact.png` (12 frames: 0, 18, 36, 54, 72, 90, 109, 127, 145, 163, 181, 199). I also opened full-resolution crops of frames 0, 20, 24, 26, 30, 32, 33, 36, 38, 44, 45, 50, 51, 54, 57, 58, 60, 62, 66, 72, 100, 120, 180 and 199. Frame numbers are output frames at 10 fps (t = frame / 10).

| Item | Result | Evidence |
|---|---|---|
| T-pose or frozen characters | **pass** | Walking legs swing with knee lift: customer f20–f44, staff f44–f62. Talk gestures (arms and head) change between f72, f90, f109 and f127. There is a test for limbs frozen over 2 s of walk or talk. |
| Floating or sunk feet | **pass** | Contact shadows touch the soles at f20, f44, f54, f100 and f199. Pelvis height is solved so the lowest sole is at z = 0 (tested to < 1 mm). |
| Sliding / moonwalking | **pass** (fixed during the run) | The first render slid: the stance foot moved ~10 cm per step, and up to ~1.5 m/s at speed ramps. **Fixed:** the stance foot now moves back at exactly body speed, each walk is a whole number of half-cycles starting and ending with the planted foot under the hip, and the swing foot is lifted by 2-link IK. Planted-foot speed in straight walking, including ramps, is now ~1e-5 m/s, and a test enforces < 0.05 m/s. **Residual:** in-place pivot turns (staff f30–f38 leaving the desk; customer f66–f72 turning to the staff) rotate both feet about the body centre at up to 0.5 m/s for 0.3–0.8 s. That reads as a shuffle-turn. Mixamo Turn clips would replace it. |
| Walking through cars | **pass** | The staff path (1.5, 5.35) → (3.12, 4.18) and the customer path (7.5, 0.45) → (3.9, 3.4) keep ≥ 0.55 m from every car footprint by construction (obstacles are inflated by body radius plus clearance). The tests assert ≥ 0.3 m (body radius) on 300 random paths and on the clip timeline. No overlap visible at f44–f62. |
| Staff not facing the customer during the greet | **pass** | Staff yaw −45° equals the bearing to the customer exactly over the whole talk. The staff arm gestures toward the customer at f60, f100 and f199. The customer turns toward the staff from f66, reaching 135° by f72 and holding it to f199. |
| Pink or missing textures | **pass** | Magenta pixels are 0.0000% in all frames. No image textures are used (procedural materials only). |
| Too dark or blown out | **pass** | Mean luminance is 175.0–178.1 and at most 0.061% of pixels are clipped. The image is bright and fairly low-contrast (light floor, area lights), but not blown out. |
| Misaligned boxes | **pass** | Boxes hug the silhouettes in every sheet frame and crop. They come straight from mask pixels, and the mask pass uses the same camera. |
| Wrong role colours | **pass** | Staff #1 is red (dark top, red vest), customer #2 is blue in all 12 sheet frames. |
| Wrong scale | **pass** | Staff 1.78 m reads slightly taller than customer 1.70 m (f100, f199). During the talk both boxes are about 310–340 px tall at 4.5–5 m. Proportions are adult. |

Other notes from the visual check:
- **The staff member at the desk (f0–f38) looks strongly tilted.** This is camera perspective, not a pose bug. A true vertical at that floor point projects at −20.7° in the image, because the point is near the left image edge with a −35° pitch and 74° HFOV. The desk edge leans the same way.
- **The customer is out of view at f0–f7 while entering.** They are labelled in 192/200 frames (96%); the staff member in 200/200 (100%). The requirement is ≥ 80%.
- The minimum box height is 40 px, at the customer's first labelled frame (entering at the right edge). `visible_frac` is 1.00 in all frames, because the two people never occlude each other in this layout.
- **Composition:** with a 3.0 m height and −35° pitch, full bodies fit only between 1.9 and 6.1 m from the camera. I moved the greet to about 4.5 m out. People are framed in the upper half of the image and the far cars are cut by the top edge. This is inherent to the specified camera, not a bug.

## Problems found and fixed during the run

1. **Action at the top edge, heads cut off** (first layout put the greet about 7 m from the camera). I re-laid out the room so that car_1, the desk and the greet spot sit 4.5–5.6 m out. I added a pre-render geometry test that both people are in view for ≥ 80% of the clip.
2. **Staff overshot and turned ~120° in place before greeting.** The approach-point cost now includes the final turn angle.
3. **Foot sliding.** See the visual check table. Root cause: sinusoidal hip swing, plus late-swing geometry where the swing leg was vertically longer than the stance leg, so it got planted.
4. **Staff walked sideways while turning from the desk.** The staff now turns to the first path segment before stepping off. The customer's spawn yaw is set to the entry heading.
5. **Projection test tolerance.** Blender's `world_to_camera_view` works in float32. Against the fixture (120 points, 3 cameras) the median error is 1e-4 px and the max is 4.6e-3 px, for off-image points close to the camera. The test tolerance is 0.01 px, with that reason documented. This is a test tolerance, not a qa.json gate.
6. **libpng `eXIf: duplicate` warnings flooded `verify.py` output.** Blender 4.2 writes the eXIf chunk twice. The render step now drops the second copy at chunk level; pixels are untouched.

## Assets and design choices

- **Stand-ins were used for everything; real assets are missing.** `assets/` does not exist, so there are no Mixamo characters, car models or HDRI.
  - People are procedural capsule mannequins with a procedural walk, talk and idle (`sim/blender/people.py`, `sim/behaviour.py`).
  - Cars are bevelled box bodies with dark glass cabins and cylinder wheels. The desk is a box.
  - The "day" lighting is 6 ceiling area lights plus a sky-coloured world. The room has no glass facade or HDRI.
- **Mask colours:** person *k* gets a pure 0/1 RGB colour (#1 = (255, 0, 0), #2 = (0, 255, 0)). These values survive the Standard view transform exactly. 0 unknown-colour pixels in 200 masks.
- **`visible_frac`** = mask pixels ÷ solo-render mask pixels (the person rendered alone), counted within the image only. Truncation at the image border is not counted.
- **`box_xyxy`** = mask bounding box with exclusive x2/y2 pixel edges. A person is dropped from a frame if the mask is under 400 px or the box is under 20 px tall (`qa.json` labels).
- **No barrel distortion**, so boxes stay aligned with the post-processed video. Post-processing is Gaussian noise σ = 3, blur σ = 0.7, a burned-in timestamp, and x264 at CRF 26.
- COCO, MOT and A-format exports were skipped, as the quick-test spec says.

## Open problems and flags for the human

- **FLAG: A's spec is missing.** `salespeople_prompts/2026-10-11-showroom-coach-design.md` is not in the repo. `sim/configs/presets.json` uses the quick-test fallback greet rule: ≤ 1.5 m, ≤ 60°, ≥ 2 s, with the event time at the onset of the run. Two details are my reading, not A's: only the *staff* member's facing is checked, and the event is timed at onset rather than at confirmation. Please confirm both against A's spec.
- **FLAG: rule-vs-scripted greet offset.** The scripted greet (staff stops) and the rule greet (staff enters 1.5 m) differ systematically by 0.83 s, inside the ±1.0 s gate. Slower deceleration or a larger stop distance would push it past 1 s. Decide whether "scripted greet" should mean the arrival or the moment the staff member comes within range.
- **Pivot turns** still move the feet; see the visual check table.
- **Mixamo, cars and HDRI are needed before S3.** Expect a higher s/frame with them, so re-measure before the full run.
- **qa.json threshold notes (not changed).**
  - The quick-test spec does not define "clipped pixel". `qa.json` uses luminance ≥ 250.
  - `min_in_view_frac` = 0.8 is an extra gate I added, from the 80% camera-coverage requirement.
