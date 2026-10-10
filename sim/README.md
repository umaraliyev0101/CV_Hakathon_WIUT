# sim/ — synthetic car-showroom CCTV with exact ground truth

Renders showroom camera clips in Blender with frame-exact labels: role (staff / customer), visible
box, `visible_frac`, floor position (m) and yaw per person per frame, car/desk/entrance/exit zones (m
and px), and greet / engagement / handoff / party / walk-out events (scripted truth plus the same
events recomputed by A's rules from true 3D states). Real-looking people (Microsoft Rocketbox,
MIT) driven by mocap, real car models, glass facade with HDRI outside, three lighting variants.

## One command per stage

```bash
source .venv/bin/activate                      # numpy, opencv, pytest, onnxruntime-gpu, rfdetr (no ultralytics)
python -m pytest -q sim/tests                  # S1 pure core: navigation, behaviour, events, projection, labels, anim plan
python sim/gen_configs.py                      # (re)generate sim/configs/scene_000..011.json (seeded, validated)
python sim/run_all.py --config sim/configs/scene_003.json           # S2-S5: one scene, all cameras x lightings
python sim/run_all.py --all --jobs 1                                # S6: every scene
python sim/export.py                           # S6: COCO (train/valid/test by scene), MOT, A-format labels
python sim/share.py                            # S6: data/synthetic/share/ (selection in sim/configs/share.json)
python sim/verify.py --all                     # THE gate chain over the share pack (exit 0 = all pass)
python -m sim.qa.contact_sheets data/synthetic/clips/<clip_id>       # re-make one contact sheet
```

Blender 4.2+ is found from `$BLENDER`, then `PATH`, then `~/opt/blender-*/blender`. Every GPU
subprocess gets `CUDA_VISIBLE_DEVICES=1`; the GPU that actually rendered is sampled with
`nvidia-smi pmon` and stored as `gpu_index` in `data/synthetic/scenes/<scene>/render_stats.json`.

## How it works

| Module | Role |
|---|---|
| `gen_configs.py` | Seeded rejection sampling of a scene: room 18–26 × 12–17 m, glass south facade with the entrance, east test-drive roller door, desk, 4–6 cars (2.5 m clearance), people + scripts per scenario, 2 cameras (2.8–3.2 m high, pitch −25…−50°, HFOV 70–90°, 4.5–9 m from the action). A candidate is kept only if every scripted event is reproduced by the rules within 0.8 s, nobody comes closer than 0.5 m, and each camera sees every person (full body, ≥ 60 px, not behind a car / desk / another person) ≥ 86 % of the clip. |
| `behaviour.py` | Frame-stepped scheduler: one generator per person, actions `enter, walk_to, idle, idle_until, sit, approach, talk, talk_until_relieved, await_staff, follow, leave`. Visibility-graph paths around inflated car/desk footprints, trapezoid speed profiles (1.15–1.4 m/s), yaw limited to 180°/s. Writes `timeline.json`; Blender only replays it. |
| `gt_events.py` + `configs/presets.json` | A's rules on true 3D states (fallback thresholds, flagged). |
| `anim_plan.py` | Which mocap clip at which clip frame: walk phase advances with distance (stride matched), in-place turns play the 90° turn clips by accumulated yaw (stepped turns, no pivot slide), idles/talks loop with a seam cross-fade, 0.25 s blends. Customers listen, staff talk. |
| `blender/build_scene.py` | Room, marble floor, glass facade + HDRI, roller door, desk, plants, real cars (length normalised, paint randomised), lighting variants `day`, `evening_dim`, `glare` (low sun through the facade). |
| `blender/people.py`, `animate.py` | Rocketbox import, texture relink, staff uniform (= red recolour of the dark suit fabric), keyframes from `anim_plan` (fast `foreach_set`). |
| `blender/render.py` | Per camera: instance masks + per-person solo masks (Workbench, glass hidden) once, then RGB (EEVEE Next, 16 samples, raytracing, AgX) per lighting variant. |
| `labels.py` | Mask → visible box (x2/y2 exclusive), drop < 400 px or < 20 px tall, `visible_frac` = mask px / solo px, `gt.json`. |
| `post.py` | Noise σ 3, blur σ 0.7, burned-in timestamp, x264 CRF 26, yuv420p, +faststart, 10 fps. No barrel distortion, so boxes stay aligned. |
| `export.py`, `share.py` | COCO (RF-DETR layout, split by scene), MOT, A-format labels (adapter), share pack. |
| `qa/` | Gates (thresholds in `configs/qa.json`), contact sheets. |

Timing: scenes run at 30 fps with `frame_step = 3` and are encoded at 10 fps, so motion is real-time.

## Design choices to know

- **Scripted greet = staff arrival** (stops 1.25 m from the customer, facing them). The rule greet is
  the onset of "within 1.5 m and facing ≤ 60° for ≥ 2 s". Choreography keeps them ≤ 0.8 s apart: the
  last approach leg is a straight 1 m walk at the customer, so only the final 0.25 m (decelerating)
  lies inside the radius; staff never pass inside a party member's radius before reaching the leader.
- **Labels follow what is drawn:** a person outside the glass facade is labelled if visible
  (`present` = inside the room is kept for the event rules).
- **Hair cards** count as mask pixels (Workbench ignores alpha): boxes can be a few px taller at the head.
- **Uniform:** Rocketbox clothing is baked into one texture, so staff uniform = dark suit fabric
  recoloured red (`uniform.suit_recolour`). Hard negatives: customers in dark suits / a burgundy blazer.

## Measured numbers, dataset stats, limitations

See `reports/synth_status.md` (measured s/frame per pass, GPU index, clip/frame/person/event counts,
open problems) and `reports/synth_visual_review.md`.
