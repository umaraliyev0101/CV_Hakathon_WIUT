# Prompt for teammate C: synthetic showroom data + model training

Saved verbatim on 2026-10-11 from the team's prompt. Part 1 has been replaced by
`prompts/teammate_c_synthetic_showroom.md`. The run prompt is `prompts/teammate_c_full_run_goal.md`.

Paste everything below the line into your coding assistant (Claude Code or similar).
Fill the [HH:MM] placeholders with real times before you start.

**Updated:**
- Part 1 (the synthetic generator) is superseded by `prompts/teammate-C-3d-synthetic-pipeline.md`
  (in this repo: `prompts/teammate_c_synthetic_showroom.md`); use that file for the 3D work.
- The contract below now follows teammate A's spec
  (`salespeople_prompts/2026-10-11-showroom-coach-design.md`):
  - The detector finds `person` only.
  - Staff vs customer comes from A's rules plus one click.
  - Event thresholds come from A's presets.

---

## Context

We are a 3-person team at the WIUT Hackathon 2026, Computer Vision track (partners: Toyota Royal
Motors, BYD Royal Motors, Ducati). The final demo is tomorrow (2026-10-12). Judges score:
Innovation 25%, Technical Complexity 25%, Impact & Practicality 25%, Presentation 15%,
Business 10%. Judges open our public web demo on their own laptops/phones and feed it their own
video, so everything must also work on real footage, not only on our data.

Product: analytics for car showrooms from the cameras a dealer already has. Two halves:
- Salesperson analysis (teammate A): per-visit sales-process checklist and per-salesperson
  scorecard — time to greet, walked the customer to a car, handover to the finance desk,
  customer left without contact. A "continuous mystery shopper".
- Customer analysis (teammate B): per-car dwell, paths, zone funnel, unattended customers, exits.

You (teammate C): the data and the model that both halves stand on:
- a synthetic showroom generator in Blender that renders multi-camera, CCTV-like 2D video with
  exact labels, and
- a fine-tuned detector that tells staff from customer, exported to ONNX, with honest metrics on
  real footage.

## Interface contract (agree with A and B first, do not change silently)

Create `contracts/README.md` with this if it does not exist, and tell A and B.
- Class: a single `person` class. Staff vs customer is decided downstream by A's roles unit
  (rules plus one click); it is not a detector class.
- Model: `models/showroom_detector.onnx`, plus `models/README.md` documenting the input tensor
  name, shape, normalisation and output format. Add a row to A's `models/MANIFEST.json` with URL,
  sha256 and licence.
- Wrapper: it must plug into A's `perception.detect` unit:
  `detect(frame) -> list[Det(bbox, score, cls)]`. Copy the exact `Det` dataclass from A's code.
- Runtime: onnxruntime on CPU.
- A's machine (i5-10210U, no GPU) picks its detector with a CPU benchmark. Your model must beat or
  match the COCO-pretrained baseline on recall at a speed that still fits there.
- Event definitions: use A's spec (§4.5 rules, §4.6 presets) as the single source of truth. Do not
  define your own thresholds.
- Synthetic output: see the 3D pipeline prompt, §4.6 (gt.json, COCO, MOT, A-format labels).

## Part 1: synthetic showroom generator

Follow the 3D pipeline prompt (the authoritative spec, with its own QA gates and stop rules).

## Part 2: detector fine-tuning

Model: RF-DETR Nano (Apache 2.0; `pip install rfdetr`). Do not use Ultralytics YOLO (AGPL), and
do not use CrowdHuman (licence unclear) or SMPL/AMASS (non-commercial).

Real data:
- Our staged clips: phone at 2.5–3 m height, team members, red vest = staff, several rooms,
  normal and dim light.
- Extract frames at 1–2 fps.
- Auto-label people with a COCO-pretrained RF-DETR Medium or Large, then verify by hand in CVAT or
  Label Studio. Make sure reflections in glass and car paint are not labelled; they are the
  negatives the fine-tune learns from. Record each person's role as an attribute too
  (red vest = staff), because A needs real role labels to evaluate its rules.
- Split by clip/session, never by random frame. Keep one unseen room plus one dim-light clip as
  test-only.
- Optional pretraining data: one or two retail scene folders from NVIDIA PhysicalAI-SmartSpaces on
  Hugging Face (CC BY 4.0). Never download the full 6 TB.

Training:
- Dataset in COCO format with `train/`, `valid/` and `test/` folders, each holding
  `_annotations.coco.json`.
- `RFDETRNano().train(dataset_dir=..., epochs=..., batch_size=..., grad_accum_steps=..., lr=1e-4, output_dir=...)`,
  keeping the total batch size around 16. Check the current rfdetr docs for argument names.
- Prefer a CUDA GPU (on this machine: **GPU 1**, `CUDA_VISIBLE_DEVICES=1`). Apple MPS is a fallback
  only.
- Goal: a showroom-robust person detector. The failure modes to fix are reflections in car paint and
  glass, people partly hidden behind cars, and steep ceiling-camera angles.

Experiments, all tested on the same held-out REAL test set:
1. Baseline: the COCO-pretrained detector A ships.
2. Fine-tuned on real only.
3. Fine-tuned on real + synthetic.
4. Fine-tuned on synthetic only, if time allows.

Report in `reports/model_eval.md` and `reports/model_eval.json`:
- person AP50 and recall
- false positives on reflections (count per 100 frames)
- recall on occluded people (visible_frac < 0.6)
- AP50 on the dim-light subset
- FPS of the ONNX model on a laptop CPU at 720p, and on the GPU
- model size
- whether synthetic data helped, said plainly either way

Export: `pip install rfdetr[onnxexport]` and `model.export()`. Check that ONNX outputs match
PyTorch on 5 frames within tolerance. Then update `models/detect.py` and run it on a real frame,
printing the detections.

## Licences and repo hygiene

Record every asset, dataset and model with its licence and author in `DATA_SOURCES.md`:
- Mixamo (royalty-free; raw files not redistributed)
- Sketchfab CC BY cars, with author names
- Poly Haven (CC0) and MPFB2 (CC0)
- COCO
- SmartSpaces (CC BY 4.0)
- our own footage, with the consent of everyone filmed
- RF-DETR (Apache 2.0)

Also record how the synthetic data was generated.

Do not commit raw Mixamo/Sketchfab files, renders or datasets. Add `assets/` and `data/` to
`.gitignore`. Publish sample clips through a Hugging Face dataset or GitHub release instead.
Ask me before spending money, uploading data to an external service, or installing system-wide
software.

## Priorities and hard stops

- **P0**, by [HH:MM]: `models/detect.py` with COCO-pretrained weights, so A and B are unblocked.
- **P0**, by [HH:MM]: real staged data labelled → RF-DETR Nano person fine-tune → ONNX → eval report
  against A's baseline. Ship it only if it beats the baseline on the real test set.
- **P1**: synthetic v0 — room, cars, people walking and idling, 3 cameras, labelled frames →
  retrain on real + synthetic → comparison table.
- **P2**: scripted scenarios with events.json for A and B, the CCTV look, lighting variants.
- **Hard stop:** if synthetic v0 is not producing correctly labelled frames by [HH:MM], stop the
  synthetic track, ship the real-only model, and note "synthetic twin" as roadmap.
- Same error three attempts in a row: stop and tell me, rather than looping.
- Before stopping for any reason, write `reports/status_C.md`: what works, what doesn't, measured
  numbers, and how to regenerate everything.

## Rules

- Never evaluate on clips that were used for training, and never "fix" a metric by changing the
  test set.
- Report measured numbers only. If something wasn't measured, say so.
- Keep commands reproducible: one command per stage, documented in `sim/README.md` and
  `training/README.md`.
