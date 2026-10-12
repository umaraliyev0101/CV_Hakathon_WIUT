# Person detector fine-tuning: results

Run on 2026-10-11, 09:30–12:06 (+05), GPU 1. All numbers below were measured. Raw results:
`data/finetune_runs/eval.json` (RF-DETR) and `data/finetune_runs/eval_yolo.json` (YOLO benchmark).
For a slide-ready summary see `reports/finetune_for_pitch.md`.

## Summary

- **Deployed model:** RF-DETR Nano fine-tuned from COCO weights → `models/showroom_detector_ft.onnx`
  (person-only head, same size and CPU speed as the stock model).
- **Gain where the product needs it:** at the product threshold (conf 0.5) it finds **88.6 %** of real
  people instead of 82.7 % (missed people 67 → 44 of 387), and **72 %** of people hidden behind cars
  instead of 59 %, with no more false alarms.
- **Not improved:** box accuracy on real footage (AP50–95 0.779 → 0.762, within noise for 106 frames).
- **YOLO11n benchmark:** worse than RF-DETR before and after fine-tuning, and AGPL-3.0, so not used.

## 1. Data

### Real footage (labelled in this run)

| Item | Value |
|---|---|
| Source | 37 Pexels dealership clips: Antoni Shkraba shoot, IDs 7154207–7154242, plus 36512656 (≈ 7 min) |
| Sampling | 1 frame per second → 429 frames |
| Pre-labels | COCO RF-DETR Large, `person`, score ≥ 0.3 → 1,184 boxes |
| Review | Every frame checked visually (4 parallel reviewers + lead spot-check of every added box) |
| Review changes | 70 boxes deleted (car-seat headrests seen through windscreens, reflections, duplicates), 27 added (missed blurred background people), 8 tightened |
| Final | 1,141 person boxes |

Labelling policy: every visible real person gets one box of their visible extent, including people
seen through glass, cut off at the frame edge, or blurred in the background. Reflections, posters and
headrests get no box. A score cut-off alone could not have produced these labels: the low-scoring
boxes included real background people *and* the headrest and reflection false positives.

Split by clip (neighbouring frames are near-duplicates, so frames from one clip never span splits):

| Split | Clips | Frames | Boxes |
|---|---|---|---|
| train | 25 | 291 | 657 |
| valid | 3 | 32 | 97 |
| test | 9 (incl. demo clips 7154214, 7154217, 7154241, 36512656) | 106 | 387 |

### Synthetic footage (labels exact from the renderer)

Blender CCTV clips from `sim/` (camera 2.8 m high, pitched down 25°). Boxes are visible-mask boxes with role,
track ID and visible fraction. Existing COCO export: train 6,300 / valid 612 / test 1,950 frames.

### Fine-tune mix (`data/finetune/`)

| Split | Real | Synthetic | Total frames | Boxes |
|---|---|---|---|---|
| train | 291 × 3 = 873 | every 4th = 1,575 | 2,448 | 6,831 |
| valid | 32 | 153 | 185 | 390 |

Real frames are repeated 3× so 291 real frames are not swamped by synthetic ones. The real test
split is kept separate and is never trained on.

Note: the synthetic COCO export was regenerated at 11:59 by the simulation work (train 6,300 → 5,756
frames; valid and test unchanged). Training used the earlier export; its exact mix is kept in
`data/finetune/`. Re-running `build_mix.py` now gives 2,312 train frames (in `data/finetune_rebuilt_1159/`).

## 2. Training

| Setting | Value |
|---|---|
| Model | RF-DETR Nano (`rfdetr` 1.11.2), from COCO weights, `class_names=["person"]` |
| Input | 384 × 384 |
| Optimiser | lr 1e-4 (encoder 1.5e-4), batch 16, EMA on |
| Time | ~25 s per epoch on one RTX PRO 6000 |
| Stopping | Validation mAP50–95 peaked at epoch 4 (0.892), then fell (0.870 by epoch 8); run stopped at epoch 9 |
| Exported | epoch-4 EMA checkpoint → ONNX, opset 17, simplified, 107.6 MB |

Validation mAP50–95 by epoch: 0.802, 0.865, 0.870, 0.892, **0.892**, 0.887, 0.889, 0.880, 0.870.

## 3. Results

Evaluation script: `models/finetune/evaluate.py`. Confidence 0.5 (product default), IoU 0.5.
Real = the 106-frame real test split. Synthetic = every 3rd synthetic test frame (650 frames).

### 3.1 Detection

| Metric | RF-DETR Nano COCO | **RF-DETR Nano fine-tuned** | YOLO11n COCO | YOLO11n fine-tuned |
|---|---|---|---|---|
| Real AP50 | **0.943** | 0.939 | 0.875 | 0.731 |
| Real AP50–95 | **0.779** | 0.762 | 0.704 | 0.439 |
| Real recall @0.5 | 0.827 | **0.886** | 0.685 | 0.651 |
| Real false positives / frame | 0.13 | 0.12 | **0.00** | 0.16 |
| Synthetic AP50 | 0.990 | **0.990** | 0.980 | 0.972 |
| Synthetic AP50–95 | 0.823 | **0.856** | 0.833 | 0.823 |
| Synthetic recall @0.5 | 0.980 | **0.988** | 0.972 | 0.973 |
| Synthetic false positives / frame | 0.018 | **0.011** | 0.015 | 0.223 |

### 3.2 Use-case checks

| Check | RF-DETR COCO | **RF-DETR fine-tuned** | YOLO11n COCO | YOLO11n fine-tuned |
|---|---|---|---|---|
| People < 60 % visible (behind cars), recall (synthetic, n = 29) | 0.59 | **0.72** | 0.28 | 0.24 |
| Floor-position error from detector boxes, median (m) | 0.19 | 0.18 | 0.17 | 0.18 |
| Floor-position error from detector boxes, p90 (m) | 0.32 | 0.29 | 0.29 | 0.32 |
| Headrest / reflection regions detected (real test) | 1 of 3 | 1 of 3 | 0 of 3 | 0 of 3 |
| CPU speed, 720p, 4 threads | 18.5 FPS | 18.0 FPS | not measured | not measured |

Floor-position error is what feeds person-to-car distance and dwell time (`analytics/`): each matched
detector box is projected onto the floor and compared with the simulator's true position.

### 3.3 Distance and dwell on exact boxes (reference)

From `analytics/validate_synthetic.py`, using simulator boxes over 61 clips (100,010 person-frames):

| Person visible | Median error | p90 error |
|---|---|---|
| ≥ 90 % | 0.18 m | 0.28 m |
| 60–90 % | 0.18 m | 0.88 m |
| < 60 % | 0.18 m | 0.52 m (4.18 m before the head-based fallback for hidden legs) |

Dwell time near cars, 198 (person, car) pairs: 6,242 s from boxes vs 6,333 s ground truth;
per-pair error median 0.2 s, p90 0.8 s.

## 4. Reading the results

- **Recall is the gain.** The fine-tune mainly raises how many people pass the product threshold, on
  real footage (+5.9 pts) and for people partly hidden behind cars (+14 pts). False positives do not rise.
- **Real AP is flat.** The changes (−0.004 AP50, −0.017 AP50–95) are within noise for 106 frames. The real
  labels started as COCO RF-DETR Large boxes, which slightly favours the COCO model's box style.
- **Distance accuracy is set by geometry, not the detector.** All four models give 0.17–0.19 m
  median floor error. The fine-tune only helps by finding more of the hidden people.
- **YOLO11n.** The stock model is cautious: zero false positives, but it misses about a third of real people
  and three quarters of people behind cars. One quick fine-tune with default settings made it worse
  on real footage, probably from forgetting real appearance on a mostly synthetic mix. That is one
  untuned run, not YOLO's best, but it starts well behind RF-DETR either way. Ultralytics is AGPL-3.0;
  RF-DETR is Apache-2.0.

## 5. Caveats

- **The test split is model-reviewed, not human-verified.** The project goal requires a human-labelled
  test set. A person should check `data/real_labels/review_videos/test_labeled.mp4` (106 frames, ~1 min)
  before these numbers are called human ground truth.
- **One shoot.** All real clips come from one Pexels shoot (same showroom, same four actors). Test
  scores overstate how well the model will do in other showrooms.
- **Small samples.** 29 hidden-person cases and 3 headrest/reflection regions; treat those rows as
  indicative.
- **Licence.** The Pexels licence does not explicitly cover ML training (`research_notes/open_footage.md`).
- **Synthetic validation dominates model selection.** 153 of 185 validation frames are synthetic.

## 6. Reproduce

From the repo root (GPU 1):

```bash
# real labels
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/label_real.py prelabel
.venv/bin/python models/label_real.py export            # applies data/real_labels/review.json
.venv/bin/python -I models/finetune/build_mix.py   # data/finetune/ (symlinks)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/train_nano.py
.venv/bin/python models/finetune/export_ft.py data/finetune_runs/nano_mix/checkpoint_best_ema.pth models/showroom_detector_ft.onnx
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/evaluate.py --ckpt data/finetune_runs/nano_mix/checkpoint_best_ema.pth
# YOLO benchmark
.venv/bin/python -I models/finetune/to_yolo.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/train_yolo.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/finetune/evaluate.py --yolo-base data/finetune_runs/yolo11n.pt \
    --yolo-ft data/finetune_runs/yolo11n_mix/weights/best.pt --out data/finetune_runs/eval_yolo.json
# distance / dwell validation
.venv/bin/python -m analytics.validate_synthetic
```

Using the model: `Detector(model_path="models/showroom_detector_ft.onnx")` in `models/detect.py`
(the person column is chosen from the head width), or `PULSE_RFDETR_FT=models/showroom_detector_ft.onnx`
in `showroom-pulse`.

## 7. Files

| Path | What |
|---|---|
| `models/showroom_detector_ft.onnx` | Fine-tuned detector |
| `models/label_real.py` | Real-footage pre-label + review export |
| `models/finetune/` | `build_mix.py`, `train_nano.py`, `export_ft.py`, `evaluate.py`, `to_yolo.py`, `train_yolo.py` |
| `analytics/` | Floor projection, car distance, dwell; validation and overlay videos |
| `data/real_labels/` | COCO labels, `review.json`, labelled images (`labeled/`), review videos |
| `data/finetune_runs/` | Checkpoints, logs, `eval.json`, `eval_yolo.json` |
