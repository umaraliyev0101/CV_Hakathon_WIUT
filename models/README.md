# models/ — P0 baseline person detector

COCO-pretrained **RF-DETR Nano** (`rfdetr.RFDETRNano`, rfdetr 1.11.2), exported to ONNX and run with
ONNX Runtime. Only the COCO `person` class is kept. No fine-tuning: these are the stock COCO weights.

```python
from models.detect import detect, Det, Detector   # from repo root
dets = detect(frame_bgr, conf=0.5)                # list[Det(bbox=(x1,y1,x2,y2), score, cls="person")]
det = Detector(device="cpu", threads=8)           # or device="cuda" (needs CUDA_VISIBLE_DEVICES set)
```

CLI (prints `person score=0.93 bbox=[x1,y1,x2,y2]`, one line per detection):

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/detect.py data/real_open/pexels_7154214_1920_1080_25fps.mp4 \
    --frame 100 --out reports/real_open_detect_1.jpg --conf 0.5 [--device cpu|cuda|auto] [--threads 8]
```

GPU policy: CUDA is used only when `CUDA_VISIBLE_DEVICES` is set explicitly (this machine: always `=1`;
GPU 0 belongs to others). If it is unset, `device="auto"` and `device="cuda"` fall back to CPU.

## File

| | |
|---|---|
| Path | `models/showroom_detector.onnx` (gitignored, 107.8 MB, fp32) |
| sha256 | `c429c147f355cd13b9650e87b334fd74abea2b6778ee0064082b988bc9f9fbfb` |
| Opset / IR | ONNX opset 17, IR 8, static batch 1, simplified with onnxsim 0.8.1 |
| Source weights | `rf-detr-nano.pth` from `https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth` (rfdetr cache: `~/.roboflow/models/`, sha256 `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`) |
| Licence | RF-DETR code and Nano COCO weights: Apache-2.0 (Roboflow). Trained on COCO (annotations CC BY 4.0) |

## Input

| name | shape | dtype |
|---|---|---|
| `input` | `[1, 3, 384, 384]` (NCHW) | float32 |

Pre-processing, matching `RFDETR.predict()`: bilinear resize of the full frame to 384x384 (no letterbox;
aspect ratio is not kept, `cv2.INTER_LINEAR`, same as torch `F.resize(antialias=False)`), BGR to RGB,
divide by 255, then ImageNet normalisation `(x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`.

## Outputs

| name | shape | dtype | meaning |
|---|---|---|---|
| `dets` | `[1, 300, 4]` | float32 | box per query, `(cx, cy, w, h)` normalised to 0..1 of the model input (and so of the original frame) |
| `labels` | `[1, 300, 91]` | float32 | class **logits** per query (not probabilities) |

Decoding (`Detector.postprocess`, the same as rfdetr `PostProcess`): `sigmoid(labels)`, flatten to
300*91 scores, take the top 300; `class = idx % 91`, `query = idx // 91`; keep `class == 1` (person) and
`score > conf`; box = `dets[query]` changed to xyxy, times `(W, H, W, H)` of the original frame, clipped.
No NMS (DETR-style, one box per query).

Class index: the 91 logit slots are **raw COCO category ids** (1..90, sparse; slot 0 unused), so person = 1
(not 0). This was checked against `RFDETRNano.predict()` output on real frames: class_id 1 is `person`,
3 is `car`, 31 `handbag`, 32 `tie`, 62 `chair`.

## Validation (ONNX vs PyTorch)

`python models/export_onnx.py --skip-export` runs rfdetr's `predict()` (PyTorch) and `Detector` (ONNX, CPU)
on 4 real frames at conf 0.5 and matches person boxes by IoU. Result (2026-10-11):

| frame | torch persons | onnx persons | max box diff | max score diff |
|---|---|---|---|---|
| pexels_7154214 @100 (1920x1080) | 4 | 4 | 0.18 px | 0.0053 |
| pexels_7154217 @150 (1920x1080) | 3 | 3 | 0.36 px | 0.0005 |
| pexels_7154241 @200 (1066x1920) | 3 | 3 | 0.18 px | 0.0003 |
| pexels_36512656 @240 (2160x3840) | 1 | 1 | 0.48 px | 0.0018 |

## Measured speed

1280x720 BGR frame (frame 100 of pexels_7154214 downscaled), end to end = preprocess + ORT run + postprocess,
mean over 100 runs (CPU) or 300 runs (GPU) after 10 warm-up runs. onnxruntime-gpu 1.31.0.
CPU: AMD Ryzen Threadripper PRO 9985WX (64 cores / 128 threads). **The machine was shared and under load
(load average about 50) during the CPU runs, so CPU numbers are noisy and probably pessimistic.**

| provider | intra-op threads | end-to-end FPS | end-to-end ms | model-only ms |
|---|---|---|---|---|
| CPU | 4 | 22.1 | 45.3 | 39.0 |
| CPU | 8 | 28.5 | 35.1 | 32.2 |
| CPU | 16 | 28.9 | 34.6 | 31.5 |
| CPU | ORT default (64, one per physical core) | 22.8 | 43.8 | 41.7 |
| CPU | 128 (all logical cores) | 14.4 | 69.7 | 58.8 |
| CUDA, GPU 1 (`CUDA_VISIBLE_DEVICES=1`) | n/a | 265.4 | 3.8 | 2.0 |

Use 8–16 threads on CPU. Using every core is slower.

## Regenerate

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/export_onnx.py          # export + simplify + validate
CUDA_VISIBLE_DEVICES=1 .venv/bin/python models/export_onnx.py --skip-export   # validate only
```

Needs `pip install rfdetr onnx onnxsim onnxruntime-gpu` (rfdetr pulls in torch). The exporter calls
`RFDETRNano().export(opset_version=17, batch_size=1)`, runs onnxsim, and writes `models/showroom_detector.onnx`.
The sha256 may change if the rfdetr, torch or onnxsim version changes.

## Real fine-tuning set (`data/real_labels/`, gitignored)

Built by `models/label_real.py` from 37 Pexels dealership clips in `data/real_open/` (36 clips of the
Antoni Shkraba shoot, IDs 7154207–7154242, plus 36512656), sampled at 1 fps.

- **Labels:** COCO RF-DETR Large pre-labels (person, score ≥ 0.3), then a visual review of every
  frame by Claude (`review.json`: 70 boxes deleted (headrests seen through windscreens, reflections,
  duplicates), 27 added (missed blurred background people), 8 fixed). Each COCO annotation records
  `attributes.source` = `model` or `review`.
- **Split by clip:** train 25 clips / 291 images / 657 boxes; valid 3 / 32 / 97; test 9 / 106 / 387.
  The test clips include the four demo clips (7154214, 7154217, 7154241, 36512656).
- **Caveats:** the test split is model-reviewed, not human-labelled; a person must verify it before
  it is reported as human ground truth. All clips share one shoot (same set, same four actors), so
  test scores overstate how well the model generalises to other showrooms. Using Pexels clips to train ML
  models is not explicitly covered by the Pexels licence (see `research_notes/open_footage.md`).

## Fine-tuned person detector (`models/showroom_detector_ft.onnx`)

RF-DETR Nano fine-tuned from the COCO weights, person-only head (logit column 0; `detect.py` picks
the column from the head width, so both ONNX files work with `Detector(model_path=...)`).

- **Data** (`data/finetune/`, symlinks): train = reviewed real Pexels frames ×3 (873) + every 4th
  synthetic CCTV frame (1,575); valid = 32 real + 153 synthetic. Script: `models/finetune/train_nano.py`.
- **Run:** batch 16, lr 1e-4, 384 px, GPU 1; ~25 s/epoch. Validation mAP50–95 peaked at epoch 4
  (0.892) and then declined, so the run was stopped at epoch 9 and the epoch-4 EMA checkpoint
  (`data/finetune_runs/nano_mix/checkpoint_best_ema.pth`) was exported (`models/finetune/export_ft.py`).
- **Speed:** unchanged, 18 FPS at 720p on 4 CPU threads (same architecture).

`models/finetune/evaluate.py` (conf 0.5, IoU 0.5; results in `data/finetune_runs/eval.json`):

| Metric | COCO Nano | Fine-tuned |
|---|---|---|
| Real test AP50 / AP50–95 (106 frames, 387 boxes) | 0.943 / 0.779 | 0.939 / 0.762 |
| Real test recall @0.5 | 0.827 | **0.886** |
| Real test false positives / frame @0.5 | 0.13 | 0.12 |
| Synthetic test AP50 / AP50–95 (650 frames) | 0.990 / 0.823 | 0.990 / **0.856** |
| Synthetic recall, people < 60 % visible (n = 29) | 0.59 | **0.72** |
| Floor-position error from detector boxes, median / p90 | 0.19 / 0.32 m | 0.18 / 0.29 m |

Reading it: the fine-tune mostly raises recall at the deployment threshold (real +6 pts) and on
people partly hidden behind cars (+14 pts, small sample), without more false positives. Real AP is
flat within noise; note the real labels started as RF-DETR Large (COCO) boxes, which favours the
COCO model's box style. The hard-negative check (headrests/reflections) had only 3 regions in the
test split, too few to conclude. The real test split is model-reviewed, not yet human-verified, and
comes from the same shoot as the training clips.

### Benchmark: YOLO11n (Ultralytics, AGPL-3.0, comparison only, not for the product)

Same data (`models/finetune/to_yolo.py`, `train_yolo.py`: 15 epochs, 640 px, ~1.5 min) and the same
`evaluate.py` checks at conf 0.5 (`data/finetune_runs/eval_yolo.json`):

| Model | Real AP50 / AP50–95 | Real recall | Real FP/frame | Occluded recall (syn) | Syn FP/frame |
|---|---|---|---|---|---|
| RF-DETR Nano COCO | 0.943 / 0.779 | 0.827 | 0.13 | 0.59 | 0.02 |
| **RF-DETR Nano fine-tuned** | 0.939 / 0.762 | **0.886** | 0.12 | **0.72** | 0.01 |
| YOLO11n COCO | 0.875 / 0.704 | 0.685 | 0.00 | 0.28 | 0.02 |
| YOLO11n fine-tuned | 0.731 / 0.439 | 0.651 | 0.16 | 0.24 | 0.22 |

The stock YOLO11n already trails stock RF-DETR Nano on real footage and on people behind cars. The
quick YOLO fine-tune got worse on real footage (default hyper-parameters, one untuned run; likely
forgetting real appearance on a synthetic-heavy mix), so it is reported as a single run, not
YOLO's best. Decision: keep RF-DETR (better here, and Apache-2.0).
