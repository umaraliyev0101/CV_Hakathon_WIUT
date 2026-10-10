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
