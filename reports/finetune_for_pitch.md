# Person detector fine-tuning: how it works and what it scored

A brief for the pitch deck. Every number below was **measured** on 2026-10-11 between 11:39 and 11:55 (+05), unless it is marked *estimate*.
Raw results: `data/finetune_runs/eval_coco_nano.json` (before) and `data/finetune_runs/eval_finetuned_nano.json` (after).

---

## 1. One-paragraph version (for the slide notes)

Everything in our product (greeting time, engagement, walk-outs, dwell near a car) starts with one question:
*where are the people in the frame?* We began with a strong off-the-shelf detector (RF-DETR Nano, trained on COCO).
Then we fine-tuned it on two kinds of showroom data: **real dealership footage** that we labelled ourselves, and
**synthetic CCTV footage** we rendered in Blender. Here, labels come free and are pixel-perfect. One fine-tune of about 5 minutes on one GPU had these results:

- the detector now misses **34% fewer real people** (recall 82.7% → **88.6%**),
- it finds **72% of people hidden behind cars**, up from 59%,
- it does not add false alarms (0.13 → 0.12 per image),
- it keeps the same model size and speed.

---

## 2. Headline numbers (safe to put on a slide)

Test data the model **never saw during training**. Confidence threshold 0.5 (the product default).

| What we measured | Before (stock COCO model) | After (our fine-tune) | Change |
|---|---|---|---|
| **Real showroom footage: people found (recall)** | 82.7% | **88.6%** | **+5.9 pts** |
| Real footage: missed people (out of 387) | 67 | **44** | **−34%** |
| Real footage: precision (detections that are real people) | 95.8% | **96.3%** | +0.5 pts |
| Real footage: F1 (balance of the two) | 0.888 | **0.923** | +0.035 |
| Real footage: false alarms per image | 0.13 | **0.12** | slightly better |
| **Synthetic CCTV: people behind cars (<60% visible)** | 58.6% | **72.4%** | **+13.8 pts** |
| Synthetic CCTV: recall, all people | 98.0% | **98.8%** | +0.8 pts |
| Synthetic CCTV: mAP50:95 (box tightness) | 82.3% | **85.6%** | +3.4 pts |
| Synthetic CCTV: false alarms per image | 0.018 | **0.011** | −40% |
| **Floor position error (where a person stands)**, median | 19.1 cm | **17.5 cm** | −1.6 cm |
| Floor position error, 90th percentile | 31.5 cm | **29.2 cm** | −2.3 cm |

**Extra operating point.** If the detector runs at confidence 0.3 instead of 0.5, real-footage recall goes up to **92.5%**,
and false alarms go up to 0.33 per image. This setting suits the "never miss a customer" alert mode.

### Suggested slide wording

> "Fine-tuned on real + synthetic showroom footage: **88.6% of people found on unseen real dealership video
> (+5.9 pts), 34% fewer missed customers, 72% of customers found even when hidden behind a car
> (+14 pts), with no extra false alarms. Positions on the floor are accurate to about 17 cm."**

Short version: **"34% fewer missed customers after a 5-minute fine-tune."**

---

## 3. A note on the word "accuracy"

Object detectors do not have one "accuracy" number. Use these terms:

- **Recall**: of all real people in the frame, the share we found. *This one matters most for us*, because a missed customer means a missed greeting alert.
- **Precision**: of everything we flagged as a person, the share that really was a person. Low precision means false alarms.
- **F1**: a single number that balances recall and precision. If the slide must say "accuracy", **F1 = 92.3%** is the honest number to use.
- **mAP50 / mAP50:95**: the standard research benchmark score. It also measures how tightly the boxes fit. It is useful for credibility with technical judges.

---

## 4. How the fine-tuning was done

### 4.1 Starting model
- **RF-DETR Nano** (Roboflow, Apache-2.0). It is a modern real-time transformer detector, pre-trained on COCO (330k images, 80 classes).
- We keep only the `person` class.
- Speed (stock model, ONNX): **265 FPS on GPU**, **28.5 FPS on CPU** at 720p. The fine-tuned model has the same architecture and size, so its speed is the same. (*Estimate*: the fine-tuned weights are not exported to ONNX yet, but nothing in the export changes.)

### 4.2 Data source 1: real dealership footage (labelled by us)
- 37 free Pexels dealership clips, one frame per second → **429 frames, 1,141 person boxes**.
- Labelling pipeline (`models/label_real.py`):
  1. A **bigger model** (RF-DETR Large) drew draft boxes automatically.
  2. **Every frame was reviewed** by an AI vision reviewer: **70 wrong boxes removed** (car headrests seen through windscreens, reflections in car paint, duplicates), **27 missed people added** (blurred background staff) and 8 boxes fixed.
- The split is **by clip, not by frame**, so the test set comes from videos the model never saw: train 25 clips / 291 frames, validation 3 / 32, **test 9 / 106 frames / 387 people**.
- The test clips include our four demo clips. The model was **not** trained on the demo videos.

### 4.3 Data source 2: synthetic CCTV footage (rendered by us)
- Blender showroom scenes with **real car models** and **animated humans** (Microsoft Rocketbox, MIT licence).
- **3 lighting conditions** (day, dim evening, window glare), **2 ceiling-camera angles**, and scripted showroom scenarios (greeting, walk-out, family party, staff handoff, crowded floor).
- **68 clips, 44,280 frames, 127,641 auto-labelled person boxes**. Every box is pixel-perfect, and we know exactly how much of each person is hidden.
- Why synthetic data helps: real stock footage is shot at eye level. Real showroom CCTV looks **down from the ceiling**, and people are often **half-hidden behind cars**. We can render as much of that as we need, at no labelling cost.

### 4.4 Training mix and settings
| | |
|---|---|
| Training set | 2,448 images: 873 real (291 frames repeated 3× so real data has enough weight) + 1,575 synthetic (every 4th frame) |
| Validation set | 185 images (32 real + 153 synthetic), used for early stopping |
| Settings | 20 epochs max, batch 16, learning rate 1e-4, EMA weights, early stopping (patience 5) |
| What happened | Best result at epoch 4. Training stopped by itself at epoch 8 because results stopped improving |
| Wall time | **About 5 minutes** on one NVIDIA RTX PRO 6000 GPU |
| Model size | 30.1 M parameters (unchanged) |
| Script | `models/finetune/train_nano.py`; evaluation: `models/finetune/evaluate.py` |

**Pitch angle:** adapting to a new showroom is cheap. Record a few clips, auto-label them, review them, and fine-tune in about 5 minutes. A dealer group can do this for each site.

---

## 5. Honest caveats (keep these in mind for Q&A)

1. **Real test labels are AI-reviewed, not human-verified.** The labels are carefully checked, but a person has not signed them off yet. If asked, say: "labelled by a large model and checked frame by frame. Human verification is the next step."
2. **All real clips come from one photo shoot** (same showroom, same 4 actors). Real-world recall in a different showroom will probably be **somewhat lower** than 88.6% (*estimate*). That is why we use the synthetic data, which has varied lighting and cameras.
3. **The occluded-people number is based on 29 people.** The +14-point gain is large, but the sample is small. Present it as "strong signal", not as a precise figure.
4. **Box tightness on real footage dipped slightly** (mAP50:95 77.9% → 76.2%; mAP50 94.3% → 93.9%). Our own labelling style differs a little from COCO's. This does not affect counting or alerts, which depend on *finding* people (recall). Don't put real mAP on the slide. Use recall, F1 and missed people.
5. **The "hard negative" test is too small to report** (3 regions such as headrests and reflections; both models trip on 1 of them).
6. **Licence:** the Pexels licence does not explicitly cover ML training. That is fine for a hackathon demo. For a commercial product we would retrain on the partner's own CCTV.

---

## 6. What we would claim in the future (label as roadmap, not results)

*Estimates only. Nothing below has been measured.*
- With a few hundred labelled frames from each partner's **own CCTV**, we would expect real recall in the **low-to-mid 90s** at the same false-alarm rate, based on the jump we saw from ~300 real frames.
- A bigger backbone (RF-DETR Small/Medium) would add a few more points, but it costs speed. Nano already runs **many cameras on one GPU** (265 FPS ÷ 10 FPS per camera ≈ 25 cameras, *estimate*).

---

## 7. Glossary for the deck

| Term | Plain meaning |
|---|---|
| Fine-tuning | Taking an already-trained model and training it a bit more on our own data so it specialises |
| Synthetic data | Computer-generated (3D-rendered) video where the labels are known exactly |
| Recall | % of real people the model found |
| Precision | % of detections that were really people |
| F1 | Balance of recall and precision (closest thing to "accuracy") |
| mAP | Standard detection benchmark score that also rewards tight boxes |
| Occluded | Partly hidden, for example behind a car |
| Confidence threshold | How sure the model must be before it reports a person. Lower means fewer misses and more false alarms |
| EMA / early stopping | Training tricks that keep the most stable weights and stop before over-fitting |
