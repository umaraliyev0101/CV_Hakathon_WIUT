# Judge Q&A prep: likely technical questions and answers

Every number below comes from a file in this repo (source in brackets). Do not quote numbers that are not here.

---

## A. Product and architecture

**1. What does the system do, in one sentence?**
One showroom camera (uploaded video or live webcam) in; per-salesperson coaching (greet time, waits, walk-outs,
engagement) from **Coach**, and anonymous customer analytics (footfall, dwell per car, unserved interest, journeys,
recommendations) from **Pulse**, merged in one site (**Studio**).

**2. Walk me through the pipeline.**
Sample video at 5 fps, ≤ 960 px wide → person detector → ByteTrack (supervision) → RTMPose-s keypoints → DINOv2-small +
HSV appearance per track → identity merge (track fragments → persons) → geometry (floor metres, facing) → roles
(staff vs customer) → event state machine / zone analytics → metrics, alerts, face-blurred annotated video, CSV/JSON.
[showroom-coach/README.md, showroom-pulse/README.md]

**3. Why cache perception?**
Models run once per video; the cache is keyed by file hash + every content-changing parameter (including the detector
weights' sha256). Flipping a role or editing a zone re-scores in a fraction of a second, no re-detection.

**4. Why two apps instead of one?**
Two lenses on the same floor: Coach = staff performance (sales manager), Pulse = customer interest per car (GM /
marketing). Built in parallel, deployed as one Docker Compose stack behind one nginx origin, one HTTPS tunnel.
Limitation: each app tracks and numbers people on its own, so staff are confirmed separately in each.

**5. How is it deployed?**
Docker Compose: nginx (`web`) + `coach` + `pulse` (+ optional ngrok). ONNX Runtime, no PyTorch at inference. Only
127.0.0.1:8090 published; optional basic auth. Coach also has a one-click `run.bat` for a Windows laptop.

**6. Does it need a GPU?**
No. Designed for CPU (ONNX Runtime). Coach was measured on an i5-10210U laptop (4 cores, no GPU). GPU makes Pulse much
faster (see Q22).

---

## B. Detection model and fine-tuning

**7. Which detector do you use and why?**
RF-DETR Nano (Roboflow, Apache-2.0), a real-time DETR-style transformer, 384×384 input, no NMS. Fine-tuned by us to a
person-only head → `models/showroom_detector_ft.onnx`. `deploy/up.sh` uses it in both apps when the file exists
(Pulse keeps YOLOX-s for cars, since our head only knows "person"). App defaults without it: Coach YOLOX-nano on
2 tiles, Pulse YOLOX-s.

**8. Why not YOLO?**
We benchmarked YOLO11n: on real footage it found 68.5% of people vs 82.7% for stock RF-DETR Nano, and 28% of people
behind cars vs 59%. Our quick fine-tune of YOLO made it worse (65.1%). Also Ultralytics is AGPL-3.0; RF-DETR is
Apache-2.0. Caveat: one untuned YOLO run, not YOLO's best. [reports/finetune_results.md]

**9. What did the fine-tune achieve?** (test data never trained on, conf 0.5)

| | Stock COCO | Fine-tuned |
|---|---|---|
| Real recall | 82.7% | **88.6%** (missed 67 → 44 of 387, −34%) |
| Real precision | 95.8% | 96.3% |
| Real F1 | 0.888 | **0.923** |
| False alarms / frame (real) | 0.13 | 0.12 |
| People < 60% visible (behind cars), recall | 59% | **72%** (n = 29, indicative) |
| Synthetic mAP50–95 | 0.823 | 0.856 |
| Floor position error, median / p90 | 19.1 / 31.5 cm | 17.5 / 29.2 cm |

[reports/finetune_for_pitch.md]

**10. What did NOT improve?**
Real-footage box tightness: AP50–95 0.779 → 0.762, AP50 0.943 → 0.939, within noise on 106 frames. The real labels
started from RF-DETR Large (COCO) boxes, which slightly favours the COCO model's box style. Say this before they find it.

**11. Why does recall matter more than mAP for you?**
A missed customer = a missed greeting alert or an uncounted walk-out. mAP mostly rewards box tightness; our product
acts at one threshold (0.5), so recall/precision at that threshold is the honest metric. At conf 0.3 real recall is
92.5% with 0.33 false alarms/frame ("never miss a customer" mode).

**12. How was training done?**
RF-DETR Nano from COCO weights, `class_names=["person"]`, 384×384, lr 1e-4 (encoder 1.5e-4), batch 16, EMA. ~25 s/epoch
on one RTX PRO 6000. Val mAP50–95 peaked at epoch 4 (0.892), stopped at epoch 9; epoch-4 EMA exported to ONNX
opset 17, simplified, ~108 MB.

**13. Training data?**
2,448 train images: 873 real (291 frames repeated 3× so they are not swamped) + 1,575 synthetic (every 4th frame).
Validation 185 (32 real + 153 synthetic). Real test split (106 frames, 387 people) never trained on.

**14. Isn't 5 epochs / 5 minutes too little?**
It is a fine-tune from strong COCO weights on a small domain set; validation peaked at epoch 4 and then fell
(0.892 → 0.870 by epoch 8), so longer training overfit. Early stopping on held-out validation picked the checkpoint.

**15. How did you avoid data leakage?**
Split **by clip, not by frame** (neighbouring frames are near-duplicates). Train 25 clips, valid 3, test 9. Demo clips
are in the test split, so the model was never trained on what we demo. Synthetic COCO export is split by scene.

**16. How were the real labels made?**
37 Pexels dealership clips at 1 fps → 429 frames. RF-DETR Large pre-labels (score ≥ 0.3, 1,184 boxes), then every
frame reviewed visually: 70 boxes removed (car headrests seen through windscreens, reflections, duplicates), 27 added
(blurred background people), 8 tightened → 1,141 boxes. A score cut-off alone could not do this: low-score boxes held
both real background people and the headrest false positives.

**17. Who reviewed the labels? Is the test set human ground truth?** *(expect this one)*
Honest answer: review was done by an AI vision reviewer (4 parallel reviewers + a lead spot-check of every added box),
not by a human annotator. The test set is model-reviewed; a human should verify it (~1 min review video exists) before
calling it human ground truth. [models/README.md, finetune_results.md §5]

**18. Will it generalise to Toyota/BYD showrooms?**
Not proven. All real clips come from **one Pexels shoot** (same showroom, same four actors), so test scores overstate
generalisation. That is why we added synthetic data with ceiling cameras, three lightings and occlusion by cars, and
why our ask is a 2-week pilot on real partner CCTV.

**19. Pexels licence for training?** *(risk)*
Pexels licence does not explicitly cover ML training (research_notes/open_footage.md). Non-commercial hackathon demo;
for a product we would train on partner footage under a data agreement. Note: DATA_SOURCES.md §3 still says
"never trained on" — that line is out of date.

**20. How fast is it?**
- RF-DETR Nano ONNX, 720p end-to-end: 28.5 FPS CPU (8 threads, shared loaded machine), 18.0–18.5 FPS (4 threads),
  265 FPS GPU. Fine-tuned model: same size and speed. [models/README.md, finetune_results.md]
- Coach file mode on the laptop: 0.87× real time on the 60 s, 12-person benchmark clip (target 1× missed); live
  3.6–4.5 fps with ~4 people. Pulse: 1.52× real time on a quiet 60 s excerpt, 0.88–1.86× on 5-min clips; live engine
  8.0 fps (1–4 people).

**21. Why are pose and DINOv2 needed if you have boxes?**
Pose: facing direction (shoulder line), torso-length fallback for depth when legs are hidden or the person sits.
DINOv2 + HSV: re-join track fragments of one person after occlusion (within one video only, never across visits) and
the "same outfit as others" staff cue. Pose and DINOv2 are most of the CPU time.

**22. Pulse on CPU vs GPU?**
On CPU, Pulse needed 260–350 s to perceive a 45–55 s clip in the deployed stack (~6× slower than real time); on GPU
3–17 s. One job at a time (single worker), so uploads queue. [reports/demo_clips.md]

---

## C. Synthetic data

**23. Why synthetic data?**
Stock footage is eye-level; real showroom CCTV looks down from the ceiling and people are half-hidden behind cars.
Rendering gives that viewpoint at zero labelling cost with pixel-perfect labels, plus exact visible fraction, floor
position and scripted events for validating analytics.

**24. How was it generated?**
Blender 4.2 LTS, EEVEE RGB + Workbench instance masks. 12 seeded scenes, 4–6 real car models, 12 Microsoft Rocketbox
avatars (MIT) driven by mocap (stride-matched walks, turn clips, talk/listen), 3 lightings (day, dim evening, glare),
2 cameras each at 2.8–3.2 m, pitch −25°, HFOV 70–90°. CCTV look + H.264. Scenario engine scripts greet, walk-out,
party, handoff, crowded, 4-minute dealership. 65 clips, 41,544 frames, 127,270 person boxes. [reports/synth_status.md]

**25. How do you know the synthetic data is any good?**
Automated gates: `python sim/verify.py --all` → PASS 32/32 on the share pack (render, events, video, scene geometry
— feet within ±5 cm of the floor — detector recall per lighting 0.985–0.997), plus a 12-item visual review per clip.
And the outcome: adding it to fine-tuning raised recall on **real** footage.

**26. Sim-to-real gap?**
Exists. We mitigate by mixing real frames (3× weighted) and evaluating only on real test data for the headline
number. Synthetic dominates validation (153 of 185 frames), so model selection is synthetic-biased — a known caveat.

**27. Labels from masks — how?**
Per-person instance masks from Workbench render → visible-extent box, visible fraction, track id, role. Floor position
and yaw come from the scene graph, events from the script and from rule re-detection (`sim/gt_events.py`).

---

## D. Geometry, distance, dwell

**28. How do you get metres from one uncalibrated camera?**
Coach: calibration-free pinhole model — assume a 1.70 m person and 70° HFOV, depth Z = f·H/h_px using full body height,
else shoulder-to-hip torso, else box height (flagged "approximate"). Optional 4-point floor homography replaces it.
Analytics module: known camera K, R, t → feet (box bottom-centre) projected to floor; if the box implies a person
< 1.4 m (legs hidden by a car), project the head onto a plane at 1.7 m instead.

**29. How accurate are positions and dwell?**
On simulator boxes over 61 clips (100k person-frames): median floor error 0.18 m; p90 0.28 m (≥ 90% visible), 0.88 m
(60–90%), 0.52 m (< 60%; was 4.18 m before the head fallback). Dwell near cars, 198 pairs: 6,242 s vs 6,333 s ground
truth, per-pair median error 0.2 s. With real detector boxes: median 0.175 m, p90 0.29 m. Distance accuracy is set by
geometry, not the detector (all models 0.17–0.19 m).

**30. How do you define "near a car" / dwell?**
Within 1.5 m of the car body footprint; gaps ≤ 1 s bridged, visits < 2 s (walking past) dropped. Car footprints are
drawn once per camera (cars don't move), optionally pre-filled by the COCO `car` class. Pulse auto-suggests car zones
from a keyframe; user can edit.

---

## E. Staff vs customer, events

**31. How do you tell staff from customers?**
Transparent logistic score from four hand-set features: `outfit_peers` (others in similar clothes), `contacts`
(distinct people within 2 m for ≥ 5 s), `presence` (share of clip on screen; ignored on clips < 60 s), `initiator`
(who walks toward whom). > 0.6 staff, < 0.4 customer, else "unsure". Each suggestion shows its top reasons; one click
overrides. The market leader (SKAIVISION) doesn't disclose its method.

**32. Why not a trained uniform classifier?**
No labelled staff data from partners, and uniforms differ per brand. Hand-set, explainable features + human
confirmation work on day one; a trained classifier is a pilot-phase upgrade once partner footage exists.

**33. What is the staff-detection accuracy?**
Not measured on showroom data — do not claim a number. Say: the human confirms each role, so errors are corrected
before scoring.

**34. How are events defined?**
Pure state machine per customer: entered → waiting → greeted → engaged/browsing → left. Greet = staff within 1.5 m
(and facing, when known) for `greet_time`; engagement = within 2.0 m for `engage_time`. Dealership preset copies
SKAIVISION's rule (2 m / 2 min); Short-clip preset (2 s greet / 8 s engagement / 10 s wait alert) for demo clips.
Also party (arrive together; greet one = greet all), handoff, already-present (excluded from time-to-greet).

**35. How well do events work on your demo clips?**
On synthetic demo clips (with staff taken from ground truth): greet at 11.6 s vs GT 11.9 s; walk-out at 49.4 s vs GT
49.9 s. Known failures: a greet missed in dim light when staff are far from the camera; a phantom customer from a box at
the frame edge. Pulse: visitor double-counted when a track splits; dwell 0 when the auto zone is in front of the car
but people stand behind it. [reports/demo_clips.md]

**36. Any evaluation on real public data?**
Yes, MEVA (Kitware/IARPA, CC BY 4.0) indoor clips — not showrooms. Results are weak and we report them: Coach talk
recall far below the 70% target; tracking IDF1 ~20% (Pulse pooled test). Reasons: MEVA annotates only people in
activities and splits GT tracks per activity, crowded school/gym scenes fragment tracks. An oracle ablation (GT boxes)
lifts talk recall to 44%, so detection/tracking is the bottleneck — the reason we fine-tuned the detector.
[showroom-coach/docs/EVAL.md, showroom-pulse/README.md]

**37. How do you handle occlusion and ID switches?**
ByteTrack keeps low-score boxes to bridge short occlusions; offline constrained clustering merges fragments by
appearance (two tracks overlapping in time can never be one person); live mode re-attaches a recently lost person.
The fine-tune adds recall on people behind cars (59% → 72%). Still the weakest part (see Q36).

---

## F. Privacy, ethics, licences

**38. Privacy?**
No face recognition, no age/gender/emotion/ethnicity models (none in the repo). Faces blurred in every output and on
the live canvas. No re-identification across visits; appearance used only inside one video/session. On-premise, no
cloud calls; live mode stores no frames; jobs deleted after 24 h. Decision support only; staff informed; signage is
the operator's legal duty (Uzbekistan AI law ZRU-1115 — counsel should review before a pilot). Don't say "legally
compliant".

**39. What if the detector misses someone — is their face shown?**
Yes, that is the honest limit: blur covers every *detected* person (down to score 0.2, plus neighbouring frames). A
missed person is not blurred, so outputs are for internal review.

**40. Licences?**
RF-DETR Apache-2.0; no Ultralytics/AGPL, no re-ID weights from Market-1501/Duke/MSMT17. Rocketbox MIT, Poly Haven CC0.
Car models: CC BY, one CC BY-NC, one unknown — demo only, replace for commercial use. Coach image bundles GPL FFmpeg;
not pushed to a public registry.

**41. Could this be used to surveil or punish staff?**
Designed as coaching/decision support: scorecards for a manager's review, roles confirmed by a human, no identity.
We recommend staff are informed and that scores are not used as automatic judgments.

---

## G. Scale, cost, business

**42. Multi-camera?**
Not yet: one camera per job, no cross-camera tracking. Roadmap item. Per-camera floor calibration already supported
(homography / known K,R,t).

**43. What hardware per showroom?**
Coach targets a CPU laptop; Pulse wants a GPU for real-time multi-person perception. Commercial reference: SKAIVISION
runs 15 SD cameras on an 8 GB RTX 3050/4060 edge box. Fine-tuned RF-DETR Nano: 265 FPS on GPU → a single small GPU
covers many cameras for detection.

**44. ROI?**
Formula, not a forecast: extra cars/month = walk-ins × walk-out rate × recovered share × (greeted − ungreeted close
rate); break-even = monthly cost / gross margin per car. With hypothetical inputs it pays for itself with about one
extra car every four months. All inputs are assumptions until partners supply real ones. [showroom-coach/docs/PITCH_NOTES.md §4]

**45. Competitors / what's new?**
SKAIVISION (US, 150+ dealerships) uses the same 2 m / 2 min logic on GPU edge boxes and is silent on staff detection.
Ours: open, explainable staff suggestion with one-click override, runs on CPU, privacy by design, plus our own
fine-tuned detector and a synthetic CCTV pipeline. Don't say "better than SKAIVISION".

**46. Join with sales data?**
Optional sales CSV (`salesperson,timestamp,sold`) → conversion per salesperson (Coach) and attention-to-sales per car
(Pulse).

**47. What would you do with 2 more weeks?**
Pilot on partner CCTV: human-labelled test set from their cameras, fine-tune on it, measure staff-role accuracy and
greet-time error, multi-camera, Uzbek/Russian UI, calibration in the UI.

---

## H. Tricky / gotcha questions

**48. "Your real test set is only 106 frames from one shoot — is +5.9 pts significant?"**
Recall moved from 320 → 343 people found of 387; consistent gains on synthetic (recall, mAP, hidden people) and no
rise in false alarms. Small and single-shoot, so we call it indicative, not a guarantee.

**49. "Did you just train on your demo clips?"**
No. The demo clips are in the test split; split is by clip.

**50. "Why is your MEVA score so low?"**
MEVA is not a showroom and annotates only activity participants with short per-activity tracks; we report it rather
than hide it. Oracle ablation shows tracking/detection is the bottleneck.

**51. "What's 'AI-reviewed labelling' — isn't that circular?"**
Pre-labels came from a different, bigger model (RF-DETR Large); the review removed 70 and added 27 boxes that a score
threshold could not separate. It is still not human ground truth (Q17).

**52. "How do you handle a mirror/reflection or a person inside a car?"**
Labelling policy: reflections, posters, headrests get no box; people through glass do. Of 3 headrest/reflection
regions in the real test, stock and fine-tuned RF-DETR each detect 1 (tiny sample).

---

## Inconsistencies to fix or be ready for

- `DATA_SOURCES.md` §3 says Pexels clips are "never trained on"; they were used for fine-tuning (train split).
- `models/README.md` top says "No fine-tuning" (stale; describes the baseline file).
- `reports/finetune_for_pitch.md` §4.1 says fine-tuned weights are "not exported to ONNX yet" (they are now).
- CPU FPS differs by source: 28.5 FPS (8 threads) vs 18 FPS (4 threads) — always say the thread count.
- `reports/demo_clips.md` results were measured with default detectors, not the fine-tuned one.
