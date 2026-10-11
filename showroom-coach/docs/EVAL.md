# Evaluation: Showroom Coach on MEVA (Jira YAK-44)

Everything here was measured on one i5-10210U laptop (8 logical CPUs, 16 GB, no GPU) with the default model stack
(`yolox_nano@2` + RTMPose-s + DINOv2-small). A target that was missed is reported as missed, with the measured value.

**Bottom line.** The pipeline does not reach the engagement target on MEVA. On the four held-out test clips it finds
**1.1% of the ground-truth talks** (twin-aware; 0.5% strict) against a target of 70%, and it matches only 17.6% of the
annotated person boxes at all. Where an actor is followed, arrival and leaving are timed closely (median absolute error
0.8 s over the 7 matched events, window 5 s), but the recall is 20.6% (2.9% strict). Staff suggestion and greet time could
not be evaluated because there is no labelled showroom footage. With the ground-truth boxes in place of the detections
(the oracle ablation) the same engagement rule reaches 44.1% talk recall (twin-aware; 32.3% strict) on test, so detection and tracking account for
most of the gap, but the rule alone is also under the target at the settings tested on test. On a near-idle machine file
mode runs at 0.87x real time on the spec's 720p benchmark clip (12 people per frame), under the 1x target; the same
footage at 960 px gave 1.07x in another run. The two runs differ in command (bench vs analyze), source cut (720p vs 960 px) and people found (31 at 960 px); neither was repeated, so the cause of the gap is not isolated. A busier 60 s clip (about 14 people per
frame) gave 0.56x while another agent was running. Live mode runs at 3.6 to 4.5 fps at 3.8 people per frame (3.61 fps in
process, 4.45 fps end to end over the WebSocket), around the 4 fps target. A 60 s upload reaches the confirm step in 60.3 s
(122 s and 253 s in earlier runs under load).
The test clips were used more than once; see "Test-set exposure".

## Data

### MEVA clips

MEVA ("Multiview Extended Video with Activities") by Kitware Inc. and the Intelligence Advanced Research Projects
Activity (IARPA), licensed **CC BY 4.0** (<http://creativecommons.org/licenses/by/4.0/>). Attribution: *"Multiview
Extended Video with Activities" (MEVA) dataset by Kitware Inc. and IARPA, CC BY 4.0.* Citation: K. Corona, K. Osterdahl,
R. Collins, A. Hoogs, "MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection", IEEE/CVF Winter
Conference on Applications of Computer Vision (WACV), 2021. The videos and the KPF annotations (`geom`, `activities`,
`types`) stay in `data/meva/` and are never committed; sources and URLs are in `data/sources.csv` and
`data/meva/clips.csv`.

Seven indoor clips of 5 minutes from five cameras (school site: G299 twice, G330 twice, G421, G423; bus site: G331). Every number below uses only the
**first 150 s of each clip** (750 sampled frames at 5 fps), with the ground truth cut to the same window.

| clip | camera | site | split | clip size | annotated talks / enters / exits (whole 5 min) | annotated actor ids in the first 150 s | person boxes scored (sampled) |
|---|---|---|---|---|---|---|---|
| `2018-03-09.10-15-00.10-20-00.school.G299` | G299 | school | dev | 207 MB | 54 / 19 / 27 | 437 | 14278 |
| `2018-03-11.16-35-00.16-40-00.school.G330` | G330 | school | dev | 193 MB | 68 / 0 / 0 | 410 | 30853 |
| `2018-03-07.11-05-00.11-10-00.bus.G331` | G331 | bus | dev | 148 MB | 26 / 3 / 4 | 109 | 6837 |
| `2018-03-11.17-15-00.17-20-00.school.G299` | G299 | school | test | 158 MB | 51 / 6 / 8 | 261 | 23347 |
| `2018-03-12.10-50-00.10-55-00.school.G330` | G330 | school | test | 205 MB | 50 / 0 / 0 | 153 | 23333 |
| `2018-03-12.10-10-01.10-15-01.school.G423` | G423 | school | test | 74 MB | 13 / 16 / 16 | 96 | 3175 |
| `2018-03-13.15-50-00.15-55-00.school.G421` | G421 | school | test | 156 MB | 35 / 12 / 13 | 128 | 4307 |

"Annotated actor ids" counts ids, not people: MEVA gives a new actor id per annotated activity, so one person often
carries several ids (see Limitations). The perception run of the seven clips took 38 minutes of wall time in total, on a
machine that was shared with other work:

| clip | split | processed length | perception wall time | real-time factor |
|---|---|---|---|---|
| 2018-03-09 10-15 G299 | dev | 150 s | 225 s | 0.67x |
| 2018-03-11 16-35 G330 | dev | 150 s | 318 s | 0.47x |
| 2018-03-07 11-05 G331 | dev | 150 s | 368 s | 0.41x |
| 2018-03-11 17-15 G299 | test | 150 s | 284 s | 0.53x |
| 2018-03-12 10-50 G330 | test | 150 s | 599 s | 0.25x |
| 2018-03-12 10-10 G423 | test | 150 s | 179 s | 0.84x |
| 2018-03-13 15-50 G421 | test | 150 s | 331 s | 0.45x |

### Split

The split was written to `data/meva/split.json` on 2026-10-11 **before any tuning** and has not changed. It is roughly
40% dev (3 clips) and 60% test (4 clips), and the test set contains two cameras that never appear in dev (G423 and
G421), so the test result is not only "the same camera on another day". Dev clips are used for tuning only. The test clips
were scored with fixed settings (five runs in all, and some earlier use); "Test-set exposure" lists everything.

```json
{
  "note": "Written 2026-10-11 before any tuning. Dev = 3 clips (tuning only), test = 4 clips (measured once). Test includes cameras G423 and G421 that are not in dev.",
  "limit_seconds": 150,
  "dev": [
    "2018-03-09.10-15-00.10-20-00.school.G299",
    "2018-03-11.16-35-00.16-40-00.school.G330",
    "2018-03-07.11-05-00.11-10-00.bus.G331"
  ],
  "test": [
    "2018-03-11.17-15-00.17-20-00.school.G299",
    "2018-03-12.10-50-00.10-55-00.school.G330",
    "2018-03-12.10-10-01.10-15-01.school.G423",
    "2018-03-13.15-50-00.15-55-00.school.G421"
  ]
}
```

The `note` field is the file's original text from before tuning and is left as written; its "measured once" understates how
the test clips were used, which "Test-set exposure" lists.

### No showroom footage

`data/stock/` is empty: the owner has not added Pexels showroom clips, and there is no `labels/` directory. There is
therefore **no labelled staff/customer data at all**. Staff suggestion accuracy, clicks needed per clip and greet-time
error are **not measured**, and the tool that would score them (`eval-labels`) has not been built because there is
nothing to run it on. They stay unevaluated until the team films staged scenes (or labelled stock clips are added; the
label format is in the design spec, section 6).

## Method

**Pipeline under test.** The offline path of `showroom-coach analyze`: frames sampled at 5 fps and resized to at most
960 px wide, person detection with `yolox_nano` on 2 overlapping tiles, ByteTrack (supervision), RTMPose-s every second
sampled frame per track, DINOv2-small and HSV appearance descriptors for tracks of at least 3 observations spanning 1 s,
then `analysis.analyze` with the `short` preset: flicker tracks dropped, track fragments merged by appearance into
*persons* (`merge_thr`, default 0.30), floor positions in metres from an uncalibrated pinhole model (70 degree
horizontal field of view, 1.70 m people), role features and events. Perception is cached per video and parameter set
(`runs/cache/`), so tuning and evaluation re-score the same detections; nothing in this document re-ran a detector.

**Ground truth.** MEVA person boxes (`types.yml` actors of type `person`) at the source frame matching each sampled
frame, scaled to the processed size, and the `person_talks_to_person`,
`person_enters_scene_through_structure` and `person_exits_scene_through_structure` activities. MEVA annotates only the
actors of its activities, not everyone in view.

**Id mapping, strict and twin-aware.** On every sampled frame the ground-truth boxes are matched one-to-one with our
boxes by IoU (at least 0.3); each match is a vote of our id for that actor. Two mappings are reported side by side,
because MEVA often annotates one physical person as two or more actors (8% to 32% of the ground-truth boxes have a
near-identical twin box in the same frame, per clip; see the appendix):

* *Strict*: an id of ours maps to the actor it votes for most when the votes number at least 3 and at least 30% of the
  id's own frames. Twin actors are separate people. A conversation annotated under the "other" id of a person we
  followed can never match.
* *Twin-aware*: an id also maps when its votes are at least 50% of that actor's annotated sampled frames (an actor that
  is annotated for only a few seconds can only collect a few votes), and actors whose boxes overlap with IoU of at
  least 0.7 on at least 50% of at least 3 co-annotated sampled frames are one person, on both sides of the comparison.
  Ground-truth talk pairs that become identical after that count once, and a "talk" between two ids of one person is
  dropped.

Strict is the conservative reading; twin-aware is what a person looking at the video would call the same person.
Neither changes the detections.

**Tracking.** MOTA and IDF1 from `motmetrics` on the sampled frames, IoU at least 0.5, once for the raw tracks and once
for the merged persons (the "with and without identity merge" comparison). People we track who are not annotated count
as false positives, so precision and MOTA are pessimistic, and MOTA can be below zero.

**Engagement against talks, roles ignored.** A group talk of k actors becomes all k(k-1)/2 pairs; a pair's interval is
the activity span intersected with both actors' own spans. Our side is every pair of persons that stood within `dist`
metres for at least `min_dur` seconds, bridging gaps of up to 1.5 s (the same streak rule as the event engine's
engagement, but for every pair regardless of role). A detection matches a ground-truth pair interval when both of its
persons map to exactly that pair of actors and the temporal IoU is at least 0.3, one-to-one, best first. Reported:

* `recall_all`: matched / all ground-truth pair intervals (the number the target is about);
* `recall_tracked`: matched / only the intervals whose two actors both map from a person of ours. It isolates the
  engagement logic from detection and tracking misses: when `recall_all` is far below `recall_tracked`, people were
  never followed;
* `precision_mapped`: matched / our intervals whose two persons both map to annotated actors (an interval between
  unannotated bystanders cannot be judged and is left out of the denominator);
* median start-time error over the matches, and the same at tIoU 0.1 as a looser line.

**Arrival and leaving.** The ground-truth time of an enter event is the start of the actor's span, of an exit its end.
Ours is the start (enter) or end (exit) of the presence interval nearest to that time, over every person mapped to the
actor, or, as a second reading, the earliest start / latest end ("first/last seen"). A match is within 5 s. Events in
the first or last second of the clip are not scored. `recall_tracked` again counts only actors we followed.

**Tuning (dev clips only).** `python -m showroom_coach.evaluation.tune` loads each dev clip's cached perception and
ground truth once, then scores the grid merge_thr in {0.20, 0.30, 0.40, 0.50} x dist in {1.5, 2.0, 2.5, 3.0} m x
min_dur in {2, 4, 6, 8} s (gap tolerance fixed at 1.5 s), pooled over the dev clips. The choice maximises the
twin-aware talk F1, the harmonic mean of `recall_all` and `precision_mapped`; ties are broken by the persons' IDF1, then
by the settings nearest the untuned defaults. The tuner works on the cached perception only, so anything that changes the detections is not tuned: the detector score
threshold (the original plan had 0.3 and 0.4 in the grid) would need a new perception run of the dev clips for every value,
about 15 minutes of CPU per value here, and was dropped. The test clips are never opened by the tuner. The chosen `dist` and
`min_dur` are settings of this evaluation's roles-ignored pair detector; they do **not** change the product presets,
which are fixed by the owner (`short`: engagement within 2.0 m for 8 s). The chosen `merge_thr` would replace the
product default only if, at the chosen `dist` and `min_dur`, it beat 0.30 on twin-aware talk F1 or persons IDF1 without
losing on the other.

## Results

### Dev tuning

Dev clips: G330 (16:35), G299 (10:15), G331 bus, 243 twin-aware ground-truth pair intervals in total (269 strict). The
identity merge threshold changes the persons, and with them the tracking numbers and enter/exit recall:

| merge_thr | persons found per dev clip | persons IDF1 | persons MOTA | ID switches | enter/exit recall (twin-aware) | best twin-aware talk F1 over the 16 pair settings |
|---|---|---|---|---|---|---|
| 0.20 | [116, 76, 64] | 21.5% | -6.1% | 400 | 14.7% | 6.5% (dist 2.5, min_dur 8) |
| 0.30 (product default) | [59, 42, 46] | 19.8% | -6.0% | 398 | 2.9% | 6.5% (dist 2.5, min_dur 8) |
| 0.40 | [48, 32, 32] | 19.3% | -6.0% | 397 | 2.9% | 5.0% (dist 2.5, min_dur 8) |
| 0.50 | [47, 31, 26] | 19.0% | -6.0% | 397 | 2.9% | 5.0% (dist 2.5, min_dur 8) |

The pair-detector settings at the default merge (0.30); the full 64-row grid is in `runs/eval/tune_dev.json`:

| dist (m) | min_dur (s) | detected pair intervals | of which both persons map to an actor | matched | recall_all | recall_tracked | precision_mapped | F1 (twin-aware) | F1 (strict) |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 2 | 367 | 164 | 8 | 3.3% | 27.6% | 4.9% | 3.9% | 2.2% |
| 1.5 | 4 | 148 | 74 | 8 | 3.3% | 27.6% | 10.8% | 5.0% | 2.5% |
| 1.5 | 6 | 73 | 41 | 7 | 2.9% | 24.1% | 17.1% | 4.9% | 2.7% |
| 1.5 | 8 | 45 | 26 | 7 | 2.9% | 24.1% | 26.9% | 5.2% | 2.8% |
| 2 | 2 | 589 | 262 | 10 | 4.1% | 34.5% | 3.8% | 4.0% | 2.0% |
| 2 | 4 | 260 | 127 | 8 | 3.3% | 27.6% | 6.3% | 4.3% | 2.4% |
| 2 | 6 | 133 | 67 | 8 | 3.3% | 27.6% | 11.9% | 5.2% | 2.6% |
| 2 | 8 | 81 | 40 | 7 | 2.9% | 24.1% | 17.5% | 4.9% | 2.8% |
| 2.5 | 2 | 792 | 346 | 15 | 6.2% | 51.7% | 4.3% | 5.1% | 2.7% |
| 2.5 | 4 | 380 | 178 | 12 | 4.9% | 41.4% | 6.7% | 5.7% | 3.3% |
| 2.5 | 6 | 211 | 101 | 11 | 4.5% | 37.9% | 10.9% | 6.4% | 3.7% |
| **2.5** | **8** | 137 | 64 | 10 | 4.1% | 34.5% | 15.6% | **6.5%** | 3.9% |
| 3 | 2 | 1035 | 440 | 15 | 6.2% | 51.7% | 3.4% | 4.4% | 2.5% |
| 3 | 4 | 514 | 240 | 15 | 6.2% | 51.7% | 6.2% | 6.2% | 3.1% |
| 3 | 6 | 305 | 147 | 12 | 4.9% | 41.4% | 8.2% | 6.2% | 3.5% |
| 3 | 8 | 200 | 102 | 10 | 4.1% | 34.5% | 9.8% | 5.8% | 3.7% |

What the grid says:

* **The chosen setting is merge_thr 0.30, dist 2.5 m, min_dur 8 s (twin-aware talk F1 6.5%, `recall_all` 4.1%,
  `precision_mapped` 15.6%).** The untuned setting (dist 2.0) scores F1 4.9% on dev.
* **The grid is nearly flat and the signal is tiny.** Every one of the 64 settings matches between 6 and 16 of the 243
  ground-truth intervals, and the six best settings (F1 from 6.2% to 6.5%) match 10 to 16, so the choice of `dist` and
  `min_dur` is within noise. Larger `dist` finds more intervals and more false ones; larger `min_dur` trades recall
  for precision. No setting comes near the 70% target.
* **merge_thr 0.20 has a better IDF1 (21.5% against 19.8%) but fragments the people (116, 76 and 64 persons in the three
  clips against 59, 42 and 46) and ties on talk F1: both match 10 of the 243 intervals, and the 0.07-point gap (6.45% against 6.51%) comes only
  from 67 against 64 mapped detections.** By the rule above it does not beat 0.30, so the product default stays at **0.30**; `analysis.py` is unchanged. 0.40 and 0.50 are worse on both.
* The enter/exit recall of 14.7% at 0.20 against 2.9% at the others is an artefact of fragmentation (more persons mean
  more presence boundaries to match), not a better identity model; it is not part of the selection.

### Test (4 clips; exposure listed below)

Settings: `short` preset, dist 2.5 m, min_dur 8 s, gap tolerance 1.5 s, merge_thr 0.30, first 150 s of each clip
(`runs/eval/meva_test.json`). 3000 sampled frames and 54 162 annotated person boxes.

**Tracking** (pooled over the four clips):

| | MOTA | IDF1 | precision | recall | ID switches |
|---|---|---|---|---|---|
| raw tracks (no identity merge) | -28.1% | 15.5% | 28.2% | 17.8% | 337 |
| persons (after the identity merge, merge_thr 0.30) | -27.4% | 13.5% | 28.3% | 17.6% | 244 |

The identity merge removes about a quarter of the ID switches (337 to 244) but lowers IDF1 by 2.0 points (15.5% to
13.5%): merging the right fragments is worth less than the wrong merges and dropped duplicates cost. Recall is 17.6%:
the detector matches fewer than one in five of the annotated person boxes (49.6% and 46.8% on G423 and G421, the two clips with few
annotated people; 11.9% to 13.5% on the two crowded school clips). MOTA is below zero because the people we track
who MEVA did not annotate are counted as false positives.

**Engagement and arrival/leaving** (pooled over the four clips):

| | strict | twin-aware |
|---|---|---|
| ground-truth talk pair intervals | 390 | 358 |
| ... of which both actors are followed by a person of ours | 15 | 30 |
| our pair intervals (both persons map to an actor / all) | 14 / 128 | 39 / 128 |
| matched (tIoU >= 0.3) | 2 | 4 |
| **talk recall_all** | **0.5%** | **1.1%** |
| talk recall_tracked | 13.3% | 13.3% |
| talk precision_mapped | 14.3% | 10.3% |
| median start-time error (matches) | 16.2 s (n=2) | 4.6 s (n=4) |
| looser tIoU >= 0.1: recall_all / precision_mapped | 0.8% / 21.4% | 1.7% / 15.4% |
| enter/exit events scored | 34 | 34 |
| **enter/exit recall_all** (within 5 s, nearest presence boundary) | **2.9%** (1/34) | **20.6%** (7/34) |
| enter/exit recall_tracked (events of actors we followed) | 100.0% (1/1) | 77.8% (7/9) |
| enter/exit median absolute timing error | 1.1 s | 0.8 s |
| enter/exit recall_all with first / last seen | 2.9% | 14.7% |

Enter and exit separately:

| | strict enter | strict exit | twin-aware enter | twin-aware exit |
|---|---|---|---|---|
| events scored | 14 | 20 | 14 | 20 |
| recall_all | 7.1% (1) | 0.0% (0) | 14.3% (2) | 25.0% (5) |
| median abs timing error | 1.1 s | n/a | 1.2 s | 0.7 s |
| median signed error (ours minus ground truth) | 1.1 s | n/a | -0.1 s | -0.3 s |

How to read it:

* Talk recall is **1.1% (4 of 358)**, far below the 70% target. Only 30 of the 358 ground-truth pair intervals (8%) have
  both actors followed by a person of ours, so for about 92% of them detection or tracking has already failed. Where
  both are followed (`recall_tracked`) 4 of 30 match. The 4-of-358 and the 2-of-390 are too few matches for the
  percentages to carry more than "very low": each of the 4 matches is a quarter of the result.
* The strict and twin-aware columns differ in what they credit, not in the detections: the same 128 intervals of ours are
  scored. The start-time errors (16.2 s strict, 4.6 s twin-aware) are medians of 2 and 4 matches.
* Where an actor is followed, arrival and leaving are timed closely: the median absolute error is 0.8 s over the 7
  matched events, and 7 of the 9 events of followed actors fall within the 5 s window (the other 2 do not). Overall
  recall is 20.6% (7 of 34) twin-aware and 2.9% (1 of 34) strict. Both G330 clips (one dev, one test)
  have no enter/exit events at all.

**Untuned settings on the same test clips**, for comparison (`runs/eval/meva_test_default.json`; tracking and
enter/exit numbers do not depend on dist and min_dur and are identical):

| settings | strict talk recall_all, precision_mapped | twin-aware talk recall_all, precision_mapped |
|---|---|---|
| untuned: dist 2.0, min_dur 8 | 0.3% (1/390), P 8.3% | 0.8% (3/358), P 12.0% |
| tuned on dev: dist 2.5, min_dur 8 | 0.5% (2/390), P 14.3% | 1.1% (4/358), P 10.3% |

The tuned setting finds 4 intervals instead of 3 (twin-aware) and 2 instead of 1 (strict): the dev tuning gain is
visible in direction but is a difference of one detection.

**Dev clips with the chosen settings**, for reference (`runs/eval/meva_dev.json`; these are the clips the settings
were chosen on, so they are optimistic):

| | MOTA | IDF1 | precision | recall | ID switches |
|---|---|---|---|---|---|
| raw tracks (no identity merge) | -6.5% | 22.9% | 45.5% | 27.4% | 559 |
| persons (after the identity merge, merge_thr 0.30) | -6.0% | 19.8% | 45.5% | 26.9% | 398 |

| | strict | twin-aware |
|---|---|---|
| ground-truth talk pair intervals | 269 | 243 |
| ... of which both actors are followed by a person of ours | 10 | 29 |
| our pair intervals (both persons map to an actor / all) | 37 / 137 | 64 / 137 |
| matched (tIoU >= 0.3) | 6 | 10 |
| **talk recall_all** | **2.2%** | **4.1%** |
| talk recall_tracked | 60.0% | 34.5% |
| talk precision_mapped | 16.2% | 15.6% |
| median start-time error (matches) | 3.1 s (n=6) | 3.4 s (n=10) |
| looser tIoU >= 0.1: recall_all / precision_mapped | 2.6% / 18.9% | 4.5% / 17.2% |
| enter/exit events scored | 34 | 34 |
| **enter/exit recall_all** (within 5 s, nearest presence boundary) | **2.9%** (1/34) | **2.9%** (1/34) |
| enter/exit recall_tracked (events of actors we followed) | 100.0% (1/1) | 50.0% (1/2) |
| enter/exit median absolute timing error | 1.2 s | 1.2 s |
| enter/exit recall_all with first / last seen | 0.0% | 0.0% |

Per-clip tables for test and dev are in the appendix.

**Test-set exposure.** The test clips were not untouched, and "measured once" would overstate it. Everything that was done
with them:

* *Before the split existed* (`data/meva/split.json` was written at 06:50 on 2026-10-11): `runs/eval/meva_smoke.json`
  (06:45) scored the first 60 s of the G421 clip, now a test clip, with the default settings (strict 1 of 21 talk pair
  intervals, twin-aware 2 of 21). The twin-aware id mapping and the twin clusters were developed in that period with that
  G421 clip in view, so the twin-aware rule is not independent of the test data.
* *G421 and G423 for other work.* A G421 cut (`meva_b_60s`, 180 to 240 s) was used for smoke tests, live-mode runs and the
  duplicate-id bug work, and the full 5 min G421 clip for a file-mode speed run (0.71x). G423, also a test clip, was used
  for the live-mode speed runs (from 123 s, and the 20 s cut used by `bench`). None of the speed runs used labels.
* *Clip selection.* The seven clips were chosen for their annotated talk counts before the split, so the annotation counts
  of all of them, test clips included, were known.
* *After the split:* five scored runs on the test clips with fixed settings: system at 2.5 m / 8 s (chosen on dev) and at
  2.0 m / 8 s (untuned), and the oracle ablation at 2.5 m / 8 s, 2.0 m / 8 s and 1.5 m / 6 s (the last picked on dev
  only). The full per-clip test tables were printed and read when they were written.
* No threshold, rule or setting of the evaluation was chosen or changed because of a test result. The pre-split G421 use is
  the main place where the test data could have influenced a design choice of the evaluation, through the twin-aware mapping.

## Ablation: oracle tracks (perfect detection and tracking)

**Why.** The system result is dominated by detection (17.6% of the annotated boxes are matched), so it cannot say
whether the engagement logic works when the people *are* found. This ablation takes detection and tracking out of the
question.

**Method.** `eval-meva --oracle` (and `tune --oracle`) replace the cached detections with the MEVA ground-truth person
boxes. The sampled frames and the video info are those of the cached run (750 frames per clip at 5 fps). On every frame
there is one observation per annotated person: the box of the matching source frame, scaled to processed pixels, score
1.0, **no keypoints** (the geometry falls back to the box height and flags every distance as approximate) and **no
embeddings** (so the identity merge has nothing to merge and persons are the tracks). The track id is the lowest actor id
of the actor's twin cluster, so the several annotations of one physical person become one track. Then the unchanged
analysis, roles-ignored pair detector and scoring run, and every oracle person maps to its own actor. The output carries
`"mode": "oracle"`.

Three pair-detector settings were run, each once on dev and once on test: the settings chosen on dev for the system
(2.5 m, 8 s), the untuned reference (2.0 m, 8 s), and the setting picked on the dev clips alone with the oracle tracks by
the same F1 rule (`tune --oracle`, 1.5 m, 6 s; grid below). Nothing was picked on test.

**What it isolates.** With detection, tracking and identity perfect, what is left is (a) the distance between two people
as the geometry computes it from the box height, and (b) the engagement rule (within `dist` metres for `min_dur` seconds,
1.5 s gaps bridged) set against what MEVA annotators marked as `person_talks_to_person`. A miss here is not a detector,
tracker or identity-merge miss.

| test clips, dist 2.5 m, min_dur 8 s | system (detections) | oracle (ground-truth tracks) |
|---|---|---|
| talk recall_all, strict | 0.5% (2/390) | 32.3% (126/390) |
| talk recall_all, twin-aware | 1.1% (4/358) | 44.1% (158/358) |
| talk precision_mapped, twin-aware | 10.3% | 26.0% |
| ground-truth pairs with both actors followed, twin-aware | 30 of 358 | 358 of 358 |
| persons IDF1 / MOTA | 13.5% / -27.4% | 92.1% / 86.5% |
| enter/exit recall_all, twin-aware | 20.6% (7/34) | 94.1% (32/34) |

Talk results with oracle tracks, per setting (twin-aware is the headline; strict also counts twin annotations of one
person as different people, so it is lower):

| pair-detector settings | split | strict recall_all | twin-aware recall_all | twin-aware precision_mapped | pair intervals detected |
|---|---|---|---|---|---|
| dist 2.5 m, min_dur 8 s (chosen on dev for the system) | dev | 34.9% (94/269) | **52.3%** (127/243) | 28.0% (127/453) | 453 |
| dist 2.5 m, min_dur 8 s (chosen on dev for the system) | test | 32.3% (126/390) | **44.1%** (158/358) | 26.0% (158/608) | 608 |
| dist 2.0 m, min_dur 8 s (untuned reference) | dev | 32.0% (86/269) | **46.1%** (112/243) | 32.7% (112/343) | 343 |
| dist 2.0 m, min_dur 8 s (untuned reference) | test | 27.4% (107/390) | **37.2%** (133/358) | 30.0% (133/444) | 444 |
| dist 1.5 m, min_dur 6 s (chosen on dev with oracle tracks) | dev | 32.7% (88/269) | **46.5%** (113/243) | 41.5% (113/272) | 272 |
| dist 1.5 m, min_dur 6 s (chosen on dev with oracle tracks) | test | 24.4% (95/390) | **31.8%** (114/358) | 33.1% (114/344) | 344 |

The dev grid with oracle tracks (`runs/eval/tune_dev_oracle.json`; the F1 pick is in bold):

| dist (m) | min_dur (s) | detected pair intervals | matched | recall_all | precision_mapped | F1 (twin-aware) | F1 (strict) |
|---|---|---|---|---|---|---|---|
| 1.5 | 2 | 547 | 136 | 56.0% | 24.9% | 34.4% | 26.0% |
| 1.5 | 4 | 352 | 130 | 53.5% | 36.9% | 43.7% | 32.5% |
| **1.5** | **6** | 272 | 113 | 46.5% | 41.5% | **43.9%** | 32.5% |
| 1.5 | 8 | 199 | 88 | 36.2% | 44.2% | 39.8% | 29.5% |
| 2 | 2 | 862 | 166 | 68.3% | 19.3% | 30.0% | 22.5% |
| 2 | 4 | 605 | 161 | 66.3% | 26.6% | 38.0% | 27.9% |
| 2 | 6 | 454 | 139 | 57.2% | 30.6% | 39.9% | 28.8% |
| 2 | 8 | 343 | 112 | 46.1% | 32.7% | 38.2% | 28.1% |
| 2.5 | 2 | 1133 | 179 | 73.7% | 15.8% | 26.0% | 19.4% |
| 2.5 | 4 | 799 | 176 | 72.4% | 22.0% | 33.8% | 24.9% |
| 2.5 | 6 | 612 | 157 | 64.6% | 25.7% | 36.7% | 26.3% |
| 2.5 | 8 | 453 | 127 | 52.3% | 28.0% | 36.5% | 26.0% |
| 3 | 2 | 1474 | 194 | 79.8% | 13.2% | 22.6% | 16.9% |
| 3 | 4 | 1056 | 192 | 79.0% | 18.2% | 29.6% | 22.0% |
| 3 | 6 | 807 | 167 | 68.7% | 20.7% | 31.8% | 22.9% |
| 3 | 8 | 594 | 136 | 56.0% | 22.9% | 32.5% | 22.9% |

Tracking quality of the oracle tracks themselves (what is left of MOTA and IDF1 is the twin duplicates: the second box of
a twin pair has no track to match):

| split (dist 2.5 m, min_dur 8 s) | MOTA | IDF1 | precision | recall | ID switches | persons found per clip |
|---|---|---|---|---|---|---|
| dev | 87.3% | 91.2% | 100.0% | 87.3% | 0 | [235, 176, 71] |
| test | 86.5% | 92.1% | 100.0% | 86.5% | 0 | [172, 97, 55, 38] |

Enter and exit with oracle tracks:

| | strict enter | strict exit | twin-aware enter | twin-aware exit |
|---|---|---|---|---|
| dev: recall_all (events) | 66.7% (10/15) | 31.6% (6/19) | 100.0% (15/15) | 94.7% (18/19) |
| dev: median abs timing error | 0.0 s | 0.1 s | 0.1 s | 0.1 s |
| test: recall_all (events) | 92.9% (13/14) | 35.0% (7/20) | 100.0% (14/14) | 90.0% (18/20) |
| test: median abs timing error | 0.1 s | 0.1 s | 0.1 s | 0.1 s |

**Reading.**

* Perfect detection and tracking lift test talk recall from 1.1% to 44.1% (twin-aware; strict
  0.5% to 32.3%) at the same settings. So most of the system's shortfall is detection and tracking, not the rule.
* The engagement logic alone is **still under the 70% target at the settings run on test**: 44.1% on test and 52.3% on dev at 2.5 m and
  8 s, with a precision of 26.0% (about three of four proximity intervals it finds on annotated people are not
  annotated talks). The oracle's start-time error (median 0.1 s) and enter/exit timing say nothing about the system: the
  oracle boxes exist only during the annotated spans, so a pair's start and an actor's arrival coincide with the annotation
  by construction (the same circularity as the enter/exit recall below), and those columns are left out of the oracle talk
  table.
* On the dev grid recall rises with `dist` and falls with `min_dur`. 70% is reached only at 2.5 m or 3.0 m with a minimum
  of 4 s or less (four of the 16 settings), and then the precision is 13% to 22%. The F1 pick (1.5 m, 6 s) is a precision-
  oriented setting: dev recall 46.5% with a precision of 41.5%, test recall 31.8% with a precision of 33.1%. The settings were not chosen for
  recall, and no recall-first setting was run on test.
* Enter/exit recall with oracle tracks (about 94% twin-aware on test) is not evidence for the system: see the caveats.

**Caveats.**

* *Distance from box height.* Every distance in this run comes from the box height (a 1.70 m person, a 70 degree field of
  view, no calibration, boxes that can be cut off at the frame edge). Real keypoints would not change that for the
  oracle, and the real-world accuracy of the distance is unmeasured.
* *MEVA talk is not showroom engagement.* The annotated talks are between 2 and 16 people in school and bus-site scenes;
  the product looks for a staff member beside a customer. A pair standing within a few metres is not necessarily a
  conversation, and a conversation is not necessarily within a few metres, so this is a measure of agreement with MEVA's
  labelling, not of showroom accuracy.
* *Ground truth exists only while annotated.* Oracle people appear and disappear with their annotations, so presence-based
  enter/exit timing is partly circular, and unannotated bystanders (who would produce false pairs on real footage) are
  absent, which flatters precision.
* *Actor ids are not people.* MEVA gives a person a new actor id per annotated activity and the oracle cannot link them
  (persons found per clip are in the table above, far more than there are people in the scene). The twin collapse only
  joins ids whose boxes coincide at the same time.
* *Test exposure.* The oracle was run three times on test (2.5 m / 8 s, 2.0 m / 8 s, 1.5 m / 6 s), after two system runs:
  five scored runs on test in all, none used to choose a setting. The headline is the system's pre-chosen 2.5 m / 8 s.
  See "Test-set exposure".

## Speed

The first two blocks were measured earlier in the project with the machine shared with other work (other agents' test
and perception runs) and are not benchmark figures; the "idle machine" section below is the later, near-idle measurement.

**File mode** (from `PROGRESS.md`, M2 and M3 summaries; wall clock, i5-10210U, other work was loading the CPU during
some runs; 5 fps sampling, at most 960 px wide):

| clip | people per frame | real-time factor | wall time |
|---|---|---|---|
| `meva_a_60s` (G299 cut, 960 px) | about 11 | 0.976x | 61.5 s for 60 s |
| full G421 clip (1920x1072, 5 min) | about 12 | 0.713x | 421 s for 300 s |
| the same two before the batching and pose-stride work | | 0.706x and 0.506x | |
| annotated-video render of `meva_a_60s` (10 fps, faces blurred) | | 7.4x real time | 8.1 s |

The perception runs that filled the cache for this evaluation (table in "Data") ran at 0.25x to 0.84x with other work on
the machine; they show the spread with crowd size and contention, not a clean figure.

**Live mode** (stage timings and people counts from `runs/live_bench_all.log`, three runs per clip, 100 frames through
`LiveSession` at 5 fps timestamps; the CPU was 33% to 100% busy before each run; measured before the latest `live.py`
changes, so indicative only):

| clip | people per frame | processed fps, runs 1 / 2 / 3 |
|---|---|---|
| `meva_a_60s` | mean 11.4, max 15 | 1.49 / 2.09 / 2.13 |
| `meva_b_60s` (G421 cut) | mean 14.5, max 19 | 0.88 / 1.23 / 1.59 |
| G423 from 123 s | mean 3.8, max 5 (90 of 100 frames with 1 to 4 people) | 2.27 / 3.32 / 2.10 overall; 2.36 / 3.53 / 2.22 on the frames with 1 to 4 people |

Appearance embedding takes about half of the live time (detect 15% to 31%, pose 18% to 35%, embed 46% to 52%). The
WebSocket client (`scripts/live_file_client.py`) on a crowded clip (15.4 people per frame) saw a server fps of 3.92 in one
run and 1.75 in another (`runs/live_file_client_run.txt`, `runs/live_file_client_run2.txt`).

### Speed (idle machine)

Measured during final verification; see PROGRESS.md Morning summary.

These numbers were measured by the controller on a near-idle machine (07:30 to 07:42, logs in `runs/verify/`; one clip
was run while another agent was working, marked below). The commands are `showroom-coach bench`,
`showroom-coach analyze`, `scripts/live_file_client.py` against the server started by `run.bat`, and `scripts/smoke_api.py`.

| measurement | clip | people | result |
|---|---|---|---|
| `bench`, file mode (fresh run, no cache) | `meva_a_60s_720p.mp4`, 720p, 60 s | 12.0 per frame | perception 68.9 s, **real-time factor 0.87x** (decode 2.6, detect 13.9, track 1.3, pose 20.0, embed 31.1 s); render 13.8 s (4.3x real time) |
| `analyze`, fresh cache | `meva_a_60s`, 960 px, 60 s | 31 persons found | perception 56.2 s, **real-time factor 1.07x** |
| `analyze`, fresh cache | `meva_b_60s`, 60 s | 33 persons found | perception 106.4 s, **real-time factor 0.56x**; another agent was running then, so possibly contended |
| `bench`, live mode, in process | `meva_g423_20s.mp4`, 100 frames | 3.8 per frame (min 3, max 7) | **3.61 fps** mean, 3.39 median (277 ms mean, p95 486 ms); embedding 14.5 s of 27.7 s |
| real `/ws/live` through `live_file_client.py` (server from `run.bat`) | same clip, 100 frames | 3.8 per frame (min 3, max 6) | **end-to-end 4.45 fps**, round trip 5.21 fps (192 ms mean) |
| web smoke, `run.bat` + `smoke_api.py --budget 180` | `meva_a_60s` | | upload to needs_confirm **60.3 s (budget met)**, roles round trip 0.14 s, video ready 12.2 s later |

The in-process and WebSocket live figures are on the same clip but are different measurements: the first times
`LiveSession.process` alone, the second counts the whole round trip of a frame (the server's own mean of 7.90 fps is an
average of per-frame rates and is not comparable). Read them as a run-to-run spread of 3.6 to 4.5 fps around the 4 fps
target: the WebSocket run was the faster one, and it slowed mid-run (4.5 to 5.8 overlays per second in its first
12 s, 3.4 to 4.2 after that; `runs/verify/live_client.txt`).

(The command that produces these figures is `showroom-coach bench`; see "Reproduce".)

## Targets vs measured

| Target (design spec, section 6) | Measured | Status |
|---|---|---|
| Talk / engagement recall >= 70% on MEVA | **System:** 1.1% test twin-aware (4 of 358), 0.5% strict (2 of 390); 13.3% when both actors are followed; dev 4.1%. **Oracle tracks (perfect detection and tracking):** 44.1% test twin-aware, 32.3% strict at 2.5 m / 8 s; dev 52.3%. A recall-first setting reaches 79% to 80% on the oracle dev grid (3.0 m, 2 to 4 s) at 13% to 18% precision; it was not run on test | ❌ missed, with and without perfect tracking |
| Staff suggestion >= 80% correct | not measurable: no labelled showroom data | not measured |
| Greet-time median error <= 2 s | not measurable: no labelled showroom data | not measured |
| File mode >= 1x real time | **0.87x on the spec benchmark** `meva_a_60s_720p` (720p, 12 people per frame, fresh run). 1.07x on the same footage at 960 px (`meva_a_60s`, 31 persons). The two runs differ in command (bench vs analyze), source cut (720p vs 960 px) and people found (31 at 960 px); neither was repeated, so the cause of the gap is not isolated; 0.56x on a 14-person 60 s clip (`meva_b_60s`, possibly contended). Earlier, with other work on the machine: 0.98x and 0.71x (5 min, 12 people) | ❌ missed on the benchmark (0.87x) |
| Live mode >= 4 fps (1 to 4 people) | On the 3.8-person clip: 3.61 fps in the in-process bench and 4.45 fps end to end over the real WebSocket, a run-to-run spread of 3.6 to 4.5 fps (the WebSocket run was the faster one and slowed mid-run). Earlier, with other work on the machine, 0.9 to 3.9 fps over all runs with retained logs (3.5 fps at best on frames with 1 to 4 people) | borderline: 3.6 to 4.5 fps around the target |
| Upload to results in under 3 minutes (spec section 12) | 60.3 s from upload to the confirm step for the 60 s clip `meva_a_60s` on a near-idle machine (roles round trip 0.14 s, annotated video 12.2 s later). Under load (other agents running), earlier runs took 122 s (`meva_a_60s`) and 253 s (`meva_b_60s`) | ✅ met when idle; ❌ under load (253 s) |

Other measured values, with no target: tracking IDF1 13.5% (persons) and 15.5% (raw tracks), MOTA -27.4% and -28.1% on
test; enter/exit recall 20.6% twin-aware (2.9% strict) with a median absolute timing error of 0.8 s over the 7 matched events.

## Limitations

* **MEVA is not a showroom.** It is indoor surveillance footage from school-site and bus-site cameras: people are small (the median
  annotated box in the detector benchmark was 47 px tall), up to 16 are in a group, cameras are high and wide, and the
  talks are mostly groups. The detector and the distance model were never checked against a showroom camera.
* **Talk is not engagement.** `person_talks_to_person` marks conversations, and it is annotated for groups; the product
  looks for a staff member next to a customer. This evaluation ignores roles and scores proximity for a minimum time
  against the talks. A group of k people is k(k-1)/2 ground-truth pairs, so one 16-person talk is 120 intervals and the
  large groups dominate `recall_all`.
* **Duplicate ground-truth actors.** 8% to 32% of the person boxes have a near-identical twin box. The twin-aware
  numbers fold them; the strict ones do not. Neither is a perfect reading of who is who, and the twin rule is our own.
* **Unannotated bystanders.** People who are not part of an annotated activity are not boxed, so they count as false
  positives: tracking precision and MOTA are pessimistic, and a precision on pair intervals is only defined for those
  whose two persons map to actors.
* **Fragmentation in crowds.** The three dev clips produced 59, 42 and 46 persons, and the test clips 42, 49, 40 and 16,
  from 75 to 414 raw tracks per clip; there are several hundred ID switches per set. Pair intervals break when either
  person's id changes.
* **Detector recall.** `yolox_nano@2` was chosen in the M0 benchmark for speed within 5 points of the best recall; its
  recall on annotated people is 17.6% on test and 26.9% on dev, and detection and tracking together account for most of the
  missed talks: with ground-truth boxes instead of detections the same rule finds 44.1% (see the oracle ablation). The
  best candidate of that benchmark, `yolox_s@2`, reached 31.0% recall (404 of 1303 boxes) at about ten times the time per
  frame and was not run on the full evaluation.
* **150 s per clip, 7 clips.** Only the first 150 s of each 5 min clip are scored, and only 4 test clips (358 twin-aware
  talk pair intervals, 34 enter/exit events). The confidence interval on a rate like 4 of 358 is wide, and the dev
  tuning differences are within noise. Enter/exit events in the first or last second are skipped.
* **Fixed settings, no failure analysis.** Each setting was run once on the test clips (five scored runs in all, see "Test-set
  exposure") and the clips were not inspected afterwards to find out why pairs were missed; the reasons above (detection recall, fragmentation, groups,
  threshold choice) are measured only in aggregate.
* **Geometry is uncalibrated.** Distances use a pinhole guess (70 degrees, 1.70 m people). Its accuracy on real footage
  is unmeasured; only synthetic people were checked (within 0.1%).
* **CPU contention.** The perception timings of the cache runs and the earlier speed numbers were produced while other
  processes used the machine; the "Speed (idle machine)" figures are near-idle, and one of them is marked as possibly
  contended.
* **Showroom parts unevaluated.** Staff suggestion, clicks per clip, greet time and walk-outs have no data here.

## Reproduce

Windows 11, Git Bash, from the repository root, Python 3.12 venv (see `CLAUDE.md`). The clips and annotations are listed
in `data/meva/clips.csv` (`video_url` for the videos; the KPF files come from
`gitlab.kitware.com/meva/meva-data-repo`, `annotation/DIVA-phase-2/MEVA/kitware`) and live in `data/meva/videos/` and
`data/meva/annotations/`. The pipeline resizes inputs to at most 960 px wide, so the 720p speed clip is processed at
960x540. The first evaluation run on a machine without the cache runs perception on the first 150 s of
every clip (about 40 minutes of CPU here); later runs and the tuner reuse `runs/cache/`.

```bash
.venv/Scripts/python -m pip install -e ".[dev,eval]"        # "eval" adds motmetrics
.venv/Scripts/python -m showroom_coach fetch-models          # the default stack's weights

# 1. tune on the dev clips only (cache only; about 2 minutes)
.venv/Scripts/python -m showroom_coach.evaluation.tune --split-file data/meva/split.json --out runs/eval/tune_dev.json

# 2. measure the test clips with the chosen settings, and with the untuned ones
.venv/Scripts/python -m showroom_coach eval-meva --split test --split-file data/meva/split.json --limit-seconds 150 \
    --dist 2.5 --min-dur 8 --merge-thr 0.3 --out runs/eval/meva_test.json
.venv/Scripts/python -m showroom_coach eval-meva --split test --split-file data/meva/split.json --limit-seconds 150 \
    --dist 2.0 --min-dur 8 --merge-thr 0.3 --out runs/eval/meva_test_default.json

# 3. the dev clips with the chosen settings
.venv/Scripts/python -m showroom_coach eval-meva --split dev --split-file data/meva/split.json --limit-seconds 150 \
    --dist 2.5 --min-dur 8 --merge-thr 0.3 --out runs/eval/meva_dev.json

# 4. speed: file-mode real-time factor (fresh run, no cache) with per-stage times and render time,
#    and live-mode fps with people per frame; writes runs/bench/bench.json
.venv/Scripts/python -m showroom_coach bench --video data/samples/meva_a_60s_720p.mp4 --frames 100

# 5. ablation: oracle tracks (ground-truth boxes instead of detections); dist/min_dur picked on dev only
.venv/Scripts/python -m showroom_coach.evaluation.tune --oracle --split-file data/meva/split.json --out runs/eval/tune_dev_oracle.json
for split in dev test; do
  .venv/Scripts/python -m showroom_coach eval-meva --oracle --split $split --split-file data/meva/split.json --limit-seconds 150 \
      --dist 2.5 --min-dur 8 --merge-thr 0.3 --out runs/eval/meva_${split}_oracle_chosen.json
  .venv/Scripts/python -m showroom_coach eval-meva --oracle --split $split --split-file data/meva/split.json --limit-seconds 150 \
      --dist 2.0 --min-dur 8 --merge-thr 0.3 --out runs/eval/meva_${split}_oracle_default.json
  .venv/Scripts/python -m showroom_coach eval-meva --oracle --split $split --split-file data/meva/split.json --limit-seconds 150 \
      --dist 1.5 --min-dur 6 --merge-thr 0.3 --out runs/eval/meva_${split}_oracle_devtuned.json
done
```

`python -m showroom_coach.evaluation.meva_eval` takes the same flags as `eval-meva`. Results are deterministic given the
cache. Test it without models: `showroom-coach bench --video <any video> --fake`.

## Appendix: per-clip results

### Test clips (dist 2.5 m, min_dur 8 s, merge_thr 0.30)

| clip | MOTA tracks | IDF1 tracks | ID sw. | MOTA persons | IDF1 persons | ID sw. | precision | recall | GT boxes with a twin |
|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | -15.1% | 14.7% | 111 | -14.7% | 12.1% | 71 | 32.6% | 13.5% | 13.6% |
| 2018-03-12 10-50 G330 | -38.1% | 9.9% | 93 | -37.5% | 10.0% | 57 | 19.5% | 11.9% | 29.3% |
| 2018-03-13 15-50 G421 | -58.4% | 27.8% | 104 | -56.7% | 22.2% | 91 | 31.6% | 46.8% | 28.0% |
| 2018-03-12 10-10 G423 | -8.5% | 31.4% | 29 | -7.1% | 26.5% | 25 | 47.0% | 49.6% | 32.2% |
| **Total** | -28.1% | 15.5% | 337 | -27.4% | 13.5% | 244 | 28.3% | 17.6% | 22.6% |

strict:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | 199 | 4 / 27 | 1 | 0.5% | 33.3% | 25.0% | 4.2 s | 6 | 0.0% | n/a | n/a |
| 2018-03-12 10-50 G330 | 121 | 3 / 38 | 0 | 0.0% | 0.0% | 0.0% | n/a | 0 | n/a | n/a | n/a |
| 2018-03-13 15-50 G421 | 41 | 0 / 51 | 0 | 0.0% | 0.0% | n/a | n/a | 12 | 8.3% | 100.0% | 1.1 s |
| 2018-03-12 10-10 G423 | 29 | 7 / 12 | 1 | 3.4% | 50.0% | 14.3% | 28.3 s | 16 | 0.0% | n/a | n/a |
| **Total** | 390 | 14 / 128 | 2 | 0.5% | 13.3% | 14.3% | 16.2 s | 34 | 2.9% | 100.0% | 1.1 s |

twin-aware:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | 189 | 10 / 27 | 1 | 0.5% | 25.0% | 10.0% | 4.2 s | 6 | 0.0% | n/a | n/a |
| 2018-03-12 10-50 G330 | 116 | 12 / 38 | 0 | 0.0% | 0.0% | 0.0% | n/a | 0 | n/a | n/a | n/a |
| 2018-03-13 15-50 G421 | 40 | 9 / 51 | 3 | 7.5% | 20.0% | 33.3% | 5.0 s | 12 | 41.7% | 71.4% | 0.7 s |
| 2018-03-12 10-10 G423 | 13 | 8 / 12 | 0 | 0.0% | n/a | 0.0% | n/a | 16 | 12.5% | 100.0% | 1.0 s |
| **Total** | 358 | 39 / 128 | 4 | 1.1% | 13.3% | 10.3% | 4.6 s | 34 | 20.6% | 77.8% | 0.8 s |

### Dev clips (same settings)

| clip | MOTA tracks | IDF1 tracks | ID sw. | MOTA persons | IDF1 persons | ID sw. | precision | recall | GT boxes with a twin |
|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 2.3% | 23.4% | 325 | 2.7% | 19.9% | 249 | 53.8% | 24.5% | 18.7% |
| 2018-03-09 10-15 G299 | -14.4% | 20.5% | 186 | -13.9% | 17.8% | 114 | 39.7% | 25.2% | 28.1% |
| 2018-03-07 11-05 G331 | -30.1% | 25.3% | 48 | -29.0% | 22.8% | 35 | 37.3% | 41.6% | 7.6% |
| **Total** | -6.5% | 22.9% | 559 | -6.0% | 19.8% | 398 | 45.5% | 26.9% | 19.8% |

strict:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 202 | 30 / 77 | 6 | 3.0% | 66.7% | 20.0% | 3.1 s | 0 | n/a | n/a | n/a |
| 2018-03-09 10-15 G299 | 50 | 1 / 13 | 0 | 0.0% | 0.0% | 0.0% | n/a | 29 | 0.0% | n/a | n/a |
| 2018-03-07 11-05 G331 | 17 | 6 / 47 | 0 | 0.0% | n/a | 0.0% | n/a | 5 | 20.0% | 100.0% | 1.2 s |
| **Total** | 269 | 37 / 137 | 6 | 2.2% | 60.0% | 16.2% | 3.1 s | 34 | 2.9% | 100.0% | 1.2 s |

twin-aware:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 181 | 36 / 77 | 7 | 3.9% | 35.0% | 19.4% | 3.3 s | 0 | n/a | n/a | n/a |
| 2018-03-09 10-15 G299 | 45 | 7 / 13 | 1 | 2.2% | 16.7% | 14.3% | 3.5 s | 29 | 0.0% | 0.0% | n/a |
| 2018-03-07 11-05 G331 | 17 | 21 / 47 | 2 | 11.8% | 66.7% | 9.5% | 12.5 s | 5 | 20.0% | 100.0% | 1.2 s |
| **Total** | 243 | 64 / 137 | 10 | 4.1% | 34.5% | 15.6% | 3.4 s | 34 | 2.9% | 50.0% | 1.2 s |

### Oracle tracks, test clips (dist 2.5 m, min_dur 8 s)

The start-error and enter/exit timing columns of the oracle tables are circular (ground-truth boxes exist only while an
actor is annotated) and say nothing about the system.

| clip | MOTA tracks | IDF1 tracks | ID sw. | MOTA persons | IDF1 persons | ID sw. | precision | recall | GT boxes with a twin |
|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | 93.1% | 95.2% | 0 | 93.0% | 95.2% | 0 | 100.0% | 93.0% | 13.6% |
| 2018-03-12 10-50 G330 | 84.9% | 91.8% | 0 | 84.6% | 91.6% | 0 | 100.0% | 84.6% | 29.3% |
| 2018-03-13 15-50 G421 | 88.8% | 92.8% | 0 | 86.7% | 91.6% | 0 | 100.0% | 86.7% | 28.0% |
| 2018-03-12 10-10 G423 | 53.6% | 69.7% | 0 | 52.9% | 69.0% | 0 | 100.0% | 52.9% | 32.2% |
| **Total** | 86.9% | 92.3% | 0 | 86.5% | 92.1% | 0 | 100.0% | 86.5% | 22.6% |

strict:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | 199 | 416 / 416 | 63 | 31.7% | 40.6% | 15.1% | 0.1 s | 6 | 33.3% | 100.0% | 0.1 s |
| 2018-03-12 10-50 G330 | 121 | 149 / 149 | 37 | 30.6% | 37.0% | 24.8% | 0.2 s | 0 | n/a | n/a | n/a |
| 2018-03-13 15-50 G421 | 41 | 36 / 36 | 21 | 51.2% | 65.6% | 58.3% | 0.0 s | 12 | 41.7% | 100.0% | 0.1 s |
| 2018-03-12 10-10 G423 | 29 | 7 / 7 | 5 | 17.2% | 45.5% | 71.4% | 0.1 s | 16 | 81.2% | 100.0% | 0.1 s |
| **Total** | 390 | 608 / 608 | 126 | 32.3% | 42.3% | 20.7% | 0.1 s | 34 | 58.8% | 100.0% | 0.1 s |

twin-aware:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 17-15 G299 | 189 | 416 / 416 | 81 | 42.9% | 42.9% | 19.5% | 0.0 s | 6 | 100.0% | 100.0% | 1.0 s |
| 2018-03-12 10-50 G330 | 116 | 149 / 149 | 43 | 37.1% | 37.1% | 28.9% | 0.2 s | 0 | n/a | n/a | n/a |
| 2018-03-13 15-50 G421 | 40 | 36 / 36 | 27 | 67.5% | 67.5% | 75.0% | 0.1 s | 12 | 83.3% | 83.3% | 0.1 s |
| 2018-03-12 10-10 G423 | 13 | 7 / 7 | 7 | 53.8% | 53.8% | 100.0% | 0.1 s | 16 | 100.0% | 100.0% | 0.1 s |
| **Total** | 358 | 608 / 608 | 158 | 44.1% | 44.1% | 26.0% | 0.1 s | 34 | 94.1% | 94.1% | 0.1 s |

### Oracle tracks, dev clips (same settings)

| clip | MOTA tracks | IDF1 tracks | ID sw. | MOTA persons | IDF1 persons | ID sw. | precision | recall | GT boxes with a twin |
|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 88.0% | 92.7% | 0 | 87.3% | 92.3% | 0 | 100.0% | 87.3% | 18.7% |
| 2018-03-09 10-15 G299 | 84.2% | 86.7% | 1 | 83.2% | 86.1% | 0 | 100.0% | 83.2% | 28.1% |
| 2018-03-07 11-05 G331 | 96.5% | 97.3% | 0 | 95.6% | 96.8% | 0 | 100.0% | 95.6% | 7.6% |
| **Total** | 88.1% | 91.7% | 1 | 87.3% | 91.2% | 0 | 100.0% | 87.3% | 19.8% |

strict:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 202 | 378 / 378 | 78 | 38.6% | 55.7% | 20.6% | 0.1 s | 0 | n/a | n/a | n/a |
| 2018-03-09 10-15 G299 | 50 | 25 / 25 | 10 | 20.0% | 30.3% | 40.0% | 0.1 s | 29 | 51.7% | 100.0% | 0.1 s |
| 2018-03-07 11-05 G331 | 17 | 50 / 50 | 6 | 35.3% | 85.7% | 12.0% | 0.1 s | 5 | 20.0% | 100.0% | 0.0 s |
| **Total** | 269 | 453 / 453 | 94 | 34.9% | 52.2% | 20.8% | 0.1 s | 34 | 47.1% | 100.0% | 0.1 s |

twin-aware:

| clip | GT pair intervals | ours (mapped / all) | matched | recall_all | recall_tracked | precision_mapped | median start error | enter/exit events | recall_all | recall_tracked | median abs timing error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018-03-11 16-35 G330 | 181 | 378 / 378 | 102 | 56.4% | 57.0% | 27.0% | 0.1 s | 0 | n/a | n/a | n/a |
| 2018-03-09 10-15 G299 | 45 | 25 / 25 | 14 | 31.1% | 31.1% | 56.0% | 0.1 s | 29 | 100.0% | 100.0% | 0.1 s |
| 2018-03-07 11-05 G331 | 17 | 50 / 50 | 11 | 64.7% | 64.7% | 22.0% | 0.1 s | 5 | 80.0% | 100.0% | 0.1 s |
| **Total** | 243 | 453 / 453 | 127 | 52.3% | 52.7% | 28.0% | 0.1 s | 34 | 97.1% | 100.0% | 0.1 s |
