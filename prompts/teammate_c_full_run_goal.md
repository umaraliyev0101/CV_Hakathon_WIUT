# Goal-run prompt for teammate C: full synthetic showroom pipeline and baseline detector

Written 2026-10-11 03:15 (+05). It builds on the quick test (`reports/synth_quicktest.md`, commit
`b33639a`), where all gates pass on stand-in assets.

**How to launch:** start Claude Code in `/mnt/raid1/projects/CV_Hakathon_WIUT`. Paste §A as the
message and §B as the `/goal` condition. Before launching, a human should do the pre-flight in §0;
every item there is optional.

---

## 0. Human pre-flight (optional, about 20 minutes; each item makes the result more realistic)

The agent cannot do these, because they need an account or a browser click.

- [ ] **Mixamo** (free Adobe ID). Put files in `assets/mixamo/`:
  - `characters/`: about 10 characters, FBX "With Skin", T-pose. Mix genders, skin tones and body
    types. Include 2–3 in business wear.
  - `anims/`: FBX "Without Skin", with "In Place" ticked for locomotion. Walking, Slow Walking,
    Idle ×2, Talking ×2, Looking Around, Pointing, Waving, Turn Left 90, Turn Right 90.
- [ ] **Cars**: 4–6 `.glb` files in `assets/cars/` from Sketchfab (CC BY, downloadable). Write the
  URL and author of each in `assets/MANIFEST.csv`.
- [ ] Have **A's spec** `salespeople_prompts/2026-10-11-showroom-coach-design.md` copied into the
  repo. Without it, the agent keeps the fallback rules and flags them.

If you skip these, the agent uses the CC0 fallback ladder in §A.4.

---

## A. Message to paste

You are continuing teammate C's synthetic-showroom work in this repo. Read these first:

1. `prompts/teammate_c_synthetic_showroom.md`. This is **the authoritative spec**: §1–§7 cover
   scenes, people, behaviour, cameras, rendering, labels, the CCTV look, stages S0–S7, gates,
   the visual review, the fix loop and the rules.
2. `reports/synth_quicktest.md`. This is what already exists: the S0–S2 skeleton, stand-in assets,
   measured s/frame, open flags and known residuals (pivot-turn foot shuffle, the 0.83 s
   rule-vs-scripted greet offset).
3. `research_notes/open_footage.md`, a survey of open real footage and datasets.

Your mission: take the pipeline from the quick test to **"Done" in spec §7**. Then add the P0
baseline detector wrapper from §A.5. Everything in the spec applies unless this message overrides
it.

### A.1 Filled-in placeholders

| Placeholder | Value |
|---|---|
| BLENDER_PATH | `~/opt/blender-4.2.23-linux-x64/blender` (4.2.23 LTS; found from `$BLENDER`, then `PATH`, then `~/opt/blender-*`) |
| Machine | Ubuntu 25.10, Linux 6.17. 2× RTX PRO 6000 Blackwell (96 GB each). 128 cores, 122 GB RAM. `/mnt/raid1` has about 666 GB free and is 91% full. |
| **GPU** | **Use GPU 1 only.** GPU 0 runs other people's vLLM and sglang services; never touch it. |
| DEADLINE | **2026-10-12 06:00 (+05)**. The demo is later that day. |
| MAX_HOURS | 20 h of wall-clock |
| RENDER_BUDGET_HOURS | 8 h |
| DISK_BUDGET_GB | 150 GB. Stop rendering if `/mnt/raid1` falls below 100 GB free (stricter than the spec's 10 GB, because the disk is shared). |
| MAX_TURNS | 400 |

### A.2 GPU 1 pinning (check it, don't assume it)

- Export `CUDA_VISIBLE_DEVICES=1` for every torch, onnxruntime-gpu or Cycles process. Put this in
  `run_all.py` and `verify.py` (set in the environment of the subprocesses), not only in your
  shell.
- Blender EEVEE runs headless through EGL. On 2026-10-11, `nvidia-smi pmon -c 1` showed a headless
  EEVEE render on **GPU index 1**, both with and without `CUDA_VISIBLE_DEVICES`. Re-check this
  once in S0 while a render runs, and record the result in `render_stats.json` (`gpu_index`) and
  the status file. If it ever lands on GPU 0, find the EGL device-selection mechanism and fix it
  before rendering at scale.
- Use parallel Blender processes on GPU 1 only if the measured throughput gain is above 1.3×.
  Measure it; don't assume it. Keep GPU 1 memory use under about 60 GB, because another vLLM
  process is also on it.

### A.3 Overrides and clarifications to the spec

- **qa.json:** you may **add** keys and new gates. You may make `expected_frames` per clip, derived
  from that clip's config, as long as the check stays an exact equality. You may **not** lower,
  loosen or delete any existing threshold. The quick-test values are the floor.
- **A's spec missing:** if `salespeople_prompts/…` is still absent, keep `presets.json` on the
  fallback rules and flag it. Build the A-format labels as a documented adapter from `gt.json`, and
  flag that too, as spec §4.6 says. For engagement, walkout, party and handoff, use the thresholds
  in spec §4.3 and the quick-test prompt (`prompts/teammate_c_synthetic_showroom_quicktest.md`).
  Put each one in `presets.json` with a `"_source"` note. Never put a threshold inside the code.
- **Greet offset:** decide on the "arrival vs entering range" definition, document it, and keep
  every scripted greet within ±1.0 s of the rule greet in every scenario. Do this through
  choreography, not tolerance.
- **Pivot-turn shuffle:** fix it with in-place turn clips (Mixamo Turn 90, or a procedural
  step-turn) so the visual review's "sliding" item passes on every shared clip.

### A.4 Asset fallback ladder (use the highest rung available, and record it in MANIFEST.csv)

**People:**
1. Mixamo in `assets/mixamo/` (human pre-flight).
2. CC0 rigged humans with walk, idle and gesture animations you can download without an account.
   For example, MakeHuman/MPFB2 (CC0 output) driven by CMU mocap BVH (free to use), or a
   Quaternius CC0 character and animation pack. Verify the licence on the source page before
   downloading, and record the URL.
3. The existing procedural mannequins. Improve them with clothing-coloured segments, a head and
   hair, and hands. If you end up here, mark the realism gates "blocked by assets". qa_detector is
   likely to fail on capsules, and that is an honest result. Do not tune the gate around it.

**Cars:**
1. `assets/cars/*.glb` (human pre-flight).
2. CC0 car models from Poly Haven, Kenney or Quaternius (no account needed).
3. The existing bevelled box stand-ins.

**HDRI and textures:** Poly Haven public API (CC0). Downloading is allowed. Spend no money, create
no accounts, upload nothing.

### A.5 P0 baseline detector, which unblocks A and B (from the training prompt)

- `models/detect.py`: `detect(frame) -> list[Det(bbox, score, cls)]` with a **single person
  class**, using COCO-pretrained **RF-DETR Nano** (Apache-2.0, `pip install rfdetr`) exported to
  ONNX (`models/showroom_detector.onnx`). Run it with onnxruntime. CPU is the target runtime;
  GPU 1 is allowed for QA speed.
- A's code is absent, so define `Det` as a dataclass in `models/detect.py` and flag that it must be
  reconciled with A's.
- `models/README.md`: the input tensor name, shape and normalisation, the output format, the
  sha256 and the licence. Measure CPU FPS at 720p, with the thread count stated.
- Use this same detector for `qa_detector`.
- Download the Pexels clips 7154214, 7154217, 7154241 and 36512656 into `data/real_open/`. They
  are listed in `research_notes/open_footage.md`. The Pexels licence allows free commercial use
  and needs no account; use the direct download link on each page. If a download needs a login,
  skip that clip and say so. Record each clip in `DATA_SOURCES.md`. These clips are for the demo
  and for evaluation only; do not train on them.
- Run `detect.py` on 3 frames from them. Save the annotated frames to
  `reports/real_open_detect_*.jpg` and look at them.
- **Out of scope for this run:** fine-tuning on real staged clips. It needs human labelling in
  CVAT. Only make the synthetic COCO export (spec §4.6) ready for it. A synthetic-only fine-tune
  is optional and only allowed if everything in §B is already green. Evaluate it only on a held-out
  set that is real and human-labelled; never on synthetic data.

### A.6 Priorities when time runs short

Rank work in this order:

1. Scenario coverage: all 5 short-clip scenarios in the share pack, with correct events.
2. Label correctness.
3. Real human assets.
4. Lighting variants.
5. Camera count.
6. Volume.

`dealership_long` and `crowded` come after the 5 short-clip scenarios. One camera and one lighting
variant per scenario is acceptable for the share pack if the budget is tight.

### A.7 Working rules

- Spec §6.3 (fix loop) and §7 (rules) apply in full. In short:
  - Find the root cause.
  - Never hand-edit outputs or labels.
  - Don't weaken gates.
  - Report only measured numbers, and label estimates as estimates.
  - Commit locally at each stage; never push.
- Use `/tmp/...` scratch space or `data/` for intermediates. Keep `assets/`, `data/`, renders and
  weights out of git.
- Look at every contact sheet with the image-reading tool before marking a visual item as pass.
  Cite frame numbers.
- Same gate, same error, 3 attempts in a row: record it, move to independent work, or stop with a
  status file.

---

## B. `/goal` condition

```
Run `python sim/verify.py --all` end to end in one go and show its full output in the
conversation; it exits 0. That chain runs pytest on sim/tests, then qa_assets, qa_scene,
qa_render, qa_events, qa_detector and qa_video over every clip in data/synthetic/share/.
data/synthetic/share/ holds at least 5 clips covering greet_fast, greet_slow, walkout, party and
handoff, each with gt.json, A-format labels and a README; one is marked as the hero clip.
COCO (train/valid/test split by scene), MOT and A-format exports exist under data/synthetic/.
`cat reports/synth_visual_review.md` is shown: it lists every shared clip with all 12 checklist
items marked pass, with frame numbers cited.
`models/detect.py` runs on a frame from data/real_open/ and prints person detections, shown in the
conversation; models/README.md records the measured CPU FPS.
reports/synth_status.md (including the measured s/frame, which GPU index rendered, dataset stats
and resume commands), sim/README.md and DATA_SOURCES.md exist and are shown.
`git status --short` shows no tracked changes, and every commit is local (no remotes added).
No test or QA check was skipped, marked xfail, deleted or mocked, and no existing threshold in
sim/configs/qa.json was lowered or removed: `git diff b33639a -- sim/configs/qa.json` is shown
and contains only additions or per-clip expected_frames.
Hard stop: at 2026-10-12 06:00 +05, after 20 h wall-clock, after 400 turns, or when the same gate
has failed with the same error 3 turns running. Before stopping, write reports/synth_status.md
(what works, last verify.py output, open problems, resume commands). That counts as
finished-with-status, not success.
```
