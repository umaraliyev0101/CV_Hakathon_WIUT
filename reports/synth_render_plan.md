# Synthetic showroom render plan (full run)

Written 2026-10-11 ~04:50 (+05) after the first full scene, with **measured** numbers (estimates are
labelled). Budget from the goal prompt: render 8 h, disk 150 GB, stop if `/mnt/raid1` < 100 GB free.

## Measured cost (Blender 4.2.23, EEVEE Next 16 samples + raytracing, 1280×720, GPU 1)

| Pass | s/frame, one Blender process alone | s/frame, 3 processes in parallel | Source |
|---|---|---|---|
| RGB (real Rocketbox people, real cars, glass facade, HDRI) | **0.333** (s000 cam1 day, 450 frames) | 0.48–0.78 (s000, 6 clips) | `render_stats.json` |
| Instance mask (Workbench) | 0.089 | 0.071–0.075 | same |
| Solo mask per person (Workbench) | 0.041 | 0.043 | same |
| Scene build (room, 4 cars, people import + keying) | 3.7 s | 2.8 s | same |

- RGB throughput with 3 parallel processes: 3 / 0.62 s ≈ 4.8 fps vs 3.0 fps alone → **≈ 1.26–1.6×**
  depending on the clip mix (masks/labelling are CPU-bound and overlap well). GPU 1 memory with 3
  processes: 46 GB (limit 60 GB). Kept at 3 jobs, never 4.
- GPU actually used: `nvidia-smi pmon` saw the Blender PID on **GPU index 1** for every scene
  (`gpu_index: [1]` in each `render_stats.json`).

## Plan (priority order of the goal prompt A.6)

| Scenes | Scenario | Cameras × lightings | Output frames (10 fps) |
|---|---|---|---|
| s000–s004 | greet_fast, greet_slow, walkout, party, handoff | 2 × 3 | ≈ 18.0 k RGB |
| s007–s011 | second seed of each short scenario | 2 × 3 | ≈ 17.4 k RGB |
| s005 | crowded (7 people) | 2 × 3 | 4.5 k RGB |
| s006 | dealership_long (240 s) | 1 × 2 (day, evening_dim) | 4.8 k RGB |

**Estimate** (not measured): ≈ 45 k RGB frames at ≈ 3.5–4.8 fps aggregate ≈ 2.6–3.6 h of render plus
labelling/encode, i.e. finished around 07:30–08:30 (+05), inside the 8 h budget.

Disk: measured 468 MB of RGB PNG per 450-frame clip and ~100 MB of masks per scene. ≈ 70 clips →
**≈ 45 GB estimate** before cleanup, inside the 150 GB budget. PNGs of non-share clips can be
deleted after export and QA (`data/synthetic/clips/*/rgb`); the share pack keeps its PNGs because
`qa_render` reads them.
