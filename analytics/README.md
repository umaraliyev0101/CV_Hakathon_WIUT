# analytics/: person-to-car distance and dwell time

These metrics need car **positions**, not per-frame car detection: cameras are fixed and parked cars
don't move, so each car footprint is marked once per camera (synthetic clips: rebuilt from
`sim/configs/scene_NNN.json`; real install: drawn at setup, optionally pre-filled by the stock COCO
`car` class).

| File | What it does |
|---|---|
| `floor.py` | `Camera` (K, R, t from `gt.json`): pixel ↔ floor metres, near-plane-clipped polygon projection. `person_floor_xy(cam, box)`: feet = box bottom-centre; if the box implies a person shorter than 1.4 m (legs hidden by a car or desk) it projects the head onto the plane at 1.7 m instead. `distance_to_polygon`. |
| `dwell.py` | `dwell(samples, cars, dt)`: per (track, car) visits while within `NEAR_M` = 1.5 m of the car body; gaps ≤ 1 s are bridged, visits < 2 s (walking past) dropped. |
| `synthetic.py` | `car_footprints(gt)`: car bodies for a synthetic clip; fails loudly if the config no longer matches the rendered zones. |
| `validate_synthetic.py` | Box-based positions and dwell vs exact simulator `pos_m`. |
| `overlay.py` | Review/demo videos → `data/synthetic/review_videos_zones/<clip>_zones.mp4`. |

## Measured on synthetic GT (61 clips, 100k person-frames)

`.venv/bin/python -m analytics.validate_synthetic`

| Person visible | Frames | Median error | p90 error |
|---|---|---|---|
| ≥ 90 % | 92,301 | 0.18 m | 0.28 m |
| 60–90 % | 5,814 | 0.18 m | 0.88 m |
| < 60 % | 1,895 | 0.18 m | 0.52 m (was 4.18 m before the head fallback) |

Dwell, 198 (person, car) pairs: GT total 6,333 s vs box-based 6,242 s; per-pair error median 0.2 s,
p90 0.8 s. These use the simulator's exact boxes; with detector boxes the error will be larger.

Tests: `.venv/bin/python -m pytest analytics/tests -q`.
