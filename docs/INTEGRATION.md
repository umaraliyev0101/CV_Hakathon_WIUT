# Merging Showroom Pulse with Showroom Coach

Audience: the teammate who owns Showroom Coach. The comparison column is Coach's design spec
(`2026-10-11-showroom-coach-design.md` sections 4.2 and 4.3), not its code: both projects were built in
parallel overnight, so check each row against Coach's actual code before merging. Where Pulse differs,
the right column says exactly what Pulse does and why.

## Merge plan

1. **One perception engine and one cache.** Both products consume the same chain: `sources` ->
   `perception.detect` -> `perception.pose` -> `tracking` -> embeddings -> `identity.merge`, cached per
   video. Take Pulse's `pipeline.perceive` and `cache` (v2 layout, row below) as the single cache; Coach
   reads `PerceptionCache` (`obs`, `cars`, `segments`, `embeddings`, `raw_dets`) and builds its event
   engine on the timeline. Cache key = sha1 of the file plus sample fps, max width, max duration,
   detector and pose model, so a clip analysed in one tab is not re-detected in the other.
2. **One `roles.json`.** Pulse reads and writes the same file Coach exports. Accepted on import:
   `{"persons": [{"person_id": 3, "role": "staff", "name": "..."}]}` or a flat `{"3": "staff"}` map;
   roles are `staff` or `customer` only (anything else is an error). Export is the `persons` form
   (`roles.export_roles_json`). Precedence in scoring: user override > imported file > suggestion. Person
   ids are only meaningful for the same cache key and the same merge settings; if Coach's ids come from a
   different merge, re-export from the shared engine.
3. **One web app, two tabs: Salespeople and Customers.** Pulse is FastAPI with a static no-build
   frontend (`src/pulse/app`). Mount Coach's routes beside Pulse's under the same app (they share
   upload, job, roles and `/ws/live`), add a tab bar to `index.html`, and keep one upload per job so
   both tabs read the same cache. The unserved-interest alert in Pulse can consume Coach's greet events
   instead of its own proximity rule (spec 4.5 "Assisted"); that hook is the join point
   (`interest.episodes`).
4. **Share the unit tests** for `geometry`, `roles` and `identity`; run both products' suites against the
   merged package. Keep PROGRESS-style decision logs; this table is the checklist.

## Interfaces that differ from Showroom Coach's spec

| Interface | Showroom Coach spec | Showroom Pulse | Merge note |
|---|---|---|---|
| Core types module | Not named in the spec section 4.2 (types listed: `Det`, `TrackObs`, `Person`, `RoleSuggestion`, `Event`, `Scorecard`) | All in `src/pulse/datatypes.py` (package `pulse`); `JsonMixin.to_dict/from_dict` | Import path is `pulse.datatypes`; re-export from Coach's module or move Coach's types here. Names and fields of the five shared types are intended to be identical; diff the fields |
| Pulse-only types | n/a | `Zone`, `Sample`, `Timeline`, `ZoneVisit`, `InterestEpisode`, `Insight`, `StableSegment`, `Stats` | Additive, no clash expected; Coach has `Scorecard`, Pulse has none |
| `Det.cls` | Class label of a detection (type not stated) | A COCO **name string**: `'person'`, `'car'`, `'truck'`, `'motorcycle'`, `'bus'` | If Coach stores COCO ids, map ids to names at the boundary |
| `Zone` coordinates | n/a (Coach has no zones in its spec) | **Normalised** [0,1] image coordinates (`polygon`), so zones survive a resolution change; `source` = `auto` or `user`; `kind` = car/desk/entrance/lounge/other | Needed if Coach wants a desk or behind-counter zone (`zone_dwell`) |
| `Sample.foot` | n/a | Normalised [0,1] foot point; `pos_m` = (X, Z) metres or None; `dist_method` | |
| Render entry point | `render(video, timeline, events, out)` | `render_timeline(video, timeline, events, out, cfg)` is the Coach-shaped wrapper (cfg added). `render.render(video, overlays, out_path, cfg, observations_by_t)` is the low-level function | Rename or alias on merge; Coach callers must pass `cfg` |
| `observations_by_t` | Not in spec | **Required** argument of `render` and `contact_sheet` (privacy fail-closed): every sampled frame's `[(bbox, keypoints or None)]`; a frame with no key means no detected person. `render_timeline` derives it from timeline samples | Coach's render must pass raw detections too, or faces of untracked people are not blurred |
| Face blur coverage | Blur the face region | Blur covers every detected person: raw detections down to score 0.2 (cache v2 `raw_dets.npz`) plus tracked boxes at t and +-1 sampled frame. A person the detector misses cannot be blurred. Face box min side = 0.12 x person-box width | Same method in spirit; keep Pulse's union rule |
| Render output | H.264 mp4 via imageio-ffmpeg | Same; writes `<out>.part.mp4` then `os.replace`; frame size cropped to even dimensions | |
| `roles.suggest` | `suggest(persons, interactions) -> list[RoleSuggestion]` | `suggest(persons, interactions, cfg=None)`: extra `cfg` argument (role weights and thresholds come from `Config`) | Make `cfg` optional on Coach's side (Pulse already defaults it) |
| Role features | `outfit_peers`, `contacts`, `presence`, `initiator`, optional `zone_dwell`, optional `looks_staff` | **Reduced set**: `outfit_peers`, `contacts`, `presence`, `initiator` only; no `zone_dwell`, no `looks_staff`. `presence` is `None` for clips shorter than `presence_min_clip_s` (60 s) | Union the feature set on merge; weights live in `Config.role_weights` |
| Role "reasons" | Top contributing features, in plain words | A feature yields a reason only when its normalised value >= 0.25 (controller ruling R3), top 3 by weight x norm, so a customer with presence 0.1 has `reasons == []` | Coach may show reasons for any positive contribution |
| Role normalisation | n/a | `outfit_peers` capped at 3, `contacts` at 4, then weighted mean over the features that are present; `staff` >= 0.6, `customer` <= 0.35, else `unsure` (`unsure` counts as customer in statistics, flagged on the dashboard) | |
| Roles skip | "Skip" allowed in the UI (everyone customer) | `PUT /api/jobs/{id}/roles {"skip": true}` sets `roles_skipped`: scoring ignores suggestions, imported roles and overrides, and adds the warning "Roles skipped: everyone counted as a customer". Cleared by any non-skip PUT | Coach's skip semantics must match or the two tabs will disagree on who is staff |
| `identity.merge` | `merge(tracks) -> persons` (offline clustering, overlapping tracks never merge) | `merge(tracks, cfg, max_gap_s=None) -> list[Person]`: default `max_gap_s` = preset `leave_after` (10 s dealership, 3 s short); `merge_sim_thr` **0.9** (tuned from 0.75 on the dev clips only); `merge_max_speed` 0.25 frame widths per second; chain rule (A's latest-ending track joins B's earliest-starting, overlap <= 0.2 s); cosine on L2-normalised embeddings, missing embedding never merges | Pulse tuned on counts only; Coach should re-check the threshold against its own metric |
| `identity.reattach` | Live: re-attach to a recently lost person | `reattach(lost, new, now, cfg)`; live uses **HSV torso histograms only** (no DINOv2), person ids per session; stricter after the 0.9 change | |
| `geometry.position_m` | `position_m(obs)` -> distance basis (X, Z) | `position_m(obs, frame_size, hfov_deg=70, person_h=1.70, kp_thr=0.3, floor_h=None) -> PosEstimate(x, z, method)`; `method` is `full`, `torso`, `box`, `floor` or **`invalid`** (NaN position, e.g. non-finite foot point); an `invalid` sample has `pos_m = None` | Callers must handle `invalid` and a dataclass return, not a bare tuple |
| `geometry` extras | `distance_m`, `faces` | Also `foot_point`, `facing_dir`, `fit_homography`, `to_floor` (optional 4-point floor calibration). `faces` returns None when facing is unknown or positions coincide | |
| Foot point | Not specified | Mean of confident ankles, else box bottom-centre; used for zones and speed | |
| `perceive` | n/a (engine API unspecified) | `pipeline.perceive(video, cache_root, cfg, progress=None, engine=None, embedder=None)`; the app passes `engine.embedder` when present; `engine_factory` is called with no args (or cfg if it declares a parameter) | Stub-friendly for tests |
| Perception cache | Per-video `.npz`/`.json` (layout unstated) | **Cache v2**, `runs/cache/<key>/`: `meta.json`, `tracks.npz`, `cars.json`, `segments.json`, `embeddings.npz`, **`raw_dets.npz`** (every raw person detection with score >= 0.2, tracked or not, used only for blurring), plus `keyframe.jpg` (raw frame, local only). `exists()` needs all six data files; v1 caches are rebuilt. Written to a temp dir then renamed | Coach must write `raw_dets.npz` or its renders cannot blur untracked people |
| Cache embeddings | Torso HSV + DINOv2-small on a few crops | 3 crops per track (score >= 0.5, one per second, evenly picked), resized to 112x112 in memory; only embeddings are cached; every track with >= 1 crop gets the full HSV+DINOv2 vector; no crop -> no embedding (never merged); DINOv2 unavailable -> HSV-only for all | |
| Tracker | ByteTrack from supervision | Same, but `sv.ByteTrack(frame_rate=30)` always: supervision scales the lost-track buffer by frame_rate/30, so passing the sampling fps (5) shrank a 10-frame buffer to 1 frame. With 30 the buffer is exactly `track_buffer_s` x sample fps frames | Copy this quirk |
| `supervision` pin | Allowed (MIT) | `supervision>=0.26,<0.31` (`ByteTrack` is deprecated since 0.28 and removed in 0.31) | Pin in the merged `pyproject.toml` |
| rtmlib | `rtmlib` with YOLOX and RTMPose ONNX weights | **Vendored subset** in `src/pulse/perception/vendor_rtmlib` (Apache-2.0, `NOTICE.md`), pip package not installed: its wheel requires `opencv-python` and `opencv-contrib-python`, which clash with `opencv-python-headless`, and its multiclass YOLOX path drops scores that ByteTrack needs. onnxruntime-only `BaseTool`, local model path, `providers` argument | Merge must not reintroduce pip `rtmlib` |
| Detector | Pluggable, picked by benchmark | `yolox_s` default (M0 benchmark, recall 0.629 on 35 GT boxes of one clip); returns person plus `car`/`truck`/`motorcycle`/`bus` in one pass; car zones come from one keyframe. Pose `rtmpose_s` | Coach's detector must also emit vehicle classes if it wants car zones |
| Sampling | 5 fps, 960 px max width | Same defaults (`sample_fps` 5, `max_width` 960); longer than `max_duration_s` (300 s) is trimmed with a warning | |
| Zone statistics | n/a | Only on camera-stable, same-view segments (`camera_motion`); non-stable samples are dropped (store stats use the unfiltered timeline) | Coach's event engine should also respect `StableSegment` |
| Presets | Coach spec 4.6 | Dealership: `leave_after` 10 s, `zone_min`/`zone_gap` 1.5/2 s, `stop_time` 10 s, `interest_time` 60 s, `assist_dist`/`assist_time` 2.0 m/5 s, `alert_after` 45 s, `desk_time` 30 s, `min_sample` 5. Short: 3 s, 1/1 s, 3 s, 10 s, 2.0 m/2 s, 8 s, 5 s, 2 (demo only) | Preset names and `leave_after` semantics (already-present exclusion) are meant to match |
| CLI | `analyze <video> --out <dir> [--preset] [--roles roles.json]`, `eval`, `bench` | `pulse analyze VIDEO --out DIR --preset {short,dealership} [--zones] [--roles] [--no-video]`, `pulse eval --out docs/EVAL.md`, `pulse bench detectors|speed` | `--zones` is Pulse-only |
| REST API | upload, job status, people/roles, results, export, `/ws/live` | `POST /api/jobs`, `GET /api/jobs/{id}`, `GET\|PUT .../zones`, `GET .../people`, `PUT .../roles`, `POST .../roles/import`, `POST .../score`, `GET .../results`, `.../export.json`, `.../export.csv`, `.../keyframe.jpg`, `.../files/{path}`, `GET /api/health`; pages `/` and `/live`. Zones and role overrides live outside `out/` so regenerating a report never deletes them | Route names may collide on merge; prefix Coach's under `/api/coach/` if needed |
| Host/Origin guard | Not in spec | `HostOriginGuard` middleware: HTTP needs Host in {localhost, 127.0.0.1, [::1], testserver, plus `PULSE_ALLOWED_HOSTS`} else 400; non-GET/HEAD/OPTIONS with a foreign Origin -> 403. `run.sh` binds 127.0.0.1 | Keep it for the merged app (DNS-rebinding and cross-site POST protection) |
| Upload and job limits | n/a | 1 GB upload limit (413 from Content-Length before spooling); jobs and uploads deleted after 24 h on app start; zones <= 50 and <= 200 points per polygon; zone file load only from `labels/zones/` or the jobs dir, .json <= 1 MB | |
| `/ws/live` transport | Webcam frames over a WebSocket | Client sends **binary JPEG frames** (limit 2 MB; `decode_jpeg` requires a JPEG SOI, rejects > 16 MP or > 8192 px per side) and **text JSON control messages** (limit 64 KB). Server `t` is its own monotonic clock since connect, never client time. One live session at a time (second client gets an error message and close 1013). uvicorn must run with `--ws-max-size 4194304` (`main.WS_MAX_SIZE`) | Idle 120 s -> close 1001; 5 bad messages in a row -> close 1009; bad Host or foreign Origin -> close 1008 before accept |
| `/ws/live` client -> server messages | Not specified | `{"type":"zones","zones":[...]}` (replace zones), `{"type":"load_zones","path":"labels/zones/<clip>.json"}` (server replies `{"type":"zones",...}`), `{"type":"preset","name":"short"\|"dealership"}`, `{"type":"toggle","x":0..1,"y":0..1}` (flip staff/customer of the person at that normalised point) | |
| `/ws/live` server -> client messages | Not specified | `{"type":"overlay","t","frame_size":[w,h],"persons":[{"person_id","bbox","role","zone_id","zone_name","dwell","alert"}],"zones":[{"zone_id","name","now","total"}],"occupancy":N,"alerts":[{"person_id","zone_id","dwell","message"}],"fps","dropped","step_ms"}` and `{"type":"error","message"}`. The reader keeps only the newest unprocessed frame (`dropped` counts the rest) | Coach's live overlay needs its own message kinds (greet, wait alert); add them as new `type` values |
| Live privacy | Faces blurred in outputs | Live mode stores no frames. The local preview pixelates each head region client-side; server never sends video back | |
| Live engine | One online engine for both modes | Live owns its own lazily-created engine, serialised behind a lock; person ids are per session | One shared engine instance is the merge goal; keep the lock |
| Event vocabulary | `events` unit: greet, engagement, wait_alert, walkout, party, handoff | No `events` unit and no `Scorecard`/`metrics`. Pulse has `journey` (ZoneVisit), `interest` (assisted / unserved episodes), `stats`, `insights` (R1-R7). `Event.kind` strings containing `unserved` or `alert` outline a person red in `render_timeline` | Interest "assisted" can be fed by Coach's greet events (spec 4.5) |
| Party / group rule | arrive within 5 s, <= 1.5 m for most of first 30 s | Same rule, used for group statistics only | |
| Report outputs | Scorecard per salesperson and store | `report.build_report`: `report.json`, `per_car.csv` (= `export.csv`), heatmap, journey flow, charts, thumbnails, contact sheets, annotated mp4. Reports store the video basename only | |
| Privacy of thumbnails | Face-blurred | Same; thumbnails and keyframe previews are blurred with the exact sampled-time detections; an image of a different aspect is resized to the analysis frame (grey, never unblurred) | |
| Webcam check | Grab one frame, assert not blank | `scripts/webcam_grab.py` checks the raw frame in memory and saves **only a 24-px pixelated copy** to `runs/verify/webcam.jpg` | Adopt the same storage rule |

## What stays out of the merge

Per-salesperson scoring stays in Coach; per-car and store-level customer analytics stay in Pulse. Pulse
never ranks individual staff (spec section 13).
