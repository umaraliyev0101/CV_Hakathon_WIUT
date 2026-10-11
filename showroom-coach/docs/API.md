# Showroom Coach — web API contract (binding for Tasks 4.1, 4.2, 5.1, 5.2)

All JSON. Times are seconds from clip start (floats). Person ids (`pid`) are ints in Python; in JSON object KEYS they are strings ("3"), in values they are numbers.

## Job lifecycle
`queued → perceiving → analyzing → needs_confirm → rendering → done`, or `failed` at any point.
- After `needs_confirm`, `POST /roles` may be called any number of times (each call re-scores instantly and re-renders the video in the background; status goes `rendering` → `done`).

## Endpoints

### `GET /api/presets`
```json
{"dealership": {"label": "Dealership (SKAIVISION thresholds)", "thresholds": {"greet_dist": 1.5, "greet_time": 5.0, "engage_dist": 2.0, "engage_time": 120.0, "gap_tol": 3.0, "wait_alert": 60.0, "leave_after": 10.0, "...": "..."}},
 "short": {"label": "Short clip (10–60 s)", "thresholds": {"...": "..."}}}
```

### `POST /api/jobs` (multipart: `file`, form field `preset` = "short" | "dealership")
→ `200 {"job_id": "a1b2c3d4e5f6"}` · `415` bad extension (allowed: mp4 mov avi mkv webm m4v) · `400` unknown preset · `413` file > 1 GB · `411` no Content-Length / chunked upload.

### `GET /api/jobs/{id}`
```json
{"id": "a1b2c3d4e5f6", "filename": "clip.mp4", "preset": "short",
 "status": "perceiving", "progress": 0.42, "message": "detecting people… 25 s of 60 s",
 "error": null, "warnings": ["video trimmed to the first 300 s"],
 "video_status": "none"}
```
`video_status`: `none | rendering | ready | failed`. `404` unknown id.

### `GET /api/jobs/{id}/people` (409 before `needs_confirm`)
```json
{"people": [
  {"pid": 1, "thumb_url": "/api/jobs/a1b2/files/people/1.jpg",
   "suggestion": {"role": "staff", "confidence": 0.81, "reasons": ["same outfit as 2 others", "talked to 4 people", "on screen 95% of the clip"]},
   "role": "staff", "name": null, "t_first": 0.0, "t_last": 59.8}
]}
```
`role` = current final role (user override, else suggestion with `unsure` → `customer`).

### `POST /api/jobs/{id}/roles`
Body: `{"roles": {"3": "staff", "5": "customer"}, "names": {"3": "Aziz"}, "preset": "short"}` (all keys optional; omitted pids keep their current role; `preset` optional).
→ the full **results** object (below). Re-scoring never re-runs detection.

### `GET /api/jobs/{id}/results` (409 before `needs_confirm`)
```json
{"preset": "short", "duration": 60.0,
 "thresholds": {"greet_dist": 1.5, "...": "..."},
 "persons": [{"pid": 1, "track_ids": [3, 9], "t_first": 0.0, "t_last": 59.8, "n_obs": 290, "thumb": "1.jpg"}],
 "suggestions": {"1": {"pid": 1, "role": "staff", "confidence": 0.81, "reasons": ["..."], "features": {"outfit_peers": 2, "contacts": 4, "presence": 0.95, "initiated": 2}}},
 "roles": {"1": "staff", "2": "customer"},
 "names": {"1": "Aziz"},
 "events": [{"type": "greet", "t": 9.0, "customer": 2, "staff": 1, "t_end": null, "data": {"time_to_greet": 4.0, "via": "direct"}}],
 "scorecard": {
   "staff": [{"staff": 1, "name": "Aziz", "greets": 3, "median_time_to_greet": 6.2, "customers_engaged": 2, "engaged_total_s": 140.0, "engaged_median_s": 70.0, "missed_while_free": 1, "walkouts_while_free": 0, "sold": null, "conversion": null}],
   "store": {"customers": 5, "customers_entry_known": 4, "greeted": 4, "greet_rate": 0.8, "median_time_to_greet": 7.5, "walkouts": 1, "unattended_wait_s": 63.0, "wait_alerts": 2, "engagements": 3}},
 "timelines": {"2": {"present": [[5.0, 40.0]], "waiting": [[5.0, 9.0]], "engaged": [[9.0, 40.0]], "greeted": 9.0}},
 "warnings": ["distance is approximate for 62% of samples"],
 "video_status": "ready",
 "video_url": "/api/jobs/a1b2/files/annotated.mp4",
 "timeline_url": "/api/jobs/a1b2/files/timeline.png",
 "export": {"json": "/api/jobs/a1b2/files/results.json", "events_csv": "/api/jobs/a1b2/files/events.csv", "scorecard_csv": "/api/jobs/a1b2/files/scorecard.csv"},
 "contact_sheets": [{"url": "/api/jobs/a1b2/files/contact_sheets/01_greet_9s.jpg?v=1", "type": "greet", "t": 9.0, "caption": "Greet at 0:09"}],
 "sales": {"rows": 12, "unmatched": ["Karim"], "skipped": 1}}
```
`contact_sheets` lists the event contact sheets (three face-blurred frames around a greet, engagement, wait_alert or walkout; at most 12) of the latest finished render, in time order; it is `[]` before the first render and while a render is running. `type` and `t` (whole seconds) come from the file name `{n:02d}_{type}_{t:.0f}s.jpg`; `caption` is "Greet / Engagement / Long wait / Walk-out at m:ss". The `?v=<render revision>` query busts the browser cache, since a re-render rewrites the sheets under the same names. Not part of `results.json`.
`sales` is present only after a sales CSV was added (see below), and then each staff card also carries `sold` (count of rows with sold = true) and `conversion` (`sold / customers_engaged`; null when `customers_engaged` is 0). A salesperson who appears in no row has `sold` and `conversion` null. Without a sales CSV `sales` is absent and both fields are null. `scorecard.csv` has `sold,conversion` columns (blank when absent); `results.json` carries the same `sales` object.
Event types: `enter` (data.entry_known), `leave` (data.greeted, data.entry_known, data.detected_at), `greet` (data.time_to_greet|null, data.via = direct|engagement|party, data.confirmed_at), `engagement_start`, `engagement` (t..t_end), `wait_alert` (data.wait_s, data.free_staff [pids]), `walkout` (data.free_staff), `party` (data.members), `handoff` (data.from_staff, data.to_staff).
Warning strings (exact): `"nobody detected — the camera may be too far or too dark"`, `"fewer than 2 people — store metrics only"`, `"nobody marked as staff — mark staff to get scorecards"`, `"distance is approximate for {p}% of samples"`, `"video trimmed to the first {n} s"`, `"video ended at {x:.0f} s of an expected {y:.0f} s — the file may be damaged or have a longer audio track; results cover the first {x:.0f} s"`.
A video whose decoding stops before 90% of the length its container promises (damaged, cut off, or an audio track longer than the picture) does not fail the job: the part that decodes is analysed, `duration` is where the picture ended (last decoded frame + one 0.2 s sample interval), and the "video ended at" warning appears in the job status `warnings`, the results `warnings` and `results.json`. A file that cannot be opened or has no frame at all still fails with `error` "cannot read video: …".

### `POST /api/jobs/{id}/sales` (optional sales CSV; multipart: `file`)
CSV with a header row `salesperson,timestamp,sold` (`timestamp` optional free text and unused; `sold` = 1/0/true/false/yes/no, any case; extra columns ignored; comma, semicolon or tab separated; UTF-8, with or without BOM; at most 10,000 data rows). Accepted when the file name ends in `.csv`/`.txt` or the type is `text/csv`, `application/csv`, `application/vnd.ms-excel` or `text/plain`.
→ `200` the full **results** object, re-scored with the sales joined: rows are matched to staff by name (the names set through `POST /roles`; case and whitespace ignored) or by `S<pid>`; `sales.rows` = valid rows kept, `sales.skipped` = invalid rows left out (no salesperson, unreadable `sold`, too few cells), `sales.unmatched` = salespeople (first spelling, once each, at most 50) who match no staff member. Two staff with the same name match nobody by name (use `S<pid>`). A second upload replaces the first. The rows are kept on the job and re-applied on every `GET /results` and `POST /roles`; no video re-render happens.
`400` no `file` field, not UTF-8, no usable header, or more than 10,000 rows (the earlier sales rows are kept) · `404` unknown job · `409` before `needs_confirm` · `411` no Content-Length / chunked · `413` larger than 1 MB · `415` not a CSV file.

### `DELETE /api/jobs/{id}/sales`
Forget the sales rows. → `200` the full **results** object without `sales` (sold and conversion null again); also `200` when there were none · `404` unknown job · `409` before `needs_confirm`.

### `GET /api/jobs/{id}/files/{name}` — whitelist only
`annotated.mp4`, `timeline.png`, `results.json`, `events.csv`, `scorecard.csv`, `people/<int>.jpg`, `contact_sheets/<file>.jpg`. Anything else → 404.

## Live WebSocket `/ws/live?preset=short`
- client → server **binary**: one JPEG frame (≤ 960 px wide). Server replies with exactly one JSON text message (overlay). Client sends the next frame only after the reply (stop-and-wait).
- client → server **text** JSON controls (no reply unless noted): `{"type":"toggle","pid":3}`, `{"type":"name","pid":3,"name":"Aziz"}`, `{"type":"preset","name":"dealership"}`, `{"type":"time","t":1.2}` (timestamp override for the next frame — tests), `{"type":"summary"}` → reply `{"type":"summary", "scorecard": {...same shape as above...}, "events": [...all so far...]}`.
- Bad JPEG (or a frame > 2 MB) → `{"type":"error","error":"bad frame"}`; an internal failure while processing a frame → `{"type":"error","error":"frame could not be processed"}`. Both are the reply to that frame — release the in-flight flag.
- Bad control (unknown pid for toggle/name, unknown preset name, malformed JSON) → `{"type":"error","error":"<reason>"}` sent as an EXTRA message — it is NOT the reply to a frame; the client must only release its in-flight frame flag on `type == "overlay"` or on `error == "bad frame"`.
- Overlay:
```json
{"type": "overlay", "t": 12.4, "w": 960, "h": 540, "fps": 4.8,
 "people": [{"pid": 3, "bbox": [x1, y1, x2, y2], "face": [x1, y1, x2, y2], "role": "staff", "suggested": "staff",
             "confidence": 0.81, "override": false, "state": "free", "wait_s": null, "label": "S3 Aziz"}],
 "faces": [[x1, y1, x2, y2]],
 "pairs": [[3, 5, "engaged"]],
 "events": [{"type": "greet", "t": 11.0, "customer": 5, "staff": 3, "t_end": null, "data": {}}],
 "alerts": ["0:42 long wait: customer C7 (free: Aziz)"],
 "store": {"customers": 2, "greeted": 1, "walkouts": 0}}
```
`faces` = the box-top face rect (top 20% of the box, full width) of EVERY detected person in this frame, including those listed in `people` (whose `face` is the keypoint region, which a pose error can misplace) — the client must blur `people[].face` AND every rect in `faces`. People listed in `people` before they are confirmed (seen in ≥ 3 frames spanning ≥ 1 s) have `"state": null` and are not counted in events yet.
`state` for customers: `waiting | greeted | engaged | browsing`; staff: `busy | free`; `null` = tentative. Coordinates are in the processed frame (`w`×`h`); the client scales to its canvas.
