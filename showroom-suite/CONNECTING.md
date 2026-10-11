# Connecting Showroom Suite to the real Coach and Pulse servers

This guide is for whoever runs the two apps. It explains how to make each section of the site show real numbers
instead of demo data.

The site is `showroom-suite/studio/`: plain HTML, CSS and JavaScript, with no build step and no framework. Out of the
box every section shows **demo data**, and each section says so in a small chip at its top right. You connect the
sections one at a time. A connected section's chip turns green and reads **Live · Coach** or **Live · Pulse**.

## 1. Run the site

```bash
# from the repo root
python -m http.server 8080 -d showroom-suite/studio
# open http://localhost:8080
```

Opening `index.html` straight from disk also works for the demo. Once you connect the real servers, serve it over
HTTP as shown above, because browsers treat `file://` pages as a different origin and block the calls.

## 2. The three files you touch

| File | What it is | Do you edit it? |
|---|---|---|
| `js/config.js` | Server addresses and one on/off flag per section | Yes |
| `js/data.js` | One function per piece of data. The `Live` object at the top is where the real calls go | Yes |
| `js/mock.js` | The demo simulation and numbers | No. It stays as the fallback |
| `js/app.js`, `js/floor.js`, `css/suite.css`, `index.html` | The views and the floor drawing | No |

Views never call a server themselves. They call `Suite.data.getTeam()` and similar functions, and each function
returns one fixed **shape**, the same in demo and live mode. Your job for each section is to return that shape from
the real server.

If a live call fails, the section does not go blank. Its chip turns red ("Server not reachable · showing demo
data"), hovering the chip shows the error, and the browser console logs a `[suite]` warning. **Before presenting,
check that every chip you expect to be green is green.**

## 3. Before anything else: let the browser reach the servers

The site runs on a different port from the apps, so the browser makes **cross-origin** requests. Both servers have
to allow them, or every call fails with a CORS error in the console.

Both apps are FastAPI. Add this to each `app/main.py` after `app = FastAPI(...)`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://<demo-machine-ip>:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Pulse also refuses unknown `Host` headers. Start it with the demo machine's address listed:

```bash
PULSE_ALLOWED_HOSTS=localhost,<demo-machine-ip> .venv/bin/uvicorn pulse.app.main:app --host 0.0.0.0 --port 8001 --ws-max-size 4194304
```

Then set the addresses in `js/config.js`:

```js
coachUrl: "http://localhost:8000",
pulseUrl: "http://localhost:8001",
```

If the demo runs from another laptop, you don't need to edit anything. Put the addresses in the link instead:
`http://<ip>:8080/?coach=http://<ip>:8000&pulse=http://<ip>:8001&live=team,visitors`

## 4. Connect one section

Every section follows the same steps. Here they are for **Team**, the simplest one.

**a. Find the data on the server.** Open Coach's API docs at `http://localhost:8000/docs` (FastAPI serves them
automatically) and find the endpoint that returns per-salesperson results. Note its path and look at what it
returns.

**b. Write the function in `data.js`.** Replace the `todo(...)` line in `Live`. Use `get(base, path)` for GET,
`send(base, path, body)` for POST, and `mmss(seconds)` to format times. Convert the server's JSON into the shape written in the comment above the
function:

```js
getTeam: async () => {
  const r = await get(C.coachUrl, "/api/<the-scorecard-endpoint>");   // <- real path from /docs
  return {
    period: `Today · ${r.people.length} salespeople`,
    lead: [
      { label: "Top score", value: [...r.people].sort((a, b) => b.score - a.score)[0].name, text: "Highest score this session." },
      { label: "Team median greet", value: mmss(r.median_greet_s), text: "Target is 1:00." },
      { label: "Engaged visits", value: `${r.engaged} of ${r.visits}`, text: "Within 2 m for over 2 minutes." },
      { label: "Walk-outs", value: String(r.walkouts), text: "Left without being greeted." },
    ],
    staff: r.people.map((p) => ({
      id: p.track_id, name: p.name || `Person ${p.track_id}`, since: "",
      greets: p.greets, medianGreetSec: p.median_greet_s, engaged: p.engaged,
      walkouts: p.walkouts, score: Math.round(p.score), trend: p.history || [p.score, p.score],
    })),
    suggestions: (r.staff_candidates || []).map((c) => ({ trackId: c.track_id, reason: c.why, confidence: c.p })),
  };
},
```

The field names on the right (`r.people`, `p.median_greet_s` and so on) are **examples**. Use whatever the real
endpoint returns. The names on the left are fixed, because the view reads them.

**c. Switch the section on.** In `config.js` set `live: { team: true, ... }`, reload, and check that the chip
says **Live · Coach**.

**d. If the server has no endpoint with that data**, add a small read-only one that returns exactly the shape. That
is often quicker than converting on the client. Keep these under a prefix such as `/suite/...` so it's clear they
exist for this site.

## 5. What each section needs

The full shapes are in the comments at the top of `js/data.js`. All times are in **seconds** and all shares are
**0–1**.

| Section | Flag | Functions | Comes from | Notes |
|---|---|---|---|---|
| Team | `team` | `getTeam`, `confirmStaff` | Coach | `confirmStaff(trackId, isStaff)` is the one-click staff confirmation. Point it at Coach's confirm endpoint. |
| Visitors | `visitors` | `getVisitors` | Pulse | `heat` is a grid: `{cols, rows, values}`, row-major, any non-negative scale. If Pulse has a heatmap image instead of numbers, downsample it to about 48×32. `zones`, `journeys` and `unserved` map onto Pulse's zones, journeys and unserved-interest output. |
| Replays | `sessions` | `getSessions`, `analyzeVideo`, `getSession` | Both | See "Uploads" and "Session results" below. |
| Demand | `demand` | `getDemand` | Pulse, plus Coach | Marketing charts, built from aggregates over a period (for example the last 4 weeks). **Funnel:** counts per stage: came in (Pulse entries) → 20 s+ at a car → 60 s+ at a car (Pulse dwell) → within 2 m for 2 min+ (Coach engagements) → reached reception (Pulse zone). **Busy hours:** average visitors per weekday × hour. **Models:** per car, weekly visits with 60 s+ dwell and the share approached. **Campaign:** weekly considered visits per car, plus the week index when a campaign started (enter that by hand). The "Next step" sentences are computed from these numbers, so you only send data. The optional `fix` text on a funnel stage is shown when that stage loses the most people. |
| Live floor | `floor` | `getFloorplan`, `getLive` | Both | See "Live floor" below. |
| Insights | `insights` | `getInsights` | Both | Pulse already has `insights.py`. Combine its findings with Coach's scorecard. Each item needs `value` (the big number), `text` (one sentence) and `act` (one next step). If you run out of time, leave this section in demo mode, but don't present it as real numbers. |

### Uploads (Replays)

`analyzeVideo(file, onStep)` receives the browser `File`. Send it to **both** servers, so the scorecard and the
customer journeys come from the same video:

```js
analyzeVideo: async (file, onStep) => {
  const form = () => { const f = new FormData(); f.append("file", file); return f; };
  onStep(0, "Uploading");
  const [coachJob, pulseJob] = await Promise.all([
    send(C.coachUrl, "/api/<upload-endpoint>", form()),
    send(C.pulseUrl, "/api/<upload-endpoint>", form()),
  ]);
  onStep(1, "Following each person across frames");
  // poll both job-status endpoints until done; call onStep(2, ...) and onStep(3, ...) as they progress
  ...
  return { id: coachJob.id, name: file.name, recorded: "Just now", lengthSec, customers, medianGreetSec, status: "ready" };
},
```

`onStep(index 0–3, label)` drives the four-segment progress bar. Call it whenever the jobs move forward, so the
bar never looks frozen.

### Session results (`getSession`)

Clicking a session, or finishing an upload, opens `#session/<id>`: the video with a timeline of moments under it,
then that session's numbers. `getSession(id)` returns everything the page shows (shape in `js/data.js`).

**Demo mode** plays the user's own upload with invented moments and numbers on top. The page labels them three
ways: a "Demo results" banner, a stamp on the video, and a "demo" tag on each caption. In live mode return
`demo: false`, and none of these labels appear.

**One id for two jobs.** An upload creates a Coach job and a Pulse job. Have `analyzeVideo` return
`id: `${coachJob.job_id}~${pulseJob.job_id}``, and split it again in `getSession`:

```js
getSession: async (id) => {
  const [cid, pid] = id.split("~");
  const [c, p] = await Promise.all([
    get(C.coachUrl, `/api/jobs/${cid}/results`),       // Coach: events, scorecard, video_url
    get(C.pulseUrl, `/api/jobs/${pid}/results`),       // Pulse: per_car, journey_flow, files.annotated_video
  ]);
  const at = (base, path) => (path ? base + path : null);  // both apps return root-relative file paths
  const name = (pidOf) => c.names?.[pidOf] || `Person ${pidOf}`;
  const moments = [
    ...c.events.filter((e) => e.type === "greet").map((e) => ({ t: e.t, kind: "greet", lens: "coach",
        text: `${name(e.staff)} greeted a customer${e.data?.time_to_greet != null ? ` after ${mmss(e.data.time_to_greet)}` : ""}.` })),
    ...c.events.filter((e) => e.type === "engagement").map((e) => ({ t: e.t, kind: "engagement", lens: "coach",
        durSec: e.t_end - e.t, text: `${name(e.staff)} stayed with the customer for ${mmss(e.t_end - e.t)}.` })),
    ...c.events.filter((e) => e.type === "walkout").map((e) => ({ t: e.t, kind: "walkout", lens: "coach",
        text: "A customer left without being greeted." })),
    ...UNSERVED_FROM_PULSE,   // see below
  ].sort((a, b) => a.t - b.t);
  const cars = [...(p.per_car || [])].sort((a, b) => b.visitors - a.visitors);
  return {
    id, name: "…", recorded: "…", lengthSec: c.duration, demo: false,
    customers: c.scorecard.store.customers, medianGreetSec: c.scorecard.store.median_time_to_greet,
    videos: [
      { label: "Salespeople", url: at(C.coachUrl, c.video_url) },                       // Coach's annotated.mp4
      { label: "Customers", url: at(C.pulseUrl, p.files?.annotated_video && `/api/jobs/${pid}/files/${p.files.annotated_video}`) },
    ].filter((v) => v.url),
    moments,
    staff: c.scorecard.staff.map((s) => ({ name: s.name || `S${s.staff}`, greets: s.greets,
      medianGreetSec: s.median_time_to_greet, engaged: s.customers_engaged, walkouts: s.walkouts_while_free,
      score: /* your formula, 0-100 */ 0 })),
    routes: /* from p.journey_flow: the most common Start > car > ... > End paths, as { path, count } */ [],
    busiestCar: cars[0] ? { label: cars[0].name, visits: cars[0].visitors, medianDwellSec: cars[0].median_dwell,
                            unserved: cars[0].unserved } : null,
  };
},
```

Notes:

- **Field names.** Check them against a real response (`/api/jobs/<id>/results`) before relying on them. The
  Coach ones are documented in `showroom-coach/docs/API.md`. Coach's video exists only after staff are confirmed
  and the render finishes (`video_status: "ready"`). Until then, leave it out of `videos`.
- **Unserved moments.** Pulse's report counts unserved interest per car (`per_car[].unserved`), but it has no
  list with times. Add a small read-only endpoint, e.g. `GET /suite/jobs/{id}/unserved`, that returns
  `[{t_start, t_end, zone}]` from `analysis.episodes` where `unserved` is true. Then map each one to
  `{ t: t_start, kind: "unserved", lens: "pulse", durSec: t_end - t_start, car: zone, text }`.
- **Two videos.** When `videos` has more than one entry, the player shows a switch above it. The switch keeps the
  current position, so the same moment can be compared in Coach's and Pulse's annotations.
- **Same origin.** Behind `deploy/`'s proxy, `coachUrl` and `pulseUrl` are this site's own `/coach` and `/pulse`,
  so the browser sends the video and API requests without CORS.

### Live floor

This is the only section that updates continuously. `getLive()` is called every `pollMs` (1 s by default) and has
to return one snapshot: everyone's position, the KPIs, and the newest events.

- **Positions.** Coach and Pulse each detect people on their own, so take positions from **one** of them, and
  staff labels from Coach. Both apps have live websocket endpoints (`live_ws.py` in Coach). Rather than polling
  over HTTP, open the websocket once and have `getLive()` return the latest message:

  ```js
  let latest = null, ws = null;
  function openLive() {
    ws = new WebSocket(C.coachUrl.replace(/^http/, "ws") + "/<live-ws-path>");
    ws.onmessage = (e) => { latest = JSON.parse(e.data); };
    ws.onclose = () => setTimeout(openLive, 1000);
  }
  // in Live:
  getLive: async () => {
    if (!ws) openLive();
    if (!latest) throw new Error("waiting for the first live frame");
    return toSnapshot(latest);   // convert to the shape in data.js
  },
  ```

- **Coordinates.** The floor drawing takes any units. The simplest option is camera pixels:
  `getFloorplan()` returns `widthM` = frame width in px, `depthM` = frame height in px, the car zones as pixel boxes,
  and `unitsPerMetre` = about how many pixels make one metre on the floor. With that set, the 2 m engagement ring
  is drawn at the true size. For a real top-down map, convert pixels to metres with a 4-point homography (OpenCV
  `getPerspectiveTransform`) and leave `unitsPerMetre` at 1.
- **KPIs and events.** `kpis` and `events` can come from Coach (greets, waits, engagements) and Pulse (zone
  dwell, unserved interest). Tag each event with `lens: "coach"` or `lens: "pulse"`. The lens toggle filters on
  that tag.
- **`heat`** is optional. Leave it out and the Customers lens draws without the heat layer.

**Webcam:** browsers only allow the camera on `localhost` or over HTTPS. For a live-camera demo, run everything
on the presenting laptop and open `http://localhost:8080`.

## 6. Checklist before the demo

- [ ] Both servers start, and `http://localhost:8000/docs` and `http://localhost:8001/docs` open
- [ ] No CORS errors in the browser console (F12 → Console)
- [ ] Every section you connected shows a green **Live** chip, and none shows red
- [ ] Any section still in demo mode is presented as demo data
- [ ] Uploading a short clip in Replays reaches "Done" and the new session appears in the list
- [ ] The Live floor moves when people move in front of the camera, and toggling the lenses changes what is drawn

## 7. Changing the look

The design tokens are at the top of `css/suite.css`: colours, fonts and spacing. The showroom name is
`showroom` in `config.js`. Don't add new colours or fonts directly in components. Add a token and use it, so the
site stays consistent.
