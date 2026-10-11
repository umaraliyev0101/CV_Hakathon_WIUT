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
| Replays | `sessions` | `getSessions`, `analyzeVideo` | Both | See "Uploads" below. |
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
