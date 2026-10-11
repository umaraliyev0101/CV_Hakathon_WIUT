# Showroom Suite: Coach and Pulse on one demo page

**New (YAK-46): `studio/` is the merged site.** One design, one navigation (Live floor, Replays, Team, Visitors,
Insights), with Coach and Pulse as two lenses on the same showroom floor instead of two embedded apps. It runs on
demo data out of the box. [CONNECTING.md](CONNECTING.md) explains how to wire each section to the real servers.
Run it with `python -m http.server 8080 -d showroom-suite/studio`.

The rest of this file describes the earlier split-view page (`index.html`), which embeds both apps side by side.

`index.html` is one static page that shows both hackathon features side by side:

- **Salespeople**: [Showroom Coach](../showroom-coach/README.md). Greet time, waits, walk-outs, engagement and a
  scorecard per salesperson.
- **Customers**: [Showroom Pulse](../showroom-pulse/README.md). Car zones, dwell, unserved interest, journeys and
  the heatmap.

Each app runs on its own server, unchanged. The page only embeds them, so it works from any host.

## Run

1. **Start Coach** on port 8000 (from `showroom-coach/`, after its own setup, see its README):
   `run.bat --host 0.0.0.0` (Windows) or `.venv/bin/python -m showroom_coach serve --host 0.0.0.0 --port 8000`
2. **Start Pulse** on port 8001 (from `showroom-pulse/`, after its own setup). Pulse refuses unknown Host headers,
   so list the server's address:
   `PULSE_ALLOWED_HOSTS=<server-ip-or-name> .venv/bin/uvicorn pulse.app.main:app --host 0.0.0.0 --port 8001 --ws-max-size 4194304`
3. **Serve this page**, for example on port 8080: `python -m http.server 8080 -d showroom-suite`. You can also
   just open `index.html` as a file.

Open `http://<server>:8080/`. The page assumes Coach on `:8000` and Pulse on `:8001` of the same host
(`localhost` when opened as a file). To point it elsewhere, edit the two URLs in the top bar and press Apply (the
browser remembers them), or share a link:
`http://<server>:8080/?coach=http://<host>:8000&pulse=http://<host>:8001`.

Each panel shows a status dot (green means the server answered, red means it is not reachable). It also has
buttons to reload the panel, open it in a new tab, and enlarge it (Esc restores).

## Limits

- **Live webcam mode** needs a secure context. Browsers allow the camera only on `localhost` or over HTTPS, so
  on `http://<ip>` use the upload flow, or open the live page on the server machine itself.
- **Mixed content:** if this page is served over HTTPS, both apps must be HTTPS too, or the browser blocks the
  panels.
- **Staff are confirmed separately in each panel.** The two apps are independent: each detects and numbers
  people on its own.
- **Ports:** binding to `0.0.0.0` exposes the apps on the office network. Keep it on a trusted LAN. Neither app
  has accounts.

`docs/superpowers/` holds an earlier single-server design (both apps mounted in one process). It was dropped for
the demo deadline in favour of this page.
