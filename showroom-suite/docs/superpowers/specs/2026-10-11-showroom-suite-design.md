# Showroom Suite: one web page for Coach and Pulse (Jira YAK-45)

> **Superseded (2026-10-11):** for the demo deadline the owner chose separate servers plus the static split-view page in `showroom-suite/index.html` (see `showroom-suite/README.md`). Not implemented.

Date: 2026-10-11. Deadline: pushed and demo-ready today, within hours. Approved in chat by the owner.

## Goal

One branch, `yak-45/showroom-suite`, holding both hackathon features. One local web page lets you upload a
video once and see both analyses:

- **Salespeople**: Showroom Coach. It measures time to greet, waits while staff are free, walk-outs and
  engagement, and gives a scorecard per salesperson.
- **Customers**: Showroom Pulse. It measures car zones, dwell, interest that no salesperson served,
  journeys and the heatmap.

## What the owner asked for, and what was assumed

- Asked: merge the teammate's feature (Pulse, `showroom-pulse/`) with Coach on one branch; build one web
  page that includes both; upload once; push today.
- Assumed: Coach is imported into `showroom-coach/` with its history kept, the same way Pulse was imported
  (`git subtree`); a new branch is pushed and `main` is not touched; the purpose is the hackathon final demo.

## Decisions

1. **Embedded UIs, not a rewrite.** Each tab embeds that app's existing UI in an `<iframe>`, opened on its
   own job. Both UIs and their tests stay as they are. Rewriting them as one native page was rejected
   because it is too large for hours. Running two servers on two ports was rejected because Pulse's Origin
   guard refuses cross-port POSTs.
2. **One process, two mounts.** `showroom_suite.app.create_app()` mounts Coach's `create_app()` at
   `/coach` and Pulse's `create_app()` at `/pulse`. Starlette does not run a mounted app's lifespan, so the
   suite's lifespan enters both apps' lifespans (Coach first, then Pulse) and exits them in reverse.
   `/coach` and `/pulse` redirect (307) to `/coach/` and `/pulse/`, so relative URLs resolve under the prefix.
3. **Relative URLs in both frontends.** Every absolute `/api/...`, `/static/...`, `/live` and `/ws/live`
   reference in Coach's and Pulse's static HTML/JS becomes relative, and the WebSocket URL is built from
   `new URL("ws/live", location.href)`. Each app still works when served alone at `/`.
4. **One environment.** Coach's dependency changes from `opencv-python` to `opencv-python-headless`, the
   build Pulse needs; the two cannot be installed together. Coach makes no OpenCV GUI calls. One Python 3.12
   venv at the repo root has all three packages installed in editable mode. Each app keeps its own
   `models/`, `runs/` and job folders inside its own subfolder.
5. **Upload once, run in sequence.** The shell page posts the chosen file and preset to `/coach/api/jobs`.
   When Coach's job leaves `queued`/`perceiving` (or fails), it posts the same `File` to `/pulse/api/jobs`.
   This avoids two ONNX pipelines fighting over the 4 CPU cores. Both `job_id`s are kept in the shell URL
   (`#coach=<id>&pulse=<id>`), so reloading the page restores both tabs.
6. **Staff are confirmed once in each tab.** The two engines number people differently, so labels are not
   carried from one tab to the other. The shell says so in one line.
7. **Live tab.** A Coach/Pulse switch loads `/coach/live` or `/pulse/live`, never both at once, so only one
   engine holds the webcam and the CPU. Switching sets the other iframe to `about:blank`. The iframes carry
   `allow="camera"`.
8. **Security carried over.** Bind to 127.0.0.1 only. Pulse's Host/Origin guard stays active on `/pulse`.
   uvicorn runs with `ws_max_size = pulse.app.main.WS_MAX_SIZE` (4 MiB), which Pulse's live mode requires.

## Components

```
showroom-suite/
  pyproject.toml           package showroom-suite (Apache-2.0), depends on showroom-coach and showroom-pulse
  run.bat, run.sh          create/check the root .venv (Python 3.12), then: python -m showroom_suite serve
  src/showroom_suite/
    __init__.py, __main__.py
    app.py                 create_app(coach_kwargs=None, pulse_kwargs=None) -> FastAPI
    cli.py                 `showroom-suite serve [--port 8000]` (host 127.0.0.1)
    static/index.html, suite.js, suite.css    the shell page (no build step, no CDN)
  tests/test_suite_app.py  shell page, mounts, lifespans, redirects, no absolute URLs in either frontend
  docs/superpowers/specs/  this file
README.md (repo root)      what is in the repo, how to run the suite, links to both READMEs
```

`create_app` passes through each app's keyword arguments (for example stub engines in tests), so the suite
tests run without models.

## Data flow

```
browser /  ── file + preset ──> /coach/api/jobs ─> Coach job (perceive -> needs_confirm -> render -> done)
          └─ when Coach leaves perceiving ─> /pulse/api/jobs ─> Pulse job (perceive -> ready_for_zones -> Score clicked -> scored)
tab Salespeople = <iframe src="/coach/#job=C">     tab Customers = <iframe src="/pulse/#job=P">
tab Live        = <iframe src="/coach/live"> or <iframe src="/pulse/live">
```

## Error handling

- If either upload fails (411/413/415/400 from the app), the shell shows that app's error message in that
  tab. The other tab is unaffected.
- If Coach's job fails, Pulse is still started, so one broken pipeline doesn't block the other.
- If polling Coach loses the connection, the shell retries with backoff for up to 60 s. After that it starts
  Pulse anyway and says so.
- A file over 1 GB is refused in the browser before upload, which is the limit both apps already enforce.

## Testing

- New `showroom-suite/tests` (FastAPI TestClient, stubbed engines):
  - `/` serves the shell;
  - `/coach/` and `/pulse/` serve each app's page, and the bare `/coach` and `/pulse` paths redirect to them;
  - `/coach/api/presets` and `/pulse/api/health` answer;
  - both lifespans ran;
  - a scan of both static folders finds no absolute `/api`, `/static`, `/live` or `/ws` URL.
- Both existing suites stay green in the shared venv: Coach `pytest -q` and Pulse `pytest -q`. A test that
  hard-codes an absolute path and fails because of the URL change is updated in the same commit.
- End to end with Playwright against `run.bat`:
  - upload `meva_a_60s.mp4` once;
  - Salespeople reaches staff confirmation and then results;
  - Customers reaches `ready_for_zones` and, after Score, its report;
  - both live pages connect with Chrome's fake camera;
  - no console errors.
  - The timings are recorded honestly in the suite README.

## Push procedure

1. Audit Coach's history before the subtree import: no videos, weights, `data/eval_only/` files or
   secrets in any commit, because the repo is public.
2. Show the owner the commit list and `git diff --stat origin/main`, and push only after they confirm.
3. Push `yak-45/showroom-suite` to `origin` with the token given only for that one command
   (`git -c http.extraHeader=...`), never written to `.git/config` or any file. `main` is not pushed.
4. Commits carry no `Co-Authored-By` trailer (Coach's CLAUDE.md rule).

## Out of scope

- A shared detection engine and cache, and one shared `roles.json` (Pulse's `docs/INTEGRATION.md` plan).
- Carrying staff labels between tabs.
- Rewriting the two UIs as one native page.
- Any change to `main`.
- Opening a pull request (offered after the push).
