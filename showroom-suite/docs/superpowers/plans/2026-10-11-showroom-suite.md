# Showroom Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One branch with Showroom Coach and Showroom Pulse, served by one local web page. One upload feeds both,
with a Salespeople tab, a Customers tab and a Live tab.

**Architecture:** A new `showroom_suite` FastAPI app mounts Coach's app at `/coach` and Pulse's app at `/pulse`,
runs both lifespans, and serves a plain-JS shell at `/`. Each tab embeds that app's existing UI in an iframe.
Both frontends are made prefix-safe: no absolute URLs, and the backend's `/api/...` URLs are rebased in the
browser. Backends are unchanged except for one Pulse middleware path check.

**Tech Stack:** Python 3.12, FastAPI 0.143 / Starlette 1.7, uvicorn, vanilla JS (no build, no CDN), pytest,
Playwright MCP for the end-to-end check.

**Spec:** `showroom-suite/docs/superpowers/specs/2026-10-11-showroom-suite-design.md`

## Global Constraints

- Work in `C:\Users\USER\Desktop\CV_Hakathon_WIUT`, branch `yak-45/showroom-suite`. Never touch `main`.
- Commits have no `Co-Authored-By` trailer. Commit messages use the prefix `feat(yak-45):`, `fix(yak-45):`,
  `docs(yak-45):` or `test(yak-45):`.
- Licences permissive only. Add no new models or datasets. Never commit videos, weights, `data/`, `models/*.onnx`,
  `runs/`, `jobs/` or `.venv/`.
- One venv at the repo root: `.venv` on Python 3.12
  (`C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe`).
- Bind to `127.0.0.1` only. uvicorn `ws_max_size = pulse.app.main.WS_MAX_SIZE`.
- Privacy is unchanged: no face recognition, and every rendered frame is blurred (both apps already do this).
- Report measured numbers honestly. A missed target is stated with its measured value.
- The push happens only after the owner confirms in chat. The token is passed for that one command only, never
  written to a file or `.git/config`.

## Review Focus

1. **Pulse's early 413 under a mount.** `request.url.path` includes the `/pulse` prefix, so the
   `== "/api/jobs"` check silently stops firing. Pinned in Task 3 (standalone) and Task 4 (through the suite).
2. **Backend-built file URLs** (`thumb_url`, `video_url`, contact sheets, exports) pointing at `/api/...` and
   giving 404 inside `/coach/` or `/pulse/`. Pinned by the rebase in Tasks 2 and 3 and checked in Task 6's
   network log.
3. **Coach's `<base href>` trick.** It must be `/coach/static/` under the suite and `/static/` standalone, or
   CSS and JS give 404 and the tab is blank. Pinned in Task 4 (asset fetch through the mount) and Task 6.
4. **Reloading the shell.** It must restore both tabs from `#coach=..&pulse=..`. Checked in Task 6.
5. **Live tab.** Switching between Coach and Pulse must release the other one's camera and engine
   (`about:blank`). Checked in Task 6.

---

### Task 1: Import Coach with its history and build one shared environment

**Files:**
- Create (via subtree): `showroom-coach/**` (Coach's 68 commits)
- Modify: `showroom-coach/pyproject.toml` (`opencv-python` → `opencv-python-headless`)
- Modify: `.gitignore` (root: `.venv/`)

- [ ] **Step 1: Audit Coach's history before it goes public.** Run in `C:\Users\USER\Desktop\showroom-coach`:

```bash
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | awk '$1=="blob" && $2>1000000'
git log --all --name-only --format= | sort -u | grep -E '\.(mp4|avi|mov|mkv|webm|onnx|npz|pt|pth|zip|jpg|png)$|^data/(eval_only|meva|samples|stock)/' || echo "no media/weights paths"
git grep -nE 'ghp_[A-Za-z0-9]{30,}|github_pat_|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' $(git rev-list --all) || echo "no secrets"
```
Expected: no blob over 1 MB, "no media/weights paths" (the SVG mock thumbnails are fine), "no secrets". On any
hit, STOP and report it to the owner. Don't import.

- [ ] **Step 2: Subtree-import Coach** (the working tree must be clean):

```bash
cd /c/Users/USER/Desktop/CV_Hakathon_WIUT
git subtree add --prefix=showroom-coach ../showroom-coach main -m "Add Showroom Coach (salespeople coaching) under showroom-coach/, history kept

Imported with git subtree from the local overnight-build repo (68 commits, last 9baa229). Code, tests and
docs only: videos, model weights, datasets and runs/ stay gitignored and local."
git log --oneline -3 && ls showroom-coach
```

- [ ] **Step 3: Switch Coach to headless OpenCV and ignore the root venv.** In `showroom-coach/pyproject.toml`
  replace `"opencv-python>=4.9",` with `"opencv-python-headless>=4.9",`. Append `.venv/` to the root `.gitignore`.
  Check that Coach makes no GUI calls: `grep -rnE "cv2\.(imshow|namedWindow|waitKey|destroy)" showroom-coach/src`
  should print nothing.

- [ ] **Step 4: Build the shared venv** (Pulse needs Python `>=3.12,<3.13`):

```bash
"/c/Users/USER/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv
.venv/Scripts/python -m pip install -e "showroom-coach[dev]" -e "showroom-pulse[dev]"
.venv/Scripts/python -c "import cv2, showroom_coach, pulse; print(cv2.__version__)"
.venv/Scripts/python -m pip list 2>/dev/null | grep -iE "^opencv"
```
Expected: the imports work and only `opencv-python-headless` is listed. If `opencv-python` was pulled in, run
`pip uninstall -y opencv-python` and reinstall headless with `--force-reinstall`.

- [ ] **Step 5: Model weights (local only, never committed).**

```bash
cp -r ../showroom-coach/models/yolox_nano ../showroom-coach/models/rtmpose_s ../showroom-coach/models/dinov2_small showroom-coach/models/
cd showroom-pulse && ../.venv/Scripts/python -c "from pulse.perception.models import ensure_model; [print(ensure_model(n, 'models')) for n in ('yolox_s', 'rtmpose_s', 'dinov2_small')]"; cd ..
git status --short   # if showroom-pulse/models/MANIFEST.json changed only in field order or dates: git checkout -- it
```

- [ ] **Step 6: Run both suites in the shared venv.** Run each from its own folder so it uses its own pytest config:

```bash
(cd showroom-coach && ../.venv/Scripts/python -m pytest -q -p no:cacheprovider) 2>&1 | tail -3
(cd showroom-pulse && ../.venv/Scripts/python -m pytest -q -p no:cacheprovider) 2>&1 | tail -3
```
Expected: both pass. Coach had 835 passed and 11 deselected before the import.

- [ ] **Step 7: Commit**

```bash
git add .gitignore showroom-coach/pyproject.toml
git commit -m "feat(yak-45): one venv for both apps: Coach uses opencv-python-headless like Pulse"
```

### Task 2: Make Coach's frontend prefix-safe

**Files:**
- Modify: `showroom-coach/src/showroom_coach/app/static/index.html` (base script, live link)
- Modify: `showroom-coach/src/showroom_coach/app/static/live.html` (base script, back link)
- Modify: `showroom-coach/src/showroom_coach/app/static/app.js` (ROOT, API paths, rebase, comments)
- Modify: `showroom-coach/src/showroom_coach/app/static/live.js` (ROOT, presets path, ws URL)
- Test: `showroom-coach/tests/test_app.py` (new test `test_frontend_has_no_absolute_urls`)

**Interfaces:**
- Produces: `window.APP_ROOT` (string ending in `/`: `/` standalone, `/coach/` under the suite), set by each
  page's inline head script before `app.js` or `live.js` load. Both scripts read
  `const ROOT = window.APP_ROOT || '/';`.

- [ ] **Step 1: Write the failing test** (append to `showroom-coach/tests/test_app.py`):

```python
def test_frontend_has_no_absolute_urls():
    """The suite mounts this app at /coach/, so the pages must not hard-code root-absolute URLs."""
    static = Path(__file__).resolve().parents[1] / "src" / "showroom_coach" / "app" / "static"
    pat = re.compile(r"""["'`]/(api|static|ws|live)\b|(?:href|src)="/""")
    hits = [f"{p.name}:{n}: {line.strip()[:80]}" for p in sorted(static.glob("*.*")) if p.suffix in {".html", ".js", ".css"}
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    assert not hits, "absolute URLs:\n" + "\n".join(hits)
```
(`re` and `Path` are already imported in `test_app.py`. If either isn't, add the import.)

- [ ] **Step 2: Run it.** `cd showroom-coach && ../.venv/Scripts/python -m pytest tests/test_app.py::test_frontend_has_no_absolute_urls -q`
  Expected: FAIL, listing about 30 lines.

- [ ] **Step 3: index.html head script.** Replace the IIFE body (`var onBackend = location.pathname === '/'; ...`) with:

```js
  (function () {
    // On the backend the page is served at the app root: "/" alone, "/coach/" under Showroom Suite. Opened in
    // place (index.html on a plain file server) relative URLs already resolve, so no <base>.
    var root = location.pathname;
    var onBackend = root.charAt(root.length - 1) === '/';
    window.APP_ROOT = onBackend ? root : '/';
    if (onBackend) document.write('<base href="' + root + 'static/">');
    document.write('<link rel="stylesheet" href="style.css">');
    window.__loadApp = function () { document.write('<script src="app.js"><\/script>'); };
  })();
```
  Update the comment above it to say "the app root (`/`, or `/coach/` under Showroom Suite)" instead of `"/"`.
  Change the Live link `href="/live"` to `href="../live"` (under `<base href="<root>static/">` it resolves to
  `<root>live`).

- [ ] **Step 4: live.html head script.** Replace `var onBackend = location.pathname === '/live' || location.pathname === '/live/';`
  and the `document.write('<base href="/static/">')` line with:

```js
    var path = location.pathname;
    var onBackend = /\/live\/?$/.test(path);
    var root = onBackend ? path.replace(/live\/?$/, '') : '/';
    window.APP_ROOT = root;
    if (onBackend) document.write('<base href="' + root + 'static/">');
```
  Update the comment ("serves this page at `<root>live` and this folder at `<root>static/`"). Change the back
  link `href="/"` to `href="../"`.

- [ ] **Step 5: app.js.**
  - Directly after the `const Q = new URLSearchParams(location.search);` line, add:

```js
// App root: "/" standalone, "/coach/" under Showroom Suite (set by index.html before this script loads).
const ROOT = window.APP_ROOT || '/';
// The server returns root-absolute file URLs ("/api/jobs/.../files/..."); give them this app's prefix.
function rebase(x) {
  if (typeof x === 'string') return x.startsWith('/api/') ? ROOT + x.slice(1) : x;
  if (Array.isArray(x)) return x.map(rebase);
  if (x && typeof x === 'object') for (const k of Object.keys(x)) x[k] = rebase(x[k]);
  return x;
}
```
  - Rewrite the API paths mechanically, then check what's left:

```bash
sed -i -e "s#\`/api/#\`\${ROOT}api/#g" -e "s#'/api/#ROOT + 'api/#g" showroom-coach/src/showroom_coach/app/static/app.js
grep -nE "[\"'\`]/(api|static|ws|live)" showroom-coach/src/showroom_coach/app/static/app.js
```
  - In `api()`: change `if (MOCK) return mockApi(method, path, body);` to
    `if (MOCK) return mockApi(method, '/' + path.slice(ROOT.length), body);`, and the final `return data;` to
    `return rebase(data);`.
  - The upload code around the old lines 103–110 (XHR or `fetch` to `/api/jobs`) is covered by the sed. If it
    parses the response with its own `JSON.parse`, it only reads `job_id`, which needs no rebase.
  - Reword the two comments on lines 8 and 963 that quote `<base href="/static/">` to say
    "a <base> under the app root", so the scan is clean.

- [ ] **Step 6: live.js.** Add `const ROOT = window.APP_ROOT || '/';` near the top, after the first `const`.
  Replace `'/api/presets'` with `ROOT + 'api/presets'`. Replace the `wsUrl()` body with:

```js
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${ROOT}ws/live?preset=${encodeURIComponent(S.preset)}`;
```

- [ ] **Step 7: Run the whole Coach suite.** `cd showroom-coach && ../.venv/Scripts/python -m pytest -q -p no:cacheprovider`
  Expected: all pass. If a test in `test_app.py`, `test_live.py` or `test_packaging.py` asserted the old literal
  (for example `'/ws/live'` in live.js or `<base href="/static/">`), update that assertion to the new form and
  note it in the commit body.

- [ ] **Step 8: Commit**

```bash
git add showroom-coach/src/showroom_coach/app/static showroom-coach/tests
git commit -m "feat(yak-45): Coach frontend works under a path prefix (/coach/ in Showroom Suite)"
```

### Task 3: Make Pulse's frontend prefix-safe and fix the early 413 under a mount

**Files:**
- Modify: `showroom-pulse/src/pulse/app/static/{app.js,dashboard.js,live.js,zones.js,index.html,live.html}`
- Modify: `showroom-pulse/src/pulse/app/main.py:198` (path check)
- Test: `showroom-pulse/tests/test_static.py` (regex in `test_js_api_paths_exist_as_routes`, new no-absolute-URL test)

**Interfaces:**
- Produces: Pulse pages use document-relative URLs (`api/...`, `static/...`, `live`, `./`), module imports
  `./x.js`, and the WebSocket URL `new URL("ws/live", location.href)` with `http` swapped for `ws`. No
  `APP_ROOT` is needed, because Pulse has no `<base>`.

- [ ] **Step 1: Write the failing tests** (append to `showroom-pulse/tests/test_static.py`):

```python
def test_frontend_has_no_absolute_urls():
    """Showroom Suite mounts Pulse at /pulse/, so pages must not hard-code root-absolute URLs."""
    pat = re.compile(r"""["'`]/(api|static|ws|live)\b|(?:href|src)="/""")
    hits = [f"{p.name}:{n}: {line.strip()[:80]}" for p in sorted(Path(STATIC).glob("*.*")) if p.suffix in {".html", ".js", ".css"}
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    assert not hits, "absolute URLs:\n" + "\n".join(hits)


def test_upload_size_check_works_under_a_mount(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pulse.app.main import create_app
    outer = FastAPI()
    outer.mount("/pulse", create_app(jobs_dir=tmp_path / "jobs"))
    with TestClient(outer) as c:
        r = c.post("/pulse/api/jobs", content=b"x", headers={"content-length": str(3 * 1024**3), "content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
```
  In `test_js_api_paths_exist_as_routes`, change the regex to match relative paths and prefix a slash:

```python
        for m in re.finditer(r"(?<![\w/.])api/[A-Za-z0-9_/.${}\-]*", js.read_text()):
            found.add(_norm("/" + m.group(0).rstrip("/")))
```

- [ ] **Step 2: Run them.** `cd showroom-pulse && ../.venv/Scripts/python -m pytest tests/test_static.py -q`
  Expected: `test_frontend_has_no_absolute_urls` FAILS with about 30 lines, `test_js_api_paths_exist_as_routes`
  FAILS with "no /api paths found", and `test_upload_size_check_works_under_a_mount` FAILS (not 413; the TestClient
  sends 1 byte and the handler errors differently).

- [ ] **Step 3: Rewrite the static URLs** (document-relative for HTML and fetch, module-relative for imports):

```bash
S=showroom-pulse/src/pulse/app/static
sed -i -E 's#from "/static/([a-z0-9_]+\.js)"#from "./\1"#g' $S/*.js
sed -i -E "s#([\"'\`])/api/#\1api/#g" $S/*.js
sed -i -E 's#(href|src)="/static/#\1="static/#g; s#href="/live"#href="live"#; s#href="/"#href="./"#' $S/*.html
grep -nE "[\"'\`]/(api|static|ws|live)|(href|src)=\"/" $S/*.*
```
  The only expected leftover is live.js's `"/ws/live"`. Replace that `wsUrl` body with:

```js
  const u = new URL("ws/live", location.href);
  u.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return u.href;
```

- [ ] **Step 4: Rebase backend URLs in Pulse's fetch helper** (`app.js`, the function around line 13 that calls
  `fetch(url, opts)`). Where it returns the parsed JSON, wrap it as `return rebase(data)` and add this above the
  function:

```js
// The server returns root-absolute file URLs ("/api/jobs/.../files/..."); make them relative so they work
// under a path prefix (Showroom Suite serves this app at /pulse/).
function rebase(x) {
  if (typeof x === "string") return x.startsWith("/api/") ? x.slice(1) : x;
  if (Array.isArray(x)) return x.map(rebase);
  if (x && typeof x === "object") for (const k of Object.keys(x)) x[k] = rebase(x[k]);
  return x;
}
```

- [ ] **Step 5: Fix the early size check** in `showroom-pulse/src/pulse/app/main.py` (line ~198):

```python
        # under a mount (Showroom Suite serves Pulse at /pulse) url.path carries the prefix too
        if request.method == "POST" and request.url.path == request.scope.get("root_path", "") + "/api/jobs":
```

- [ ] **Step 6: Run the whole Pulse suite.** `cd showroom-pulse && ../.venv/Scripts/python -m pytest -q -p no:cacheprovider`
  Expected: all pass. If `test_index_has_banner_and_live_link`, `test_i18n.py`, `test_api.py` or `test_docs.py`
  asserted an old literal (`href="/live"`, `/static/...`), update it to the relative form.

- [ ] **Step 7: Commit**

```bash
git add showroom-pulse/src/pulse/app showroom-pulse/tests
git commit -m "feat(yak-45): Pulse frontend works under a path prefix; early 413 also under a mount"
```

### Task 4: The suite app (mounts, lifespans, CLI)

**Files:**
- Create: `showroom-suite/pyproject.toml`, `showroom-suite/src/showroom_suite/{__init__.py,__main__.py,app.py,cli.py}`
- Create: `showroom-suite/src/showroom_suite/static/index.html` (placeholder that Task 5 replaces: a `<title>Showroom Suite</title>` page with the words Salespeople and Customers)
- Test: `showroom-suite/tests/test_suite_app.py`

**Interfaces:**
- Consumes: `showroom_coach.app.main.create_app(stack_factory=None, jobs_root=None, live_session_factory=None)`,
  `pulse.app.main.create_app(jobs_dir=Path("jobs"), cache_root=None, engine_factory=None)`, `pulse.app.main.WS_MAX_SIZE`.
- Produces: `showroom_suite.app.create_app(coach_kwargs: dict | None = None, pulse_kwargs: dict | None = None) -> FastAPI`
  with `app.state.coach` and `app.state.pulse`; `showroom_suite.app.PULSE_ROOT: Path`;
  `showroom_suite.cli.main(argv=None) -> int` (`serve [--port 8000]`).

- [ ] **Step 1: pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "showroom-suite"
version = "0.1.0"
description = "One local web page for Showroom Coach (salespeople) and Showroom Pulse (customers) (YAK-45)"
requires-python = ">=3.12,<3.13"
license = {text = "Apache-2.0"}
dependencies = ["showroom-coach", "showroom-pulse", "fastapi>=0.115", "uvicorn[standard]>=0.29"]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[project.scripts]
showroom-suite = "showroom_suite.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
showroom_suite = ["static/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
  Install: `.venv/Scripts/python -m pip install -e "showroom-suite[dev]" --no-deps` (both apps are already installed
  in editable mode; `--no-deps` stops pip from looking for them on PyPI).

- [ ] **Step 2: Write the failing tests** (`showroom-suite/tests/test_suite_app.py`):

```python
"""Showroom Suite: the shell page, both apps under their prefixes, both lifespans, upload limits through the mounts."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from showroom_suite.app import create_app


def _no_models(*_a, **_k):
    raise RuntimeError("suite tests never run perception")


@pytest.fixture
def app(tmp_path):
    return create_app(coach_kwargs={"stack_factory": _no_models, "jobs_root": tmp_path / "coach"},
                      pulse_kwargs={"jobs_dir": tmp_path / "pulse", "cache_root": tmp_path / "cache",
                                    "engine_factory": _no_models})


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_shell_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Salespeople" in r.text and "Customers" in r.text


@pytest.mark.parametrize("prefix", ["/coach", "/pulse"])
def test_bare_prefix_redirects_to_slash(client, prefix):
    r = client.get(prefix, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == prefix + "/"


@pytest.mark.parametrize("path, needle", [("/coach/", "Showroom Coach"), ("/coach/live", "<html"),
                                          ("/pulse/", "Showroom Pulse"), ("/pulse/live", "<html")])
def test_app_pages_under_prefix(client, path, needle):
    r = client.get(path)
    assert r.status_code == 200 and needle in r.text


@pytest.mark.parametrize("path", ["/coach/static/app.js", "/coach/static/style.css", "/coach/static/live.js",
                                  "/pulse/static/app.js", "/pulse/static/style.css", "/pulse/static/live.js",
                                  "/coach/api/presets", "/pulse/api/health", "/api/health"])
def test_assets_and_apis_under_prefix(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/coach/api/jobs", "/pulse/api/jobs"])
def test_oversize_upload_refused_through_the_mount(client, path):
    r = client.post(path, content=b"x", headers={"content-length": str(3 * 1024**3),
                                                 "content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_both_lifespans_run(app, tmp_path):
    seen = []
    mgr = app.state.coach.state.jobs
    real = mgr.cleanup
    mgr.cleanup = lambda: (seen.append("coach"), real())[1]
    old = tmp_path / "pulse" / "stale"
    old.mkdir(parents=True)
    import os, time
    os.utime(old, (time.time() - 3 * 86400,) * 2)
    with TestClient(app):
        pass
    assert seen == ["coach"] and not old.exists()     # Pulse's lifespan deletes job folders older than 24 h
```
  Run: `cd showroom-suite && ../.venv/Scripts/python -m pytest -q`. Expected: FAIL (`ModuleNotFoundError: showroom_suite.app`).
  If Pulse's `jobs_mod.cleanup` only deletes folders matching its job-id pattern or containing `job.json`, make
  the stale folder look like one. Read `showroom-pulse/src/pulse/app/jobs.py` `cleanup` and mirror it, keeping the
  assertion.

- [ ] **Step 3: app.py**

```python
"""Showroom Suite: one local web app with Showroom Coach at /coach/ and Showroom Pulse at /pulse/, plus the
shell page at / that uploads one clip to both and shows them as tabs (spec: docs/superpowers/specs/...)."""
from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pulse
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pulse.app.main import create_app as create_pulse_app
from showroom_coach.app.main import create_app as create_coach_app

STATIC_DIR = Path(__file__).resolve().parent / "static"
PULSE_ROOT = Path(pulse.__file__).resolve().parents[2]      # showroom-pulse/: Pulse resolves models/ etc. here
NO_STORE = {"Cache-Control": "no-store"}


def create_app(coach_kwargs: dict | None = None, pulse_kwargs: dict | None = None) -> FastAPI:
    coach = create_coach_app(**(coach_kwargs or {}))
    pulse_app = create_pulse_app(**{"jobs_dir": PULSE_ROOT / "jobs", "cache_root": PULSE_ROOT / "runs" / "cache",
                                    **(pulse_kwargs or {})})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Starlette never runs a mounted app's lifespan: enter both here (startup cleanup, shutdown of workers).
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(coach.router.lifespan_context(coach))
            await stack.enter_async_context(pulse_app.router.lifespan_context(pulse_app))
            yield

    app = FastAPI(title="Showroom Suite", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.coach, app.state.pulse = coach, pulse_app

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html", headers=NO_STORE)

    @app.get("/api/health")
    def health():
        return {"ok": True, "apps": ["coach", "pulse"]}

    for prefix in ("/coach", "/pulse"):            # relative URLs in each app need the trailing slash
        app.add_api_route(prefix, lambda prefix=prefix: RedirectResponse(prefix + "/", status_code=307),
                          include_in_schema=False)

    app.mount("/suite-static", StaticFiles(directory=STATIC_DIR), name="suite-static")
    app.mount("/coach", coach)
    app.mount("/pulse", pulse_app)
    return app
```
  Before relying on `cache_root`, check that `PULSE_ROOT / "runs" / "cache"` matches what Pulse itself uses when
  `cache_root` is None: read the line in `create_app` that resolves it. If it differs, use Pulse's own default.

- [ ] **Step 4: cli.py, `__main__.py`, `__init__.py`**

```python
# cli.py
"""`showroom-suite serve`: start the suite on http://127.0.0.1:PORT (local only)."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="showroom-suite", description="Showroom Coach + Showroom Pulse on one local page")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the suite on 127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if not 0 < args.port < 65536:
        print("--port must be between 1 and 65535", file=sys.stderr)
        return 2
    import uvicorn
    from pulse.app.main import WS_MAX_SIZE

    from .app import PULSE_ROOT, create_app
    os.chdir(PULSE_ROOT)        # Pulse resolves models/, labels/zones/ and runs/ against the working directory
    print(f"Showroom Suite: open http://127.0.0.1:{args.port}/  (Ctrl+C to stop)", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, ws_max_size=WS_MAX_SIZE)
    return 0
```
```python
# __main__.py
import sys

from .cli import main

sys.exit(main())
```
```python
# __init__.py
"""Showroom Suite: Showroom Coach and Showroom Pulse on one local web page (WIUT Hackathon 2026, YAK-45)."""
__version__ = "0.1.0"
```

- [ ] **Step 5: Run the tests.** `cd showroom-suite && ../.venv/Scripts/python -m pytest -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add showroom-suite/pyproject.toml showroom-suite/src showroom-suite/tests
git commit -m "feat(yak-45): Showroom Suite app mounts Coach at /coach and Pulse at /pulse"
```

### Task 5: The shell page and launchers

**Files:**
- Create: `showroom-suite/src/showroom_suite/static/{index.html,suite.js,suite.css}`
- Create: `showroom-suite/run.bat`, `showroom-suite/run.sh`, `README.md` (repo root), `showroom-suite/README.md`
- Test: `showroom-suite/tests/test_suite_app.py` (shell assets and the no-external-hosts check)

**Interfaces:**
- Consumes: `POST {coach|pulse}/api/jobs` (multipart `file`, `preset` → `{"job_id"}`; errors are
  `{"detail"}` from Coach and `{"error"}` from Pulse), `GET coach/api/jobs/{id}` → `{status, message}`, and the
  iframe deep links `coach/#job=ID` and `pulse/#job=ID`.

- [ ] **Step 1: Add the tests** (append to `test_suite_app.py`):

```python
def test_shell_assets_served_and_local_only(client):
    for path in ("/suite-static/suite.js", "/suite-static/suite.css"):
        r = client.get(path)
        assert r.status_code == 200
        assert "http://" not in r.text and "https://" not in r.text     # no CDN, works offline
    html = client.get("/").text
    assert html.count('src="about:blank"') == 3                        # frames load on demand, not at startup
```
  Run: FAIL (404 on suite.js).

- [ ] **Step 2: index.html** (the shell; plain HTML, three tabs, one upload form):

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Showroom Suite</title>
<link rel="stylesheet" href="suite-static/suite.css">
</head>
<body>
<header class="top">
  <h1>Showroom Suite</h1>
  <p class="sub">One showroom clip, two views: how salespeople serve customers, and what customers do. Runs on this computer only; faces are blurred.</p>
</header>
<form id="upload" class="card">
  <label class="file">Video <input type="file" id="file" name="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,video/*" required></label>
  <label>Preset
    <select id="preset" name="preset">
      <option value="short" selected>Short clip (10–60 s)</option>
      <option value="dealership">Dealership</option>
    </select>
  </label>
  <button type="submit" id="go">Analyse with both</button>
  <ul class="status" aria-live="polite">
    <li>Salespeople: <span id="st-coach">waiting for a clip</span></li>
    <li>Customers: <span id="st-pulse">waiting for a clip</span></li>
  </ul>
  <p class="note">Confirm staff in each tab: the two engines number people separately, so labels don't carry over.</p>
</form>
<nav class="tabs" role="tablist">
  <button role="tab" id="tab-coach" aria-controls="panel-coach" aria-selected="true">Salespeople</button>
  <button role="tab" id="tab-pulse" aria-controls="panel-pulse" aria-selected="false">Customers</button>
  <button role="tab" id="tab-live" aria-controls="panel-live" aria-selected="false">Live</button>
</nav>
<section id="panel-coach" role="tabpanel"><iframe id="f-coach" title="Salespeople (Showroom Coach)" src="about:blank"></iframe></section>
<section id="panel-pulse" role="tabpanel" hidden><iframe id="f-pulse" title="Customers (Showroom Pulse)" src="about:blank"></iframe></section>
<section id="panel-live" role="tabpanel" hidden>
  <div class="switch">
    <button id="live-coach" aria-pressed="false">Salespeople live</button>
    <button id="live-pulse" aria-pressed="false">Customers live</button>
    <span class="note">One at a time: each uses the camera and the CPU.</span>
  </div>
  <iframe id="f-live" title="Live camera" src="about:blank" allow="camera"></iframe>
</section>
<script src="suite-static/suite.js"></script>
</body>
</html>
```
  (The Live tab has one iframe. Switching sets its `src`, so the previous page unloads and releases the camera.)

- [ ] **Step 3: suite.js**

```js
// Showroom Suite shell. One upload goes to Coach (coach/api/jobs). Once Coach has finished detection, the same
// file goes to Pulse (pulse/api/jobs), so the two ONNX pipelines don't fight over the CPU. Both job ids live
// in the URL hash (#coach=..&pulse=..) so a reload restores the tabs.
'use strict';
const MAX_BYTES = 1024 ** 3;
const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jobs = { coach: null, pulse: null };

function setStatus(app, text, bad = false) {
  const el = $(`#st-${app}`);
  el.textContent = text;
  el.classList.toggle('bad', bad);
}

function writeHash() {
  const p = new URLSearchParams();
  for (const k of ['coach', 'pulse']) if (jobs[k]) p.set(k, jobs[k]);
  history.replaceState(null, '', '#' + p);
}

function showJob(app) {
  $(`#f-${app}`).src = jobs[app] ? `${app}/#job=${encodeURIComponent(jobs[app])}` : 'about:blank';
}

async function upload(app, file, preset) {
  const form = new FormData();
  form.append('file', file);
  form.append('preset', preset);
  let res;
  try { res = await fetch(`${app}/api/jobs`, { method: 'POST', body: form }); }
  catch { throw new Error('could not reach the server'); }
  const data = await res.json().catch(() => null);
  const msg = data && (typeof data.detail === 'string' ? data.detail : data.error);
  if (!res.ok) throw new Error(msg || `the server answered ${res.status}`);
  return data.job_id;
}

// Wait until Coach leaves queued/perceiving. Gives up after 60 s without contact, so Pulse still starts.
async function coachDetectionDone(id) {
  let lostSince = 0;
  for (;;) {
    try {
      const res = await fetch(`coach/api/jobs/${encodeURIComponent(id)}`, { cache: 'no-store' });
      if (res.status === 404) return 'missing';
      const j = await res.json();
      lostSince = 0;
      setStatus('coach', j.message || j.status, j.status === 'failed');
      if (!['queued', 'perceiving'].includes(j.status)) return j.status;
    } catch {
      lostSince = lostSince || Date.now();
      if (Date.now() - lostSince > 60000) return 'unreachable';
    }
    await sleep(1000);
  }
}

async function start(file, preset) {
  if (file.size > MAX_BYTES) { setStatus('coach', 'The file is larger than 1 GB.', true); return; }
  $('#go').disabled = true;
  jobs.coach = jobs.pulse = null;
  writeHash(); showJob('coach'); showJob('pulse');
  setStatus('coach', 'uploading…'); setStatus('pulse', 'starts when Salespeople has finished detection');
  try {
    jobs.coach = await upload('coach', file, preset);
    writeHash(); showJob('coach');
    const s = await coachDetectionDone(jobs.coach);
    if (s === 'unreachable') setStatus('coach', 'lost contact with the server; starting Customers anyway', true);
  } catch (e) { setStatus('coach', `upload failed: ${e.message}`, true); }
  try {
    setStatus('pulse', 'uploading…');
    jobs.pulse = await upload('pulse', file, preset);
    writeHash(); showJob('pulse');
    setStatus('pulse', 'analysing; open the Customers tab');
  } catch (e) { setStatus('pulse', `upload failed: ${e.message}`, true); }
  $('#go').disabled = false;
}

function selectTab(name) {
  for (const t of ['coach', 'pulse', 'live']) {
    $(`#tab-${t}`).setAttribute('aria-selected', String(t === name));
    $(`#panel-${t}`).hidden = t !== name;
  }
  if (name !== 'live') setLive(null);          // leaving Live releases the camera
}

function setLive(app) {
  $('#f-live').src = app ? `${app}/live` : 'about:blank';
  $('#live-coach').setAttribute('aria-pressed', String(app === 'coach'));
  $('#live-pulse').setAttribute('aria-pressed', String(app === 'pulse'));
}

$('#upload').addEventListener('submit', (e) => {
  e.preventDefault();
  const file = $('#file').files[0];
  if (file) start(file, $('#preset').value);
});
for (const t of ['coach', 'pulse', 'live']) $(`#tab-${t}`).addEventListener('click', () => selectTab(t));
$('#live-coach').addEventListener('click', () => setLive('coach'));
$('#live-pulse').addEventListener('click', () => setLive('pulse'));

const saved = new URLSearchParams(location.hash.slice(1));
for (const k of ['coach', 'pulse']) {
  jobs[k] = saved.get(k);
  if (jobs[k]) { showJob(k); setStatus(k, 'restored from the link'); }
}
```

- [ ] **Step 4: suite.css** (light/dark tokens, phone width, iframes fill the remaining height):

```css
:root { --bg: #f6f7f9; --fg: #1b2430; --muted: #5b6573; --card: #fff; --line: #d9dee5; --accent: #1478e6; --bad: #c62828; }
@media (prefers-color-scheme: dark) { :root { --bg: #11161d; --fg: #e8edf3; --muted: #9aa5b3; --card: #1a212b; --line: #2c3542; --accent: #5aa6ff; --bad: #ff6b6b; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { margin: 0; font-size: 1.4rem; }
.sub, .note { color: var(--muted); margin: 4px 0 0; }
.card { display: flex; flex-wrap: wrap; gap: 12px 16px; align-items: end; margin: 12px 0; padding: 12px 16px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
.card label { display: flex; flex-direction: column; gap: 4px; }
.card .note { flex-basis: 100%; }
button { font: inherit; padding: 6px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }
#go { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: .6; cursor: progress; }
.status { list-style: none; margin: 0; padding: 0; flex-basis: 100%; }
.status .bad { color: var(--bad); }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); }
.tabs [role=tab] { border-radius: 8px 8px 0 0; border-bottom: none; }
.tabs [aria-selected=true] { background: var(--accent); border-color: var(--accent); color: #fff; }
.switch { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0; }
.switch [aria-pressed=true] { outline: 2px solid var(--accent); }
iframe { display: block; width: 100%; height: calc(100vh - 120px); min-height: 480px; border: 1px solid var(--line); border-top: none; background: var(--card); }
```

- [ ] **Step 5: Launchers.** Write `showroom-suite/run.bat` in the style of `showroom-coach/run.bat` (no labels,
  `pause` on every error path):
  - `cd /d "%~dp0.."`;
  - create `.venv` with `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` or `py -3.12`, and require 3.12;
  - if `import showroom_suite` fails, run
    `pip install -e showroom-coach -e showroom-pulse && pip install -e showroom-suite --no-deps`;
  - if `showroom-coach\models\yolox_nano\yolox_nano.onnx` is missing, run
    `python -m showroom_coach fetch-models yolox_nano rtmpose_s dinov2_small` from inside `showroom-coach`;
  - in `showroom-pulse`, run the `ensure_model` one-liner from Task 1 Step 5 (it skips files that are already
    there);
  - finally run `python -m showroom_suite serve %*`.

  Write `run.sh` with the same steps for bash.

- [ ] **Step 6: READMEs.** Root `README.md`, 20–30 lines:
  - what is in the repo: `showroom-coach/`, `showroom-pulse/` and `showroom-suite/`, each one line with a link to
    its README;
  - the one-command start (`showroom-suite\run.bat`, then open http://127.0.0.1:8000/);
  - privacy (local only, faces blurred, no face recognition);
  - licences ("Apache-2.0; model licences in each project's README table").

  `showroom-suite/README.md`: how the page works (the upload sequence, three tabs, staff confirmed per tab) and
  the Task 6 measurements once they exist.

- [ ] **Step 7: Run the tests.** `cd showroom-suite && ../.venv/Scripts/python -m pytest -q` → all pass.

- [ ] **Step 8: Commit**

```bash
git add README.md showroom-suite
git commit -m "feat(yak-45): Showroom Suite page: one upload, Salespeople, Customers and Live tabs; launchers"
```

### Task 6: End-to-end check in a real browser, then finish

**Files:**
- Modify: `showroom-suite/README.md` (measured timings)

- [ ] **Step 1: Start the suite.** Run `showroom-suite/run.bat` in the background and wait for the printed URL.

- [ ] **Step 2: Playwright (MCP) at 1280 px.**
  - Open `http://127.0.0.1:8000/` and upload `C:\Users\USER\Desktop\showroom-coach\data\samples\meva_a_60s.mp4`
    with preset Short.
  - Record:
    - the time to Coach's staff confirmation;
    - that Customers starts only after Coach has finished detection;
    - the time to Pulse's `ready_for_zones`.
  - In the Salespeople iframe: thumbnails load, confirm staff, then results arrive with the video playing
    (Range requests 206).
  - In the Customers iframe: go zones → roles → Score, and the report appears.
  - Check `browser_network_requests` for any 404 and `browser_console_messages` for errors. Expect none.
  - Reload the shell: both tabs come back from the hash.
  - Live tab:
    - each live page loads and its WebSocket connects (`/coach/ws/live` and `/pulse/ws/live` show 101);
    - with no camera available, each page shows its own camera error, not a blank frame;
    - switching between them leaves only one live page loaded.
  - At 390 px width: no horizontal page scroll in the shell.

- [ ] **Step 3: Failure path.** Upload a text file renamed `bad.mp4`. Coach's tab shows its failure message,
  Customers is still started and shows its own failure, and the shell stays usable.

- [ ] **Step 4: Final test runs.** Run all three suites. Record the counts and times in
  `showroom-suite/README.md` with the measured E2E timings, and commit:
  `docs(yak-45): Showroom Suite verified end to end`. Run `git status` and confirm it's clean (no models, jobs or
  runs files).

- [ ] **Step 5: Ask before pushing.** Show the owner `git log --oneline origin/main..HEAD | head -20`, the commit
  count and `git diff --stat origin/main | tail -5`, then STOP and wait for their yes.

- [ ] **Step 6: Push** (only after that yes). The token goes through an environment variable for this one command:

```bash
TOKEN='<the token from chat>' && git -c http.https://github.com/.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)" push -u origin yak-45/showroom-suite
git config --get-regexp 'http\..*extraheader' || echo "no stored credentials"
```
  Then tell the owner the branch URL, offer to open a PR, and remind them to revoke the token.
