"""Static frontend: pages are served, banner shown, no external hosts, JS endpoints exist."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pulse.app.main import STATIC, create_app

BANNER = "Anonymous, aggregated analytics; decision support only."


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return create_app(jobs_dir=tmp_path_factory.mktemp("jobs"))


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/live", "/static/app.js", "/static/zones.js",
                                  "/static/dashboard.js", "/static/i18n.js", "/static/style.css"])
def test_pages_served(client, path):
    assert client.get(path).status_code == 200


def test_index_has_banner_and_live_link(client):
    html = client.get("/").text
    assert BANNER in html
    assert 'href="/live"' in html


def test_no_external_hosts():
    files = [p for p in Path(STATIC).rglob("*") if p.is_file()]
    assert files
    pat = re.compile(r"https?://", re.I)
    for p in files:
        for m in pat.finditer(p.read_text(errors="ignore")):
            line = p.read_text(errors="ignore")[m.start():m.start() + 60]
            # the SVG namespace URI is an identifier, not a network request
            assert line.startswith("http://www.w3.org/"), f"{p.name}: external URL {line!r}"


def _norm(path: str) -> str:
    return re.sub(r"\$\{[^}]*\}|\{[^}]*\}", "{}", path)


def test_js_api_paths_exist_as_routes(app):
    routes = {_norm(r.path) for r in app.routes if r.path.startswith("/api/")}
    found = set()
    for js in Path(STATIC).glob("*.js"):
        for m in re.finditer(r"/api/[A-Za-z0-9_/.${}\-]*", js.read_text()):
            found.add(_norm(m.group(0).rstrip("/")))
    assert found, "no /api paths found in JS"
    missing = found - routes
    assert not missing, f"JS references unknown endpoints: {sorted(missing)}"


def test_live_page_never_displays_raw_video():
    css = (Path(STATIC) / "style.css").read_text()
    js = (Path(STATIC) / "live.js").read_text()
    assert re.search(r"\.stage video\s*\{[^}]*visibility:\s*hidden", css)
    assert "pendingFrame" in js and "shownFrame" in js
    assert not re.search(r"drawImage\(\s*video\b", js.split("function pump")[1].split("// ----")[1]), \
        "the visible canvas must be drawn from shownFrame, never the raw video"
    # only the hidden grab canvas may read the raw video
    assert len(re.findall(r"drawImage\(\s*video\b", js)) == 1


def test_stepper_badge_selector_only_matches_number_span():
    css = (Path(STATIC) / "style.css").read_text()
    assert not re.search(r"\.steps li(\.\w+)? span\b", css)


def test_short_preset_checked_by_default():
    html = (Path(STATIC) / "index.html").read_text()
    assert re.search(r'value="short"[^>]*checked', html)
    assert not re.search(r'value="dealership"[^>]*checked', html)
