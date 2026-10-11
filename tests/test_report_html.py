"""Printable one-page HTML report (Task 24)."""
import base64
import json
import re
from html.parser import HTMLParser

import pytest

from pulse import pipeline
from pulse.config import get_config
from pulse.datatypes import Zone
from tests.test_report import _check_module, _setup

BANNER_TEXT = "Anonymous, aggregated analytics; decision support only."

EVIL = "<script>alert(1)</script>"


@pytest.fixture
def built_html(tmp_path):
    video, pc, zones = _setup(tmp_path)
    zones = [Zone(z.zone_id, EVIL, z.kind, z.polygon, z.source) for z in zones]
    cfg = get_config("short")
    a = pipeline.score(pc, cfg, zones=zones)
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=False)
    return out, a


def test_report_html_listed_and_validated(built_html, capsys):
    out, _ = built_html
    rep = json.loads((out / "report.json").read_text())
    assert rep["files"]["report_html"] == "report.html"
    assert (out / "report.html").is_file()
    assert _check_module().main([str(out), "--no-video"]) == 0
    assert "OK report.html" in capsys.readouterr().out


def test_report_html_is_self_contained_and_has_content(built_html):
    out, a = built_html
    html = (out / "report.html").read_text()
    assert BANNER_TEXT in html
    assert "@media print" in html and "A4" in html
    assert "v.mp4" in html                                   # generated-from line names the clip
    assert "<script" not in html.lower()
    assert not re.search(r"(?i)(src|href)\s*=\s*[\"']?\s*(https?:)?//", html)
    m = re.search(r'<img[^>]+src="data:image/png;base64,([A-Za-z0-9+/=]+)"', html)
    assert m, "heatmap must be embedded as a data URI"
    assert base64.b64decode(m.group(1)) == (out / "heatmap.png").read_bytes()   # the blurred heatmap
    for k in ("Visitors", "Per-car"):
        assert k in html


def test_zone_names_are_escaped(built_html):
    out, _ = built_html
    html = (out / "report.html").read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert EVIL not in html


def test_html_parses_and_has_no_script_tags(built_html):
    out, _ = built_html
    tags = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            tags.append(tag)
    P().feed((out / "report.html").read_text())
    assert "table" in tags and "script" not in tags


def test_insights_rendered_escaped(tmp_path):
    from pulse.report_html import render_html
    rep = {"banner": "Anonymous, aggregated analytics; decision support only.",
           "meta": {"video": "<b>x</b>.mp4", "preset": "short", "analysed_duration": 10},
           "store": {}, "per_car": [], "warnings": ["<i>warn</i>"],
           "insights": [{"title": "T<1>", "evidence": {"a": "<e>"}, "action": "do <it>",
                         "confidence": "low", "sample": 3}] * 5}
    html = render_html(rep, b"\x89PNG")
    assert html.count("T&lt;1&gt;") == 3                      # top three only
    assert "&lt;b&gt;x&lt;/b&gt;.mp4" in html and "&lt;i&gt;warn&lt;/i&gt;" in html
    assert "&lt;e&gt;" in html and "do &lt;it&gt;" in html
