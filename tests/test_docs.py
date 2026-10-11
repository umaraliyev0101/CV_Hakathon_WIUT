"""Docs checks (Task 21): licence table completeness, privacy heading, pitch/integration content."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
HEADER = "| Component | Licence | Trained on | Where used |"


def _table_rows(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    i = lines.index(header)
    rows = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        rows.append(ln)
    return rows


def test_readme_licence_table_header_and_manifest_rows():
    assert HEADER in README.splitlines()
    rows = "\n".join(_table_rows(README, HEADER))
    manifest = json.loads((ROOT / "models" / "MANIFEST.json").read_text())["models"]
    assert manifest
    for m in manifest:
        assert f"`models/{m['file'].split('/')[0]}`" in rows, m["name"]
    assert "MEVA" in rows and "CC BY 4.0" in rows and "Corona" in rows


def test_readme_licence_table_covers_pyproject_dependencies():
    toml = (ROOT / "pyproject.toml").read_text()
    deps = re.search(r"dependencies = \[(.*?)\]", toml, re.S).group(1)
    names = [re.match(r"\s*\"([A-Za-z0-9_.\-]+)", ln).group(1)
             for ln in deps.splitlines() if ln.strip().startswith('"')]
    rows = "\n".join(_table_rows(README, HEADER)).lower()
    for n in names:
        assert n.lower() in rows, n
    for extra in ("rtmlib", "ffmpeg", "pyav", "wikimedia"):
        assert extra in rows, extra


def test_readme_privacy_and_results_sections():
    assert re.search(r"^#+ Privacy\b", README, re.M)
    assert re.search(r"^#+ Results\b", README, re.M)
    assert re.search(r"^#+ Limitations\b", README, re.M)
    assert "Anonymous, aggregated analytics; decision support only." in README
    assert "docs/EVAL.md" in README


def test_readme_results_copy_eval_headlines():
    eval_md = (ROOT / "docs" / "EVAL.md").read_text()
    row = next(ln for ln in eval_md.splitlines() if ln.startswith("| entry/exit count error"))
    measured = row.split("|")[3].strip()
    assert measured in README


def test_pitch_notes():
    t = (ROOT / "docs" / "PITCH_NOTES.md").read_text()
    for c in ("V-Count", "FootfallCam", "Deepingsource", "Palexy"):
        assert c in t
    assert "ASSUMPTION" in t
    assert "Royal Motors" in t


def test_integration_table():
    t = (ROOT / "docs" / "INTEGRATION.md").read_text()
    assert "| Interface | Showroom Coach spec | Showroom Pulse |" in t
    for k in ("datatypes.py", "render_timeline", "observations_by_t", "roles.suggest",
              "identity.merge", "position_m", "/ws/live", "raw_dets.npz", "roles.json"):
        assert k in t, k


def test_run_sh_is_executable():
    p = ROOT / "run.sh"
    assert p.is_file() and os.access(p, os.X_OK)
    s = p.read_text()
    assert s.startswith("#!/usr/bin/env bash")
    assert "--ws-max-size 4194304" in s and "127.0.0.1" in s
