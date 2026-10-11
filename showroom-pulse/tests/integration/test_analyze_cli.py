import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PULSE = ROOT / ".venv" / "bin" / "pulse"

pytestmark = pytest.mark.slow


def _run(*args):
    return subprocess.run([str(PULSE), *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_analyze_meva_10s_and_check_outputs(meva_10s, tmp_path):
    out = tmp_path / "out"
    r = _run("analyze", meva_10s, "--out", out, "--cache", tmp_path / "cache")
    assert r.returncode == 0, r.stderr
    assert "visitors" in r.stdout
    c = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_outputs.py"), str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert c.returncode == 0, c.stdout + c.stderr


def test_analyze_garbage_file_exits_2(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    r = _run("analyze", bad, "--out", tmp_path / "o", "--cache", tmp_path / "cache")
    assert r.returncode == 2
    assert "Unreadable" in r.stderr
