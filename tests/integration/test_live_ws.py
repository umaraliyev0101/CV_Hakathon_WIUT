import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_smoke_live_end_to_end(meva_dev_clip, capsys):
    """Real models over the websocket: overlays, dwell timers and an alert from a MEVA dev window."""
    spec = importlib.util.spec_from_file_location("smoke_live", ROOT / "scripts" / "smoke_live.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--seconds", "20", "--max-seconds", "3"]) == 0
    out = capsys.readouterr().out
    assert "SMOKE LIVE OK" in out and "processed fps:" in out
