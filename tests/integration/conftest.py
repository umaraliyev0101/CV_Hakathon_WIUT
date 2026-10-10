import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def meva_dev_clip() -> Path:
    """First dev clip from labels/split.json (Task 12 extends this fixture)."""
    split = ROOT / "labels" / "split.json"
    try:
        video = ROOT / json.loads(split.read_text())["dev"][0]["video"]
    except (OSError, KeyError, IndexError, ValueError):
        video = None
    if video is None or not video.exists():
        pytest.fail("MEVA dev clip missing: run data download (see data/TODO_DOWNLOADS.md)")
    return video
