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


@pytest.fixture(scope="session")
def meva_10s(meva_dev_clip, tmp_path_factory) -> Path:
    """10 s H.264 cut of the dev clip from the 60 s mark, width 960."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_clip", ROOT / "scripts" / "make_clip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dst = tmp_path_factory.mktemp("meva10") / "meva_10s.mp4"
    return mod.make_clip(meva_dev_clip, dst, start=60, dur=10, width=960)


@pytest.fixture(scope="session")
def showroom_clip() -> Path:
    """First stock showroom clip, else first synthetic share clip."""
    stock = ROOT / "data" / "stock"
    found = sorted(p for ext in ("mp4", "mov", "webm") for p in stock.glob(f"*.{ext}"))
    if not found:
        found = sorted((ROOT / "data" / "synthetic" / "share").glob("*.mp4"))
    if not found:
        pytest.fail("No showroom clip: put one in data/stock/ (*.mp4|*.mov|*.webm) "
                    "or data/synthetic/share/ (*.mp4)")
    return found[0]
