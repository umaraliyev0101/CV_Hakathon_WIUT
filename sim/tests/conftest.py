import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import behaviour, schema  # noqa: E402

SCENE_CFG = REPO / "sim" / "configs" / "scene_000.json"


@pytest.fixture(scope="session")
def cfg():
    return schema.load_scene(SCENE_CFG)


@pytest.fixture(scope="session")
def presets():
    return json.loads((REPO / "sim" / "configs" / "presets.json").read_text())


@pytest.fixture(scope="session")
def timeline(cfg):
    return behaviour.build_timeline(cfg)
