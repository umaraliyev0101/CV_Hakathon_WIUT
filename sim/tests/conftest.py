import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import behaviour, schema  # noqa: E402

CFG_DIR = REPO / "sim" / "configs"
SCENE_CFG = CFG_DIR / "scene_000.json"
ALL_SCENES = sorted(CFG_DIR.glob("scene_*.json"))


@pytest.fixture(scope="session")
def cfg():
    return schema.load_scene(SCENE_CFG)


@pytest.fixture(scope="session")
def presets():
    return json.loads((CFG_DIR / "presets.json").read_text())


@pytest.fixture(scope="session")
def timeline(cfg):
    return behaviour.build_timeline(cfg)


_TL_CACHE = {}


def scene_and_timeline(path):
    if path not in _TL_CACHE:
        c = schema.load_scene(path)
        _TL_CACHE[path] = (c, behaviour.build_timeline(c))
    return _TL_CACHE[path]


@pytest.fixture(scope="session", params=ALL_SCENES, ids=[p.stem for p in ALL_SCENES])
def scene(request):
    return scene_and_timeline(request.param)
