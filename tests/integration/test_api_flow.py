import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "smoke_api.py"


def test_smoke_api_main_returns_zero():
    spec = importlib.util.spec_from_file_location("smoke_api", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
