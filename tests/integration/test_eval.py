import json

import pytest

from pulse.config import get_config
from pulse.eval import run_eval

pytestmark = pytest.mark.slow

HEADINGS = ["Entries and exits", "Occupancy", "Zone dwell", "Journeys", "Tracking", "Speed"]


def test_evaluation_is_deterministic_without_the_memo():
    """Two cold evaluations (no runs/eval memo read) of the first dev clip give identical results. This is
    what proves `evaluate` itself is deterministic; the file-level test below only proves rendering."""
    cfg = get_config("dealership")
    entry = json.loads((run_eval.ROOT / "labels" / "split.json").read_text())["dev"][0]
    a = run_eval.run_clip(entry, "dev", cfg, use_cache=False)
    b = run_eval.run_clip(entry, "dev", cfg, use_cache=False)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["n_gt_persons"] > 0 and a["entries"]["gt"] > 0


def test_run_eval_writes_all_sections(tmp_path):
    """The generated document has every required section; the second run (which may reuse the per-clip
    memo) renders byte-identical text. Evaluation determinism is covered by the test above."""
    out = tmp_path / "EVAL.md"
    assert run_eval.main(out) == 0
    text = out.read_text()
    for h in HEADINGS:
        assert f"## {h}" in text
    assert "| metric | target | measured (test) | measured (dev) | met? |" in text
    assert "## Stops" in text and "**Many-to-many matching**" in text
    assert "| clip | length | usable (stable, same view) | precision |" in text
    assert "| wm-bangalore-04 |" in text and "| wm-bangalore-07 |" in text
    assert "Limitations" in text and "Task 20" not in text
    assert "Test numbers were visible during tuning" in text
    out2 = tmp_path / "EVAL2.md"
    assert run_eval.main(out2) == 0
    assert out2.read_text() == text
