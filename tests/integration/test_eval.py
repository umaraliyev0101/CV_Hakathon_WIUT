import pytest

from pulse.eval import run_eval

pytestmark = pytest.mark.slow

HEADINGS = ["Entries and exits", "Occupancy", "Zone dwell", "Journeys", "Tracking", "Speed"]


def test_run_eval_writes_all_sections_deterministically(tmp_path):
    out = tmp_path / "EVAL.md"
    assert run_eval.main(out) == 0
    text = out.read_text()
    for h in HEADINGS:
        assert f"## {h}" in text
    assert "| metric | target | measured (test) | measured (dev) | met? |" in text
    assert "Stops precision/recall: no labelled showroom clip yet" in text
    assert "Limitations" in text and "Task 20" not in text
    out2 = tmp_path / "EVAL2.md"
    assert run_eval.main(out2) == 0
    assert out2.read_text() == text                      # byte-identical across runs
