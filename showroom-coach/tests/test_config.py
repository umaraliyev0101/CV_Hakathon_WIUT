import pytest
from showroom_coach.config import PRESETS, get_preset


def test_presets_match_spec():
    d, s = PRESETS["dealership"], PRESETS["short"]
    assert (d.greet_dist, d.greet_time, d.engage_dist, d.engage_time) == (1.5, 5.0, 2.0, 120.0)
    assert (d.gap_tol, d.wait_alert, d.leave_after) == (3.0, 60.0, 10.0)
    assert (s.greet_dist, s.greet_time, s.engage_dist, s.engage_time) == (1.5, 2.0, 2.0, 8.0)
    assert (s.gap_tol, s.wait_alert, s.leave_after) == (1.5, 10.0, 3.0)


def test_unknown_preset():
    with pytest.raises(ValueError):
        get_preset("nope")


def test_repo_root_contains_pyproject():
    from showroom_coach.config import repo_root
    assert (repo_root() / "pyproject.toml").is_file()


def test_dirs_default_and_env_override(monkeypatch, tmp_path):
    from showroom_coach.config import cache_dir, jobs_dir, repo_root
    monkeypatch.delenv("SHOWROOM_JOBS", raising=False)
    monkeypatch.delenv("SHOWROOM_CACHE", raising=False)
    assert jobs_dir() == repo_root() / "jobs"
    assert cache_dir() == repo_root() / "runs" / "cache"
    monkeypatch.setenv("SHOWROOM_JOBS", str(tmp_path / "j"))
    monkeypatch.setenv("SHOWROOM_CACHE", str(tmp_path / "c"))
    assert jobs_dir() == tmp_path / "j"
    assert cache_dir() == tmp_path / "c"
