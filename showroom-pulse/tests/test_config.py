from pulse.config import PRESETS, get_config

def test_presets_match_spec_table():
    d, s = PRESETS["dealership"], PRESETS["short"]
    assert (d.leave_after, s.leave_after) == (10, 3)
    assert (d.zone_min, d.zone_gap, s.zone_min, s.zone_gap) == (1.5, 2, 1, 1)
    assert (d.stop_speed, d.stop_time, s.stop_speed, s.stop_time) == (0.3, 10, 0.3, 3)
    assert (d.interest_time, s.interest_time) == (60, 10)
    assert (d.assist_dist, d.assist_time, s.assist_dist, s.assist_time) == (2.0, 5, 2.0, 2)
    assert (d.alert_after, s.alert_after) == (45, 8)
    assert (d.desk_time, s.desk_time) == (30, 5)
    assert (d.min_sample, s.min_sample) == (5, 2)
    assert s.demo_only and not d.demo_only

def test_get_config_overrides_and_preset():
    c = get_config("dealership", sample_fps=10)
    assert c.preset.name == "dealership" and c.sample_fps == 10 and c.max_width == 960

def test_unknown_preset_raises():
    import pytest
    with pytest.raises(KeyError):
        get_config("nope")
