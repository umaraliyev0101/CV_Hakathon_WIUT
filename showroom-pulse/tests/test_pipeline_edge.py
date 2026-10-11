from pulse import cache, pipeline
from pulse.config import get_config


def synthetic_cache(tmp_path, obs, duration=20.0, trimmed=False):
    meta = {"video": "synthetic", "sha1": "x", "key": "k", "src_w": 960, "src_h": 540, "src_fps": 25.0,
            "duration": duration, "analysed_duration": duration, "trimmed": trimmed, "frame_w": 960,
            "frame_h": 540, "sample_fps": 5.0, "n_samples": int(duration * 5), "detector": "test",
            "pose_model": "test", "providers": ["CPUExecutionProvider"], "perception_seconds": 1.0,
            "rtf": duration}
    from pulse.datatypes import StableSegment
    return cache.PerceptionCache(tmp_path, meta, obs, [], duration / 2,
                                 [StableSegment(0, duration, True, True)], {})


def test_score_with_no_people(tmp_path):
    a = pipeline.score(synthetic_cache(tmp_path, []), get_config("short"), zones=[])
    assert a.stats.store["visitors"] == 0 and a.insights == []
    assert any("Nobody detected" in w for w in a.stats.warnings)


def test_score_with_no_zones(tmp_path):
    from pulse.datatypes import TrackObs
    obs = [TrackObs(round(i * 0.2, 3), i, 1, (400, 200, 460, 400), 0.9, None) for i in range(50)]
    a = pipeline.score(synthetic_cache(tmp_path, obs), get_config("short"), zones=[])
    assert a.stats.store["visitors"] == 1 and a.stats.per_car == []
    assert any("No zones" in w for w in a.stats.warnings)


def test_trim_warning_in_stats(tmp_path):
    a = pipeline.score(synthetic_cache(tmp_path, [], trimmed=True), get_config("short"), zones=[])
    assert any("trimmed" in w for w in a.stats.warnings)


def test_rescore_does_not_run_models(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model path touched during re-scoring")
    monkeypatch.setattr(pipeline, "PerceptionEngine", boom)
    monkeypatch.setattr(pipeline, "perceive", boom)
    before = cache.perception_runs()
    pipeline.score(synthetic_cache(tmp_path, []), get_config("dealership"), zones=[])
    assert cache.perception_runs() == before


def test_entry_during_unstable_segment_still_counted(tmp_path):
    from pulse.datatypes import StableSegment, TrackObs
    obs = [TrackObs(round(4 + i * 0.2, 3), i, 1, (0, 200, 60, 400), 0.9, None) for i in range(20)]
    pc = synthetic_cache(tmp_path, obs)
    pc.segments = [StableSegment(0, 10, False, False), StableSegment(10, 20, True, True)]
    a = pipeline.score(pc, get_config("short"), zones=[])
    assert a.stats.store["visitors"] == 1 and a.stats.store["entries"] == 1
    assert any("analysed" in w for w in a.stats.warnings)
