import pytest

from pulse import cache, pipeline
from pulse.config import get_config
from pulse.sources import VideoError


def test_missing_and_corrupt_video_raise_and_leave_no_cache(tmp_path):
    cfg = get_config("short")
    root = tmp_path / "cache"
    runs = cache.perception_runs()
    with pytest.raises(VideoError):
        pipeline.perceive(tmp_path / "nope.mp4", root, cfg, engine=object())
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    with pytest.raises(VideoError):
        pipeline.perceive(bad, root, cfg, engine=object())
    assert not root.exists() or not list(root.iterdir())
    assert cache.perception_runs() == runs
