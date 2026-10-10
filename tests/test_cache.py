from dataclasses import replace

import numpy as np

from pulse import cache
from pulse.config import get_config
from pulse.datatypes import Det, StableSegment, TrackObs


def make_pc(d):
    kp = [[float(i), float(i) + 0.5, 0.9] for i in range(17)]
    obs = [TrackObs(0.0, 0, 1, (1.0, 2.0, 30.0, 60.0), 0.9, kp),
           TrackObs(0.2, 1, 1, (2.0, 2.0, 31.0, 60.0), 0.8, None),
           TrackObs(0.2, 1, 2, (5.0, 5.0, 40.0, 70.0), 0.7, None)]
    return cache.PerceptionCache(
        dir=d, meta={"video": "x.mp4", "n_samples": 2, "rtf": 3.5, "frame_w": 64, "frame_h": 48},
        obs=obs, cars=[Det((1, 2, 3, 4), 0.8, "car")], keyframe_t=0.2,
        segments=[StableSegment(0.0, 0.2, True, True)],
        embeddings={1: np.arange(5, dtype=float) / 10, 2: np.ones(5)})


def test_round_trip(tmp_path):
    d = tmp_path / "abc"
    assert not cache.exists(d)
    pc = make_pc(d)
    cache.save(pc)
    assert cache.exists(d)
    assert not [p for p in tmp_path.iterdir() if p.name != "abc"]  # no temp leftovers
    got = cache.load(d)
    assert got.meta == pc.meta and got.keyframe_t == 0.2
    assert got.obs == pc.obs
    assert got.obs[1].keypoints is None and got.obs[0].keypoints is not None
    assert got.cars == pc.cars and got.segments == pc.segments
    assert set(got.embeddings) == {1, 2}
    assert np.allclose(got.embeddings[1], pc.embeddings[1])


def test_partial_dir_is_not_a_cache(tmp_path):
    d = tmp_path / "half"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    assert not cache.exists(d)


def test_video_key_depends_on_config(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"abc" * 1000)
    cfg = get_config("short")
    k = cache.video_key(v, cfg)
    assert k == cache.video_key(v, cfg)
    assert k != cache.video_key(v, replace(cfg, sample_fps=2.0))
    v.write_bytes(b"abd" * 1000)
    assert k != cache.video_key(v, cfg)


def test_raw_detections_round_trip(tmp_path):
    pc = make_pc(tmp_path / "abc")
    pc.raw_dets = [(0.0, (1.0, 2.0, 30.0, 60.0), 0.25), (0.2, (5.0, 5.0, 40.0, 70.0), 0.3)]
    cache.save(pc)
    got = cache.load(tmp_path / "abc")
    assert got.raw_dets == pc.raw_dets
    (tmp_path / "abc" / "raw_dets.npz").unlink()
    assert not cache.exists(tmp_path / "abc")            # an old cache without raw detections is rebuilt


def test_video_key_changes_with_cache_version(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"abc" * 1000)
    assert cache.CACHE_VERSION >= 2
