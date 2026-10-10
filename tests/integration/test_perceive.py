import pytest

from pulse import cache, pipeline
from pulse.config import get_config

pytestmark = pytest.mark.slow


def test_perceive_caches_and_builds_people(meva_10s, tmp_path, capsys):
    cfg = get_config("short")
    fracs = []
    pc = pipeline.perceive(meva_10s, tmp_path, cfg, progress=fracs.append)
    assert cache.exists(pc.dir)
    assert fracs and fracs[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fracs)
    runs = cache.perception_runs()
    pc2 = pipeline.perceive(meva_10s, tmp_path, cfg)
    assert cache.perception_runs() == runs
    assert len(pc2.obs) == len(pc.obs)
    persons, timeline, t2p = pipeline.build_people(pc, cfg)
    assert len(persons) >= 1 and any(len(s) > 0 for s in timeline.values())
    assert pc.meta["rtf"] > 0
    with capsys.disabled():
        print("\nMETA", {k: pc.meta[k] for k in ("rtf", "perception_seconds", "embed_seconds", "n_samples")})
