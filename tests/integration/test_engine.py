import pytest

from pulse import sources
from pulse.config import get_config
from pulse.engine import PerceptionEngine

pytestmark = pytest.mark.slow


def test_engine_on_meva_10s(meva_10s, capsys):
    cfg = get_config("short")
    eng = PerceptionEngine(cfg)
    per_frame = []
    for k, (t, frame) in enumerate(sources.iter_frames(meva_10s, cfg.sample_fps, cfg.max_width)):
        per_frame.append(eng.process(t, frame, k))
    assert len(per_frame) >= 49
    all_obs = [o for f in per_frame for o in f]
    assert len(all_obs) >= 1, "no persons tracked"
    # stable ids: some id seen at frame k reappears in >= 80% of the next 10 frames
    stable = False
    for k, obs in enumerate(per_frame[:-10]):
        for o in obs:
            hits = sum(any(p.track_id == o.track_id for p in per_frame[k + j]) for j in range(1, 11))
            if hits >= 8:
                stable = True
    assert stable
    kp = next(o.keypoints for o in all_obs if o.keypoints is not None)
    assert len(kp) == 17 and all(len(r) == 3 for r in kp)
    with capsys.disabled():
        print("\nTIMINGS ms", {k: round(v, 1) for k, v in eng.timings.items()},
              "persons-obs", len(all_obs), "frames", len(per_frame))


def test_vehicles_on_showroom_middle_frame(showroom_clip):
    cfg = get_config("short")
    eng = PerceptionEngine(cfg)
    info = sources.probe(showroom_clip)
    target = max(0.0, info.duration / 2)
    frame = None
    for t, f in sources.iter_frames(showroom_clip, cfg.sample_fps, cfg.max_width):
        frame = f
        if t >= target:
            break
    assert frame is not None
    assert len(eng.detect_vehicles(frame)) >= 1
