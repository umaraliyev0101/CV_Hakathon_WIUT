"""Pre-render check from geometry: both people in the camera view for >= 80 % of the clip."""
import numpy as np

from sim import projection, schema


def test_both_people_in_view(cfg, timeline):
    cam = projection.Camera.from_params(cfg["camera"])
    step = cfg["frame_step"]
    for pid, agent in timeline["agents"].items():
        h = schema.person(cfg, int(pid))["height_m"]
        fr = agent["frames"][::step]
        seen = 0
        for st in fr:
            x, y = st["pos_m"]
            pts = [np.array([x, y, z]) for z in (0.1, 0.5 * h, 0.9 * h)]
            if st["present"] and sum(cam.in_view(p) for p in pts) >= 2:
                seen += 1
        assert seen / len(fr) >= 0.8, f"person {pid} in view {seen}/{len(fr)}"
