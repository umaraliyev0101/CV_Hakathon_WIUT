"""Pre-render geometry check: in every short-clip scene, every camera sees every person (full body in
frame) for >= 80 % of the clip's output frames."""
import numpy as np

from sim import projection, schema
from sim.tests.conftest import scene_and_timeline, ALL_SCENES

SHORT = ("greet_fast", "greet_slow", "walkout", "party", "handoff")


def test_people_in_view_short_scenes():
    for path in ALL_SCENES:
        cfg, tl = scene_and_timeline(path)
        if cfg["scenario"] not in SHORT:
            continue
        for cp in cfg["cameras"]:
            cam = projection.Camera.from_params(cp)
            for pid, agent in tl["agents"].items():
                h = schema.person(cfg, int(pid))["height_m"]
                fr = agent["frames"][::cfg["frame_step"]]
                seen = sum(1 for st in fr if st["rendered"] and
                           sum(cam.in_view(np.array([st["pos_m"][0], st["pos_m"][1], z])) for z in (0.1, 0.5 * h, 0.9 * h)) >= 2)
                assert seen / len(fr) >= 0.8, f"{path.name} cam {cp['id']} person {pid}: {seen}/{len(fr)}"
