import math

import numpy as np

from sim import behaviour, schema


def test_timeline_deterministic_for_seed(cfg):
    h1 = behaviour.timeline_hash(behaviour.build_timeline(cfg))
    h2 = behaviour.timeline_hash(behaviour.build_timeline(schema.load_scene_dict(cfg)))
    assert h1 == h2


def test_timeline_changes_with_seed(cfg):
    other = dict(cfg, seed=cfg["seed"] + 1)
    assert behaviour.timeline_hash(behaviour.build_timeline(other)) != behaviour.timeline_hash(
        behaviour.build_timeline(cfg))


def test_frame_count(cfg, timeline):
    n = int(round(cfg["duration_s"] * cfg["fps_scene"]))
    for agent in timeline["agents"].values():
        assert len(agent["frames"]) == n


def _ang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_walking_speed_and_smooth_yaw(cfg, timeline):
    fps = cfg["fps_scene"]
    rate = cfg["navigation"]["turn_rate_dps"]
    for pid, agent in timeline["agents"].items():
        lo, hi = schema.person(cfg, int(pid))["speed_mps"]
        fr = agent["frames"]
        speeds = []
        for a, b in zip(fr[:-1], fr[1:]):
            v = math.dist(a["pos_m"], b["pos_m"]) * fps
            assert v <= hi + 1e-6, f"person {pid} too fast: {v}"
            if a["action"] == "walk":
                speeds.append(v)
            assert _ang(a["yaw_deg"], b["yaw_deg"]) <= rate / fps + 1e-6, f"person {pid} yaw jump"
        if speeds:
            assert lo - 1e-6 <= max(speeds) <= hi + 1e-6


def test_gait_feet_on_floor(cfg, timeline):
    for pid, agent in timeline["agents"].items():
        h = schema.person(cfg, int(pid))["height_m"]
        for st in agent["frames"][::5]:
            pose = behaviour.mannequin_pose(st, h)
            feet = behaviour.foot_heights(pose, h)
            assert abs(min(feet)) < 1e-3, f"lowest foot at {min(feet)}"
            assert max(feet) < 0.25


def test_gait_moves_while_walking_and_talking(cfg, timeline):
    """Guard against frozen mannequins: limb angles must change over 2 s of walk or talk."""
    fps = cfg["fps_scene"]
    for pid, agent in timeline["agents"].items():
        h = schema.person(cfg, int(pid))["height_m"]
        fr = agent["frames"]
        for i in range(0, len(fr) - 2 * fps, fps):
            win = fr[i:i + 2 * fps]
            if all(s["action"] in ("walk", "talk") for s in win):
                poses = np.array([list(behaviour.mannequin_pose(s, h)["joints"].values()) for s in win])
                assert np.ptp(poses, axis=0).max() > 2.0, f"person {pid} frozen near frame {i}"


def test_planted_foot_does_not_slide(cfg, timeline):
    """While walking (including the speed ramps) in a straight line, the planted foot stays put."""
    fps = cfg["fps_scene"]
    for pid, agent in timeline["agents"].items():
        h = schema.person(cfg, int(pid))["height_m"]
        fr = agent["frames"]
        slides = []
        for a, b in zip(fr[:-1], fr[1:]):
            if not ("walk" in (a["action"], b["action"]) and abs(a["yaw_deg"] - b["yaw_deg"]) < 1e-6):
                continue
            fa = behaviour.foot_world(a, behaviour.mannequin_pose(a, h), h)
            fb = behaviour.foot_world(b, behaviour.mannequin_pose(b, h), h)
            for side in ("l", "r"):
                if fa[side][2] < 1e-4 and fb[side][2] < 1e-4:   # planted in both frames
                    slides.append(math.dist(fa[side][:2], fb[side][:2]) * fps)
        assert slides, f"person {pid}: no planted-foot frames found"
        assert max(slides) < 0.05, f"person {pid}: planted foot slides at {max(slides):.3f} m/s"
