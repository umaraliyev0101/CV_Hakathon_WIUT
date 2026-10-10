"""Scene config loading/validation and small output helpers (pure, no Blender)."""
import copy
import json
from pathlib import Path

REQUIRED = ("scene_id", "seed", "scenario", "duration_s", "fps_scene", "frame_step", "lighting",
            "room", "entrance", "cars", "zone_apron_m", "desk", "camera", "navigation", "persons", "script")
ROLES = ("staff", "customer")


class ConfigError(ValueError):
    pass


def load_scene(path):
    cfg = json.loads(Path(path).read_text())
    cfg["_config_path"] = str(path)
    return load_scene_dict(cfg)


def load_scene_dict(cfg):
    cfg = copy.deepcopy(cfg)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"scene config missing keys: {missing}")
    if cfg["fps_scene"] % cfg["frame_step"]:
        raise ConfigError("fps_scene must be a multiple of frame_step")
    ids = [p["id"] for p in cfg["persons"]]
    if len(set(ids)) != len(ids):
        raise ConfigError("duplicate person ids")
    for p in cfg["persons"]:
        if p["role"] not in ROLES:
            raise ConfigError(f"person {p['id']}: bad role {p['role']}")
        lo, hi = p["speed_mps"]
        if not (1.1 <= lo <= hi <= 1.5):
            raise ConfigError(f"person {p['id']}: speed range must lie in 1.1-1.5 m/s")
    cam = cfg["camera"]
    for k in ("position_m", "yaw_deg", "pitch_deg", "hfov_deg", "width_px", "height_px"):
        if k not in cam:
            raise ConfigError(f"camera missing {k}")
    return cfg


def person(cfg, pid):
    for p in cfg["persons"]:
        if p["id"] == pid:
            return p
    raise KeyError(pid)


def out_fps(cfg):
    return cfg["fps_scene"] // cfg["frame_step"]


def n_scene_frames(cfg):
    return int(round(cfg["duration_s"] * cfg["fps_scene"]))


def output_scene_frames(cfg):
    """Scene-frame indices that are rendered (frame_step sampling), starting at 0."""
    return list(range(0, n_scene_frames(cfg), cfg["frame_step"]))


def clip_id(cfg):
    return f"{cfg['scene_id']}_{cfg['scenario']}_cam{cfg['camera']['id']}_{cfg['lighting']}"


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False))
