"""Scene config loading/validation and small output helpers (pure, no Blender)."""
import copy
import json
from pathlib import Path

REQUIRED = ("scene_id", "seed", "scenario", "duration_s", "fps_scene", "frame_step", "lightings",
            "room", "entrance", "cars", "zone_apron_m", "desk", "cameras", "navigation", "persons")
LIGHTINGS = ("day", "evening_dim", "glare")
CAM_LIMITS = {"height_m": (2.6, 3.5), "pitch_deg": (-50.0, -25.0), "hfov_deg": (60.0, 100.0)}
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
        if "script" not in p:
            raise ConfigError(f"person {p['id']}: missing script")
    for lt in cfg["lightings"]:
        if lt not in LIGHTINGS:
            raise ConfigError(f"unknown lighting {lt}")
    if not cfg["cameras"]:
        raise ConfigError("no cameras")
    for cam in cfg["cameras"]:
        for k in ("id", "position_m", "yaw_deg", "pitch_deg", "hfov_deg", "width_px", "height_px"):
            if k not in cam:
                raise ConfigError(f"camera missing {k}")
        lo, hi = CAM_LIMITS["height_m"]
        if not lo <= cam["position_m"][2] <= hi:
            raise ConfigError(f"camera {cam['id']}: height outside {lo}-{hi} m")
        for k in ("pitch_deg", "hfov_deg"):
            lo, hi = CAM_LIMITS[k]
            if not lo <= cam[k] <= hi:
                raise ConfigError(f"camera {cam['id']}: {k} {cam[k]} outside {lo}..{hi}")
    return cfg


def camera(cfg, cam_id):
    for c in cfg["cameras"]:
        if c["id"] == cam_id:
            return c
    raise KeyError(cam_id)


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


def clip_id(cfg, cam_id, lighting):
    return f"{cfg['scene_id']}_{cfg['scenario']}_cam{cam_id}_{lighting}"


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False))
