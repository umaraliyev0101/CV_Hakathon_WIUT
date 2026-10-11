"""Thresholds, presets, camera/role settings and filesystem locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    greet_dist: float = 1.5
    greet_time: float = 5.0
    engage_dist: float = 2.0
    engage_time: float = 120.0
    gap_tol: float = 3.0
    wait_alert: float = 60.0
    leave_after: float = 10.0
    party_arrive: float = 5.0
    party_dist: float = 1.5
    party_window: float = 30.0
    facing_max_deg: float = 60.0
    edge_leave: float = 1.0  # grace before "left" when the last box touched a frame edge


PRESETS: dict[str, Thresholds] = {
    "dealership": Thresholds(),
    "short": Thresholds(greet_time=2.0, engage_time=8.0, gap_tol=1.5, wait_alert=10.0,
                        leave_after=3.0, party_window=10.0),
}
PRESET_LABELS = {
    "dealership": "Dealership (SKAIVISION thresholds)",
    "short": "Short clip (10–60 s)",
}


def get_preset(name: str) -> Thresholds:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}") from None


@dataclass(frozen=True)
class CameraConfig:
    hfov_deg: float = 70.0
    person_height_m: float = 1.70
    floor_points: tuple[tuple[float, float, float, float], ...] | None = None  # (u, v, X, Z) x4


@dataclass(frozen=True)
class RoleWeights:
    bias: float = -1.5
    outfit_peers: float = 1.2
    contacts: float = 1.6
    presence: float = 1.0
    initiator: float = 1.0
    staff_thr: float = 0.6
    customer_thr: float = 0.4
    outfit_sim_thr: float = 0.85
    contact_dist: float = 2.0
    contact_min_s: float = 5.0


# Detector specs are `<model>[@<tiles>]` (see perception.detect.parse_detector_spec). Both come from the
# M0 benchmark (runs/bench/detectors.json, 7 candidates): DEFAULT_DETECTOR = fastest whose recall is within
# 5 points of the best; LIVE_DETECTOR = fastest whose recall on people >= 80 px tall is within 5 points of
# the best on that bucket (webcam subjects are near the camera).
# Deployments can override both without code changes: SHOWROOM_DETECTOR sets the file-mode spec and
# SHOWROOM_LIVE_DETECTOR the live one (falling back to SHOWROOM_DETECTOR). Unset, empty or whitespace = default.
# e.g. SHOWROOM_DETECTOR=rfdetr_ft@2 with SHOWROOM_RFDETR_FT=/path/to/our_finetuned.onnx.
_BUILTIN_DETECTOR = "yolox_nano@2"


def detector_from_env(var: str, default: str, environ=None) -> str:
    """The detector spec in environment variable `var`, stripped; `default` when unset or blank."""
    value = (os.environ if environ is None else environ).get(var, "").strip()
    return value or default


DEFAULT_DETECTOR = detector_from_env("SHOWROOM_DETECTOR", _BUILTIN_DETECTOR)
LIVE_DETECTOR = detector_from_env("SHOWROOM_LIVE_DETECTOR", DEFAULT_DETECTOR)
SAMPLE_FPS = 5.0
MAX_WIDTH = 960
MAX_SECONDS = 300.0
# A track is "real" (worth describing, and kept by analysis) with at least this many observations spanning at
# least this many seconds; shorter ones are detector flicker. Perception skips their appearance work and analysis
# drops them (build_persons min_track_obs / min_track_s), so both use these constants.
MIN_TRACK_OBS = 3
MIN_TRACK_S = 1.0


def repo_root() -> Path:
    """Directory containing pyproject.toml, found by walking up from this file.

    Falls back to the current directory when the package is installed without the
    source tree (no pyproject.toml above it).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def jobs_dir() -> Path:
    env = os.environ.get("SHOWROOM_JOBS")
    return Path(env) if env else repo_root() / "jobs"


def cache_dir() -> Path:
    env = os.environ.get("SHOWROOM_CACHE")
    return Path(env) if env else repo_root() / "runs" / "cache"
