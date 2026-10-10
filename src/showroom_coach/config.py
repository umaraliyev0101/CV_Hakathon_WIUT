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
DEFAULT_DETECTOR = "yolox_nano@2"
LIVE_DETECTOR = "yolox_nano@2"
SAMPLE_FPS = 5.0
MAX_WIDTH = 960
MAX_SECONDS = 300.0


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
