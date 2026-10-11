from __future__ import annotations
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Preset:
    name: str
    leave_after: float
    zone_min: float
    zone_gap: float
    stop_speed: float
    stop_time: float
    interest_time: float
    assist_dist: float
    assist_time: float
    alert_after: float
    desk_time: float
    min_sample: int
    demo_only: bool = False


PRESETS: dict[str, Preset] = {
    "dealership": Preset("dealership", 10, 1.5, 2, 0.3, 10, 60, 2.0, 5, 45, 30, 5, False),
    "short": Preset("short", 3, 1, 1, 0.3, 3, 10, 2.0, 2, 8, 5, 2, True),
}


@dataclass(frozen=True)
class Config:
    preset: Preset
    sample_fps: float = 5.0
    max_width: int = 960
    max_duration_s: float = 300.0          # longer videos are trimmed with a warning
    person_height_m: float = 1.70
    hfov_deg: float = 70.0
    kp_thr: float = 0.3                     # keypoint confidence threshold
    detector: str = "yolox_s"               # chosen by the M0 benchmark (PROGRESS.md)
    pose_model: str = "rtmpose_s"
    det_score_thr: float = 0.3
    det_nms_thr: float = 0.45
    track_activation_thr: float = 0.35
    track_buffer_s: float = 2.0
    track_match_thr: float = 0.8
    edge_margin: float = 0.02               # normalised distance to frame edge counted as "at edge"
    speed_window_s: float = 1.0             # foot-point smoothing window for speed
    group_arrival_s: float = 5.0            # party rule (Coach §4.5)
    group_dist_m: float = 1.5
    group_window_s: float = 30.0
    contact_dist_m: float = 2.0             # roles: contacts
    contact_time_s: float = 5.0
    outfit_sim_thr: float = 0.85            # roles: outfit_peers (cosine)
    presence_min_clip_s: float = 60.0
    role_weights: dict = field(default_factory=lambda: {
        "outfit_peers": 1.0, "contacts": 1.0, "presence": 1.0, "initiator": 0.5})
    role_staff_thr: float = 0.6
    role_customer_thr: float = 0.35
    merge_sim_thr: float = 0.9              # identity merge cosine threshold (was 0.75; tuned on dev, PROGRESS.md M6)
    merge_max_speed: float = 0.25           # normalised frame widths per second
    crowding_ratio: float = 3.0             # R5
    crowding_time_s: float = 120.0          # R5
    heatmap_w: int = 48
    heatmap_h: int = 27
    sat_overlap: float = 0.6                # sat-in-car: share of the person box inside a car box
    sat_speed: float = 0.1                  # sat-in-car (a): foot speed below this, m/s
    sat_time_s: float = 5.0                 # sat-in-car (a): still for at least this long
    sat_reappear_s: float = 60.0            # sat-in-car (b): a track starts at the car again within this
    sat_gap_s: float = 2.0                  # observation gap that splits a person into two track segments


def get_config(preset_name: str = "short", **overrides) -> Config:
    preset = PRESETS[preset_name]
    return replace(Config(preset=preset), **overrides)
