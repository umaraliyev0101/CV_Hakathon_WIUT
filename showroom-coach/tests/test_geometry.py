import logging
import math
import numpy as np
import pytest
from showroom_coach.geometry import (Camera, position_m, distance_m, facing_vector, faces,
                                     body_height_px, at_frame_edge)
from tests.synth import SIT, STAND, project_person, unit

CAM = Camera(1280, 720, hfov_deg=70.0)


def test_focal_length():
    assert CAM.f == pytest.approx(1280 / (2 * math.tan(math.radians(35))))


@pytest.mark.parametrize("sep", [1.0, 1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("z0", [4.0, 6.0, 8.0])
def test_pair_distance_within_15pct_standing(sep, z0):
    a = position_m(*project_person(CAM, -sep / 2, z0), CAM)
    b = position_m(*project_person(CAM, sep / 2, z0 + 0.5), CAM)
    true = math.hypot(sep, 0.5)
    assert a.source == b.source == "full"
    assert distance_m(a, b) == pytest.approx(true, rel=0.15)


def test_depth_separation_within_15pct():
    a = position_m(*project_person(CAM, 0.0, 4.0), CAM)
    b = position_m(*project_person(CAM, 0.3, 7.0), CAM)
    assert distance_m(a, b) == pytest.approx(math.hypot(0.3, 3.0), rel=0.15)


def test_sitting_uses_torso_and_stays_accurate():
    bbox, k = project_person(CAM, 0.5, 5.0, sitting=True)
    p = position_m(bbox, k, CAM)
    assert p.source == "torso"
    assert p.z == pytest.approx(5.0, rel=0.15)


def test_cutoff_uses_torso():
    bbox, k = project_person(CAM, 0.0, 2.2)      # feet below the image at 2.2 m
    assert k[15, 2] == 0.0 and k[16, 2] == 0.0
    p = position_m(bbox, k, CAM)
    assert p.source == "torso"
    assert p.z == pytest.approx(2.2, rel=0.15)


def test_no_kpts_box_fallback():
    bbox, _ = project_person(CAM, 1.0, 5.0)
    p = position_m(bbox, None, CAM)
    assert p.source == "box" and p.approx
    assert p.z == pytest.approx(5.0, rel=0.25)


def test_floor_homography_overrides():
    from showroom_coach.config import CameraConfig
    pts = ((0, 720, -2.0, 2.0), (1280, 720, 2.0, 2.0), (1280, 400, 2.0, 8.0), (0, 400, -2.0, 8.0))
    cam = Camera.from_config(1280, 720, CameraConfig(floor_points=pts))
    p = position_m((600, 300, 680, 720), None, cam)
    assert p.source == "floor" and p.z == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("deg", [0, 45, 135, 180, 270, 300])
def test_facing_vector_recovers_direction(deg):
    want = unit(deg)
    bbox, k = project_person(CAM, 0.0, 5.0, facing=want)
    got = facing_vector(bbox, k, CAM)
    if got is None:                       # only allowed when facing away with the nose hidden
        assert want[1] >= 0.3
        return
    ang = math.degrees(math.acos(np.clip(got[0] * want[0] + got[1] * want[1], -1, 1)))
    assert ang < 30


def test_faces_true_false_none():
    assert faces((0, 5), (1, 0), (1.5, 5)) is True
    assert faces((0, 5), (1, 0), (-1.5, 5)) is False
    assert faces((0, 5), None, (1.5, 5)) is None


def test_body_height_sources():
    bbox, k = project_person(CAM, 0.0, 5.0)
    h, src = body_height_px(bbox, k)
    assert src == "full" and h == pytest.approx(CAM.f * 1.70 / 5.0, rel=0.1)


def test_at_frame_edge():
    assert at_frame_edge((0, 10, 50, 200), 1280, 720)
    assert not at_frame_edge((100, 10, 150, 200), 1280, 720)


# ---- extra coverage beyond the brief: input shapes, partial keypoints, degenerate cases ----

def test_kpts_as_nested_list_matches_array():
    bbox, k = project_person(CAM, 0.5, 5.0)
    assert position_m(bbox, k.tolist(), CAM) == position_m(bbox, k, CAM)
    assert facing_vector(bbox, k.tolist(), CAM) == facing_vector(bbox, k, CAM)


@pytest.mark.parametrize("bad", [np.zeros((17, 3)), np.zeros((5, 3)), np.zeros((17, 2)), [[1, 2, 3]], "abc"])
def test_unusable_kpts_fall_back_to_box(bad):
    bbox, _ = project_person(CAM, 0.0, 5.0)
    assert position_m(bbox, bad, CAM).source == "box"
    assert facing_vector(bbox, bad, CAM) is None


def test_single_ankle_is_enough_for_full():
    bbox, k = project_person(CAM, 0.0, 5.0)
    k[16, 2] = 0.0
    assert position_m(bbox, k, CAM).source == "full"


def test_one_sided_torso_is_enough():
    bbox, k = project_person(CAM, 0.0, 5.0, sitting=True)
    k[[6, 12], 2] = 0.0                              # right shoulder and hip hidden
    p = position_m(bbox, k, CAM)
    assert p.source == "torso" and p.z == pytest.approx(5.0, rel=0.15)


def test_facing_vector_is_a_unit_vector():
    for deg in (0, 90, 180, 270, 300):
        bbox, k = project_person(CAM, 0.0, 5.0, facing=unit(deg))
        got = facing_vector(bbox, k, CAM)
        assert got is not None and math.hypot(*got) == pytest.approx(1.0, abs=1e-9)


def test_facing_straight_away_without_nose_still_known():
    bbox, k = project_person(CAM, 0.0, 5.0, facing=(0.0, 1.0))
    assert k[0, 2] < 0.3                              # nose hidden
    got = facing_vector(bbox, k, CAM)
    assert got == pytest.approx((0.0, 1.0), abs=1e-6)


def test_facing_needs_both_shoulders():
    bbox, k = project_person(CAM, 0.0, 5.0)
    k[5, 2] = 0.0
    assert facing_vector(bbox, k, CAM) is None


def test_distance_accepts_positions_and_tuples():
    from showroom_coach.geometry import Position
    assert distance_m((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert distance_m(Position(0.0, 0.0, "full"), (3.0, 4.0)) == pytest.approx(5.0)
    assert distance_m(Position(1.0, 1.0, "box"), Position(1.0, 1.0, "box")) == 0.0


def test_faces_boundary_and_degenerate_cases():
    edge = (math.cos(math.radians(60)), math.sin(math.radians(60)))
    assert faces((0, 0), (1, 0), edge) is True        # exactly max_deg counts as facing
    assert faces((0, 0), (1, 0), (0, 0)) is None      # same spot: direction undefined
    assert faces((0, 0), (0, 0), (1, 0)) is None      # zero facing vector


def test_camera_without_floor_points_has_no_homography():
    from showroom_coach.config import CameraConfig
    assert Camera.from_config(1280, 720, CameraConfig()).floor_h is None
    assert Camera.from_config(1280, 720, CameraConfig(floor_points=((0, 0, 0, 0),))).floor_h is None


# ---- fix round 1: hip-based standing test, canonical seated pose, calibration warnings, box-height facing ----

CANON_SIT = list(SIT)                 # the seated pose of the original brief: shins vertical, knees AND ankles 0.45 m ahead
for _i in (15, 16):
    CANON_SIT[_i] = (SIT[_i][0], SIT[_i][1], 0.45)


def _project(x, z, joints, cam_h, H=1.70):
    """Local variant of tests.synth.project_person (person facing the camera) with its own joint table and camera height."""
    k = np.zeros((17, 3))
    for i, (hf, lat, fwd) in enumerate(joints):
        X, Z, Y = x + lat, z - fwd, cam_h - hf * H
        u, v = CAM.cx + CAM.f * X / Z, CAM.cy + CAM.f * Y / Z
        k[i] = (u, v, 0.9 if (0 <= u < CAM.width and 0 <= v < CAM.height) else 0.0)
    top = (0.75 if joints is not STAND else 1.0) * H
    us = [CAM.cx + CAM.f * (x + s * 0.25) / z for s in (-1, 1)]
    bbox = (max(0.0, us[0]), max(0.0, CAM.cy + CAM.f * (cam_h - top) / z),
            min(CAM.width - 1.0, us[1]), min(CAM.height - 1.0, CAM.cy + CAM.f * cam_h / z))
    return bbox, k


# (camera height m, depth m) pairs where the whole torso is in frame; ankles are in frame for most of them
SEATED_VIEWS = [(1.6, 3.0), (1.6, 4.5), (1.6, 5.0), (1.6, 6.0), (1.6, 8.0),
                (1.0, 3.0), (1.0, 4.5), (1.0, 6.0), (1.0, 8.0), (2.2, 6.0), (2.2, 8.0)]
STANDING_VIEWS = [(1.6, 4.0), (1.6, 6.0), (1.6, 8.0), (1.0, 3.0), (1.0, 5.0), (1.0, 8.0), (2.2, 6.0), (2.2, 8.0)]


@pytest.mark.parametrize("cam_h,z", SEATED_VIEWS)
def test_canonical_seated_pose_uses_torso_at_every_depth(cam_h, z):
    bbox, k = _project(0.5, z, CANON_SIT, cam_h)
    p = position_m(bbox, k, CAM)
    assert p.source == "torso"
    assert p.z == pytest.approx(z, rel=0.15)


def test_canonical_seated_pose_exercises_the_ankle_visible_band():
    # guard against the parametrisation silently testing only cases with the ankles out of frame
    visible = [(h, z) for h, z in SEATED_VIEWS if _project(0.5, z, CANON_SIT, h)[1][15, 2] > 0]
    assert (1.6, 4.5) in visible and (1.6, 5.0) in visible and (1.0, 3.0) in visible


@pytest.mark.parametrize("cam_h,z", STANDING_VIEWS)
def test_standing_person_still_uses_full_height(cam_h, z):
    bbox, k = _project(0.5, z, STAND, cam_h)
    assert k[15, 2] > 0 and k[0, 2] > 0
    p = position_m(bbox, k, CAM)
    assert p.source == "full"
    assert p.z == pytest.approx(z, rel=0.15)


def test_standing_test_needs_only_one_ankle_and_a_hip():
    bbox, k = _project(0.5, 5.0, STAND, 1.6)
    k[16, 2] = 0.0                                    # one ankle hidden
    k[[11, 5], 2] = 0.0                               # left hip and left shoulder hidden: right side is the torso
    assert position_m(bbox, k, CAM).source == "full"


def test_full_without_any_torso_keypoints():
    bbox, k = _project(0.5, 5.0, STAND, 1.6)
    k[[5, 6, 11, 12], 2] = 0.0                        # no torso at all: head and ankle decide
    p = position_m(bbox, k, CAM)
    assert p.source == "full" and p.z == pytest.approx(5.0, rel=0.15)


def _bad_calibration_cases():
    return [
        pytest.param(((0, 0, 0, 0),), id="one-point"),
        pytest.param(tuple((i * 10, i * 10, i, i) for i in range(5)), id="five-points"),
        pytest.param(((0, 0, 0, 0), (100, 100, 1, 1), (200, 200, 2, 2), (300, 300, 3, 3)), id="collinear"),
        pytest.param(((0, 0, 0, 0), (0, 0, 0, 0), (100, 100, 1, 1), (200, 200, 2, 2)), id="duplicate-point"),
        pytest.param(((0, 0, 0, 0), (1, 1, 1, float("nan")), (2, 2, 2, 2), (3, 0, 3, 3)), id="non-finite"),
    ]


@pytest.mark.parametrize("pts", _bad_calibration_cases())
def test_unusable_floor_points_warn_and_fall_back(pts, caplog):
    from showroom_coach.config import CameraConfig
    with caplog.at_level(logging.WARNING):
        cam = Camera.from_config(1280, 720, CameraConfig(floor_points=pts))
    assert cam.floor_h is None
    assert any(r.levelno == logging.WARNING and "floor_points" in r.getMessage() for r in caplog.records)
    assert position_m((600, 300, 680, 720), None, cam).source == "box"


def test_good_or_absent_floor_points_do_not_warn(caplog):
    from showroom_coach.config import CameraConfig
    pts = ((0, 720, -2.0, 2.0), (1280, 720, 2.0, 2.0), (1280, 400, 2.0, 8.0), (0, 400, -2.0, 8.0))
    with caplog.at_level(logging.WARNING):
        assert Camera.from_config(1280, 720, CameraConfig()).floor_h is None
        assert Camera.from_config(1280, 720, CameraConfig(floor_points=pts)).floor_h is not None
    assert not caplog.records


def test_facing_vector_refuses_when_height_is_only_the_box():
    bbox, k = project_person(CAM, 0.0, 5.0)
    only_upper = np.zeros_like(k)
    only_upper[[0, 5, 6]] = k[[0, 5, 6]]              # nose and shoulders only: no torso, no ankles
    assert body_height_px(bbox, only_upper)[1] == "box"
    assert facing_vector(bbox, only_upper, CAM) is None
    assert facing_vector(bbox, k, CAM) is not None    # the same person with every keypoint is fine
