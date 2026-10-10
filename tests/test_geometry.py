import math
import numpy as np
import pytest
from showroom_coach.geometry import (Camera, position_m, distance_m, facing_vector, faces,
                                     body_height_px, at_frame_edge)
from tests.synth import project_person, unit

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
