import json
from pathlib import Path

import numpy as np
import pytest

from analytics.dwell import car_distances, dwell
from analytics.floor import Camera, distance_to_polygon, foot_point

GT = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "clips" / "s000_greet_fast_cam1_day" / "gt.json"
SQUARE = [[0, 0], [2, 0], [2, 1], [0, 1]]


@pytest.fixture(scope="module")
def gt():
    if not GT.exists():
        pytest.skip("synthetic data not present")
    return json.loads(GT.read_text())


def test_camera_center_matches_gt_position(gt):
    cam = Camera.from_gt(gt["camera"])
    assert np.allclose(cam.center, gt["camera"]["position_m"], atol=1e-3)


def test_floor_pixel_roundtrip(gt):
    cam = Camera.from_gt(gt["camera"])
    xy = np.array([[3.0, 4.0], [6.5, 2.0], [1.0, 7.5]])
    assert np.allclose(cam.pixel_to_floor(cam.floor_to_pixel(xy)), xy, atol=1e-6)


def test_zone_image_corners_project_to_floor_corners(gt):
    cam = Camera.from_gt(gt["camera"])
    z = gt["zones"]["car_1"]
    assert np.allclose(cam.pixel_to_floor(z["image_px"]), z["floor_m"], atol=0.01)


def test_pixel_above_horizon_is_nan(gt):
    cam = Camera.from_gt(gt["camera"])
    assert np.isnan(cam.pixel_to_floor([[640, -5000]])).all()


def test_foot_point_is_bottom_centre():
    assert foot_point([10, 20, 30, 100]) == (20.0, 100.0)


@pytest.mark.parametrize("p, d", [([1, 0.5], 0.0), ([3, 0.5], 1.0), ([3, 2], np.hypot(1, 1)), ([1, -0.25], 0.25)])
def test_distance_to_polygon(p, d):
    assert distance_to_polygon(p, SQUARE) == pytest.approx(d)


def test_car_distances_nan_point():
    assert np.isnan(car_distances(np.array([np.nan, np.nan]), {"c": SQUARE})["c"])


def _walk(points, dt=0.1, tid=1):
    return [(i * dt, tid, np.array(p, float)) for i, p in enumerate(points)]


def test_dwell_counts_time_near_car_and_ignores_far():
    pts = [[5, 5]] * 10 + [[2.5, 0.5]] * 30 + [[5, 5]] * 10  # 3 s at 0.5 m
    (cd,) = dwell(_walk(pts), {"car": SQUARE}, dt=0.1)
    assert cd.track_id == 1 and cd.car == "car"
    assert cd.total_s == pytest.approx(3.0)
    assert cd.visits[0].min_dist_m == pytest.approx(0.5)


def test_dwell_bridges_short_gap_and_drops_walk_past():
    gap = [[2.5, 0.5]] * 20 + [[5, 5]] * 5 + [[2.5, 0.5]] * 20  # 0.5 s dropout -> one visit
    (cd,) = dwell(_walk(gap), {"car": SQUARE}, dt=0.1)
    assert len(cd.visits) == 1 and cd.total_s == pytest.approx(4.5)
    walk_past = [[5, 5]] * 10 + [[2.5, 0.5]] * 10 + [[5, 5]] * 10  # 1 s < min visit
    assert dwell(_walk(walk_past), {"car": SQUARE}, dt=0.1) == []


def test_height_above_recovers_true_height(gt):
    cam = Camera.from_gt(gt["camera"])
    xy = np.array([5.0, 4.0])
    top = cam.floor_to_pixel(xy)[0]  # pixel of the floor point, then lift by projecting a 1.75 m point
    head_w = np.append(xy, 1.75)
    p = cam.K @ (cam.R @ head_w + cam.t)
    u, v = p[:2] / p[2]
    assert cam.height_above(xy, v, u) == pytest.approx(1.75, abs=1e-6)
    assert top[1] > v  # feet image below the head


def test_person_floor_xy_uses_head_when_feet_hidden(gt):
    from analytics.floor import person_floor_xy
    cam = Camera.from_gt(gt["camera"])
    xy = np.array([5.0, 4.0])
    def px(z):
        p = cam.K @ (cam.R @ np.append(xy, z) + cam.t)
        return p[:2] / p[2]
    (uh, vh), (uf, vf), (um, vm) = px(1.7), px(0.0), px(0.9)
    # a tilted camera makes a standing person lean in the image, so a real box spans head and feet x
    full, how = person_floor_xy(cam, [min(uh, uf) - 20, vh, max(uh, uf) + 20, vf])
    assert how == "feet" and np.allclose(full, xy, atol=0.15)
    cut, how = person_floor_xy(cam, [min(uh, um) - 20, vh, max(uh, um) + 20, vm])  # legs hidden by a car
    assert how == "head" and np.allclose(cut, xy, atol=0.15)


def test_polygon_partly_behind_camera_is_clipped(gt):
    cam = Camera.from_gt(gt["camera"])
    c = cam.center[:2]
    fwd = (cam.R.T @ np.array([0, 0, 1.0]))[:2]
    fwd /= np.linalg.norm(fwd)
    side = np.array([-fwd[1], fwd[0]])
    poly = [c - 3 * fwd - side, c + 5 * fwd - side, c + 5 * fwd + side, c - 3 * fwd + side]  # straddles camera
    px = cam.floor_polygon_to_pixel(poly)
    assert px is not None and np.isfinite(px).all()
    assert cam.floor_polygon_to_pixel([c - 5 * fwd, c - 4 * fwd, c - 4 * fwd + side]) is None


def test_synthetic_footprints_inside_zones(gt):
    from analytics.synthetic import car_footprints
    fp = car_footprints(gt)
    for name, poly in fp.items():
        zone = gt["zones"][name]["floor_m"]
        assert all(distance_to_polygon(p, zone) == 0.0 for p in poly)
        assert np.ptp(np.asarray(zone)[:, 0]) > np.ptp(np.asarray(poly)[:, 0])
