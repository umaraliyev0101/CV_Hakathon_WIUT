import math
import numpy as np
import pytest
from pulse import geometry as g
from pulse.datatypes import TrackObs

W, H = 960, 540
F = g.focal_px(W, 70.0)
CX, CY = W / 2, H / 2
CAM_H = 1.5

def synth_person(X, Z, height=1.70, mode="full", tid=1):
    """Project a standing person at ground (X, Z) through a level pinhole camera at CAM_H."""
    u = CX + F * X / Z
    v_foot = CY + F * CAM_H / Z
    v_head = CY - F * (height - CAM_H) / Z
    h = v_foot - v_head
    def y(frac): return v_head + frac * h
    kps = [[u, y(0.07), 0.9]] + [[u, y(0.05), 0.9]] * 2 + [[u, y(0.06), 0.9]] * 2 \
        + [[u + 0.1 * h, y(0.18), 0.9], [u - 0.1 * h, y(0.18), 0.9]] \
        + [[u, y(0.33), 0.9]] * 4 + [[u + 0.05 * h, y(0.48), 0.9], [u - 0.05 * h, y(0.48), 0.9]] \
        + [[u, y(0.72), 0.9]] * 2 + [[u, y(0.95), 0.9]] * 2
    if mode == "torso":          # cut off below the hips (e.g. sitting / frame bottom)
        for i in (0, 13, 14, 15, 16):
            kps[i][2] = 0.05
    if mode == "box":
        kps = None
    bbox = (u - 0.15 * h, v_head, u + 0.15 * h, v_foot)
    return TrackObs(0.0, 0, tid, bbox, 0.9, kps)

@pytest.mark.parametrize("mode", ["full", "torso", "box"])
@pytest.mark.parametrize("sep", [1.0, 2.0, 3.0, 4.0])
def test_distance_within_15pct(mode, sep):
    a = synth_person(-sep / 2, 4.0, mode=mode)
    b = synth_person(sep / 2, 4.5, mode=mode, tid=2)
    pa = g.position_m(a, (W, H)); pb = g.position_m(b, (W, H))
    true = math.hypot(sep, 0.5)
    assert abs(g.distance_m((pa.x, pa.z), (pb.x, pb.z)) - true) / true <= 0.15
    assert pa.method == mode

def test_depth_full_body_close_to_truth():
    p = g.position_m(synth_person(0.5, 3.0), (W, H))
    assert p.z == pytest.approx(3.0, rel=0.05) and p.x == pytest.approx(0.5, rel=0.1, abs=0.05)

def test_foot_point_uses_ankles_else_box_bottom():
    o = synth_person(0, 3.0)
    u, v = g.foot_point(o.bbox, o.keypoints)
    assert v == pytest.approx(o.keypoints[15][1])
    u2, v2 = g.foot_point(o.bbox, None)
    assert (u2, v2) == pytest.approx(((o.bbox[0] + o.bbox[2]) / 2, o.bbox[3]))

def test_homography_roundtrip():
    img = [(100, 400), (800, 400), (900, 530), (50, 530)]
    floor = [(0, 10), (10, 10), (10, 0), (0, 0)]
    h = g.fit_homography(img, floor)
    for p, q in zip(img, floor):
        assert g.to_floor(h, p) == pytest.approx(q, abs=1e-6)

def test_position_uses_floor_homography_when_given():
    img = [(100, 400), (800, 400), (900, 530), (50, 530)]
    h = g.fit_homography(img, [(0, 10), (10, 10), (10, 0), (0, 0)])
    o = TrackObs(0, 0, 1, (440, 300, 460, 400), 0.9, None)  # foot (450, 400) on the far edge
    p = g.position_m(o, (W, H), floor_h=h)
    assert p.method == "floor" and p.z == pytest.approx(10, abs=1e-6)

def test_facing_front_and_unknown():
    o = synth_person(0, 3.0)                           # l_sh.x > r_sh.x → faces camera
    assert g.facing_dir(o.keypoints) == (0, -1)
    assert g.facing_dir(None) is None
    assert g.faces(o, (0, 3.0), (0, 1.5)) is True     # B between person and camera
    assert g.faces(o, (0, 3.0), (0, 5.0)) is False
