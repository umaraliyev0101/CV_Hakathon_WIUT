import numpy as np

from sim import layout, navigation


def _inside_any(pt, polys):
    return any(layout.point_in_polygon(pt, p) for p in polys)


def _dense(path, step=0.02):
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        for s in np.linspace(0.0, 1.0, n):
            pts.append(a + s * (b - a))
    return pts


def test_random_paths_never_enter_car_footprints(cfg):
    lay = layout.build_layout(cfg)
    nav = navigation.NavGraph(lay, cfg["navigation"])
    cars = [c["footprint_m"] for c in lay["cars"]]
    rng = np.random.default_rng(123)
    n_ok = 0
    for _ in range(300):
        a = rng.uniform([0.5, 0.5], [lay["room"]["width_m"] - 0.5, lay["room"]["depth_m"] - 0.5])
        b = rng.uniform([0.5, 0.5], [lay["room"]["width_m"] - 0.5, lay["room"]["depth_m"] - 0.5])
        if not (nav.is_free(a) and nav.is_free(b)):
            continue
        path = nav.shortest_path(a, b)
        assert path is not None, f"no path {a} -> {b}"
        for p in _dense(path):
            assert not _inside_any(p, cars), f"path {a}->{b} enters a car at {p}"
            # keep the body (radius) out of the footprint too
            assert layout.dist_to_polygons(p, cars) >= cfg["navigation"]["body_radius_m"] - 1e-6
        n_ok += 1
    assert n_ok > 100


def test_straight_line_through_car_is_rerouted(cfg):
    lay = layout.build_layout(cfg)
    nav = navigation.NavGraph(lay, cfg["navigation"])
    car = lay["cars"][0]
    cx, cy = car["center_m"]
    a, b = (cx - 4.0, cy), (cx + 4.0, cy)
    path = nav.shortest_path(a, b)
    assert len(path) > 2
    for p in _dense(path):
        assert not layout.point_in_polygon(p, car["footprint_m"])


def test_timeline_positions_never_inside_cars(cfg, timeline):
    lay = layout.build_layout(cfg)
    cars = [c["footprint_m"] for c in lay["cars"]]
    for agent in timeline["agents"].values():
        for st in agent["frames"]:
            if st["present"]:
                assert layout.dist_to_polygons(st["pos_m"], cars) >= cfg["navigation"]["body_radius_m"] - 1e-6
