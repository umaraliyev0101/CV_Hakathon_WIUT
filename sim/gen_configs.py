"""Generate the scene configs (sim/configs/scene_NNN.json) by seeded rejection sampling (pure).

python sim/gen_configs.py            # all scenes in SCENES
python sim/gen_configs.py --only 2   # one scene

A candidate scene (room, doors, desk, 4-6 cars, script, cameras) is accepted only when:
  * every scripted path exists and the timeline builds;
  * every scripted event is matched by a rule event of the same type and subject within
    MATCH_MARGIN_S (stricter than the 1.0 s qa gate), and no unscripted greet/walkout/handoff;
  * no two rendered people come closer than MIN_PAIR_DIST_M;
  * camera selection: cameras meet the spec limits (height 2.6-3.5 m, pitch -25..-50 deg, HFOV
    60-100 deg); each selected camera sees every person (full body, >= 60 px tall, not hidden
    behind a car) in >= SHARE_VIS of the clip's frames (short-clip scenarios), or, for the
    crowded / long scenes, every agent >= 0.8 of its lifetime in at least one camera and every
    camera sees someone >= 0.5 of the frames.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim import behaviour, gt_events, layout, navigation, projection, schema  # noqa: E402

CFG_DIR = REPO / "sim" / "configs"
MATCH_MARGIN_S = 0.8
MIN_PAIR_DIST_M = 0.5
SHARE_VIS = 0.86
MIN_BOX_H_PX = 60
CAR_META = CFG_DIR / "assets_cars.json"
PEOPLE_META = CFG_DIR / "assets_people.json"
TYPES_CHECKED = ("greet", "handoff", "engagement", "walkout", "party")

SCENES = [  # scene_id, scenario, seed
    ("s000", "greet_fast", 0), ("s001", "greet_slow", 1), ("s002", "walkout", 2), ("s003", "party", 3),
    ("s004", "handoff", 4), ("s005", "crowded", 5), ("s006", "dealership_long", 6),
    ("s007", "greet_fast", 7), ("s008", "greet_slow", 8), ("s009", "walkout", 9), ("s010", "party", 10),
    ("s011", "handoff", 11),
]
SHORT = ("greet_fast", "greet_slow", "walkout", "party", "handoff")

UNIFORM = {"top": [0.03, 0.03, 0.04], "vest": [0.7, 0.02, 0.02], "trousers": [0.04, 0.04, 0.05],
           "suit_recolour": [0.42, 0.02, 0.03],
           "_doc": "Rocketbox clothing is baked into one body texture, so the uniform is a shader recolour of the dark suit fabric of staff characters (suit_recolour, linear RGB). top/vest/trousers are used by the capsule fallback only."}
CASUAL_TOPS = [[0.1, 0.3, 0.65], [0.75, 0.75, 0.72], [0.15, 0.45, 0.2], [0.8, 0.55, 0.1], [0.45, 0.1, 0.4],
               [0.6, 0.62, 0.66], [0.85, 0.85, 0.85], [0.3, 0.18, 0.1], [0.05, 0.25, 0.3], [0.65, 0.3, 0.25]]
CASUAL_BOTTOMS = [[0.2, 0.17, 0.12], [0.08, 0.1, 0.18], [0.35, 0.33, 0.3], [0.05, 0.05, 0.06], [0.55, 0.5, 0.4]]
HARD_NEG = [{"kind": "suit", "top": [0.025, 0.025, 0.03], "trousers": [0.03, 0.03, 0.035]},
            {"kind": "dark_top_red", "top": [0.04, 0.03, 0.03], "trousers": [0.04, 0.04, 0.05], "accent": [0.6, 0.03, 0.03]}]


def _car_models():
    if CAR_META.exists():
        return json.loads(CAR_META.read_text())["cars"]
    return [{"model": None, "width_over_length": 0.41, "height_over_length": 0.31}]


def _characters():
    if PEOPLE_META.exists():
        return json.loads(PEOPLE_META.read_text())["characters"]
    return []


def sample_layout(rng):
    W = float(rng.uniform(18.0, 26.0))
    D = float(rng.uniform(12.0, 17.0))
    H = float(rng.uniform(4.5, 5.5))
    x_e = float(rng.uniform(6.0, 9.0))
    y_d = float(rng.uniform(5.0, 7.0))
    cfg = {
        "room": {"width_m": round(W, 2), "depth_m": round(D, 2), "height_m": round(H, 2), "wall_thickness_m": 0.2},
        "entrance": {"wall": "south", "center_x_m": round(x_e, 2), "width_m": 2.4, "height_m": 2.6},
        "exit": {"wall": "east", "center_y_m": round(D * 0.6, 2), "width_m": 3.5, "height_m": 3.0},
        "desk": {"center_m": [0.95, round(y_d, 2)], "length_m": 1.6, "width_m": 0.7, "height_m": 0.75, "yaw_deg": 0.0},
        "zone_apron_m": 1.2,
    }
    models = _car_models()
    n_cars = int(rng.integers(4, 7))
    cars = []
    desk_fp = layout.rect(cfg["desk"]["center_m"], 1.6, 0.7, 0.0)
    corridor = [[x_e - 1.8, -1.0], [x_e + 1.8, -1.0], [x_e + 1.8, 3.2], [x_e - 1.8, 3.2]]
    tries = 0
    while len(cars) < n_cars and tries < 400:
        tries += 1
        m = models[int(rng.integers(len(models)))]
        L = float(rng.uniform(4.0, 5.2))
        Wc, Hc = L * m["width_over_length"], L * m["height_over_length"]
        if not cars:   # car_1: the action car, within ~5-7 m of the south-west corner
            c = [float(rng.uniform(x_e - 3.8, x_e - 1.2)), float(rng.uniform(4.0, 5.6))]
        else:
            c = [float(rng.uniform(2.5, W - 2.5)), float(rng.uniform(2.5, D - 2.5))]
        yaw = float(rng.choice([-45.0, -30.0, 0.0, 30.0, 45.0, 90.0]))
        fp = layout.rect(c, L, Wc, yaw)
        if any(not (1.2 <= x <= W - 1.2 and 1.2 <= y <= D - 1.2) for x, y in fp):
            continue
        if any(_poly_dist(fp, o["fp"]) < 2.5 for o in cars):
            continue
        if _poly_dist(fp, desk_fp) < 2.5 or _poly_dist(fp, corridor) < 0.3:
            continue
        cars.append({"fp": fp, "cfg": {"id": len(cars) + 1, "center_m": [round(c[0], 3), round(c[1], 3)],
                                       "length_m": round(L, 3), "width_m": round(Wc, 3), "height_m": round(Hc, 3),
                                       "yaw_deg": yaw, "model": m["model"],
                                       "paint": [round(float(v), 3) for v in PAINTS[int(rng.integers(len(PAINTS)))]]}})
    if len(cars) < 4:
        return None
    cfg["cars"] = [c["cfg"] for c in cars]
    return cfg


PAINTS = [[0.55, 0.03, 0.03], [0.05, 0.12, 0.35], [0.75, 0.75, 0.78], [0.02, 0.02, 0.025], [0.9, 0.9, 0.9],
          [0.3, 0.32, 0.35], [0.05, 0.25, 0.12], [0.6, 0.45, 0.2]]


def _poly_dist(a, b):
    if any(layout.point_in_polygon(p, b) for p in a) or any(layout.point_in_polygon(p, a) for p in b):
        return 0.0
    return min(min(layout.dist_to_polygon(p, b) for p in a), min(layout.dist_to_polygon(p, a) for p in b))


def view_spot(lay, nav, car, toward, dist=1.0):
    """Free floor point `dist` m from the car footprint on the side facing `toward` (angle search)."""
    c = np.asarray(car["center_m"])
    base = math.atan2(toward[1] - c[1], toward[0] - c[0])
    for k in range(0, 13):
        for sgn in (1, -1):
            a = base + sgn * math.radians(15 * k)
            u = np.array([math.cos(a), math.sin(a)])
            for r in np.arange(1.0, 6.0, 0.05):
                p = c + r * u
                if layout.dist_to_polygon(p, car["footprint_m"]) >= dist:
                    break
            p = [round(float(p[0]), 3), round(float(p[1]), 3)]
            if nav.is_free(p) and p[1] > 1.5:
                return p
    return None


def free_spot_near(nav, pt, avoid, min_d, rng, r_max=4.0):
    for _ in range(200):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.0, r_max)
        p = [round(pt[0] + r * math.cos(a), 3), round(pt[1] + r * math.sin(a), 3)]
        if p[1] > 1.5 and nav.is_free(p) and all(math.dist(p, q) >= min_d for q in avoid):
            return p
    return None


def _person(pid, role, rng, chars, outfit=None, hard=None, **kw):
    h = float(np.clip(rng.normal(1.73 if role == "staff" else 1.71, 0.08), 1.58, 1.92))
    p = {"id": pid, "role": role, "height_m": round(h, 3), "speed_mps": [1.15, 1.4], "hard_negative": False}
    if role == "staff":
        p.update(outfit="uniform")
    elif hard is not None:
        p.update(outfit=hard["kind"], hard_negative=True, colors={k: v for k, v in hard.items() if k != "kind"})
    else:
        p.update(outfit="casual", colors={"top": CASUAL_TOPS[int(rng.integers(len(CASUAL_TOPS)))],
                                          "trousers": CASUAL_BOTTOMS[int(rng.integers(len(CASUAL_BOTTOMS)))]})
    p.update(kw)
    return p


def make_script(scenario, lay, nav, cfg, rng):
    """Returns (persons, parties, duration_s, eval_presets) or None."""
    W = lay["room"]["width_m"]
    cam0 = [0.4, 0.4]
    cars = sorted(lay["cars"], key=lambda c: math.dist(c["center_m"], cam0))
    car1 = next(c for c in lay["cars"] if c["id"] == 1)
    others = [c for c in cars if c["id"] != 1]
    V1 = view_spot(lay, nav, car1, cam0)
    V2 = view_spot(lay, nav, others[0], cam0)
    V3 = view_spot(lay, nav, others[1], cam0)
    V4 = view_spot(lay, nav, others[2], cam0)
    if None in (V1, V2, V3, V4):
        return None
    if min(math.dist(a, b) for a, b in ((V1, V2), (V1, V3), (V2, V3), (V1, V4), (V2, V4), (V3, V4))) < 2.5:
        return None
    dx, dy = lay["desk"]["center_m"]
    desk_spot = [round(dx + 0.55, 3), round(dy - 0.95, 3)]
    if not nav.is_free(desk_spot) or min(math.dist(desk_spot, v) for v in (V1, V2, V3, V4)) < 2.6:
        return None
    H2 = free_spot_near(nav, [(V1[0] + V2[0]) / 2 + 1.5, (V1[1] + V2[1]) / 2 + 2.0], [V1, V2, V3, V4, desk_spot], 2.6, rng)
    if H2 is None:
        return None
    chars = _characters()
    hard = HARD_NEG[int(rng.integers(len(HARD_NEG)))]
    face_desk_room = "entrance"
    S = lambda pid, spot, script, **kw: _person(pid, "staff", rng, chars, spawn_m=spot, spawn_yaw_deg=-90.0,  # noqa: E731
                                                enter_t=0.0, present_at_start=True, script=script,
                                                default_face=kw.pop("default_face", face_desk_room), **kw)
    x_e = lay["entrance"]["center_x_m"]

    def C(pid, script, enter_t=0.0, hard_neg=None, door=0.0):
        return _person(pid, "customer", rng, chars, hard=hard_neg, spawn_m=[round(x_e + door, 3), behaviour.OUTSIDE_Y],
                       spawn_yaw_deg=90.0, enter_t=enter_t, present_at_start=enter_t <= 0.0 and False,
                       door_x_offset=door, script=([{"do": "enter", "door_x_offset": door}] if script is not None else []) + (script or []),
                       default_face="car_1")

    t_app = round(float(rng.uniform(2.5, 5.0)), 2)
    talk = round(float(rng.uniform(14.0, 18.0)), 2)
    if scenario == "greet_fast":
        persons = [S(1, desk_spot, [{"do": "approach", "customer": 2, "after_t": t_app},
                                    {"do": "talk", "customer": 2, "s": talk},
                                    {"do": "walk_to", "target": desk_spot}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   C(2, [{"do": "walk_to", "target": V1}, {"do": "await_staff", "face": "car_1"},
                         {"do": "idle", "s": 4.0, "face": "car_1", "anim": "look_around"},
                         {"do": "walk_to", "target": V2}, {"do": "idle", "s": 999, "face": f"car_{others[0]['id']}",
                                                            "anim": "look_around"}], hard_neg=hard)]
        return persons, [], 45.0, ["short_clip"]
    if scenario == "greet_slow":
        t2 = round(float(rng.uniform(44.0, 52.0)), 2)
        persons = [S(1, desk_spot, [{"do": "sit", "until_t": 999}]),
                   S(2, H2, [{"do": "approach", "customer": 3, "after_t": t2, "face_wait": f"car_{others[0]['id']}"},
                             {"do": "talk", "customer": 3, "s": talk},
                             {"do": "walk_to", "target": H2}, {"do": "idle", "s": 999, "face": "entrance"}],
                     default_face="entrance"),
                   C(3, [{"do": "walk_to", "target": V1}, {"do": "idle", "s": 16.0, "face": "car_1", "anim": "look_around"},
                         {"do": "walk_to", "target": V2}, {"do": "await_staff", "face": f"car_{others[0]['id']}"},
                         {"do": "idle", "s": 999, "face": f"car_{others[0]['id']}", "anim": "look_around"}], hard_neg=hard)]
        return persons, [], 78.0, ["short_clip"]
    if scenario == "walkout":
        persons = [S(1, desk_spot, [{"do": "idle", "s": 999, "face": "desk"}], default_face="desk"),
                   C(2, [{"do": "walk_to", "target": V1}, {"do": "idle", "s": float(round(rng.uniform(12, 16), 2)), "face": "car_1", "anim": "look_around"},
                         {"do": "walk_to", "target": V2}, {"do": "idle", "s": float(round(rng.uniform(9, 13), 2)),
                                                            "face": f"car_{others[0]['id']}", "anim": "look_around"},
                         {"do": "leave"}], hard_neg=hard)]
        return persons, [], "exit+2.5", ["short_clip"]
    if scenario == "party":
        persons = [S(1, desk_spot, [{"do": "approach", "customer": 2, "after_t": t_app},
                                    {"do": "talk", "customer": 2, "s": talk + 4},
                                    {"do": "walk_to", "target": desk_spot}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   C(2, [{"do": "walk_to", "target": V1}, {"do": "await_staff", "face": "car_1"},
                         {"do": "idle", "s": 3.0, "face": "car_1", "anim": "look_around"},
                         {"do": "walk_to", "target": V2}, {"do": "idle", "s": 999, "face": f"car_{others[0]['id']}",
                                                            "anim": "look_around"}], door=-0.45),
                   _person(3, "customer", rng, chars, hard=hard, spawn_m=[round(x_e + 0.45, 3), behaviour.OUTSIDE_Y],
                           spawn_yaw_deg=90.0, enter_t=0.0, present_at_start=False, door_x_offset=0.45,
                           default_face="car_1", script=[{"do": "follow", "leader": 2, "offset_m": [0.9, 0.0]}])]
        return persons, [[2, 3]], 55.0, ["short_clip"]
    if scenario == "handoff":
        t2 = round(float(rng.uniform(24.0, 28.0)), 2)
        persons = [S(1, desk_spot, [{"do": "approach", "customer": 3, "after_t": t_app},
                                    {"do": "talk_until_relieved", "customer": 3, "extra_s": 1.5},
                                    {"do": "walk_to", "target": desk_spot}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   S(2, H2, [{"do": "approach", "customer": 3, "after_t": t2, "face_wait": f"car_{others[0]['id']}"},
                             {"do": "talk", "customer": 3, "s": talk},
                             {"do": "walk_to", "target": H2}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   C(3, [{"do": "walk_to", "target": V1}, {"do": "await_staff", "face": "car_1"},
                         {"do": "idle", "s": 999, "face": "car_1", "anim": "look_around"}], hard_neg=hard)]
        return persons, [], 62.0, ["short_clip"]
    if scenario == "crowded":
        persons = [S(1, desk_spot, [{"do": "approach", "customer": 3, "after_t": t_app},
                                    {"do": "talk", "customer": 3, "s": talk},
                                    {"do": "walk_to", "target": desk_spot}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   S(2, H2, [{"do": "approach", "customer": 4, "after_t": 14.0, "face_wait": "entrance"},
                             {"do": "talk", "customer": 4, "s": 18.0},
                             {"do": "walk_to", "target": H2}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   C(3, [{"do": "walk_to", "target": V1}, {"do": "await_staff", "face": "car_1"},
                         {"do": "idle", "s": 999, "face": "car_1", "anim": "look_around"}]),
                   C(4, [{"do": "walk_to", "target": V2}, {"do": "await_staff", "face": f"car_{others[0]['id']}"},
                         {"do": "idle", "s": 999, "face": f"car_{others[0]['id']}", "anim": "look_around"}], enter_t=8.0, door=-0.45),
                   _person(5, "customer", rng, chars, spawn_m=[round(x_e + 0.45, 3), behaviour.OUTSIDE_Y],
                           spawn_yaw_deg=90.0, enter_t=8.0, present_at_start=False, door_x_offset=0.45,
                           default_face="car_1", script=[{"do": "follow", "leader": 4, "offset_m": [0.9, 0.0]}]),
                   C(6, [{"do": "walk_to", "target": V3}, {"do": "idle", "s": 14.0, "face": f"car_{others[1]['id']}", "anim": "look_around"},
                         {"do": "walk_to", "target": V4}, {"do": "idle", "s": 8.0, "face": f"car_{others[2]['id']}", "anim": "look_around"},
                         {"do": "leave"}], enter_t=3.0),
                   C(7, [{"do": "walk_to", "target": free_spot_near(nav, V4, [V1, V2, V3, V4, H2, desk_spot], 1.6, rng, 3.0) or V4},
                         {"do": "idle", "s": 999, "face": f"car_{others[2]['id']}", "anim": "look_around"}], enter_t=20.0, hard_neg=hard)]
        return persons, [[4, 5]], 75.0, ["short_clip"]
    if scenario == "dealership_long":
        persons = [S(1, desk_spot, [{"do": "approach", "customer": 3, "after_t": t_app},
                                    {"do": "talk", "customer": 3, "s": 150.0},
                                    {"do": "walk_to", "target": desk_spot}, {"do": "idle", "s": 999, "face": "entrance"}]),
                   S(2, H2, [{"do": "idle", "s": 999, "face": "entrance"}]),
                   C(3, [{"do": "walk_to", "target": V1}, {"do": "await_staff", "face": "car_1"},
                         {"do": "idle", "s": 5.0, "face": "car_1", "anim": "look_around"},
                         {"do": "walk_to", "target": V2}, {"do": "idle", "s": 30.0, "face": f"car_{others[0]['id']}", "anim": "look_around"},
                         {"do": "leave"}]),
                   C(4, [{"do": "walk_to", "target": V3}, {"do": "idle", "s": 40.0, "face": f"car_{others[1]['id']}", "anim": "look_around"},
                         {"do": "walk_to", "target": V4}, {"do": "idle", "s": 30.0, "face": f"car_{others[2]['id']}", "anim": "look_around"},
                         {"do": "leave"}], enter_t=60.0, hard_neg=hard)]
        return persons, [], 240.0, ["dealership", "short_clip"]
    raise ValueError(scenario)


def _cam_candidates(lay, target):
    """Wall / corner mounts 4.5-9 m from the action centroid, aimed at it."""
    W, D = lay["room"]["width_m"], lay["room"]["depth_m"]
    pos = []
    for x in np.arange(0.4, W - 0.3, 1.0):
        pos += [(float(x), 0.35), (float(x), D - 0.3)]
    for y in np.arange(0.4, D - 0.3, 1.0):
        pos += [(0.3, float(y)), (W - 0.3, float(y))]
    out = []
    k = 0
    for (x, y) in pos:
        dist = math.dist((x, y), target)
        if not 4.5 <= dist <= 9.0 or math.dist((x, y), lay["desk"]["center_m"]) < 3.0:
            continue
        for h in (2.8, 3.2):
            for hfov in (70.0, 80.0, 90.0):
                yaw = math.degrees(math.atan2(target[1] - y, target[0] - x))
                pitch = -math.degrees(math.atan2(h - 0.9, dist))
                pitch = float(np.clip(pitch, -50.0, -25.0))
                k += 1
                out.append({"id": k, "position_m": [round(x, 3), round(y, 3), h], "yaw_deg": round(yaw, 2),
                            "pitch_deg": round(pitch, 2), "hfov_deg": hfov, "width_px": 1280, "height_px": 720})
    return out


def _seg_hits_box(p0, p1, fp, hz):
    """Segment p0->p1 (3D) passes through the prism (convex footprint fp, 0..hz)."""
    for s in np.linspace(0.05, 0.95, 40):
        q = p0 + s * (p1 - p0)
        if 0.0 <= q[2] <= hz and layout.point_in_polygon(q[:2], fp):
            return True
    return False


def _person_occluded(C, st, f, pid, agents, h, r=0.22):
    """Torso and head rays both pass within r of another (nearer) person's body axis."""
    P = np.array([st["pos_m"][0], st["pos_m"][1]])
    d_self = np.linalg.norm(P - C[:2])
    hits = 0
    for z in (0.55 * h, 0.9 * h):
        T = np.array([P[0], P[1], z])
        blocked = False
        for qid, q in agents.items():
            if qid == pid or not q["frames"][f]["rendered"]:
                continue
            Q = np.asarray(q["frames"][f]["pos_m"], float)
            if np.linalg.norm(Q - C[:2]) >= d_self:
                continue
            # closest approach of the ray C->T to the vertical axis through Q (in xy), at that height
            d2 = T[:2] - C[:2]
            u = float(np.clip(np.dot(Q - C[:2], d2) / np.dot(d2, d2), 0.0, 1.0))
            if np.linalg.norm(C[:2] + u * d2 - Q) < r and (C[2] + u * (T[2] - C[2])) < 1.75:
                blocked = True
                break
        hits += blocked
    return hits == 2


def visibility(cam_p, tl, cfg, lay, step):
    """Per person: fraction of output frames with full body in view, >= MIN_BOX_H_PX tall, not hidden
    behind a car; also per-person lifetime fraction and any-person fraction."""
    cam = projection.Camera.from_params(cam_p)
    C = np.asarray(cam_p["position_m"], float)
    cars = [(c["footprint_m"], c["height_m"]) for c in lay["cars"]]
    cars.append((lay["desk"]["footprint_m"], 1.25))     # desk + monitor
    res, life, any_seen, heights = {}, {}, None, []
    agents = tl["agents"]
    for pid, ag in agents.items():
        h = schema.person(cfg, int(pid))["height_m"]
        fr = ag["frames"][::step]
        seen = []
        for k, st in enumerate(fr):
            ok = False
            if st["rendered"]:
                x, y = st["pos_m"]
                (u0, v0, z0), (u1, v1, z1) = cam.project([x, y, 0.02]), cam.project([x, y, h])
                if z0 > 0.3 and z1 > 0.3 and 0 <= u0 < cam.width and 0 <= u1 < cam.width and 0 <= v1 and v0 < cam.height \
                        and (v0 - v1) >= MIN_BOX_H_PX:
                    blocked = any(any(_seg_hits_box(C, np.array([x, y, z]), fp, hz) for fp, hz in cars)
                                  for z in (0.3 * h, 0.55 * h, 0.9 * h))
                    ok = not blocked and not _person_occluded(C, st, k * step, pid, agents, h)
                    if ok:
                        heights.append(v0 - v1)
            seen.append(ok)
        res[pid] = float(np.mean(seen))
        alive = [s["rendered"] for s in fr]
        life[pid] = float(sum(a and b for a, b in zip(seen, alive)) / max(1, sum(alive)))
        any_seen = np.array(seen) if any_seen is None else (any_seen | np.array(seen))
    return res, life, float(any_seen.mean()), (float(np.median(heights)) if heights else 0.0)


def check_events(tl, cfg, presets, eval_presets):
    for pr in eval_presets:
        rule = gt_events.events_by_rule(tl, cfg, presets[pr])
        un_s, un_r = gt_events.match_events(tl["events_scripted"], rule, MATCH_MARGIN_S, TYPES_CHECKED)
        un_r = [r for r in un_r if r["type"] in ("greet", "walkout", "handoff")]
        if un_s or un_r:
            return f"[{pr}] unmatched scripted {un_s} rule {un_r}"
    return None


def min_pair_dist(tl):
    ags = list(tl["agents"].values())
    best = math.inf
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            for a, b in zip(ags[i]["frames"], ags[j]["frames"]):
                if a["rendered"] and b["rendered"]:
                    best = min(best, math.dist(a["pos_m"], b["pos_m"]))
    return best


def generate(scene_id, scenario, seed, presets, max_tries=300, verbose=False):
    rng = np.random.default_rng(1000 + seed)
    base = json.loads((CFG_DIR / "defaults.json").read_text())
    for attempt in range(max_tries):
        lay_cfg = sample_layout(rng)
        if lay_cfg is None:
            continue
        cfg = dict(base, scene_id=scene_id, scenario=scenario, seed=seed, **lay_cfg)
        cfg["cameras"] = [{"id": 0, "position_m": [0.4, 0.4, 3.0], "yaw_deg": 45.0, "pitch_deg": -35.0,
                           "hfov_deg": 74.0, "width_px": 1280, "height_px": 720}]
        cfg["persons"] = []
        lay = layout.build_layout(dict(cfg, persons=[]))
        nav = navigation.NavGraph(lay, cfg["navigation"])
        sc = make_script(scenario, lay, nav, cfg, rng)
        if sc is None:
            continue
        persons, parties, dur, eval_presets = sc
        cfg.update(persons=persons, parties=parties, eval_presets=eval_presets)
        cfg["uniform"] = UNIFORM
        assign_characters(cfg, rng)
        try:
            if dur == "exit+2.5":
                tl = behaviour.build_timeline(dict(cfg, duration_s=120.0))
                exits = [e["t"] for e in tl["events_scripted"] if e["type"] == "walkout"]
                if not exits:
                    continue
                dur = math.ceil((max(exits) + 2.5) * 10) / 10.0
            cfg["duration_s"] = float(dur)
            tl = behaviour.build_timeline(cfg)
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"  attempt {attempt}: {e}")
            continue
        err = check_events(tl, cfg, presets, eval_presets)
        if err:
            if verbose:
                print(f"  attempt {attempt}: events {err}")
            continue
        if min_pair_dist(tl) < MIN_PAIR_DIST_M:
            continue
        if not _scenario_timing_ok(scenario, tl):
            if verbose:
                print(f"  attempt {attempt}: timing {[(e['type'], e['t']) for e in tl['events_scripted']]}")
            continue
        # camera selection
        targets = [p["pos_m"] for a in tl["agents"].values() for p in a["frames"][::30] if p["rendered"] and p["pos_m"][1] > 0]
        centroid = np.mean(np.asarray(targets), axis=0).tolist()
        cands = []
        lay = layout.build_layout(cfg)
        for cp in _cam_candidates(lay, centroid):
            vis, life, anyf, box_h = visibility(cp, tl, cfg, lay, cfg["frame_step"] * 5)
            cands.append((min(vis.values()), anyf, cp, vis, life, box_h))
        # rank: enough visibility first, then bigger people (median box height, capped at 380 px)
        cands.sort(key=lambda c: (-(c[0] >= SHARE_VIS), -min(c[5], 380.0), -c[0]))
        chosen = []
        need = 2
        for mn, anyf, cp, vis, life, box_h in cands:
            if any(math.dist(cp["position_m"][:2], c[2]["position_m"][:2]) < 4.0 for c in chosen):
                continue
            if scenario in SHORT and mn < SHARE_VIS:
                continue
            if anyf < 0.5:
                continue
            chosen.append((mn, anyf, cp, vis, life, box_h))
            if len(chosen) == need:
                break
        if scenario not in SHORT and chosen:
            # each agent >= 0.8 of its lifetime in at least one chosen camera
            pids = chosen[0][4].keys()
            if not all(max(c[4][p] for c in chosen) >= 0.8 for p in pids):
                chosen = []
        if len(chosen) < need:
            if verbose:
                print(f"  attempt {attempt}: cameras best min-vis {cands[0][0]:.2f}")
            continue
        cfg["cameras"] = []
        for i, c in enumerate(chosen):
            cam = dict(c[2], id=i + 1)
            cfg["cameras"].append(cam)
        # re-aim the greeting choreography at the chosen main camera (pair axis across its view ray),
        # then re-check events and visibility with the final timeline
        cfg["script"] = {"staff_stop_dist_m": 1.25, "staff_approach_angles_deg": 15.0,
                         "approach_cam_m": cfg["cameras"][0]["position_m"][:2]}
        try:
            tl2 = behaviour.build_timeline(cfg)
        except (RuntimeError, ValueError):
            continue
        if check_events(tl2, cfg, presets, eval_presets) or min_pair_dist(tl2) < MIN_PAIR_DIST_M \
                or not _scenario_timing_ok(scenario, tl2):
            continue
        final = [visibility(cam, tl2, cfg, lay, cfg["frame_step"] * 5) for cam in cfg["cameras"]]
        if scenario in SHORT and min(min(v[0].values()) for v in final) < SHARE_VIS:
            if verbose:
                print(f"  attempt {attempt}: final vis {[round(min(v[0].values()), 3) for v in final]}")
            continue
        cfg["_generator"] = {"attempt": attempt,
                             "camera_min_person_vis": [round(min(v[0].values()), 3) for v in final],
                             "camera_any_person_vis": [round(v[2], 3) for v in final],
                             "camera_median_box_h_px": [round(v[3], 1) for v in final],
                             "note": "visibility estimated from geometry (full body in frame, >= 60 px, "
                                     "not behind a car or a nearer person)"}
        return cfg, tl2
    raise RuntimeError(f"{scene_id}: no valid scene after {max_tries} attempts")


def assign_characters(cfg, rng):
    """Deterministic character per person: staff from the staff pool, hard negatives from the
    suit/blazer pool (never the same character as a staff member), others casual. Height = person
    height (uniform scale in Blender)."""
    chars = _characters()
    if not chars:
        return
    pool = lambda k: [c["name"] for c in chars if k in c["pool"]]  # noqa: E731
    used = set()

    def pick(names):
        free = [n for n in names if n not in used] or names
        n = free[int(rng.integers(len(free)))]
        used.add(n)
        return n
    for p in sorted(cfg["persons"], key=lambda q: (q["role"] != "staff", not q["hard_negative"], q["id"])):
        if p["role"] == "staff":
            p["character"] = pick(pool("staff"))
        elif p["hard_negative"]:
            p["character"] = pick(pool("hard_negative"))
        else:
            p["character"] = pick(pool("casual"))
        g = next(c["gender"] for c in chars if c["name"] == p["character"])
        mu, lo, hi = (1.77, 1.62, 1.95) if g == "M" else (1.65, 1.55, 1.80)
        p["height_m"] = round(float(np.clip(rng.normal(mu, 0.06), lo, hi)), 3)


def _scenario_timing_ok(scenario, tl):
    ev = tl["events_scripted"]
    g = [e["t"] for e in ev if e["type"] == "greet"]
    if scenario == "greet_fast":
        return len(g) == 1 and 5.0 <= g[0] <= 15.0
    if scenario == "greet_slow":
        return len(g) == 1 and 40.0 <= g[0] <= 70.0
    if scenario == "walkout":
        w = [e["t"] for e in ev if e["type"] == "walkout"]
        return not g and len(w) == 1 and 30.0 <= w[0] <= 60.0
    if scenario == "party":
        return len(g) == 2 and any(e["type"] == "party" for e in ev)
    if scenario == "handoff":
        return len(g) == 1 and sum(e["type"] == "handoff" for e in ev) == 1
    if scenario == "dealership_long":
        return any(e["type"] == "engagement" and e["t_end"] - e["t"] > 120.0 for e in ev)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    presets = json.loads((CFG_DIR / "presets.json").read_text())
    for i, (sid, scen, seed) in enumerate(SCENES):
        if a.only is not None and i not in a.only:
            continue
        cfg, tl = generate(sid, scen, seed, presets, verbose=a.v)
        path = CFG_DIR / f"scene_{sid[1:]}.json"
        schema.dump_json(cfg, path)
        evs = [(e["type"], e["t"]) for e in tl["events_scripted"] if e["type"] != "engagement"]
        print(f"{path.name}: {scen} dur={cfg['duration_s']} persons={len(cfg['persons'])} cams="
              f"{cfg['_generator']['camera_min_person_vis']} events={evs}")


if __name__ == "__main__":
    main()
