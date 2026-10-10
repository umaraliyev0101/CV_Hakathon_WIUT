"""Scenario -> per-agent timeline at scene fps (pure). Blender only replays this.

Each frame state: t, present, pos_m [x, y], yaw_deg (facing, CCW from +x), action
(walk / idle / talk), anim_clip, clip_time (s into the current clip), gait_phase (cycles,
advances with distance walked so the stride matches ground speed), speed_mps.

`mannequin_pose` turns a state into joint angles for the procedural capsule mannequin used
while Mixamo assets are missing. Pelvis height is solved so the lowest foot touches z = 0.
"""
import hashlib
import json
import math

import numpy as np

from sim import layout, navigation, schema


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _bearing(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


class _Agent:
    """Accumulates per-frame states; yaw is rate limited toward a per-frame target."""

    def __init__(self, person, fps, turn_rate_dps):
        self.p = person
        self.fps = fps
        self.max_dyaw = turn_rate_dps / fps
        self.frames = []
        self.pos = list(person["spawn_m"])
        self.yaw = float(person["spawn_yaw_deg"])
        self.phase = 0.25
        self.stride = stride_m(person["height_m"])
        self.clip, self.clip_t = None, 0.0

    def _push(self, action, target_yaw, speed=0.0, present=True):
        if target_yaw is not None:
            d = _wrap(target_yaw - self.yaw)
            self.yaw = _wrap(self.yaw + max(-self.max_dyaw, min(self.max_dyaw, d)))
        clip = action
        if clip != self.clip:
            self.clip, self.clip_t = clip, 0.0
        else:
            self.clip_t += 1.0 / self.fps
        self.frames.append({
            "t": round(len(self.frames) / self.fps, 6), "present": present,
            "pos_m": [round(self.pos[0], 6), round(self.pos[1], 6)], "yaw_deg": round(self.yaw, 6),
            "action": action, "anim_clip": clip, "clip_time": round(self.clip_t, 6),
            "gait_phase": round(self.phase, 6), "stride_m": round(self.stride, 6), "speed_mps": round(speed, 6),
        })

    def idle_until(self, n_frames, face=None, action="idle"):
        while len(self.frames) < n_frames:
            tgt = None if face is None else _bearing(self.pos, face() if callable(face) else face)
            self._push(action, tgt)

    def walk(self, path, v, accel, decel, stride, pre_turn_tol_deg=15.0):
        # face the first leg of the path before stepping off: no sideways walking
        self.turn_to(path[1], tol_deg=pre_turn_tol_deg)
        length = navigation.path_length(path)
        prof = navigation.trapezoid_profile(length, v, accel, self.fps, decel)
        # whole half-cycles per walk: it starts and ends at phase 0.25 mod 0.5, where the planted
        # foot is under the hip and the swing foot is mid-air (scaled to zero at rest)
        half_cycles = max(1, int(round(2.0 * length / stride)))
        self.stride = 2.0 * length / half_cycles
        self.phase = math.floor(self.phase) + 0.25
        stride = self.stride
        prev = prof[0]
        for s in prof[1:]:
            pt, heading = navigation.point_at(path, s)
            speed = (s - prev) * self.fps
            self.phase += (s - prev) / stride
            self.pos = pt
            self._push("walk", heading, speed)
            prev = s

    def turn_to(self, point, tol_deg, action="idle"):
        while abs(_wrap(_bearing(self.pos, point) - self.yaw)) > tol_deg:
            self._push(action, _bearing(self.pos, point))


def stride_m(height_m):
    return 0.85 * height_m


def _approach_point(nav, cam_pos, cust_pos, staff_pos, r, step_deg):
    """Free point at distance r from the customer, cheap to reach and not occluding along the
    camera ray (pair axis roughly perpendicular to the ray keeps both people visible)."""
    ray = np.asarray(cust_pos) - np.asarray(cam_pos[:2])
    ray /= np.linalg.norm(ray)
    best = None
    for k in range(int(round(360 / step_deg))):
        a = math.radians(k * step_deg)
        cand = [cust_pos[0] + r * math.cos(a), cust_pos[1] + r * math.sin(a)]
        if not nav.is_free(cand):
            continue
        path = nav.shortest_path(staff_pos, cand)
        if path is None:
            continue
        axis = np.array([math.cos(a), math.sin(a)])
        # turning on the spot after arrival looks unnatural: prefer arriving already facing them
        turn = abs(_wrap(_bearing(path[-2], path[-1]) - _bearing(cand, cust_pos)))
        cost = navigation.path_length(path) + 4.0 * abs(float(np.dot(axis, ray))) + 2.0 * turn / 180.0
        if best is None or cost < best[0]:
            best = (cost, cand, path)
    if best is None:
        raise RuntimeError("no free approach point around customer")
    return best[1], best[2]


def build_timeline(cfg):
    cfg = schema.load_scene_dict(cfg)
    if cfg["scenario"] != "greet_fast":
        raise NotImplementedError("quick test implements greet_fast only")
    rng = np.random.default_rng(cfg["seed"])
    fps = cfg["fps_scene"]
    n = schema.n_scene_frames(cfg)
    lay = layout.build_layout(cfg)
    navc = cfg["navigation"]
    nav = navigation.NavGraph(lay, navc)
    sc = cfg["script"]
    staff_p = schema.person(cfg, sc["staff"])
    cust_p = schema.person(cfg, sc["customer"])
    v_cust = float(rng.uniform(*cust_p["speed_mps"]))
    v_staff = float(rng.uniform(*staff_p["speed_mps"]))

    car = next(c for c in lay["cars"] if f"car_{c['id']}" == sc["customer_look_at"])
    cust = _Agent(cust_p, fps, navc["turn_rate_dps"])
    staff = _Agent(staff_p, fps, navc["turn_rate_dps"])

    # customer: enter at t = 0, walk to the car apron, look at the car
    cust_path = nav.shortest_path(cust_p["spawn_m"], sc["customer_target_m"])
    cust.walk([cust_p["spawn_m"]] + cust_path[1:], v_cust, navc["accel_mps2"], navc["decel_mps2"], stride_m(cust_p["height_m"]))
    cust_arrive = len(cust.frames)

    # staff: idle at desk, then walk to an approach point next to the customer
    desk_c = lay["desk"]["center_m"]
    staff.idle_until(int(round(sc["staff_start_walk_t"] * fps)), face=desk_c)
    goal, path = _approach_point(nav, cfg["camera"]["position_m"], sc["customer_target_m"],
                                 staff.pos, sc["staff_stop_dist_m"], sc["staff_approach_angles_deg"])
    staff.walk(path, v_staff, navc["accel_mps2"], navc["decel_mps2"], stride_m(staff_p["height_m"]))
    staff.turn_to(sc["customer_target_m"], tol_deg=3.0)
    talk_start = len(staff.frames)
    if talk_start < cust_arrive:
        raise RuntimeError("staff arrived before the customer; adjust staff_start_walk_t")

    cust.idle_until(talk_start, face=car["center_m"])
    staff.idle_until(n, face=sc["customer_target_m"], action="talk")
    cust.idle_until(n, face=staff.pos, action="talk")
    for a in (staff, cust):
        del a.frames[n:]
    if (n - talk_start) / fps < sc["min_talk_s"]:
        raise RuntimeError("talk shorter than min_talk_s")

    return {
        "scene_id": cfg["scene_id"], "scenario": cfg["scenario"], "seed": cfg["seed"],
        "fps": fps, "n_frames": n,
        "agents": {str(a.p["id"]): {"role": a.p["role"], "frames": a.frames} for a in (staff, cust)},
        "events_scripted": [{"type": "greet", "staff": staff_p["id"], "customer": cust_p["id"],
                             "t": round(talk_start / fps, 6)}],
        "paths_m": {str(staff_p["id"]): path, str(cust_p["id"]): cust_path},
    }


def timeline_hash(tl):
    return hashlib.sha256(json.dumps(tl, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- procedural mannequin

PROPORTIONS = {  # fractions of body height
    "thigh": 0.25, "shin": 0.245, "foot_h": 0.035, "hip_w": 0.09,
    "torso": 0.29, "shoulder_w": 0.115, "arm": 0.33, "head_r": 0.065, "neck": 0.04,
}


def _leg_drop(theta, knee, h):
    """Vertical distance hip -> foot sole for hip flexion theta, knee flexion (deg); feet stay flat."""
    P = PROPORTIONS
    t, k = math.radians(theta), math.radians(knee)
    return P["thigh"] * h * math.cos(t) + P["shin"] * h * math.cos(t - k) + P["foot_h"] * h


def _foot_x(p, S):
    """Foot position along the walking direction relative to the hip for leg phase p in [0, 1).
    Stance (p < 0.5): back linearly from +S/4 to -S/4, i.e. at exactly body speed while the phase
    advances with distance, so the planted foot does not slide. Swing: smoothstep forward."""
    if p < 0.5:
        return S / 4.0 - S * p
    q = (p - 0.5) * 2.0
    return -S / 4.0 + (S / 2.0) * (q * q * (3.0 - 2.0 * q))


def _leg_ik(x, down, a, b):
    """Two-link leg (thigh a, shin b) reaching a foot point x forward and `down` below the hip.
    Returns hip flexion and knee flexion in degrees (knee bends backward)."""
    d = min(math.hypot(x, down), a + b - 1e-9)
    knee = math.pi - math.acos(max(-1.0, min(1.0, (a * a + b * b - d * d) / (2 * a * b))))
    alpha = math.acos(max(-1.0, min(1.0, (a * a + d * d - b * b) / (2 * a * d))))
    return math.degrees(math.atan2(x, down) + alpha), math.degrees(knee)


def mannequin_pose(state, h):
    P = PROPORTIONS
    a, b, fh = P["thigh"] * h, P["shin"] * h, P["foot_h"] * h
    L = a + b
    S = stride_m(h)
    act, ct = state["action"], state["clip_time"]
    env = min(1.0, state["speed_mps"] / 0.4) if act == "walk" else 0.0
    ph = state["gait_phase"]
    phases = {"l": ph % 1.0, "r": (ph + 0.5) % 1.0}
    # stance leg is straight and sets the pelvis height; the swing foot gets explicit clearance
    stance = min(phases, key=lambda k: phases[k])
    S = state.get("stride_m", S)
    xs = _foot_x(phases[stance], S)   # never scaled: a planted foot must not move
    pelvis_z = math.sqrt(L * L - xs * xs) + fh
    leg = {}
    for side, p in phases.items():
        if side == stance:
            x, lift = xs, 0.0
        else:
            x = env * _foot_x(p, S)
            lift = env * 0.07 * h * math.sin(math.pi * (p - 0.5) * 2.0)
        leg[side] = _leg_ik(x, pelvis_z - fh - lift, a, b)
    (lh, lk), (rh, rk) = leg["l"], leg["r"]
    ls, rs = -0.7 * lh, -0.7 * rh   # arms swing opposite to legs
    l_el = r_el = 10.0 * env
    head = 0.0
    if act == "talk":
        ls = 12.0 + 14.0 * math.sin(2 * math.pi * 0.6 * ct)
        rs = 6.0 + 10.0 * math.sin(2 * math.pi * 0.45 * ct + 1.3)
        l_el = 45.0 + 20.0 * math.sin(2 * math.pi * 0.6 * ct + 0.5)
        r_el = 25.0 + 15.0 * math.sin(2 * math.pi * 0.45 * ct + 2.0)
        head = 6.0 * math.sin(2 * math.pi * 0.5 * ct)
    elif act == "idle":
        ls = 2.0 * math.sin(2 * math.pi * 0.25 * ct)
        rs = -ls
        l_el = r_el = 8.0
        head = 2.0 * math.sin(2 * math.pi * 0.2 * ct + 0.7)
    return {
        "pelvis_z": pelvis_z,
        "joints": {"l_hip": lh, "r_hip": rh, "l_knee": lk, "r_knee": rk,
                   "l_shoulder": ls, "r_shoulder": rs, "l_elbow": l_el, "r_elbow": r_el, "head": head},
    }


def foot_world(state, pose, h):
    """World (x, y, z) of each foot sole centre under the ankle: {"l": ..., "r": ...}."""
    P = PROPORTIONS
    J = pose["joints"]
    yaw = math.radians(state["yaw_deg"])
    out = {}
    for side, sy in (("l", 1.0), ("r", -1.0)):
        t, k = math.radians(J[f"{side}_hip"]), math.radians(J[f"{side}_knee"])
        fwd = P["thigh"] * h * math.sin(t) + P["shin"] * h * math.sin(t - k)
        lat = sy * P["hip_w"] * h
        x = state["pos_m"][0] + math.cos(yaw) * fwd - math.sin(yaw) * lat
        y = state["pos_m"][1] + math.sin(yaw) * fwd + math.cos(yaw) * lat
        out[side] = (x, y, pose["pelvis_z"] - _leg_drop(J[f"{side}_hip"], J[f"{side}_knee"], h))
    return out


def foot_heights(pose, h):
    j = pose["joints"]
    return [pose["pelvis_z"] - _leg_drop(j["l_hip"], j["l_knee"], h),
            pose["pelvis_z"] - _leg_drop(j["r_hip"], j["r_knee"], h)]
