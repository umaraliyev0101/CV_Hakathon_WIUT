"""Scenario -> per-agent timeline at scene fps (pure). Blender only replays this.

Every person has a `script`: a list of actions run by a deterministic frame-stepped scheduler
(one Python generator per agent, stepped in person-id order, exactly one state per frame):

  enter()                        appear outside the entrance and walk in through the doorway
  walk_to(target)                navigation path around car/desk footprints, trapezoid speed profile
  idle(s, face, anim)            stand; anim idle / look_around; face a point, zone or person id
  idle_until(t, face, anim)
  approach(customer, after_t)    staff: wait until t >= after_t and the customer is waiting, then
                                 walk to a free point stop_dist_m from them, turn to face them
                                 -> scripted greet (first staff) or handoff (later staff), engagement start
  talk(customer, s)              staff: talk facing the customer, then release them
  talk_until_relieved(customer, extra_s)   staff: talk until another staff member has taken over
  await_staff(face, anim)        customer: browse until a staff member arrives, then talk facing
                                 the current engager until released
  follow(leader, offset_m)       customer (party member): mirror the leader's walks (target +
                                 offset), waits and exit
  sit(desk)                      faked as idle at the desk (no sitting clip)
  leave()                        walk out through the entrance and disappear -> scripted walkout if
                                 never greeted

Frame state: t, present (inside the room: y >= 0, used by the event rules), rendered (in the
scene: drawn and labelled), pos_m [x, y], yaw_deg (CCW from +x), action (walk / idle / talk),
anim_clip (walk, idle_a, idle_b, look_around, talk_a, talk_b, turn_left, turn_right), clip_time,
gait_phase (cycles; advances with distance walked so the stride matches the ground speed, and
with yaw while turning on the spot so turns are stepped), stride_m, speed_mps.

Scripted-greet definition (documented choice): the greet time is the staff member's ARRIVAL,
i.e. the first talk frame after stopping stop_dist_m from the customer and facing them. The
choreography keeps the rule greet (staff enters greet_dist_m while facing) within about 0.6 s of
it: the last approach leg is straight toward the customer (>= 1.0 m long, <= 45 deg turn at the
end) and the staff stops at 1.25 m, so only the last 0.25 m (decelerating) lies inside 1.5 m.
"""
import hashlib
import json
import math

import numpy as np

from sim import layout, navigation, schema

OUTSIDE_Y = -2.6      # spawn / despawn line outside the glass facade (entrance on the south wall)
DOOR_IN_Y = 0.7       # first waypoint inside the doorway
APPROACH_MIN_FINAL_SEG = 1.0
APPROACH_MAX_FINAL_TURN = 45.0
MIN_SEPARATION_PLAN = 0.9


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _bearing(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def stride_m(height_m):
    return 0.85 * height_m


class Board:
    """Shared scenario state: who waits, who is engaged with whom, the scripted event log."""

    def __init__(self, parties):
        self.awaiting = set()
        self.engager = {}            # customer -> staff currently talking with them
        self.last_engager = {}
        self.greeted = set()
        self.events = []
        self.party_of = {}
        for g in parties:
            for c in g:
                self.party_of[c] = list(g)
        self.leader_intent = {}      # leader id -> ("walk", target, v, frame) | ("await",) | ("leave", frame)

    def members(self, c):
        return self.party_of.get(c, [c])

    def engager_of(self, c):
        for m in self.members(c):
            if self.engager.get(m) is not None:
                return self.engager[m]
        return None


class Agent:
    def __init__(self, sim, person, rng):
        self.sim = sim
        self.p = person
        self.id = person["id"]
        self.fps = sim.fps
        self.max_dyaw = sim.navc["turn_rate_dps"] / sim.fps
        self.frames = []
        self.pos = list(person["spawn_m"])
        self.yaw = float(person["spawn_yaw_deg"])
        self.phase = 0.25
        self.stride = stride_m(person["height_m"])
        self.anim, self.clip_t = None, 0.0
        self.rendered = person.get("enter_t", 0.0) <= 0.0 and person.get("present_at_start", True)
        self.v = float(rng.uniform(*person["speed_mps"]))
        self.idle_var = "idle_a" if rng.uniform() < 0.5 else "idle_b"
        self.talk_var = "talk_a" if rng.uniform() < 0.5 else "talk_b"

    # ------------------------------------------------------------------ frame output
    def _push(self, action, target_yaw, speed=0.0, anim=None):
        anim = anim or action
        dy = 0.0
        if target_yaw is not None:
            d = _wrap(target_yaw - self.yaw)
            dy = max(-self.max_dyaw, min(self.max_dyaw, d))
            self.yaw = _wrap(self.yaw + dy)
        if action != "walk":
            if abs(dy) > 0.25 * self.max_dyaw:
                anim = "turn_left" if dy > 0 else "turn_right"
                self.phase += abs(dy) / 360.0      # stepped turn: one step per 90 deg
            elif anim == "idle":
                anim = self.idle_var
            elif anim == "talk":
                anim = self.talk_var
        if anim != self.anim:
            self.anim, self.clip_t = anim, 0.0
        else:
            self.clip_t += 1.0 / self.fps
        present = self.rendered and self.pos[1] >= 0.0
        if self.frames and self.frames[-1]["present"] and not present and getattr(self, "leaving", False):
            self.sim.on_exit(self.id, len(self.frames))
        self.frames.append({
            "t": round(len(self.frames) / self.fps, 6), "present": present, "rendered": self.rendered,
            "pos_m": [round(self.pos[0], 6), round(self.pos[1], 6)], "yaw_deg": round(self.yaw, 6),
            "action": action, "anim_clip": anim, "clip_time": round(self.clip_t, 6),
            "gait_phase": round(self.phase, 6), "stride_m": round(self.stride, 6), "speed_mps": round(speed, 6),
        })

    def _face_yaw(self, face):
        if face is None:
            return None
        pt = self.sim.resolve_point(face)
        if math.dist(pt, self.pos) < 1e-6:
            return None
        return _bearing(self.pos, pt)

    @property
    def f(self):
        return len(self.frames)

    # ------------------------------------------------------------------ primitive generators
    def g_hold(self, face=None, anim="idle", action="idle"):
        while True:
            self._push(action, self._face_yaw(face), anim=anim)
            yield

    def g_idle_frames(self, n, face=None, anim="idle", action="idle"):
        for _ in range(int(n)):
            self._push(action, self._face_yaw(face), anim=anim)
            yield

    def g_turn_to(self, point, tol_deg=3.0):
        pt = self.sim.resolve_point(point)
        while abs(_wrap(_bearing(self.pos, pt) - self.yaw)) > tol_deg:
            self._push("idle", _bearing(self.pos, pt))
            yield

    def g_walk_path(self, path, v=None):
        v = self.v if v is None else v
        navc = self.sim.navc
        if len(path) < 2 or navigation.path_length(path) < 0.05:
            return
        yield from self.g_turn_to(path[1], tol_deg=15.0)
        length = navigation.path_length(path)
        prof = navigation.trapezoid_profile(length, v, navc["accel_mps2"], self.fps, navc["decel_mps2"])
        half_cycles = max(1, int(round(2.0 * length / stride_m(self.p["height_m"]))))
        self.stride = 2.0 * length / half_cycles
        self.phase = math.floor(self.phase) + 0.25
        prev = prof[0]
        for s in prof[1:]:
            pt, heading = navigation.point_at(path, s)
            speed = (s - prev) * self.fps
            self.phase += (s - prev) / self.stride
            self.pos = pt
            self._push("walk", heading, speed)
            prev = s
            yield

    def g_walk_to(self, target, v=None):
        path = self.sim.nav.shortest_path(self.pos, target)
        if path is None:
            raise RuntimeError(f"person {self.id}: no path {self.pos} -> {target}")
        path = [list(self.pos)] + path[1:]
        self.last_path = path
        yield from self.g_walk_path(path, v)

    # ------------------------------------------------------------------ script actions
    def run(self):
        enter_f = int(round(self.p.get("enter_t", 0.0) * self.fps))
        if not self.rendered:
            while self.f < enter_f:
                self._push("idle", None)
                yield
            self.rendered = True
        for act in self.p["script"]:
            yield from getattr(self, "a_" + act["do"])(**{k: v for k, v in act.items() if k != "do"})
        yield from self.g_hold(face=self.sim.default_face(self))

    def a_enter(self, door_x_offset=0.0):
        e = self.sim.lay["entrance"]
        x = e["center_x_m"] + door_x_offset
        self.pos = [x, OUTSIDE_Y]
        self.yaw = 90.0
        self.sim.log_intent(self.id, ("walk", [x, DOOR_IN_Y], self.v, self.f))
        yield from self.g_walk_path([[x, OUTSIDE_Y], [x, DOOR_IN_Y]])

    def a_walk_to(self, target):
        target = self.sim.resolve_point(target)
        path = self.sim.nav.shortest_path(self.pos, target)
        if path is None:
            raise RuntimeError(f"person {self.id}: no path {self.pos} -> {target}")
        path = [list(self.pos)] + path[1:]
        self.sim.log_intent(self.id, ("walk", target, self.v, self.f, path))
        yield from self.g_walk_path(path)

    def a_idle(self, s, face=None, anim="idle"):
        yield from self.g_idle_frames(round(s * self.fps), face, anim)

    def a_idle_until(self, t, face=None, anim="idle"):
        yield from self.g_idle_frames(max(0, round(t * self.fps) - self.f), face, anim)

    def a_sit(self, desk="desk", s=None, until_t=None):
        """No sitting clip: faked as idle at the desk facing the room."""
        face = self.sim.default_face(self)
        if until_t is not None:
            yield from self.a_idle_until(until_t, face)
        else:
            yield from self.a_idle(s or 0.0, face)

    def a_await_staff(self, face=None, anim="look_around"):
        b = self.sim.board
        b.awaiting.add(self.id)
        self.sim.log_intent(self.id, ("await",))
        while b.engager_of(self.id) is None:
            self._push("idle", self._face_yaw(face), anim=anim)
            yield
        b.awaiting.discard(self.id)
        while b.engager_of(self.id) is not None:
            s = self.sim.agents[b.engager_of(self.id)]
            self._push("talk", _bearing(self.pos, s.pos))
            yield

    def a_approach(self, customer, after_t=0.0, face_wait=None):
        b = self.sim.board
        while self.f < round(after_t * self.fps) or customer not in b.awaiting and b.engager_of(customer) is None:
            self._push("idle", self._face_yaw(face_wait or self.sim.default_face(self)))
            yield
        c = self.sim.agents[customer]
        goal, path = self.sim.approach_point(self, c)
        yield from self.g_walk_path(path)
        yield from self.g_turn_to(c.pos, tol_deg=3.0)
        self.sim.engage(self.id, customer, self.f)

    def a_talk(self, customer, s):
        c = self.sim.agents[customer]
        for _ in range(int(round(s * self.fps))):
            self._push("talk", _bearing(self.pos, c.pos))
            yield
        self.sim.release(self.id, customer, self.f)

    def a_talk_until_relieved(self, customer, extra_s=1.0):
        b = self.sim.board
        c = self.sim.agents[customer]
        while b.engager.get(customer) == self.id:
            self._push("talk", _bearing(self.pos, c.pos))
            yield
        for _ in range(int(round(extra_s * self.fps))):
            self._push("talk", _bearing(self.pos, c.pos))
            yield
        self.sim.release(self.id, customer, self.f)

    def a_leave(self):
        e = self.sim.lay["entrance"]
        x = e["center_x_m"] + self.p.get("door_x_offset", 0.0)
        self.sim.log_intent(self.id, ("leave", self.f))
        yield from self.g_walk_to([x, DOOR_IN_Y])
        self.leaving = True
        yield from self.g_walk_path([list(self.pos), [x, OUTSIDE_Y]])
        self.rendered = False
        while True:
            self._push("idle", None)
            yield

    def a_follow(self, leader, offset_m):
        """Party member: mirror the leader's intents (walk targets + offset, waits, exit)."""
        b = self.sim.board
        seen = None
        while True:
            it = b.leader_intent.get(leader)
            if it is not None and it is not seen:
                seen = it
                if it[0] == "walk":
                    tgt = [it[1][0] + offset_m[0], it[1][1] + offset_m[1]]
                    if self.pos[1] < 0.0 and tgt[1] > 0.0:            # entering through the door
                        yield from self.g_walk_path([list(self.pos), tgt], v=it[2])
                        continue
                    # walk the leader's route (same corners) so the pair stays together
                    route = [list(self.pos)] + [list(w) for w in (it[4][1:-1] if len(it) > 4 else [])] + [tgt]
                    nav = self.sim.nav
                    if nav.is_free(tgt) and all(nav.visible(a, b) for a, b in zip(route[:-1], route[1:])):
                        yield from self.g_walk_path(route, v=it[2])
                    else:
                        yield from self.g_walk_to(tgt, v=it[2])
                    continue
                if it[0] == "await":
                    b.awaiting.add(self.id)
                if it[0] == "leave":
                    b.awaiting.discard(self.id)
                    yield from self.a_leave()
                    return
            eng = b.engager_of(self.id)
            if eng is not None:
                b.awaiting.discard(self.id)
                self._push("talk", _bearing(self.pos, self.sim.agents[eng].pos))
            else:
                L = self.sim.agents[leader]
                self._push("idle", L.yaw if L.rendered else None, anim="look_around")
            yield


class Sim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fps = cfg["fps_scene"]
        self.n = schema.n_scene_frames(cfg)
        self.lay = layout.build_layout(cfg)
        self.navc = cfg["navigation"]
        self.nav = navigation.NavGraph(self.lay, self.navc)
        self.board = Board(cfg.get("parties", []))
        rng = np.random.default_rng(cfg["seed"])
        self.agents = {}
        for p in sorted(cfg["persons"], key=lambda q: q["id"]):
            self.agents[p["id"]] = Agent(self, p, rng)
        for g in cfg.get("parties", []):     # party members walk at the leader's speed
            for c in g[1:]:
                self.agents[c].v = self.agents[g[0]].v
        self.script_cfg = cfg.get("script", {})

    # ------------------------------------------------------------------ helpers
    def resolve_point(self, ref):
        if isinstance(ref, (int, np.integer)) and not isinstance(ref, bool):
            return list(self.agents[int(ref)].pos)
        if isinstance(ref, str):
            if ref == "desk":
                return list(self.lay["desk"]["center_m"])
            if ref.startswith("car_"):
                car = next(c for c in self.lay["cars"] if f"car_{c['id']}" == ref)
                return list(car["center_m"])
            if ref == "entrance":
                return [self.lay["entrance"]["center_x_m"], 0.0]
            raise KeyError(ref)
        return [float(ref[0]), float(ref[1])]

    def default_face(self, agent):
        return agent.p.get("default_face", "entrance")

    def log_intent(self, pid, intent):
        self.board.leader_intent[pid] = intent

    def approach_point(self, staff, cust):
        sc = self.script_cfg
        r = sc.get("staff_stop_dist_m", 1.25)
        step = sc.get("staff_approach_angles_deg", 15.0)
        cam = sc.get("approach_cam_m", self.cfg["cameras"][0]["position_m"][:2])
        ray = np.asarray(cust.pos) - np.asarray(cam[:2])
        ray /= np.linalg.norm(ray)
        others = [a.pos for a in self.agents.values() if a.rendered and a.id not in (staff.id, cust.id)]
        # other party members: stay out of their greet radius until arriving at the leader
        mates = [self.agents[m].pos for m in self.board.members(cust.id) if m != cust.id]
        best = None
        for strict in (True, False):
            for k in range(int(round(360 / step))):
                a = math.radians(k * step)
                cand = [cust.pos[0] + r * math.cos(a), cust.pos[1] + r * math.sin(a)]
                if not self.nav.is_free(cand) or cand[1] < 0.3:
                    continue
                if any(math.dist(cand, o) < MIN_SEPARATION_PLAN for o in others):
                    continue
                if strict and any(math.dist(cand, m) < 1.6 for m in mates):
                    continue
                if strict:
                    # radial final leg: walk straight at the customer for the last APPROACH_MIN_FINAL_SEG m
                    pre = [cust.pos[0] + (r + APPROACH_MIN_FINAL_SEG) * math.cos(a),
                           cust.pos[1] + (r + APPROACH_MIN_FINAL_SEG) * math.sin(a)]
                    if not self.nav.is_free(pre) or not self.nav.visible(pre, cand) \
                            or any(math.dist(pre, o) < MIN_SEPARATION_PLAN for o in others) \
                            or any(math.dist(pre, m) < 2.0 for m in mates):
                        continue
                    path = self.nav.shortest_path(staff.pos, pre)
                    if path is None:
                        continue
                    path = [list(staff.pos)] + path[1:] + [cand]
                else:
                    path = self.nav.shortest_path(staff.pos, cand)
                    if path is None:
                        continue
                    path = [list(staff.pos)] + path[1:]
                turn = abs(_wrap(_bearing(path[-2], path[-1]) - _bearing(cand, cust.pos)))
                axis = np.array([math.cos(a), math.sin(a)])
                cost = navigation.path_length(path) + 4.0 * abs(float(np.dot(axis, ray))) + 2.0 * turn / 180.0
                if best is None or cost < best[0]:
                    best = (cost, cand, path)
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"no free approach point around customer {cust.id}")
        return best[1], best[2]

    def t_of(self, f):
        return round(f / self.fps, 6)

    def engage(self, staff, customer, f):
        b = self.board
        t = self.t_of(f)
        prev = b.engager_of(customer)
        if prev is None:
            prev = next((b.last_engager[m] for m in b.members(customer) if m in b.last_engager), None)
        for m in b.members(customer):
            if m not in b.greeted:
                b.events.append({"type": "greet", "staff": staff, "customer": m, "t": t})
                b.greeted.add(m)
            elif prev is not None and prev != staff:
                b.events.append({"type": "handoff", "customer": m, "from_staff": prev, "to_staff": staff, "t": t})
            b.events.append({"type": "engagement", "staff": staff, "customer": m, "t": t, "t_end": None})
        b.engager[customer] = staff
        b.last_engager[customer] = staff

    def release(self, staff, customer, f):
        b = self.board
        for m in b.members(customer):
            for e in b.events:
                if e["type"] == "engagement" and e["staff"] == staff and e["customer"] == m and e["t_end"] is None:
                    e["t_end"] = self.t_of(f)
        if b.engager.get(customer) == staff:
            b.engager[customer] = None

    def on_exit(self, pid, f):
        """Called with the first non-present frame index (the person crossed y < 0 leaving)."""
        a = self.agents[pid]
        if a.p["role"] == "customer" and pid not in self.board.greeted:
            self.board.events.append({"type": "walkout", "customer": pid, "t": self.t_of(f)})

    def run(self):
        gens = {pid: a.run() for pid, a in self.agents.items()}
        for f in range(self.n):
            for pid, a in self.agents.items():
                next(gens[pid])
                if len(a.frames) != f + 1:
                    raise AssertionError(f"agent {pid} produced {len(a.frames)} frames at step {f}")
        for e in self.board.events:
            if e["type"] == "engagement" and e["t_end"] is None:
                e["t_end"] = self.t_of(self.n)
        for g in self.cfg.get("parties", []):
            onset = max(next(i for i, s in enumerate(self.agents[c].frames) if s["present"]) for c in g)
            self.board.events.append({"type": "party", "customers": sorted(g), "t": self.t_of(onset)})
        return sorted(self.board.events, key=lambda e: (e["t"], e["type"]))


def build_timeline(cfg):
    cfg = schema.load_scene_dict(cfg)
    sim = Sim(cfg)
    events = sim.run()
    return {
        "scene_id": cfg["scene_id"], "scenario": cfg["scenario"], "seed": cfg["seed"],
        "fps": sim.fps, "n_frames": sim.n,
        "agents": {str(pid): {"role": a.p["role"], "frames": a.frames} for pid, a in sim.agents.items()},
        "events_scripted": events,
        "parties": cfg.get("parties", []),
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
