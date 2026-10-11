"""The live session engine: one camera, frame by frame, to overlays, events and a running scorecard (spec §3 live flow).

`LiveSession` is the online half of the one engine that serves both modes. Each `process(frame, t)` runs the perception
stack on that single frame (detect, track, pose), gives every track a person id (`identity.LiveIdentity`: a new track
may take over someone who was lost a moment ago), turns positions into floor metres (`geometry`), steps the event
engine, and returns the overlay the browser draws (api-contract.md, "Live WebSocket"). Nothing here keeps or returns
pixels: the overlay carries a `face` rectangle per listed person, plus `faces` for every other detection, and the
client blurs them all on its own canvas.

Confirmation. A new track may be detector noise, so a person feeds the event engine (and the replay history) only once
confirmed: one of their tracks has `MIN_TRACK_OBS` observations spanning `MIN_TRACK_S` seconds, the same bar as the
offline noise filter. The flag is per person and sticky, so a re-attached person stays confirmed. Until then the person
is still listed in `people` (so the client can draw, blur and click them) with `state: null`; a click sets the override
that applies once they are confirmed.

Roles. Everyone starts as a customer with `suggested = "unsure"`. Once the session is `suggest_after` seconds old, and
then every `suggest_every` seconds, the staff suggestion (`roles.suggest_roles`) is recomputed from the last five
minutes of trajectories and outfits; anyone without a user override follows it (`unsure` counts as customer). A click
(`toggle`) is an override and always wins.

Cost control, because live must keep up on a CPU. A new track is embedded at once (so it can re-attach to someone
just lost) but known people are refreshed at most `max_embeds_per_frame` crops per frame, oldest first, and pose runs
for a track on its first frame and then every `pose_every`-th frame, with the last keypoints riding along with the box
in between (the same scheme as the offline pass). `fps` is the rate the engine processes frames, not the rate they
arrive.

A `LiveSession` is not thread-safe: call it from one thread at a time (the WebSocket handler does).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import MIN_TRACK_OBS, MIN_TRACK_S, SAMPLE_FPS, CameraConfig, RoleWeights, get_preset
from .events import EventEngine, PersonState
from .geometry import Camera, at_frame_edge, facing_vector, position_m
from .identity import LiveIdentity
from .metrics import scorecard as build_scorecard
from .perception.embed import hsv_hist, person_crop, torso_crop, usable_crop
from .perception.pipeline import HSV_WEIGHT, carry_keypoints
from .perception.stack import PerceptionStack
from .perception.tracking import PERSON
from .render import face_region
from .roles import interaction_features, outfit_peers, presence, suggest_roles
from .types import Event, RoleSuggestion, to_jsonable

__all__ = ["LiveSession"]

WINDOW_S = 300.0              # trajectories, frame times and outfit statistics are kept for this long
HISTORY_FRAMES = 9000         # per-frame states kept to re-score on a preset change (30 min at 5 fps)
MAX_ALERTS = 10               # alerts shown
MAX_ALERT_RECORDS = 50        # alert records kept, so that hiding some (their customer became staff) leaves enough to show
FPS_ALPHA = 0.2               # smoothing of the processed-fps moving average
MIN_REFRESH_ALPHA = 0.1       # an identity descriptor never adapts slower than this (1/n for the first few)
UNSURE_CONFIDENCE = 0.5       # P(staff) shown before the first suggestion: no evidence either way
_EPS = 1e-6
_UNIT_MIN = 1e-12


# ------------------------------------------------------------------------------------------------ small helpers

def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, np.float64).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > _UNIT_MIN and math.isfinite(n) else None


@dataclass
class _Person:
    """What the session remembers about one person for the staff suggestion (all bounded by `WINDOW_S`)."""

    traj: deque = field(default_factory=deque)        # (t, x_m, z_m)
    last_t: float = 0.0
    hsv_sum: np.ndarray | None = None                 # torso colour histograms, summed
    hsv_n: int = 0
    dino_sum: np.ndarray | None = None                # appearance embeddings (when the stack has an embedder), summed
    dino_n: int = 0
    emb_n: int = 0                                    # how many descriptors went into the identity's descriptor
    last_embed: int = -(10 ** 9)                      # frame counter when this person was last embedded
    confirmed: bool = False                           # sticky: some track of theirs passed the noise bar

    def outfit(self) -> np.ndarray | None:
        """Torso colour mean, plus half-weight embedding mean when there is one (as in the offline pass)."""
        if self.hsv_n == 0 or self.hsv_sum is None:
            return None
        hsv = self.hsv_sum / self.hsv_n
        if self.dino_sum is not None and self.dino_n:
            return np.concatenate([hsv, HSV_WEIGHT * (self.dino_sum / self.dino_n)])
        return hsv


# ----------------------------------------------------------------------------------------------------- the session

class LiveSession:
    def __init__(self, stack: PerceptionStack, preset: str = "short", camera: CameraConfig = CameraConfig(),
                 suggest_after: float = 30.0, suggest_every: float = 5.0, embed_every: int = 5, *,
                 pose_every: int = 2, max_embeds_per_frame: int = 1, weights: RoleWeights = RoleWeights()):
        """`preset` is "short" or "dealership" (ValueError otherwise). `suggest_after` / `suggest_every` are seconds;
        `embed_every` is the number of frames between refreshes of one person's appearance descriptor."""
        th = get_preset(preset)
        if pose_every < 1 or embed_every < 1 or max_embeds_per_frame < 0:
            raise ValueError("pose_every and embed_every must be >= 1 and max_embeds_per_frame >= 0")
        self.stack = stack
        self.preset = preset
        self.camera = camera
        self.suggest_after = float(suggest_after)
        self.suggest_every = float(suggest_every)
        self.embed_every = int(embed_every)
        self.pose_every = int(pose_every)
        self.max_embeds_per_frame = int(max_embeds_per_frame)
        self.weights = weights

        self._tracker = stack.new_tracker(SAMPLE_FPS)
        self._identity = LiveIdentity()
        self._engine = EventEngine(th, t0=0.0)
        self._cameras: dict[tuple[int, int], Camera] = {}

        self._start: float | None = None              # origin of the default clock (perf_counter)
        self._last_t: float | None = None
        self._n = 0                                    # frames processed
        self._fps: float | None = None
        self._track_pid: dict[int, int] = {}           # track id -> person id
        self._pose_state: dict[int, tuple[int, np.ndarray, tuple]] = {}   # track -> (frame no., keypoints, box)
        self._pids: set[int] = set()                   # every person id handed out
        self._people: dict[int, _Person] = {}
        self._frame_times: deque[float] = deque()
        self._overrides: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self._suggestions: dict[int, RoleSuggestion] = {}
        self._last_suggest: float | None = None
        self._events: list[Event] = []
        self._alerts: list[Event] = []                 # wait_alert / walkout records, newest first (rendered on demand)
        self._tracks: dict[int, list[float]] = {}      # unconfirmed people's tracks: id -> [first t, observations, last t]
        self._history: deque[tuple[float, list[tuple]]] = deque(maxlen=HISTORY_FRAMES)
        self._t0 = 0.0                                 # the engine's "clip start": session start, or the oldest kept frame
        self._store_key: tuple | None = None           # (events so far, staff ids) the cached store counters are for
        self._store: dict = {"customers": 0, "greeted": 0, "walkouts": 0}

    # ---- controls

    def toggle(self, pid: int) -> str:
        """Flip a person staff <-> customer as a user override; returns the new role. KeyError for an unknown pid."""
        self._require(pid)
        new = "customer" if self._role(pid) == "staff" else "staff"
        self._overrides[pid] = new
        return new

    def set_name(self, pid: int, name: str) -> None:
        """Name a person for labels and alerts (blank removes it). KeyError for an unknown pid."""
        self._require(pid)
        name = str(name or "").strip()
        if name:
            self._names[pid] = name
        else:
            self._names.pop(pid, None)

    def set_preset(self, name: str) -> None:
        """Switch thresholds: a new event engine replays the session so far (same people, roles and times), so the
        events, alerts and scorecard are what the new preset would have produced. ValueError for an unknown name,
        in which case nothing changes. At most the last `HISTORY_FRAMES` frames are replayed; people already there at
        the start of the replay count as "already present", as in any clip."""
        th = get_preset(name)
        engine = EventEngine(th, t0=self._t0)
        events: list[Event] = []
        for t, rows in self._history:
            events += engine.step(t, [PersonState(pid, xz, self._role(pid), facing, edge)
                                      for pid, xz, facing, edge in rows])
        self.preset = name
        self._engine = engine
        self._events = events
        self._alerts = []
        for e in events:
            self._note_alert(e)
        self._store_key = None

    def summary(self) -> dict:
        """`{"type": "summary", "scorecard": ..., "events": [...]}` for the session so far. Engagements still going
        are included as open events (`t_end` null), counted up to now."""
        t = self._last_t or 0.0
        events = sorted(self._current_events(self._events + self._open_engagements()), key=lambda e: (e.t, e.type))
        return {"type": "summary", "scorecard": to_jsonable(self._scorecard(events, t)), "events": to_jsonable(events)}

    # ---- one frame

    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> dict:
        """Run one BGR frame (already resized, at most 960 px wide) and return the overlay dict. `t` is the time in
        seconds since the session began; by default the session's own clock (starting at the first frame)."""
        started = time.perf_counter()
        t = self._clock(t)
        self._n += 1
        h, w = frame_bgr.shape[:2]
        cam = self._camera(w, h)

        persons = [d for d in self.stack.detect(frame_bgr) if d.cls == PERSON]
        tracked = self._tracker.update(persons)
        kpts = self._keypoints(frame_bgr, tracked)
        pids = self._identify(frame_bgr, t, tracked)
        self._refresh_descriptors(frame_bgr, tracked, pids, kpts)

        rows = []                                      # confirmed people only: what the engine and the replay see
        for j, (tid, det) in enumerate(tracked):
            person = self._people[pids[j]]
            person.last_t = t
            if not person.confirmed:
                self._observe_unconfirmed(tid, person, t)
            if person.confirmed:
                pos = position_m(det.bbox, kpts[j], cam)
                rows.append((pids[j], pos.xz, facing_vector(det.bbox, kpts[j], cam), at_frame_edge(det.bbox, w, h)))
                person.traj.append((t, pos.x, pos.z))
        self._frame_times.append(t)
        self._history.append((t, rows))
        if len(self._history) == self._history.maxlen:
            self._t0 = self._history[0][0]
        self._prune(t)
        self._maybe_suggest(t)

        new_events = self._engine.step(t, [PersonState(pid, xz, self._role(pid), facing, edge)
                                           for pid, xz, facing, edge in rows])
        self._events += new_events
        for e in new_events:
            self._note_alert(e)
        status = self._engine.status(t)

        people = []
        for j, (_, det) in enumerate(tracked):
            pid = pids[j]
            info = status.get(pid, {})
            role = self._role(pid)
            sug = self._suggestions.get(pid)
            wait = info.get("wait_s")
            people.append({
                "pid": pid, "bbox": [round(float(v), 1) for v in det.bbox],
                "face": [int(v) for v in face_region(det.bbox, kpts[j])],
                "role": role, "suggested": sug.role if sug else "unsure",
                "confidence": round(sug.confidence, 3) if sug else UNSURE_CONFIDENCE,
                "override": pid in self._overrides, "state": info.get("state"),
                "wait_s": None if wait is None else round(wait, 1), "label": self._label(pid, role, wait),
            })
        people.sort(key=lambda p: p["pid"])
        listed = {id(det) for _, det in tracked}
        faces = [[int(v) for v in face_region(d.bbox, None)] for d in persons if id(d) not in listed]
        overlay = {
            "type": "overlay", "t": round(t, 3), "w": w, "h": h, "fps": 0.0, "people": people, "faces": faces,
            "pairs": [list(p) for p in self._engine.active_pairs()], "events": new_events,
            "alerts": self._alert_lines(), "store": dict(self._store_counters(t)),
        }
        elapsed = max(time.perf_counter() - started, 1e-3)       # the whole frame's processing time, this one included
        self._fps = 1.0 / elapsed if self._fps is None else (1 - FPS_ALPHA) * self._fps + FPS_ALPHA / elapsed
        overlay["fps"] = round(self._fps, 1)
        return to_jsonable(overlay)

    def _observe_unconfirmed(self, tid: int, person: _Person, t: float) -> None:
        """Count one more observation of track `tid`; the person is confirmed when that track has `MIN_TRACK_OBS`
        observations spanning `MIN_TRACK_S` seconds. Judged per track, like the offline pass: several short blips of
        one look that keep re-attaching to the same person never add up to a real visit."""
        rec = self._tracks.setdefault(tid, [t, 0, t])
        rec[1] += 1
        rec[2] = t
        if rec[1] >= MIN_TRACK_OBS and t - rec[0] >= MIN_TRACK_S - _EPS:
            person.confirmed = True
            self._tracks.pop(tid, None)

    # ---- time and cameras

    def _clock(self, t: float | None) -> float:
        """Session-relative seconds: `t` if given, else a monotonic clock (started at the first frame, or continuing
        from the last given time). Never goes backwards: the event engine needs non-decreasing time. The clock is
        `perf_counter`, not `monotonic`, which ticks only every 15.6 ms on Windows with Python 3.12."""
        if t is None:
            now = time.perf_counter()
            if self._start is None:
                self._start = now - (self._last_t or 0.0)
            t = now - self._start
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be a finite number of seconds")
        t = max(t, 0.0 if self._last_t is None else self._last_t)
        self._last_t = t
        return t

    def _camera(self, w: int, h: int) -> Camera:
        cam = self._cameras.get((w, h))
        if cam is None:
            cam = self._cameras[(w, h)] = Camera.from_config(w, h, self.camera)
        return cam

    # ---- pose, identity, descriptors

    def _keypoints(self, frame: np.ndarray, tracked: list) -> np.ndarray:
        """(N, 17, 3) keypoints per tracked box: fresh on a track's first frame and every `pose_every` frames,
        carried along with the box in between. Zeros (confidence 0) when the stack has no pose model."""
        kpts = np.zeros((len(tracked), 17, 3), np.float32)
        if tracked and self.stack.pose is not None:
            due = []
            for j, (tid, det) in enumerate(tracked):
                last = self._pose_state.get(tid)
                if last is None or self._n - last[0] >= self.pose_every:
                    due.append(j)
                else:
                    kpts[j] = carry_keypoints(last[1], last[2], det.bbox)
            if due:
                boxes = np.asarray([tracked[j][1].bbox for j in due], np.float64)
                fresh = np.asarray(self.stack.pose(frame, boxes), np.float32).reshape(len(due), 17, 3)
                for n, j in enumerate(due):
                    tid, det = tracked[j]
                    kpts[j] = fresh[n]
                    x1, y1, x2, y2 = (float(v) for v in det.bbox)
                    if fresh[n, :, 2].max() > 0 and x2 > x1 and y2 > y1:
                        self._pose_state[tid] = (self._n, fresh[n].copy(), (x1, y1, x2, y2))
                    else:                                  # nothing came back (box too small): try again next frame
                        self._pose_state.pop(tid, None)
        in_view = {tid for tid, _ in tracked}
        self._pose_state = {tid: v for tid, v in self._pose_state.items()
                            if tid in in_view or self._n - v[0] < self.pose_every}
        return kpts

    def _identify(self, frame: np.ndarray, t: float, tracked: list) -> list[int]:
        """Person id per tracked box, all different. Call order matters to `LiveIdentity`: end the step, then resolve
        each track. New tracks are embedded first (all in one call) so they can re-attach to someone just lost. A
        tracker can revive an old track whose person was re-attached to a newer one meanwhile: the identity layer
        then gives the revived track a new person, and a second guard here splits any id that still repeats."""
        self._identity.end_step(t, {tid for tid, _ in tracked})
        new = [j for j, (tid, _) in enumerate(tracked) if tid not in self._track_pid]
        described = dict(zip(new, self._describe(frame, [tracked[j][1].bbox for j in new])))
        pids: list[int] = []
        for j, (tid, _) in enumerate(tracked):
            emb, dino = described.get(j, (None, None))
            pid = self._identity.resolve(tid, t, emb)
            if pid in pids:   # one id must never label two boxes (LiveIdentity already prevents it; this keeps it so)
                pid = self._identity.assign_new(tid, t)
            self._track_pid[tid] = pid
            self._pids.add(pid)
            person = self._people.setdefault(pid, _Person(last_t=t))
            if j in described:
                self._learn(pid, person, emb, dino)
            pids.append(pid)
        return pids

    def _refresh_descriptors(self, frame: np.ndarray, tracked: list, pids: list[int], kpts: np.ndarray) -> None:
        """Each frame: torso colour for everyone (cheap), and the oldest-refreshed few appearance embeddings."""
        for j, (_, det) in enumerate(tracked):
            torso = torso_crop(frame, det.bbox, kpts[j])
            hist = hsv_hist(torso)
            if usable_crop(torso) and float(hist.any()):
                person = self._people[pids[j]]
                person.hsv_sum = hist.astype(np.float64) if person.hsv_sum is None else person.hsv_sum + hist
                person.hsv_n += 1
        due = sorted((self._people[pid].last_embed, pid, j) for j, pid in enumerate(pids)
                     if self._n - self._people[pid].last_embed >= self.embed_every)[:self.max_embeds_per_frame]
        if due:
            for (_, pid, _), (emb, dino) in zip(due, self._describe(frame, [tracked[j][1].bbox for _, _, j in due])):
                self._learn(pid, self._people[pid], emb, dino)

    def _describe(self, frame: np.ndarray, bboxes: list) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        """`(identity descriptor, embedding)` per box: the descriptor is unit(embedding + half-weight colour histogram)
        as in the offline pass, or the colour histogram alone when the stack has no embedder. (None, None) for a box
        that gives no usable crop. Appearance of the whole body, never a face match."""
        crops = [person_crop(frame, b) for b in bboxes]
        ok = [i for i, c in enumerate(crops) if usable_crop(c)]
        dino: dict[int, np.ndarray] = {}
        if ok and self.stack.embed is not None:
            out = np.asarray(self.stack.embed([crops[i] for i in ok]), np.float32)
            dino = {i: out[n] for n, i in enumerate(ok)}
        result: list[tuple[np.ndarray | None, np.ndarray | None]] = [(None, None)] * len(crops)
        for i in ok:
            hist = hsv_hist(crops[i]).astype(np.float64)
            d = dino.get(i)
            if d is not None and float(np.linalg.norm(d)) <= _UNIT_MIN:
                d = None                                   # the embedder gave nothing for this crop
            vec = np.concatenate([d, HSV_WEIGHT * hist]) if d is not None else hist
            result[i] = (_unit(vec), d)
        return result

    def _learn(self, pid: int, person: _Person, emb: np.ndarray | None, dino: np.ndarray | None) -> None:
        """Blend one fresh descriptor into a person's running means and into the identity's stored descriptor."""
        person.last_embed = self._n
        if emb is None:
            return
        self._identity.update_emb(pid, emb, alpha=max(1.0 / (person.emb_n + 1), MIN_REFRESH_ALPHA))
        person.emb_n += 1
        if dino is not None:
            person.dino_sum = dino.astype(np.float64) if person.dino_sum is None else person.dino_sum + dino
            person.dino_n += 1

    # ---- window bookkeeping and staff suggestion

    def _prune(self, t: float) -> None:
        cutoff = t - WINDOW_S
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
        self._tracks = {tid: rec for tid, rec in self._tracks.items() if rec[2] >= cutoff}
        for pid in list(self._people):
            person = self._people[pid]
            while person.traj and person.traj[0][0] < cutoff:
                person.traj.popleft()
            if not person.traj and person.last_t < cutoff:
                del self._people[pid]                      # unseen for the whole window; ids, roles, names are kept

    def _maybe_suggest(self, t: float) -> None:
        """Recompute the staff suggestion from the last five minutes, from `suggest_after` on, every `suggest_every`."""
        if t < self.suggest_after - _EPS:
            return
        if self._last_suggest is not None and t - self._last_suggest < self.suggest_every - _EPS:
            return
        self._last_suggest = t
        people = {pid: p for pid, p in self._people.items() if p.traj}
        if not people:
            return
        w = self.weights
        traj = {pid: list(p.traj) for pid, p in people.items()}
        frame_times = list(self._frame_times)
        talk = interaction_features(traj, w.contact_dist, w.contact_min_s)
        seen = presence(traj, frame_times)
        descr = {pid: o for pid, p in people.items() if (o := p.outfit()) is not None}
        peers = outfit_peers(descr, w.outfit_sim_thr)
        features = {pid: {"outfit_peers": peers.get(pid, 0), "contacts": talk[pid]["contacts"],
                          "presence": seen[pid], "initiated": talk[pid]["initiated"]} for pid in people}
        window = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0
        self._suggestions.update({s.pid: s for s in suggest_roles(features, window, w)})

    # ---- roles, labels, alerts, scorecard

    def _require(self, pid: int) -> None:
        if pid not in self._pids:
            raise KeyError(pid)

    def _role(self, pid: int) -> str:
        """The user's override, else the suggestion, with "unsure" (and no suggestion yet) counted as a customer."""
        override = self._overrides.get(pid)
        if override:
            return override
        sug = self._suggestions.get(pid)
        return "staff" if sug is not None and sug.role == "staff" else "customer"

    def _label(self, pid: int, role: str, wait_s: float | None) -> str:
        if role == "staff":
            name = self._names.get(pid)
            return f"S{pid} {name}" if name else f"S{pid}"
        return f"C{pid}" if wait_s is None else f"C{pid} ⏱ {_mmss(wait_s)}"

    def _staff_name(self, pid: int) -> str:
        return self._names.get(pid) or f"S{pid}"

    def _note_alert(self, e: Event) -> None:
        """Keep wait alerts and walk-outs as records, newest first; the text is made when the overlay is built."""
        if e.type in ("wait_alert", "walkout"):
            self._alerts.insert(0, e)
            del self._alerts[MAX_ALERT_RECORDS:]

    def _alert_lines(self) -> list[str]:
        """Human lines for the newest `MAX_ALERTS` alerts whose customer is not (now) staff, with current names."""
        lines = []
        for e in self._current_events(self._alerts):
            free = e.data.get("free_staff") or []
            who = "free: " + ", ".join(self._staff_name(p) for p in free) if free else "no staff free"
            what = "long wait" if e.type == "wait_alert" else "walk-out"
            lines.append(f"{_mmss(e.t)} {what}: customer C{e.customer} ({who})")
            if len(lines) == MAX_ALERTS:
                break
        return lines

    def _current_events(self, events: list[Event]) -> list[Event]:
        """`events` without those about a customer who is now marked staff: a role change corrects the past too."""
        return [e for e in events if e.customer is None or self._role(e.customer) != "staff"]

    def _open_engagements(self) -> list[Event]:
        """Engagements started but not yet closed, as events with `t_end` None (the metrics count them to 'now')."""
        closed = {(e.customer, e.staff, e.t) for e in self._events if e.type == "engagement"}
        return [Event("engagement", e.t, customer=e.customer, staff=e.staff)
                for e in self._events if e.type == "engagement_start" and (e.customer, e.staff, e.t) not in closed]

    def _store_counters(self, t: float) -> dict:
        """customers / greeted / walkouts so far; re-scored only when an event arrives or a role changes, so the cost
        of a long session does not grow with every frame."""
        staff = tuple(sorted(pid for pid in self._pids if self._role(pid) == "staff"))
        key = (len(self._events), staff)
        if key != self._store_key:
            card = self._scorecard(self._current_events(self._events), t).store
            self._store = {"customers": card.customers, "greeted": card.greeted, "walkouts": card.walkouts}
            self._store_key = key
        return self._store

    def _scorecard(self, events: list[Event], t: float):
        staff = sorted(pid for pid in self._pids if self._role(pid) == "staff")
        customers = sorted(pid for pid in self._pids if self._role(pid) != "staff")
        return build_scorecard(events, staff, customers, t, self._names)
