"""Event engine: turns floor positions and roles into coaching events (spec §4.5).

The engine is a pure state machine with no I/O and no clock. Call ``step(t, states)`` once per sampled frame,
with positions in metres on the floor plane and roles already decided, then call ``finish(t_end)`` once at the
end of the clip. ``run_events`` does both for a whole timeline.

A customer's live state (``status``) goes waiting -> greeted -> engaged <-> browsing: waiting until greeted,
greeted until the first engagement, then engaged while an engagement is active and browsing between them. Entering
and leaving are events. The engine emits enter, leave, greet, engagement_start, engagement, wait_alert, walkout,
party and handoff.

Each step runs in a fixed order, and the rules depend on it:
  1. observe  - update presence and start a visit for every customer who isn't in one (or who has been
                unseen for leave_after, which ends the old visit)
  2. leave    - customers absent long enough leave (a walkout if they were never greeted)
  3. pairs    - update the greet and engage streaks of every staff-customer pair, then confirm greets and
                engagements
  4. free     - record which staff are free, for walkouts and wait alerts
  5. parties  - count party-window samples and confirm parties
  6. wait     - at most one wait_alert per visit, for present customers who aren't being greeted
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .config import Thresholds
from .geometry import distance_m, faces
from .types import Event

EPS = 1e-6  # time tolerance: sample times such as 20.2 are not exact in binary floating point

__all__ = ["EventEngine", "PersonState", "run_events"]


@dataclass
class PersonState:
    """One person at one step: floor position in metres, role, and optional heading and frame-edge flag."""

    pid: int
    xz: tuple[float, float]
    role: str = "customer"  # "staff" | "customer" | "unsure" (unsure is treated as customer)
    facing: tuple[float, float] | None = None
    at_edge: bool = False

    @property
    def is_staff(self) -> bool:
        return self.role == "staff"


# --------------------------------------------------------------------------------------------- building blocks

@dataclass(frozen=True)
class _Run:
    """A streak that has closed: first and last True sample, and whether its rule had fired."""

    start: float
    end: float
    reached: bool


class _Streak:
    """Runs of True samples of one pair condition, bridging gaps of up to ``gap_tol`` seconds.

    A streak starts at the first True sample. ``last_true`` follows each True sample, and the streak closes at
    the first sample more than ``gap_tol`` after ``last_true``. ``reached`` means the rule that uses this streak
    has fired (a greet was confirmed or an engagement became active), so it fires only once per streak.
    """

    def __init__(self, gap_tol: float):
        self.gap_tol = gap_tol
        self.start: float | None = None
        self.last_true: float | None = None
        self.reached = False

    @property
    def is_open(self) -> bool:
        return self.start is not None

    def lasted(self, seconds: float) -> bool:
        return self.is_open and self.last_true - self.start >= seconds - EPS

    def update(self, t: float, cond: bool) -> _Run | None:
        """Add the sample at ``t``. Return the run that this sample closed, if any."""
        closed = None
        if self.is_open and t - self.last_true > self.gap_tol + EPS:
            closed = self.close()
        if cond:
            if not self.is_open:
                self.start = t
            self.last_true = t
        return closed

    def close(self) -> _Run | None:
        if not self.is_open:
            return None
        run = _Run(self.start, self.last_true, self.reached)
        self.start = self.last_true = None
        self.reached = False
        return run


@dataclass
class _Pair:
    """Proximity history of one staff-customer pair."""

    staff: int
    customer: int
    greet: _Streak  # within greet_dist, and facing is OK
    engage: _Streak  # within engage_dist

    @property
    def idle(self) -> bool:
        return not (self.greet.is_open or self.engage.is_open)


@dataclass
class _Visit:
    """One customer's stay, from first seen to leaving. A pid seen again after leaving starts a new visit."""

    vid: int
    pid: int
    enter_t: float
    entry_known: bool  # False if first seen within leave_after of the clip start ("already present")
    last_seen: float
    at_edge: bool = False  # the last box touched a frame edge
    free_at_last_seen: list[int] = field(default_factory=list)
    greeted_by: int | None = None
    greet_t: float | None = None
    engaged_by: list[int] = field(default_factory=list)  # staff for each engagement, in the order they began
    wait_alerted: bool = False

    @property
    def greeted(self) -> bool:
        return self.greeted_by is not None


@dataclass
class _PartyCandidate:
    """Two customers who arrived close together, watched over the party window."""

    a: _Visit
    b: _Visit
    window_start: float
    window_end: float
    both: int = 0  # samples where both are present
    close: int = 0  # the subset of those within party_dist

    def involves(self, visit: _Visit) -> bool:
        return visit is self.a or visit is self.b

    @property
    def confirmed(self) -> bool:
        return self.both >= 3 and self.close >= 0.5 * self.both


class _UnionFind:
    """Groups visit ids into parties. Joining two parties merges them."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self._parent[self.find(a)] = self.find(b)

    def group(self, x: int) -> list[int]:
        if x not in self._parent:  # never joined a party
            return [x]
        root = self.find(x)
        return [y for y in list(self._parent) if self.find(y) == root]


# ------------------------------------------------------------------------------------------------------ engine

class EventEngine:
    """Applies the §4.5 rules step by step. See the module docstring for the order within a step."""

    def __init__(self, th: Thresholds, t0: float = 0.0):
        self.th = th
        self.t0 = t0
        self._present: dict[int, PersonState] = {}  # everyone in the last step
        self._staff: set[int] = set()  # pids last seen as staff
        self._visits: dict[int, _Visit] = {}  # customers in a visit now, keyed by pid
        self._all_visits: dict[int, _Visit] = {}  # every visit, keyed by vid (party members may have left)
        self._pairs: dict[tuple[int, int], _Pair] = {}  # keyed by (staff, customer)
        self._candidates: list[_PartyCandidate] = []
        self._parties = _UnionFind()
        self._next_vid = 0
        self._out: list[Event] = []

    # ---- public API

    def step(self, t: float, states: Sequence[PersonState]) -> list[Event]:
        """Advance to time ``t`` with everyone visible now. Return the events this step produced."""
        self._present = {s.pid: s for s in states}
        for s in self._present.values():
            if s.is_staff:
                self._observe_staff(s)
            else:
                self._observe_customer(t, s)
        self._process_leaves(t)
        self._update_pairs(t)
        free = self._free_staff()
        for pid in self._present_customers():
            self._visits[pid].free_at_last_seen = free
        self._update_party_candidates(t)
        self._check_wait_alerts(t, free)
        return self._flush()

    def finish(self, t_end: float) -> list[Event]:
        """End the clip: apply the leave rule at ``t_end``, close open engagements and decide pending parties.

        Customers still present get no leave or walkout, because they were still in the showroom.
        """
        self._process_leaves(t_end)
        self._end_pairs(lambda p: True)
        for cand in list(self._candidates):  # the clip ending also ends their party window
            self._resolve_party(cand, t_end)
        return self._flush()

    def status(self, t: float) -> dict[int, dict]:
        """Live state of everyone present in the last step, keyed by pid."""
        busy = self._busy_staff()
        engaged = {p.customer for p in self._pairs.values() if p.engage.reached}
        out: dict[int, dict] = {}
        for pid, s in sorted(self._present.items()):
            if s.is_staff:
                out[pid] = {"role": "staff", "state": "busy" if pid in busy else "free"}
                continue
            v = self._visits[pid]
            state = self._customer_state(v, engaged=pid in engaged)
            wait_s = t - v.enter_t if state == "waiting" and v.entry_known else None
            out[pid] = {"role": "customer", "state": state, "wait_s": wait_s,
                        "entry_known": v.entry_known, "greeted_by": v.greeted_by}
        return out

    @staticmethod
    def _customer_state(visit: _Visit, engaged: bool) -> str:
        if engaged:
            return "engaged"
        if visit.engaged_by:
            return "browsing"  # between engagements
        return "greeted" if visit.greeted else "waiting"

    def active_pairs(self) -> list[tuple[int, int, str]]:
        """(staff, customer, "engaged" | "close") for pairs with an open engage streak and both present."""
        return sorted(
            (p.staff, p.customer, "engaged" if p.engage.reached else "close")
            for p in self._pairs.values()
            if p.engage.is_open and p.staff in self._present and p.customer in self._present
        )

    # ---- 1. observe

    def _observe_staff(self, s: PersonState) -> None:
        visit = self._visits.pop(s.pid, None)
        if visit is not None:  # relabelled from customer: no leave or walkout, but engagements still close
            self._end_pairs(lambda p: p.customer == s.pid)
            self._candidates = [c for c in self._candidates if not c.involves(visit)]
        self._staff.add(s.pid)

    def _observe_customer(self, t: float, s: PersonState) -> None:
        if s.pid in self._staff:  # relabelled from staff: their pairs end, and a fresh visit starts below
            self._staff.discard(s.pid)
            self._end_pairs(lambda p: p.staff == s.pid)
        visit = self._visits.get(s.pid)
        # Unseen for leave_after with no step in between (a live stall): that was a leave, so end the old visit.
        # The faster frame-edge grace is not used here, or someone standing at the edge would be split at 1 fps.
        if visit is not None and t - visit.last_seen >= self.th.leave_after - EPS:
            self._end_visit(visit, t)
            visit = None
        visit = visit or self._start_visit(t, s.pid)
        visit.last_seen = t
        visit.at_edge = bool(s.at_edge)

    def _start_visit(self, t: float, pid: int) -> _Visit:
        visit = _Visit(vid=self._next_vid, pid=pid, enter_t=t, last_seen=t,
                       entry_known=bool(t - self.t0 >= self.th.leave_after - EPS))
        self._next_vid += 1
        self._visits[pid] = visit
        self._all_visits[visit.vid] = visit
        self._emit("enter", t, customer=pid, data={"entry_known": visit.entry_known})
        self._add_party_candidates(visit)
        return visit

    def _present_customers(self) -> list[int]:
        return [pid for pid, s in self._present.items() if not s.is_staff]

    # ---- 2. leave

    def _process_leaves(self, t: float) -> None:
        for visit in list(self._visits.values()):
            if visit.pid in self._present:
                continue
            if t - visit.last_seen >= self._leave_grace(visit) - EPS:
                self._end_visit(visit, t)

    def _leave_grace(self, visit: _Visit) -> float:
        """Seconds unseen before a customer has left: shorter if their last box touched a frame edge."""
        return min(self.th.edge_leave, self.th.leave_after) if visit.at_edge else self.th.leave_after

    def _end_visit(self, visit: _Visit, t: float) -> None:
        # Decide the party first: a companion's greeting covers this customer too, so it is not a walkout.
        for cand in [c for c in self._candidates if c.involves(visit)]:
            self._resolve_party(cand, t)
        del self._visits[visit.pid]
        self._emit("leave", visit.last_seen, customer=visit.pid,
                   data={"greeted": visit.greeted, "entry_known": visit.entry_known, "detected_at": t})
        if not visit.greeted and visit.entry_known:
            self._emit("walkout", visit.last_seen, customer=visit.pid,
                       data={"free_staff": list(visit.free_at_last_seen)})
        self._end_pairs(lambda p: p.customer == visit.pid)

    # ---- 3. pairs: greet, engagement, handoff

    def _update_pairs(self, t: float) -> None:
        conditions = self._pair_conditions()
        for key in sorted(set(self._pairs) | set(conditions)):
            greet_ok, engage_ok = conditions.get(key, (False, False))  # absent pairs count as False
            pair = self._pairs.get(key) or _Pair(*key, _Streak(self.th.gap_tol), _Streak(self.th.gap_tol))
            pair.greet.update(t, greet_ok)
            run = pair.engage.update(t, engage_ok)
            if run is not None and run.reached:
                self._emit_engagement(pair, run)
            if pair.idle:
                self._pairs.pop(key, None)
            else:
                self._pairs[key] = pair
        self._confirm_greets(t)
        self._confirm_engagements(t)

    def _pair_conditions(self) -> dict[tuple[int, int], tuple[bool, bool]]:
        """(greet condition, engage condition) for every staff-customer pair present now."""
        staff = [s for s in self._present.values() if s.is_staff]
        customers = [s for s in self._present.values() if not s.is_staff]
        out = {}
        for s in staff:
            for c in customers:
                d = distance_m(s.xz, c.xz)
                out[(s.pid, c.pid)] = (d <= self.th.greet_dist and self._facing_ok(s, c), d <= self.th.engage_dist)
        return out

    def _facing_ok(self, s: PersonState, c: PersonState) -> bool:
        """Fails only when both are known to face away. One facing the other, or an unknown heading, is OK."""
        max_deg = self.th.facing_max_deg
        return not (faces(s.xz, s.facing, c.xz, max_deg) is False and faces(c.xz, c.facing, s.xz, max_deg) is False)

    def _confirm_greets(self, t: float) -> None:
        ready = [p for p in self._pairs.values() if p.greet.lasted(self.th.greet_time)]
        for pair in sorted(ready, key=lambda p: (p.greet.start, p.staff)):  # earliest approach wins ties
            visit = self._visits[pair.customer]
            if not visit.greeted:
                self._greet(visit, pair.staff, pair.greet.start, t, via="direct")

    def _confirm_engagements(self, t: float) -> None:
        ready = [p for p in self._pairs.values() if not p.engage.reached and p.engage.lasted(self.th.engage_time)]
        for pair in sorted(ready, key=lambda p: (p.engage.start, p.staff)):
            self._start_engagement(pair, t)

    def _start_engagement(self, pair: _Pair, t: float) -> None:
        streak, visit = pair.engage, self._visits[pair.customer]
        streak.reached = True
        self._emit("engagement_start", streak.start, customer=pair.customer, staff=pair.staff,
                   data={"confirmed_at": t})
        if not visit.greeted:
            self._greet(visit, pair.staff, streak.start, t, via="engagement")
        if visit.engaged_by and visit.engaged_by[-1] != pair.staff:
            self._emit("handoff", streak.start, customer=pair.customer, staff=pair.staff,
                       data={"from_staff": visit.engaged_by[-1], "to_staff": pair.staff})
        visit.engaged_by.append(pair.staff)

    def _close_engagement(self, pair: _Pair) -> None:
        run = pair.engage.close()
        if run is not None and run.reached:
            self._emit_engagement(pair, run)

    def _emit_engagement(self, pair: _Pair, run: _Run) -> None:
        self._emit("engagement", run.start, customer=pair.customer, staff=pair.staff, t_end=run.end)

    def _greet(self, visit: _Visit, staff: int, t_greet: float, now: float, via: str) -> None:
        visit.greeted_by, visit.greet_t = staff, t_greet
        time_to_greet = t_greet - visit.enter_t if visit.entry_known else None
        self._emit("greet", t_greet, customer=visit.pid, staff=staff,
                   data={"time_to_greet": time_to_greet, "confirmed_at": now, "via": via})
        if via != "party":
            self._greet_party_mates(visit, now)

    def _end_pairs(self, match) -> None:
        """Remove the matching pairs, emitting ``engagement`` for any engagement that was active."""
        for key in sorted(k for k, p in self._pairs.items() if match(p)):
            self._close_engagement(self._pairs.pop(key))

    # ---- 4. busy / free

    def _busy_staff(self) -> set[int]:
        """Staff with an active engagement, or an engage streak that has lasted greet_time."""
        return {p.staff for p in self._pairs.values() if p.engage.reached or p.engage.lasted(self.th.greet_time)}

    def _free_staff(self) -> list[int]:
        busy = self._busy_staff()
        return sorted(pid for pid, s in self._present.items() if s.is_staff and pid not in busy)

    # ---- 5. parties

    def _add_party_candidates(self, visit: _Visit) -> None:
        for other in self._visits.values():
            if other is not visit and abs(other.enter_t - visit.enter_t) <= self.th.party_arrive + EPS:
                start = max(other.enter_t, visit.enter_t)
                self._candidates.append(_PartyCandidate(other, visit, start, start + self.th.party_window))

    def _update_party_candidates(self, t: float) -> None:
        for cand in list(self._candidates):
            if cand.window_start - EPS <= t <= cand.window_end + EPS:
                self._count_party_sample(cand)
            if t >= cand.window_end - EPS:
                self._resolve_party(cand, t)

    def _count_party_sample(self, cand: _PartyCandidate) -> None:
        a, b = self._present.get(cand.a.pid), self._present.get(cand.b.pid)
        if a is None or b is None:
            return
        cand.both += 1
        if distance_m(a.xz, b.xz) <= self.th.party_dist:
            cand.close += 1

    def _resolve_party(self, cand: _PartyCandidate, t: float) -> None:
        self._candidates.remove(cand)
        if not cand.confirmed or self._parties.find(cand.a.vid) == self._parties.find(cand.b.vid):
            return  # not a party, or already in the same one (so nothing changes)
        self._parties.union(cand.a.vid, cand.b.vid)
        members = self._party_of(cand.a)
        self._emit("party", cand.window_start,
                   data={"members": sorted({m.pid for m in members}), "confirmed_at": t})
        greeted = [m for m in members if m.greeted]
        if greeted:
            self._greet_party_mates(min(greeted, key=lambda m: m.greet_t), t)

    def _party_of(self, visit: _Visit) -> list[_Visit]:
        return [self._all_visits[vid] for vid in self._parties.group(visit.vid)]

    def _greet_party_mates(self, greeted: _Visit, now: float) -> None:
        """Greeting one member greets every ungreeted member still in its visit, with the same staff and time.

        The time is clamped to the mate's own visit, so a greet never comes before their enter or after they were
        last seen. A mate who arrived after the greeting counts as greeted on arrival. A mate out of view at the
        greeting, often hidden behind the companion, counts as greeted at their last sighting.
        """
        for mate in self._party_of(greeted):
            if mate is not greeted and not mate.greeted and self._visits.get(mate.pid) is mate:
                t_greet = min(max(greeted.greet_t, mate.enter_t), mate.last_seen)
                self._greet(mate, greeted.greeted_by, t_greet, now, via="party")

    # ---- 6. wait alerts

    def _check_wait_alerts(self, t: float, free: list[int]) -> None:
        """Alert once per visit for a present, ungreeted customer whose known wait has passed wait_alert.

        The check is held while a greet streak for the customer is open. Greets are backdated to the streak start,
        so alerting now could contradict a greet that is about to be confirmed. If the streak closes without a
        greet, the alert fires at the next step.
        """
        being_greeted = {p.customer for p in self._pairs.values() if p.greet.is_open}
        for pid in self._present_customers():
            visit = self._visits[pid]
            if (visit.entry_known and not visit.greeted and not visit.wait_alerted and pid not in being_greeted
                    and t - visit.enter_t > self.th.wait_alert + EPS):
                visit.wait_alerted = True
                self._emit("wait_alert", t, customer=visit.pid,
                           data={"wait_s": t - visit.enter_t, "free_staff": list(free)})

    # ---- output

    def _emit(self, kind: str, t: float, customer: int | None = None, staff: int | None = None,
              t_end: float | None = None, data: dict | None = None) -> None:
        self._out.append(Event(kind, t, customer=customer, staff=staff, t_end=t_end, data=data or {}))

    def _flush(self) -> list[Event]:
        out, self._out = self._out, []
        return out


def run_events(timeline: Iterable[tuple[float, Sequence[PersonState]]], th: Thresholds,
               t_end: float | None = None, t0: float | None = None) -> list[Event]:
    """Run the engine over a whole timeline of ``(t, states)``. Events are sorted by (t, type), stably.

    ``t0`` defaults to the first timeline time and ``t_end`` to the last one.
    """
    engine: EventEngine | None = None
    events: list[Event] = []
    last_t = None
    for t, states in timeline:
        if engine is None:
            engine = EventEngine(th, t if t0 is None else t0)
        events += engine.step(t, states)
        last_t = t
    if engine is None:
        return []
    events += engine.finish(last_t if t_end is None else t_end)
    events.sort(key=lambda e: (e.t, e.type))
    return events
