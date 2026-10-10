"""Per-salesperson and store scorecards from an event list (spec section 4.7).

Pure functions: no I/O, no detection. Events may arrive in any order; they are sorted by
time before visits are built. A customer can have several visits (re-entry after leaving);
every greet and leave is matched to the latest `enter` of that customer at or before it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from .types import Event, Scorecard, StaffCard, StoreCard


@dataclass
class _Visit:
    customer: int
    t_enter: float
    entry_known: bool
    t_greet: float | None = None  # first greet of this visit
    time_to_greet: float | None = None  # that greet's time_to_greet (may be None)
    t_leave: float | None = None  # first leave of this visit


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _visits(events: list[Event], exclude: frozenset[int] = frozenset()) -> list[_Visit]:
    """One visit per `enter` event; greets/leaves attach to the latest enter at or before them.

    Customers in `exclude` (people finally marked staff) get no visits: final roles win.
    """
    # Enters sort before other events at the same instant so "at or before" holds on ties.
    ordered = sorted(events, key=lambda e: (e.t, e.type != "enter"))
    visits: list[_Visit] = []
    latest: dict[int, _Visit] = {}
    for e in ordered:
        if e.customer is None or e.customer in exclude:
            continue
        if e.type == "enter":
            v = _Visit(e.customer, e.t, bool(e.data.get("entry_known", False)))
            visits.append(v)
            latest[e.customer] = v
            continue
        v = latest.get(e.customer)
        if v is None:
            continue  # greet/leave with no earlier enter: not part of any visit
        if e.type == "greet" and v.t_greet is None:
            v.t_greet = e.t
            v.time_to_greet = e.data.get("time_to_greet")
        elif e.type == "leave" and v.t_leave is None:
            v.t_leave = e.t
    return visits


def _wait_s(v: _Visit, t_end: float) -> float:
    """Time from entry to first greet, else to leaving, else to the end of the clip."""
    if v.t_greet is not None:
        stop = v.t_greet
    elif v.t_leave is not None:
        stop = v.t_leave
    else:
        stop = t_end
    return max(0.0, stop - v.t_enter)


def unattended_wait(
    events: list[Event], t_end: float, exclude_customers: Iterable[int] = ()
) -> float:
    """Total seconds customers with a known entry waited before a greet, leave or clip end.

    `exclude_customers` drops the visits of people who are staff in the final roles.
    """
    visits = _visits(events, frozenset(exclude_customers))
    return float(sum(_wait_s(v, t_end) for v in visits if v.entry_known))


def _staff_card(sid: int, name: str | None, events: list[Event], t_end: float) -> StaffCard:
    greets = [e for e in events if e.type == "greet" and e.staff == sid]
    engagements = [e for e in events if e.type == "engagement" and e.staff == sid]
    durations = [
        max(0.0, (e.t_end if e.t_end is not None else t_end) - e.t) for e in engagements
    ]
    return StaffCard(
        staff=sid,
        name=name,
        greets=len(greets),
        median_time_to_greet=_median(
            [e.data["time_to_greet"] for e in greets if e.data.get("time_to_greet") is not None]
        ),
        customers_engaged=len({e.customer for e in engagements if e.customer is not None}),
        engaged_total_s=float(sum(durations)),
        engaged_median_s=_median(durations),
        missed_while_free=sum(
            1 for e in events if e.type == "wait_alert" and sid in (e.data.get("free_staff") or [])
        ),
        walkouts_while_free=sum(
            1 for e in events if e.type == "walkout" and sid in (e.data.get("free_staff") or [])
        ),
    )


def scorecard(
    events: list[Event],
    staff_ids: Iterable[int],
    customer_ids: Iterable[int],
    t_end: float,
    names: dict[int, str] | None = None,
) -> Scorecard:
    """Build per-salesperson cards (staff id ascending) and the store card.

    `customer_ids` is accepted for API symmetry and future use; visits come from `enter` events,
    except that visits of anyone in `staff_ids` are skipped.
    """
    names = names or {}
    staff_set = frozenset(staff_ids)
    staff = [_staff_card(sid, names.get(sid), events, t_end) for sid in sorted(staff_set)]

    # A person relabelled customer -> staff can leave an orphan `enter`; final roles win.
    visits = _visits(events, staff_set)
    greeted = [v for v in visits if v.t_greet is not None]
    known = [v for v in visits if v.entry_known]
    store = StoreCard(
        customers=len(visits),
        customers_entry_known=len(known),
        greeted=len(greeted),
        greet_rate=len(greeted) / len(visits) if visits else None,
        median_time_to_greet=_median(
            [v.time_to_greet for v in greeted if v.time_to_greet is not None]
        ),
        walkouts=sum(1 for e in events if e.type == "walkout"),
        unattended_wait_s=float(sum(_wait_s(v, t_end) for v in known)),
        wait_alerts=sum(1 for e in events if e.type == "wait_alert"),
        engagements=sum(1 for e in events if e.type == "engagement"),
    )
    return Scorecard(staff=staff, store=store)
