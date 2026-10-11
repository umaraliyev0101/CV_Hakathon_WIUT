"""Per-salesperson and store scorecards from an event list (spec section 4.7).

Pure functions: no I/O, no detection. Events may arrive in any order; they are sorted by
time before visits are built. A customer can have several visits (re-entry after leaving);
every greet and leave is matched to the latest `enter` of that customer at or before it.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import statistics
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .types import Event, Scorecard, StaffCard, StoreCard

__all__ = ["MAX_SALES_ROWS", "SalesCsvError", "SalesRows", "parse_sales_csv", "sales_conversion", "scorecard",
           "unattended_wait", "unmatched_sales"]

MAX_SALES_ROWS = 10_000      # data rows in one sales CSV
MAX_SALESPERSON_CHARS = 80   # a longer "name" is not a name: the row is skipped
MAX_TIMESTAMP_CHARS = 40
_SALES_FORMAT = "salesperson,timestamp,sold"
_SOLD_TRUE = frozenset({"1", "true", "yes"})
_SOLD_FALSE = frozenset({"0", "false", "no"})
# Uzbek writes o' and g' with a modifier letter, a typographic quote, an acute or a grave; all count as one apostrophe.
_APOSTROPHES = str.maketrans({chr(c): "'" for c in (0x2BB, 0x2BC, 0x2018, 0x2019, 0xB4, 0x60)})


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


# ------------------------------------------------------------------------------ optional sales CSV (spec 4.7)

class SalesCsvError(ValueError):
    """The sales CSV cannot be used at all (no header, wrong columns, too many rows, not text)."""


class SalesRows(list):
    """The valid rows of a sales CSV, each ``{"salesperson": str, "timestamp": str, "sold": bool}``.

    It is a plain list of dicts; `skipped` is how many data rows were left out as invalid."""

    def __init__(self, rows: Iterable[dict] = (), skipped: int = 0):
        super().__init__(rows)
        self.skipped = int(skipped)


def _sold_flag(value) -> bool | None:
    """True/False for 1/0/true/false/yes/no (any case, spaces around), else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().casefold()
        if v in _SOLD_TRUE:
            return True
        if v in _SOLD_FALSE:
            return False
    return None


def _clean_name(value) -> str:
    """The salesperson as typed, with surrounding and repeated whitespace removed."""
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _match_key(value) -> str:
    """A name as compared: apostrophe variants unified, NFKC (composed forms, full-width letters), spacing collapsed,
    case folded. Only used for matching; names are reported as typed."""
    text = unicodedata.normalize("NFKC", _clean_name(value).translate(_APOSTROPHES)).translate(_APOSTROPHES)
    return _clean_name(text).casefold()


def parse_sales_csv(text: str) -> SalesRows:
    """Rows of a sales CSV: a header with `salesperson` and `sold` (and optionally `timestamp`), then one row per
    lead or sale. Header names are case-insensitive, extra columns are ignored, and comma, semicolon or tab
    separators are accepted (spreadsheets in some locales write semicolons). `sold` is 1/0/true/false/yes/no.

    A data row with no salesperson (or one longer than 80 characters), an unreadable `sold` or too few cells is
    skipped and counted in the result's `skipped`; blank lines are ignored. SalesCsvError when there is no usable
    header, more than 10,000 data rows, or the text is not a CSV at all.
    """
    text = text.lstrip("\ufeff")
    if "\x00" in text:
        raise SalesCsvError("this is not a text CSV file")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    if not first:
        raise SalesCsvError(f"the file is empty; the first line must be the header {_SALES_FORMAT}")
    delimiter = max(",;\t", key=first.count)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next((r for r in reader if any(c.strip() for c in r)), None)
        if header is None:                                  # only separators, e.g. a blank sheet saved as CSV
            raise SalesCsvError(f"no header row found; the first line must be the header {_SALES_FORMAT}")
        cols = {c.strip().casefold(): i for i, c in enumerate(header)}
        if "salesperson" not in cols or "sold" not in cols:
            raise SalesCsvError(f"the first line must be the header {_SALES_FORMAT} "
                                f"(columns salesperson and sold are required)")
        who_i, sold_i, ts_i = cols["salesperson"], cols["sold"], cols.get("timestamp")
        rows, skipped, seen = [], 0, 0
        for r in reader:
            if not any(c.strip() for c in r):
                continue                                    # blank line
            seen += 1
            if seen > MAX_SALES_ROWS:
                raise SalesCsvError(f"too many rows: at most {MAX_SALES_ROWS:,} are accepted")
            name = _clean_name(r[who_i]) if who_i < len(r) else ""
            sold = _sold_flag(r[sold_i]) if sold_i < len(r) else None
            if not name or len(name) > MAX_SALESPERSON_CHARS or sold is None:
                skipped += 1
                continue
            stamp = r[ts_i].strip()[:MAX_TIMESTAMP_CHARS] if ts_i is not None and ts_i < len(r) else ""
            rows.append({"salesperson": name, "timestamp": stamp, "sold": sold})
    except csv.Error as exc:
        raise SalesCsvError(f"this is not a readable CSV file ({exc})") from None
    return SalesRows(rows, skipped)


def _staff_by_label(card: Scorecard, names: dict[int, str] | None) -> dict[str, int]:
    """Normalised label -> staff id, for the labels that point at exactly one salesperson.

    A salesperson answers to their typed name (case, spacing, Unicode form and apostrophe style ignored) and to
    `S<id>`, the label used in the video. Two people with the same name answer to nobody by name, so their rows are reported as unmatched
    rather than credited to the wrong person; their `S<id>` labels still work."""
    names = names or {}
    claims: dict[str, set[int]] = {}
    for c in card.staff:
        for label in (_match_key(names.get(c.staff) or c.name or ""), f"s{c.staff}"):
            if label:
                claims.setdefault(label, set()).add(c.staff)
    return {label: next(iter(ids)) for label, ids in claims.items() if len(ids) == 1}


def _readable_rows(rows: Iterable) -> Iterable[tuple[str, bool]]:
    """``(salesperson as typed, sold)`` for the rows that have both; anything else is not a sales row."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, sold = _clean_name(row.get("salesperson")), _sold_flag(row.get("sold"))
        if name and sold is not None:
            yield name, sold


def sales_conversion(card: Scorecard, rows: list[dict], names: dict[int, str]) -> Scorecard:
    """A new scorecard whose staff cards carry `sold` (rows with sold = true) and `conversion`
    (`sold / customers_engaged`, None when nobody was engaged).

    `rows` are ``{"salesperson", "timestamp", "sold"}`` dicts (`parse_sales_csv`; `sold` may also be a raw 1/0/yes/no
    string). A row belongs to the salesperson whose name it matches, ignoring case, spacing, Unicode form (NFKC) and
    the style of apostrophe (o', a modifier letter, a typographic quote or a grave accent), or whose label is
    `S<id>`. A salesperson with no row at all gets None for both: the CSV says nothing about them, which is not
    the same as no sales. Rows of people who are not staff in `card` are ignored here (`unmatched_sales` lists
    them).
    Sales are not tied to the camera's engagements, so `conversion` can exceed 1; it is reported as computed.
    """
    owner = _staff_by_label(card, names)
    sold: dict[int, int] = {}
    for name, flag in _readable_rows(rows):
        sid = owner.get(_match_key(name))
        if sid is not None:
            sold[sid] = sold.get(sid, 0) + int(flag)
    staff = []
    for c in card.staff:
        n = sold.get(c.staff)
        rate = None if n is None or c.customers_engaged <= 0 else n / c.customers_engaged
        staff.append(dataclasses.replace(c, sold=n, conversion=rate))
    return Scorecard(staff=staff, store=dataclasses.replace(card.store))


def unmatched_sales(card: Scorecard, rows: list[dict], names: dict[int, str]) -> list[str]:
    """The salespeople in `rows` that `sales_conversion` could not give to anyone in `card`: each once, in order of
    first appearance, spelled as in their first row. Rows with an unreadable `sold` do not count."""
    owner = _staff_by_label(card, names)
    out: dict[str, str] = {}
    for name, _ in _readable_rows(rows):
        key = _match_key(name)
        if key not in owner:
            out.setdefault(key, name)
    return list(out.values())
