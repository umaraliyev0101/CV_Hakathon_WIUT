"""Results export: ``results.json``, ``events.csv`` and ``scorecard.csv``.

``results.json`` is the web API's results object (``Analysis.to_dict()``) plus whatever the web layer adds (URLs,
video status). It is written as plain ASCII JSON (non-ASCII names are ``\\uXXXX`` escapes), so any reader can open
it whatever its default encoding is. The CSVs are UTF-8 with a header row and a byte-order mark (``utf-8-sig``), which
is what makes Excel on Windows show Cyrillic and Uzbek names correctly; read them with ``encoding="utf-8-sig"``.

``scorecard.csv`` has one row per salesperson and a final ``STORE`` row. The first eleven columns are the staff card
(the last two, ``sold`` and ``conversion``, are blank unless a sales CSV was added); the last six are store-only and
blank on staff rows. On the STORE row only the two staff columns that mean the same thing store-wide are filled:
``greets`` (customers greeted) and ``median_time_to_greet``; the other staff columns stay blank, because summing them
would double count (a missed customer is counted once for every free salesperson).

Names are typed by the user and these files are opened in spreadsheets, so a name that starts with ``=``, ``+``,
``-`` or ``@`` gets a leading apostrophe in the CSVs (formula injection); the JSON keeps the exact text.
Every file is written to a temporary name and moved into place, so a reader never sees half a file.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .types import to_jsonable

if TYPE_CHECKING:
    from .analysis import Analysis

__all__ = ["atomic_replace", "results_dict", "write_all", "write_events_csv", "write_scorecard_csv"]

EVENT_COLUMNS = ("type", "t", "t_end", "customer", "staff", "staff_name", "detail")
STAFF_COLUMNS = ("staff", "name", "greets", "median_time_to_greet", "customers_engaged", "engaged_total_s",
                 "engaged_median_s", "missed_while_free", "walkouts_while_free", "sold", "conversion")
STORE_COLUMNS = ("customers", "greeted", "greet_rate", "walkouts", "unattended_wait_s", "wait_alerts")
STORE_ROW = "STORE"
DECIMALS = 3
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
_REPLACE_TRIES = 10       # Windows refuses to replace a file that a reader (the web server) holds open for a moment
_REPLACE_DELAY_S = 0.2


def atomic_replace(src: Path, dst: Path) -> None:
    """``os.replace`` that waits out a reader holding ``dst`` open (PermissionError on Windows)."""
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(_REPLACE_DELAY_S)


def _write_atomic(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        atomic_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# ----------------------------------------------------------------------------------------------- JSON

def results_dict(an: Analysis, extra: dict | None = None) -> dict:
    """The results object: ``an.to_dict()`` with the entries of ``extra`` (URLs, video status, ...) added on top."""
    out = an.to_dict()
    if extra:
        out.update(to_jsonable(extra))
    return out


# ----------------------------------------------------------------------------------------------- CSV cells

def _num(v) -> str:
    """A number as text (floats rounded to 3 decimals); None and non-finite values become an empty cell."""
    if v is None or isinstance(v, bool):
        return "" if v is None else str(v).lower()
    if isinstance(v, float):
        return str(round(v, DECIMALS)) if math.isfinite(v) else ""
    return str(v)


def _text(s) -> str:
    """User-typed text for a cell: a leading formula character gets an apostrophe in front."""
    s = "" if s is None else str(s)
    return "'" + s if s.startswith(_FORMULA_STARTS) else s


def _detail_value(v) -> str:
    if isinstance(v, (list, tuple, set, frozenset)):
        return "|".join(_detail_value(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v))
    return _num(v)


def _detail(data: dict) -> str:
    """``key=value`` pairs of an event's data joined by "; " (lists joined by "|", empty values left out)."""
    parts = []
    for key, v in data.items():
        text = _detail_value(v)
        if text != "":
            parts.append(f"{key}={text}")
    return "; ".join(parts)


def _csv_bytes(header: tuple[str, ...], rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def write_events_csv(an: Analysis, path) -> Path:
    """One row per event, in event order. Columns: type,t,t_end,customer,staff,staff_name,detail.
    ``staff_name`` is the typed name, or "S<pid>" (the label used in the video) when none was typed."""
    rows = [[e.type, _num(e.t), _num(e.t_end), _num(e.customer), _num(e.staff),
             _text(an.names.get(e.staff) or f"S{e.staff}") if e.staff is not None else "", _detail(e.data)]
            for e in an.events]
    return _write_atomic(path, _csv_bytes(EVENT_COLUMNS, rows))


def write_scorecard_csv(an: Analysis, path) -> Path:
    """One row per salesperson, then the STORE row (see the module docstring for the column mapping)."""
    blank_store = [""] * len(STORE_COLUMNS)
    rows = [[_num(c.staff), _text(c.name), _num(c.greets), _num(c.median_time_to_greet), _num(c.customers_engaged),
             _num(c.engaged_total_s), _num(c.engaged_median_s), _num(c.missed_while_free),
             _num(c.walkouts_while_free), _num(c.sold), _num(c.conversion), *blank_store]
            for c in an.scorecard.staff]
    s = an.scorecard.store
    rows.append([STORE_ROW, "", _num(s.greeted), _num(s.median_time_to_greet), "", "", "", "", "", "", "",
                 _num(s.customers), _num(s.greeted), _num(s.greet_rate), _num(s.walkouts),
                 _num(s.unattended_wait_s), _num(s.wait_alerts)])
    return _write_atomic(path, _csv_bytes(STAFF_COLUMNS + STORE_COLUMNS, rows))


def write_all(an: Analysis, out_dir, extra: dict | None = None) -> dict[str, Path]:
    """Write results.json, events.csv and scorecard.csv into ``out_dir`` (created) and return their paths by name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results_dict(an, extra), indent=2, allow_nan=False)      # ensure_ascii: plain ASCII on disk
    return {
        "results.json": _write_atomic(out_dir / "results.json", text.encode("ascii")),
        "events.csv": write_events_csv(an, out_dir / "events.csv"),
        "scorecard.csv": write_scorecard_csv(an, out_dir / "scorecard.csv"),
    }
