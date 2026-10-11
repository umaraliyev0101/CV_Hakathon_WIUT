"""Optional sales CSV (spec 4.8): header exactly `car_model,period,sold`, joined to car zones by name.

Match rule: case-insensitive after stripping a leading "Car:" and surrounding whitespace. `sold`
is summed over periods per car; attention_to_sales = stops / sold (None when sold is 0 or the car
has no rows)."""
from __future__ import annotations

import csv
import io

HEADER = ["car_model", "period", "sold"]
MAX_ROWS = 20000
MAX_REASONS = 10
MAX_SOLD = 1_000_000


class SalesError(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def norm_name(name: str) -> str:
    n = " ".join(str(name).split())
    if n.lower().startswith("car:"):
        n = n[4:].strip()
    return n.casefold()


def parse_sales(text: str) -> list[tuple[str, str, int]]:
    """-> [(car_model, period, sold)]. Raises SalesError listing up to 10 reasons."""
    text = text.lstrip("﻿")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or [c.strip() for c in rows[0]] != HEADER:
        raise SalesError(["header must be exactly: car_model,period,sold"])
    out: list[tuple[str, str, int]] = []
    errs: list[str] = []
    for n, r in enumerate(rows[1:], start=2):
        if not r or all(not c.strip() for c in r):
            continue
        if len(r) != 3:
            errs.append(f"row {n}: expected 3 columns, got {len(r)}")
            continue
        model, period, sold = r[0].strip(), r[1].strip(), r[2].strip()
        if not model:
            errs.append(f"row {n}: car_model is empty")
        elif not sold.isascii() or not sold.isdigit():
            errs.append(f"row {n}: sold must be a whole number >= 0, got {sold!r}")
        elif len(sold) > 7 or int(sold) > MAX_SOLD:
            errs.append(f"row {n}: sold is too large (max {MAX_SOLD})")
        else:
            out.append((model, period, int(sold)))
        if len(out) > MAX_ROWS:
            errs.append(f"more than {MAX_ROWS} rows")
            break
    if not errs and not out:
        errs.append("no data rows")
    if errs:
        raise SalesError(errs[:MAX_REASONS])
    return out


def parse_sales_bytes(data: bytes) -> list[tuple[str, str, int]]:
    try:
        return parse_sales(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise SalesError(["file is not valid UTF-8 text"]) from None


def apply_sales(per_car: list[dict], rows) -> list[str]:
    """Set `sold` and `attention_to_sales` on every per_car dict; returns report warnings."""
    for c in per_car:
        c["sold"], c["attention_to_sales"] = None, None
    if not rows:
        return []
    totals: dict[str, int] = {}
    first_label: dict[str, str] = {}
    for model, _, sold in rows:
        k = norm_name(model)
        totals[k] = totals.get(k, 0) + sold
        first_label.setdefault(k, model)
    matched: set[str] = set()
    for c in per_car:
        k = norm_name(c.get("name", ""))
        if k in totals:
            matched.add(k)
            c["sold"] = totals[k]
            c["attention_to_sales"] = c["stops"] / totals[k] if totals[k] > 0 else None
    missing = [first_label[k] for k in first_label if k not in matched]
    if not missing:
        return []
    return [f"Sales CSV rows with no matching car zone: {', '.join(missing)}"]
