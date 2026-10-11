"""Printable one-page HTML report: inline CSS, no scripts, no external resources.

PRIVACY: the only image is the already face-blurred heatmap.png, embedded as a data URI; raw frames
never reach this module. Every dynamic string (zone names are user input) goes through `esc`."""
from __future__ import annotations

import base64
import html as _html

BANNER = "Anonymous, aggregated analytics; decision support only."

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:11px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1c1c1c;background:#fff;margin:0;padding:12mm}
.banner{border:1px solid #888;background:#f3f3f3;padding:4px 8px;font-weight:600;margin-bottom:8px}
h1{font-size:18px;margin:0 0 2px}
h2{font-size:12px;margin:10px 0 4px;border-bottom:1px solid #ccc;padding-bottom:2px}
.from{color:#555;margin-bottom:6px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.kpi{border:1px solid #ccc;padding:4px 6px}
.kpi b{display:block;font-size:15px}
.kpi span{color:#555}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:2px 5px;text-align:left}
th{background:#f3f3f3}
.warn{border:1px solid #c80;background:#fff7e6;padding:3px 8px;margin:6px 0}
.warn ul{margin:2px 0;padding-left:16px}
.insight{border:1px solid #ccc;padding:4px 8px;margin-bottom:4px;break-inside:avoid}
.insight h3{font-size:12px;margin:0 0 2px}
.ev{color:#444}
.conf{font-weight:600;text-transform:uppercase;font-size:9px;color:#555}
.heat{max-width:100%;max-height:70mm;display:block;margin:0 auto;border:1px solid #ccc}
.foot{margin-top:8px;color:#555;font-size:9px}
@page{size:A4;margin:10mm}
@media print{body{padding:0;font-size:10px}.heat{max-height:62mm}h2{break-after:avoid}}
"""


def esc(v) -> str:
    return _html.escape("" if v is None else str(v), quote=True)


def _num(v, nd=2) -> str:
    if isinstance(v, bool) or v is None:
        return "n/a" if v is None else str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if v == v else "n/a"
    return str(v)


def _pct(v) -> str:
    return "n/a" if not isinstance(v, (int, float)) or isinstance(v, bool) else f"{round(v * 100)}%"


def _mmss(v) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "n/a"
    s = int(round(v))
    return f"{s // 60}:{s % 60:02d}"


def _kpis(st: dict) -> str:
    items = [("Visitors (excl. staff)", _num(st.get("visitors"))), ("Groups", _num(st.get("groups"))),
             ("Median visit", _mmss(st.get("median_visit_s"))), ("Browse rate", _pct(st.get("browse_rate"))),
             ("Assisted share", _pct(st.get("assisted_share"))),
             ("Unserved-interest episodes", _num(st.get("unserved_episodes"))),
             ("Peak occupancy", _num(st.get("peak_occupancy"))),
             ("Peak at", _mmss(st.get("peak_t")))]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in items) + "</div>"


def _cars(rows: list[dict]) -> str:
    if not rows:
        return "<p>No car zones defined.</p>"
    sales = any(c.get("sold") is not None for c in rows)
    head = ["Rank", "Car", "Stops", "Stop rate", "Median dwell", "Total dwell", "Repeat visits",
            "Unassisted share", "Sat in car"] + (["Sold", "Stops per sale"] if sales else [])
    out = ["<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for c in rows:
        cells = [f"#{_num(c.get('rank'))}", c.get("name") or c.get("zone_id"), _num(c.get("stops")),
                 _pct(c.get("stop_rate")), _mmss(c.get("median_dwell")), _mmss(c.get("total_dwell")),
                 _num(c.get("repeat_visits")), _pct(c.get("unassisted_share")), _num(c.get("sat_in_car", 0))]
        if sales:
            cells += [_num(c.get("sold")), _num(c.get("attention_to_sales"))]
        out.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in cells) + "</tr>")
    return "".join(out) + "</tbody></table>"


def _insights(ins: list[dict]) -> str:
    if not ins:
        return "<p>No recommendations: not enough visits for confident insights.</p>"
    out = []
    for i in ins[:3]:
        ev = "; ".join(f"{k.replace('_', ' ')}: {_num(v)}" for k, v in (i.get("evidence") or {}).items())
        out.append(f'<div class="insight"><span class="conf">{esc(i.get("confidence"))} confidence'
                   f' (n = {esc(i.get("sample"))})</span><h3>{esc(i.get("title"))}</h3>'
                   f'<div class="ev">Evidence: {esc(ev) or "n/a"}</div>'
                   f'<div><b>Suggested action:</b> {esc(i.get("action"))}</div></div>')
    return "".join(out)


def render_html(report: dict, heatmap_png: bytes) -> str:
    meta, st = report.get("meta") or {}, report.get("store") or {}
    warns = report.get("warnings") or []
    wblock = ('<div class="warn"><b>Warnings</b><ul>' + "".join(f"<li>{esc(w)}</li>" for w in warns)
              + "</ul></div>") if warns else ""
    uri = "data:image/png;base64," + base64.b64encode(heatmap_png).decode("ascii")
    src = (f'Generated from clip {esc(meta.get("video"))}, preset {esc(meta.get("preset"))}, '
           f'{esc(_mmss(meta.get("analysed_duration")))} analysed.')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Showroom Pulse report</title><style>{CSS}</style></head><body>'
            f'<div class="banner">{esc(BANNER)}</div><h1>Showroom Pulse report</h1>'
            f'<div class="from">{src}</div>{wblock}{_kpis(st)}'
            f'<h2>Per-car engagement</h2>{_cars(report.get("per_car") or [])}'
            f'<h2>Top recommendations</h2>{_insights(report.get("insights") or [])}'
            f'<h2>Where people walk and linger</h2>'
            f'<img class="heat" alt="Heatmap of visitor positions (faces blurred)" src="{uri}">'
            f'<div class="foot">Heatmap is drawn on a face-blurred keyframe. {esc(BANNER)}</div>'
            f'</body></html>')
