"""Results export: results.json, events.csv and scorecard.csv."""
import csv
import dataclasses
import json

import numpy as np
import pytest

from showroom_coach.analysis import Analysis, analyze, rescore
from showroom_coach.config import get_preset
from showroom_coach.export import results_dict, write_all, write_events_csv, write_scorecard_csv
from showroom_coach.metrics import scorecard
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack
from showroom_coach.types import Event, Person, RoleSuggestion, Scorecard, StaffCard, StoreCard


def test_write_all(meet_video, tmp_path):
    res = run_perception(meet_video, fake_stack())
    an = rescore(analyze(res, "short"), res, roles={1: "staff"}, names={1: "Aziz"})
    files = write_all(an, tmp_path)
    data = json.loads(files["results.json"].read_text())
    assert data["names"]["1"] == "Aziz" and "scorecard" in data
    rows = list(csv.DictReader(open(files["scorecard.csv"], newline="", encoding="utf-8-sig")))
    assert rows[-1]["staff"] == "STORE" and rows[0]["name"] == "Aziz"
    ev = list(csv.DictReader(open(files["events.csv"], newline="", encoding="utf-8-sig")))
    assert {"type", "t", "customer", "staff"} <= set(ev[0])


# ----------------------------------------------------------------------------------- a hand-built analysis

EVENTS = [
    Event("enter", 4.0, customer=2, data={"entry_known": True}),
    Event("enter", 5.0, customer=3, data={"entry_known": True}),
    Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": 5.0, "confirmed_at": 11.0, "via": "direct"}),
    Event("engagement_start", 9.0, customer=2, staff=1, data={"confirmed_at": 17.0}),
    Event("engagement", 9.0, customer=2, staff=1, t_end=30.0),
    Event("wait_alert", 15.0, customer=3, data={"wait_s": 10.2, "free_staff": []}),
    Event("walkout", 22.0, customer=3, data={"free_staff": [1]}),
]


def _analysis(names=None, events=EVENTS) -> Analysis:
    """An `Analysis` made by hand (no video, no models): staff 1, customers 2 (greeted, engaged) and 3 (walked out)."""
    names = {1: "Aziz"} if names is None else names
    events = sorted(events, key=lambda e: (e.t, e.type))
    persons = [Person(pid, [pid], 4.0 * pid, 40.0, 100, f"{pid}.jpg") for pid in (1, 2, 3)]
    roles = {1: "staff", 2: "customer", 3: "customer"}
    return Analysis(
        preset="short", thresholds=get_preset("short"), duration=60.0, persons=persons,
        suggestions={p.pid: RoleSuggestion(p.pid, roles[p.pid], 0.8, ["demo"], {}) for p in persons},
        roles=roles, overrides=dict(roles), names=names, events=events,
        scorecard=scorecard(events, [1], [2, 3], 60.0, names), timelines={}, warnings=["demo warning"],
        approx_share=0.0, obs_pid=np.zeros(0, np.int64), obs_xz=np.zeros((0, 2)), obs_src=np.zeros(0, np.int8),
        obs_facing=np.zeros((0, 2)), obs_edge=np.zeros(0, bool), features={})


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_write_all_returns_exactly_the_three_files(tmp_path):
    files = write_all(_analysis(), tmp_path / "out")           # the directory is created
    assert set(files) == {"results.json", "events.csv", "scorecard.csv"}
    assert all(p.is_file() and p.parent == tmp_path / "out" for p in files.values())
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["events.csv", "results.json", "scorecard.csv"]


def test_results_json_is_the_results_object_plus_the_extras(tmp_path):
    an = _analysis()
    assert results_dict(an) == an.to_dict()
    extra = {"video_url": "/api/jobs/x/files/annotated.mp4", "video_status": "ready"}
    merged = results_dict(an, extra)
    assert merged["video_url"] == extra["video_url"] and merged["video_status"] == "ready"
    assert {k: v for k, v in merged.items() if k not in extra} == an.to_dict()
    files = write_all(an, tmp_path, extra)
    assert json.loads(files["results.json"].read_text()) == merged
    assert json.loads(files["results.json"].read_text())["names"] == {"1": "Aziz"}      # pid keys are strings in JSON


def test_results_json_is_plain_ascii_and_survives_non_ascii_names(tmp_path):
    an = _analysis({1: "Zoë Хабиб"})
    files = write_all(an, tmp_path)
    files["results.json"].read_bytes().decode("ascii")      # any reader, whatever its default encoding, can open it
    assert json.loads(files["results.json"].read_text("ascii"))["names"]["1"] == "Zoë Хабиб"
    assert _rows(files["scorecard.csv"])[0]["name"] == "Zoë Хабиб"                       # the CSVs are UTF-8


def test_events_csv_columns_and_rows(tmp_path):
    an = _analysis()
    path = write_events_csv(an, tmp_path / "e.csv")
    assert path == tmp_path / "e.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == ["type", "t", "t_end", "customer", "staff", "staff_name", "detail"]
    rows = _rows(path)
    assert [r["type"] for r in rows] == [e.type for e in an.events]
    greet = next(r for r in rows if r["type"] == "greet")
    assert (greet["t"], greet["customer"], greet["staff"], greet["staff_name"]) == ("9.0", "2", "1", "Aziz")
    assert greet["detail"] == "time_to_greet=5.0; confirmed_at=11.0; via=direct"
    eng = next(r for r in rows if r["type"] == "engagement")
    assert (eng["t"], eng["t_end"]) == ("9.0", "30.0")
    enter = next(r for r in rows if r["type"] == "enter" and r["customer"] == "2")
    assert (enter["t_end"], enter["staff"], enter["staff_name"], enter["detail"]) == ("", "", "", "entry_known=true")
    assert next(r for r in rows if r["type"] == "wait_alert")["detail"] == "wait_s=10.2"       # an empty list is left out


def test_events_csv_names_unnamed_staff_by_their_label(tmp_path):
    an = _analysis(names={})                                   # nobody typed a name
    rows = _rows(write_events_csv(an, tmp_path / "e.csv"))
    greet = next(r for r in rows if r["type"] == "greet")
    assert (greet["staff"], greet["staff_name"]) == ("1", "S1")
    assert all(r["staff_name"] == "" for r in rows if r["staff"] == "")           # no staff on the row, no name
    assert _rows(write_scorecard_csv(an, tmp_path / "s.csv"))[0]["name"] == ""    # the scorecard keeps a blank name


def test_csvs_start_with_a_byte_order_mark_so_excel_reads_utf8_but_json_does_not(tmp_path):
    files = write_all(_analysis({1: "Zo\u00eb \u0425\u0430\u0431\u0438\u0431 o\u2019g\u2019li"}), tmp_path)
    for name in ("events.csv", "scorecard.csv"):
        raw = files[name].read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf") and not raw.startswith(b"\xef\xbb\xbf\xef\xbb\xbf")
        assert raw.decode("utf-8-sig").split(",")[0] in ("type", "staff")           # a clean first header cell
    assert _rows(files["scorecard.csv"])[0]["name"] == "Zo\u00eb \u0425\u0430\u0431\u0438\u0431 o\u2019g\u2019li"
    assert not files["results.json"].read_bytes().startswith(b"\xef\xbb\xbf")     # JSON must not have one


def test_events_csv_lists_are_joined_not_python_reprs_and_floats_rounded(tmp_path):
    an = _analysis(events=[Event("wait_alert", 12.0, customer=2, data={"wait_s": 12.3456, "free_staff": [1, 3]})])
    row = _rows(write_events_csv(an, tmp_path / "e.csv"))[0]
    assert row["detail"] == "wait_s=12.346; free_staff=1|3"


def test_scorecard_csv_columns_and_store_row(tmp_path):
    an = _analysis()
    path = write_scorecard_csv(an, tmp_path / "s.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert header == ["staff", "name", "greets", "median_time_to_greet", "customers_engaged", "engaged_total_s",
                      "engaged_median_s", "missed_while_free", "walkouts_while_free",
                      "customers", "greeted", "greet_rate", "walkouts", "unattended_wait_s", "wait_alerts"]
    rows = _rows(path)
    assert len(rows) == 2
    staff, store = rows
    card, s = an.scorecard.staff[0], an.scorecard.store
    assert (staff["staff"], staff["name"], staff["greets"]) == ("1", "Aziz", str(card.greets))
    assert float(staff["engaged_total_s"]) == pytest.approx(21.0) and float(staff["median_time_to_greet"]) == 5.0
    assert staff["walkouts_while_free"] == "1" and staff["missed_while_free"] == "0"
    assert all(staff[c] == "" for c in ("customers", "greeted", "greet_rate", "walkouts", "unattended_wait_s", "wait_alerts"))
    assert (store["staff"], store["name"]) == ("STORE", "")
    assert (store["customers"], store["greeted"], store["walkouts"], store["wait_alerts"]) == ("2", "1", "1", "1")
    assert float(store["greet_rate"]) == pytest.approx(s.greet_rate, abs=1e-3)
    assert float(store["unattended_wait_s"]) == pytest.approx(s.unattended_wait_s, abs=1e-3)
    assert store["greets"] == "1" and float(store["median_time_to_greet"]) == 5.0       # the two shared meanings
    assert all(store[c] == "" for c in ("customers_engaged", "engaged_total_s", "engaged_median_s",
                                        "missed_while_free", "walkouts_while_free"))


def test_none_values_become_blank_cells(tmp_path):
    card = StaffCard(staff=4, name=None, greets=0, median_time_to_greet=None, customers_engaged=0, engaged_total_s=0.0,
                     engaged_median_s=None, missed_while_free=0, walkouts_while_free=0)
    store = StoreCard(customers=0, customers_entry_known=0, greeted=0, greet_rate=None, median_time_to_greet=None,
                      walkouts=0, unattended_wait_s=0.0, wait_alerts=0, engagements=0)
    an = dataclasses.replace(_analysis(), scorecard=Scorecard([card], store))
    rows = _rows(write_scorecard_csv(an, tmp_path / "s.csv"))
    assert rows[0]["name"] == "" and rows[0]["median_time_to_greet"] == "" and rows[0]["engaged_median_s"] == ""
    assert rows[-1]["greet_rate"] == "" and rows[-1]["median_time_to_greet"] == ""
    assert "None" not in (tmp_path / "s.csv").read_text("utf-8-sig")


def test_names_that_look_like_formulas_are_neutralised(tmp_path):
    files = write_all(_analysis({1: '=HYPERLINK("http://x","y")'}), tmp_path)
    assert _rows(files["scorecard.csv"])[0]["name"].startswith("'=")
    assert any(r["staff_name"].startswith("'=") for r in _rows(files["events.csv"]) if r["staff"] == "1")
    assert json.loads(files["results.json"].read_text())["names"]["1"].startswith("=HYPERLINK")  # the JSON is exact
    for name in ("+1", "-2", "@x"):
        assert _rows(write_all(_analysis({1: name}), tmp_path)["scorecard.csv"])[0]["name"] == "'" + name


def test_empty_analysis_exports_without_errors(tmp_path, make_video):
    v = make_video(tmp_path / "empty.avi", seconds=3, movers=[])
    an = analyze(run_perception(v, fake_stack()), preset="short")
    files = write_all(an, tmp_path / "out")
    assert _rows(files["events.csv"]) == []
    assert (tmp_path / "out" / "events.csv").read_text("utf-8-sig").startswith("type,t,t_end,customer,staff,staff_name,detail")
    rows = _rows(files["scorecard.csv"])
    assert [r["staff"] for r in rows] == ["STORE"] and rows[0]["customers"] == "0" and rows[0]["greet_rate"] == ""
    assert "warnings" in json.loads(files["results.json"].read_text())


def test_write_all_replaces_existing_files(tmp_path):
    (tmp_path / "events.csv").write_text("old", "utf-8")
    write_all(_analysis(), tmp_path)
    assert (tmp_path / "events.csv").read_text("utf-8-sig").startswith("type,")
    assert not [p for p in tmp_path.iterdir() if ".part" in p.name]
