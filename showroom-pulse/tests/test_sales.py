"""Sales CSV parser, car join, report/CLI/API integration (Task 25)."""
import csv
import json
from pathlib import Path

import pytest

from pulse import pipeline, sales
from pulse.config import get_config
from pulse.datatypes import Zone
from tests.test_report import _check_module, _setup

ROOT = Path(__file__).resolve().parents[1]
HEAD = "car_model,period,sold\n"


def test_parse_ok_and_normalised_header():
    rows = sales.parse_sales("﻿" + HEAD + "Hatch,2026-09,3\r\n hatch ,2026-10,4\n\n")
    assert rows == [("Hatch", "2026-09", 3), ("hatch", "2026-10", 4)]


@pytest.mark.parametrize("text, reason", [
    ("car,period,sold\nA,x,1\n", "header"),
    ("", "header"),
    (HEAD, "no data rows"),
    (HEAD + "A,x\n", "row 2"),
    (HEAD + "A,x,-1\n", "row 2"),
    (HEAD + "A,x,1.5\n", "row 2"),
    (HEAD + "A,x,abc\n", "row 2"),
    (HEAD + ",x,1\n", "row 2"),
])
def test_parse_rejects_with_reasons(text, reason):
    with pytest.raises(sales.SalesError) as e:
        sales.parse_sales(text)
    assert any(reason in r for r in e.value.reasons)


def test_parse_non_utf8_bytes_rejected():
    with pytest.raises(sales.SalesError):
        sales.parse_sales_bytes(b"\xff\xfe\x00bad")


def _cars():
    return [{"zone_id": "car-1", "name": "Car: White Hatchback", "stops": 6},
            {"zone_id": "car-2", "name": "Car: SUV", "stops": 2},
            {"zone_id": "car-3", "name": "Sedan", "stops": 5}]


def test_join_case_insensitive_prefix_sum_and_ratio():
    rows = [("white hatchback", "a", 2), ("  WHITE HATCHBACK ", "b", 1), ("car: sedan", "a", 0),
            ("Coupe", "a", 4), ("Coupe", "b", 1)]
    cars = _cars()
    warns = sales.apply_sales(cars, rows)
    assert cars[0]["sold"] == 3 and cars[0]["attention_to_sales"] == 2.0
    assert cars[1]["sold"] is None and cars[1]["attention_to_sales"] is None      # unmatched car
    assert cars[2]["sold"] == 0 and cars[2]["attention_to_sales"] is None          # zero sold
    assert len(warns) == 1 and "Coupe" in warns[0] and warns[0].count("Coupe") == 1


def test_join_no_sales_leaves_none():
    cars = _cars()
    assert sales.apply_sales(cars, None) == []
    assert all(c["sold"] is None and c["attention_to_sales"] is None for c in cars)


def test_sample_csv_parses_and_joins_stock_zones():
    rows = sales.parse_sales_bytes((ROOT / "docs" / "sample_sales_SYNTHETIC.csv").read_bytes())
    zones = json.loads((ROOT / "labels" / "zones" / "wm-bangalore-07.json").read_text())["zones"]
    cars = [{"zone_id": z["zone_id"], "name": z["name"], "stops": 1} for z in zones if z["kind"] == "car"]
    assert sales.apply_sales(cars, rows) == []
    assert all(c["sold"] and c["attention_to_sales"] for c in cars)
    assert (ROOT / "docs" / "sample_sales_SYNTHETIC.md").read_text().lower().count("synthetic") >= 2


def test_per_car_csv_and_report_with_sales(tmp_path):
    video, pc, zones = _setup(tmp_path)
    cfg = get_config("short")
    rows = sales.parse_sales(HEAD + "car 1,a,2\nGhost,a,5\n")
    a = pipeline.score(pc, cfg, zones=zones, sales=rows)
    out = tmp_path / "out"
    pipeline.write_outputs(pc, a, out, cfg, video=video, render_video=False)
    head = list(csv.reader((out / "per_car.csv").open()))
    assert head[0][-3:] == ["sat_in_car", "sold", "attention_to_sales"]
    rep = json.loads((out / "report.json").read_text())
    car = rep["per_car"][0]
    assert car["sold"] == 2 and car["attention_to_sales"] == car["stops"] / 2
    assert any("Ghost" in w for w in rep["warnings"])
    assert "Stops per sale" in (out / "report.html").read_text()
    assert _check_module().main([str(out), "--no-video"]) == 0


def test_without_sales_columns_empty(tmp_path):
    video, pc, zones = _setup(tmp_path)
    a = pipeline.score(pc, get_config("short"), zones=zones)
    assert a.stats.per_car[0]["sold"] is None


def test_cli_sales_flag_rejects_bad_file(tmp_path, capsys):
    from pulse import cli
    bad = tmp_path / "s.csv"
    bad.write_text("x,y\n")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    assert cli.main(["analyze", str(video), "--out", str(tmp_path / "o"), "--sales", str(bad)]) == 2
    assert "header" in capsys.readouterr().err


def test_parse_huge_sold_rejected_not_crash():
    for v in ("1" * 5000, "1000001", "99999999"):
        with pytest.raises(sales.SalesError) as e:
            sales.parse_sales(HEAD + f"A,x,{v}\n")
        assert any("sold is too large" in r for r in e.value.reasons)
    assert sales.parse_sales(HEAD + "A,x,1000000\n") == [("A", "x", 1000000)]


def test_parse_edge_formats():
    assert sales.parse_sales('﻿car_model,period,sold\r\n"A, B",Q1,1\r\n') == [("A, B", "Q1", 1)]
    with pytest.raises(sales.SalesError) as e:
        sales.parse_sales(HEAD + "A,x,1,extra\n")
    assert any("row 2" in r for r in e.value.reasons)
    cars = [{"zone_id": "c", "name": "Car: A", "stops": 6}]
    sales.apply_sales(cars, sales.parse_sales(HEAD + "A,Q1,1\na,Q2,2\n"))
    assert cars[0]["sold"] == 3 and cars[0]["attention_to_sales"] == 2.0
