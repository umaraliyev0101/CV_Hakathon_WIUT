"""Optional sales CSV (spec 4.7): `salesperson,timestamp,sold` rows joined to the scorecard -> sold and conversion.

Pure-function tests use hand-built scorecards; the API tests run on the fake perception stack and the synthetic
`small_meet` video of test_app (pid 1 = the blue stander = staff; with the `short` preset it engages one customer).
"""
import csv
import dataclasses
import io

import pytest
from fastapi.testclient import TestClient

from showroom_coach import metrics
from showroom_coach.app.main import create_app
from showroom_coach.metrics import (MAX_SALES_ROWS, SalesCsvError, parse_sales_csv, sales_conversion,
                                    unmatched_sales)
from showroom_coach.perception.stack import fake_stack
from showroom_coach.types import Scorecard, StaffCard, StoreCard
from tests.test_app import _gated_factory, small_meet, upload, wait_status  # noqa: F401  (small_meet is a fixture)


def _card(staff, name, engaged):
    return StaffCard(staff=staff, name=name, greets=0, median_time_to_greet=None, customers_engaged=engaged,
                     engaged_total_s=0.0, engaged_median_s=None, missed_while_free=0, walkouts_while_free=0)


STORE = StoreCard(customers=3, customers_entry_known=3, greeted=2, greet_rate=2 / 3, median_time_to_greet=4.0,
                  walkouts=1, unattended_wait_s=9.0, wait_alerts=0, engagements=3)


def _scorecard(*cards):
    return Scorecard(staff=list(cards), store=STORE)


def row(who, sold, ts="2026-03-02 10:00"):
    return {"salesperson": who, "timestamp": ts, "sold": sold}


# ------------------------------------------------------------------------------------- sales_conversion


def test_sold_count_and_conversion_per_salesperson():
    card = _scorecard(_card(1, "Aziz", 4), _card(2, "Bek", 2))
    rows = [row("Aziz", True), row("Aziz", True), row("Aziz", False), row("Bek", False), row("Bek", True)]
    out = sales_conversion(card, rows, {1: "Aziz", 2: "Bek"})
    aziz, bek = out.staff
    assert (aziz.sold, aziz.conversion) == (2, pytest.approx(0.5))
    assert (bek.sold, bek.conversion) == (1, pytest.approx(0.5))


def test_zero_sold_is_zero_and_no_engagements_gives_no_conversion():
    card = _scorecard(_card(1, "Aziz", 3), _card(2, "Bek", 0))
    out = sales_conversion(card, [row("Aziz", False), row("Bek", True)], {1: "Aziz", 2: "Bek"})
    assert (out.staff[0].sold, out.staff[0].conversion) == (0, 0.0)
    assert (out.staff[1].sold, out.staff[1].conversion) == (1, None)           # division by zero is not a number


def test_names_match_ignoring_case_and_surrounding_or_repeated_whitespace():
    card = _scorecard(_card(1, "Aziz", 1), _card(2, "Bek Karim", 1))
    rows = [row("  AZIZ ", True), row("aziz", True), row("bek   KARIM", True), row("\tBek Karim\n", False)]
    out = sales_conversion(card, rows, {1: "Aziz", 2: "Bek Karim"})
    assert [c.sold for c in out.staff] == [2, 1]


def test_cyrillic_names_match_case_insensitively():
    card = _scorecard(_card(1, "Азиз", 2))
    out = sales_conversion(card, [row("азиз", True), row("АЗИЗ", False)], {1: "Азиз"})
    assert (out.staff[0].sold, out.staff[0].conversion) == (1, pytest.approx(0.5))


def test_s_pid_label_is_accepted_for_named_and_unnamed_staff():
    card = _scorecard(_card(1, "Aziz", 2), _card(3, None, 2))
    out = sales_conversion(card, [row("S1", True), row(" s3 ", True), row("S3", True), row("S2", True)], {1: "Aziz"})
    assert [c.sold for c in out.staff] == [1, 2]
    assert unmatched_sales(card, [row("S2", True)], {1: "Aziz"}) == ["S2"]


def test_name_comes_from_names_when_the_card_has_none():
    card = _scorecard(_card(1, None, 2))
    out = sales_conversion(card, [row("Aziz", True)], {1: "Aziz"})
    assert out.staff[0].sold == 1


@pytest.mark.parametrize("flag", [True, 1, "1", "true", "TRUE", "Yes", " yes ", "True"])
def test_every_accepted_spelling_of_sold(flag):
    out = sales_conversion(_scorecard(_card(1, "A", 1)), [row("A", flag)], {1: "A"})
    assert out.staff[0].sold == 1


@pytest.mark.parametrize("flag", [False, 0, "0", "false", "FALSE", "No", " no "])
def test_every_accepted_spelling_of_not_sold(flag):
    out = sales_conversion(_scorecard(_card(1, "A", 1)), [row("A", flag)], {1: "A"})
    assert out.staff[0].sold == 0


def test_rows_with_an_unreadable_sold_value_or_no_salesperson_are_ignored():
    card = _scorecard(_card(1, "A", 1))
    rows = [row("A", "maybe"), row("A", None), row("", True), {"sold": True}, {"salesperson": "A"}, "junk", None,
            row("A", True)]
    assert sales_conversion(card, rows, {1: "A"}).staff[0].sold == 1


def test_unmatched_salespeople_are_ignored_by_the_counts_and_reported():
    card = _scorecard(_card(1, "Aziz", 2))
    rows = [row("Aziz", True), row("Karim", True), row("  karim ", False), row("Dilnoza", True), row("S9", True)]
    out = sales_conversion(card, rows, {1: "Aziz"})
    assert out.staff[0].sold == 1
    assert unmatched_sales(card, rows, {1: "Aziz"}) == ["Karim", "Dilnoza", "S9"]      # first spelling, once each


def test_unmatched_ignores_rows_that_are_unreadable_anyway():
    card = _scorecard(_card(1, "Aziz", 2))
    assert unmatched_sales(card, [row("Karim", "maybe"), row("Aziz", True)], {1: "Aziz"}) == []


def test_two_staff_with_the_same_name_are_ambiguous_and_unmatched_but_s_labels_still_work():
    card = _scorecard(_card(1, "Aziz", 2), _card(2, "aziz", 2))
    rows = [row("Aziz", True), row("S2", True)]
    out = sales_conversion(card, rows, {1: "Aziz", 2: "aziz"})
    assert [c.sold for c in out.staff] == [None, 1]
    assert unmatched_sales(card, rows, {1: "Aziz", 2: "aziz"}) == ["Aziz"]


def test_staff_who_appear_in_no_row_have_no_numbers_not_zero():
    card = _scorecard(_card(1, "Aziz", 2), _card(2, "Bek", 2))
    out = sales_conversion(card, [row("Aziz", True)], {1: "Aziz", 2: "Bek"})
    assert (out.staff[1].sold, out.staff[1].conversion) == (None, None)


def test_pure_function_returns_a_new_scorecard_and_is_idempotent():
    card = _scorecard(_card(1, "Aziz", 2))
    rows = [row("Aziz", True)]
    out = sales_conversion(card, rows, {1: "Aziz"})
    assert out is not card and out.staff[0] is not card.staff[0]
    assert card.staff[0].sold is None and card.staff[0].conversion is None          # the input is untouched
    assert out.store == card.store
    assert sales_conversion(out, rows, {1: "Aziz"}) == out                          # re-applying changes nothing
    assert sales_conversion(out, [], {1: "Aziz"}).staff[0].sold is None             # no rows: numbers cleared


def test_other_card_fields_are_kept():
    card = _scorecard(dataclasses.replace(_card(1, "Aziz", 2), greets=5, engaged_total_s=61.5))
    got = sales_conversion(card, [row("Aziz", True)], {1: "Aziz"}).staff[0]
    assert (got.staff, got.name, got.greets, got.customers_engaged, got.engaged_total_s) == (1, "Aziz", 5, 2, 61.5)


# ------------------------------------------------------------------------------------- parse_sales_csv


GOOD = "salesperson,timestamp,sold\nAziz,2026-03-02 09:41,1\nBek,2026-03-02 10:05,no\n"


def test_parse_reads_rows_with_a_boolean_sold():
    rows = parse_sales_csv(GOOD)
    assert rows == [{"salesperson": "Aziz", "timestamp": "2026-03-02 09:41", "sold": True},
                    {"salesperson": "Bek", "timestamp": "2026-03-02 10:05", "sold": False}]
    assert rows.skipped == 0


def test_parse_header_is_case_and_space_insensitive_and_extra_columns_are_ignored():
    text = "\ufeff Salesperson ,Note,SOLD,Timestamp\nAziz,hi,yes,noon\n"
    assert parse_sales_csv(text) == [{"salesperson": "Aziz", "timestamp": "noon", "sold": True}]


def test_parse_timestamp_column_is_optional():
    assert parse_sales_csv("salesperson,sold\nAziz,1\n") == [{"salesperson": "Aziz", "timestamp": "", "sold": True}]


@pytest.mark.parametrize("sep", [";", "\t"])
def test_parse_semicolon_and_tab_separated_files_from_spreadsheets(sep):
    text = sep.join(["salesperson", "timestamp", "sold"]) + "\r\n" + sep.join(["Aziz", "9:41", "1"]) + "\r\n"
    assert parse_sales_csv(text) == [{"salesperson": "Aziz", "timestamp": "9:41", "sold": True}]


def test_parse_quoted_names_with_commas_and_cyrillic():
    rows = parse_sales_csv('salesperson,timestamp,sold\n"Karimov, Aziz",t,1\nАзиз,t,0\n')
    assert [r["salesperson"] for r in rows] == ["Karimov, Aziz", "Азиз"]


def test_parse_collapses_whitespace_in_names_and_skips_blank_lines():
    rows = parse_sales_csv("salesperson,timestamp,sold\n\n  Bek   Karim ,t,1\n   \n,,\n")
    assert [r["salesperson"] for r in rows] == ["Bek Karim"] and rows.skipped == 0


def test_parse_bad_rows_are_skipped_and_counted():
    text = ("salesperson,timestamp,sold\n"
            "Aziz,t,1\n"
            ",t,1\n"                    # no salesperson
            "Bek,t,maybe\n"             # unreadable sold
            "Bek,t\n"                   # too short
            "Bek\n"                     # too short
            + "X" * 81 + ",t,1\n"       # absurd name
            "Bek,t,0\n")
    rows = parse_sales_csv(text)
    assert [r["salesperson"] for r in rows] == ["Aziz", "Bek"] and rows.skipped == 5


@pytest.mark.parametrize("text", [
    "", "   \n\n", "Aziz,2026-03-02,1\nBek,2026-03-02,0\n",           # no header row
    "name,when,result\nAziz,t,1\n", "salesperson,timestamp\nAziz,t\n",   # header without the needed columns
    "salesperson,timestamp,sold\x00\nAziz,t,1\n",                       # binary rubbish
])
def test_parse_requires_a_header_and_text(text):
    with pytest.raises(SalesCsvError):
        parse_sales_csv(text)


def test_parse_header_only_gives_no_rows_but_no_error():
    rows = parse_sales_csv("salesperson,timestamp,sold\n")
    assert rows == [] and rows.skipped == 0


def test_parse_row_limit():
    line = "Aziz,t,1\n"
    ok = parse_sales_csv("salesperson,timestamp,sold\n" + line * MAX_SALES_ROWS)
    assert len(ok) == MAX_SALES_ROWS == 10_000
    with pytest.raises(SalesCsvError, match="10,000"):
        parse_sales_csv("salesperson,timestamp,sold\n" + line * (MAX_SALES_ROWS + 1))


def test_parse_oversized_cell_is_a_clean_error():
    with pytest.raises(SalesCsvError):
        parse_sales_csv("salesperson,timestamp,sold\n" + "A" * 200_000 + ",t,1\n")


def test_module_exports():
    assert {"sales_conversion", "unmatched_sales", "parse_sales_csv", "SalesCsvError", "MAX_SALES_ROWS"} <= set(
        metrics.__all__)


# ------------------------------------------------------------------------------------- API


def csv_file(text, name="sales.csv", ctype="text/csv"):
    return {"file": (name, text.encode("utf-8"), ctype)}


API_CSV = ("salesperson,timestamp,sold\n"
           "Aziz,2026-03-02 09:41,1\n"
           "aziz ,2026-03-02 10:05,0\n"
           "S1,2026-03-02 11:20,0\n"
           "Karim,2026-03-02 11:40,1\n"
           "Karim,2026-03-02 12:02,yes\n"
           "Bek,2026-03-02 12:30,maybe\n")                      # one bad row


@pytest.fixture(scope="module")
def api(tmp_path_factory, small_meet):  # noqa: F811
    """One app and one analysed job (staff = pid 1 "Aziz" with one engaged customer under `short`): (client, id)."""
    root = tmp_path_factory.mktemp("sales") / "jobs"
    with TestClient(create_app(stack_factory=fake_stack, jobs_root=root)) as c:
        jid = upload(c, small_meet)
        assert wait_status(c, jid, {"needs_confirm", "failed"})["status"] == "needs_confirm"
        yield c, jid


@pytest.fixture
def job(api):
    """The shared job reset to a known state: Aziz is staff, no sales CSV."""
    c, jid = api
    assert c.delete(f"/api/jobs/{jid}/sales").status_code == 200
    r = c.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff", "2": "customer"}, "names": {"1": "Aziz"},
                                               "preset": "short"})
    assert r.status_code == 200
    return c, jid


def staff_card(res):
    (card,) = res["scorecard"]["staff"]
    return card


def test_without_a_sales_csv_nothing_changes(job):
    c, jid = job
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert "sales" not in res
    assert staff_card(res)["customers_engaged"] == 1
    assert (staff_card(res)["sold"], staff_card(res)["conversion"]) == (None, None)


def test_upload_sales_csv_joins_by_name(job):
    c, jid = job
    r = c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["sales"] == {"rows": 5, "unmatched": ["Karim"], "skipped": 1}
    card = staff_card(res)
    assert (card["name"], card["sold"], card["conversion"]) == ("Aziz", 1, pytest.approx(1.0))       # 1 sold / 1 engaged
    assert res["video_status"] in ("rendering", "ready") and "scorecard" in res and "timelines" in res


def test_results_keep_the_sales_after_the_upload(job):
    c, jid = job
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert res["sales"] == {"rows": 5, "unmatched": ["Karim"], "skipped": 1}
    assert staff_card(res)["sold"] == 1


def test_sales_are_reapplied_when_roles_names_or_preset_change(job):
    c, jid = job
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    res = c.post(f"/api/jobs/{jid}/roles", json={"names": {"1": "Karim"}}).json()          # the rows now fit "Karim"
    assert res["sales"]["unmatched"] == ["Aziz"]
    assert (staff_card(res)["name"], staff_card(res)["sold"]) == ("Karim", 2)
    res = c.post(f"/api/jobs/{jid}/roles", json={"names": {"1": "Nobody"}}).json()          # only the "S1" row fits
    assert res["sales"]["unmatched"] == ["Aziz", "Karim"]
    assert (staff_card(res)["sold"], staff_card(res)["conversion"]) == (0, 0.0)
    res = c.post(f"/api/jobs/{jid}/roles", json={"names": {"1": "Aziz"}, "preset": "dealership"}).json()
    assert res["preset"] == "dealership" and staff_card(res)["customers_engaged"] == 0
    assert (staff_card(res)["sold"], staff_card(res)["conversion"]) == (1, None)           # nobody engaged: no rate
    res = c.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "customer"}}).json()       # nobody is staff now
    assert res["scorecard"]["staff"] == [] and res["sales"]["unmatched"] == ["Aziz", "S1", "Karim"]


def test_a_second_upload_replaces_the_first(job):
    c, jid = job
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    res = c.post(f"/api/jobs/{jid}/sales", files=csv_file("salesperson,timestamp,sold\nAziz,t,0\n")).json()
    assert res["sales"] == {"rows": 1, "unmatched": [], "skipped": 0}
    assert (staff_card(res)["sold"], staff_card(res)["conversion"]) == (0, 0.0)


def test_delete_clears_the_sales(job):
    c, jid = job
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    r = c.delete(f"/api/jobs/{jid}/sales")
    assert r.status_code == 200 and "sales" not in r.json()
    assert (staff_card(r.json())["sold"], staff_card(r.json())["conversion"]) == (None, None)
    assert "sales" not in c.get(f"/api/jobs/{jid}/results").json()
    assert c.delete(f"/api/jobs/{jid}/sales").status_code == 200                           # clearing twice is fine


def _read_csv(c, url):
    text = c.get(url).content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def test_exports_carry_sold_and_conversion(job):
    c, jid = job
    url = f"/api/jobs/{jid}/files/scorecard.csv"
    aziz, store = _read_csv(c, url)
    assert (aziz["sold"], aziz["conversion"]) == ("", "")                                  # blank without a sales CSV
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    aziz, store = _read_csv(c, url)
    assert (aziz["name"], aziz["sold"], float(aziz["conversion"])) == ("Aziz", "1", 1.0)
    assert (store["staff"], store["sold"], store["conversion"]) == ("STORE", "", "")
    exported = c.get(f"/api/jobs/{jid}/files/results.json").json()
    assert exported["sales"]["unmatched"] == ["Karim"] and exported["scorecard"]["staff"][0]["sold"] == 1
    c.delete(f"/api/jobs/{jid}/sales")
    assert _read_csv(c, url)[0]["sold"] == ""
    assert "sales" not in c.get(f"/api/jobs/{jid}/files/results.json").json()


def test_bad_csv_is_a_400_and_keeps_the_earlier_sales(job):
    c, jid = job
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    for text in ("", "Aziz,t,1\n", "name,when,result\nAziz,t,1\n"):
        r = c.post(f"/api/jobs/{jid}/sales", files=csv_file(text))
        assert r.status_code == 400 and "salesperson" in r.json()["detail"], text
    r = c.post(f"/api/jobs/{jid}/sales", files=csv_file("salesperson,timestamp,sold\n" + "A,t,1\n" * 10_001))
    assert r.status_code == 400 and "10,000" in r.json()["detail"]
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("sales.csv", b"\xff\xfe\x00bad", "text/csv")})
    assert r.status_code == 400 and "UTF-8" in r.json()["detail"]
    assert c.get(f"/api/jobs/{jid}/results").json()["sales"]["rows"] == 5


def test_missing_file_field_is_a_400(job):
    c, jid = job
    assert c.post(f"/api/jobs/{jid}/sales", data={"x": "1"}, files={"other": ("a.csv", b"x", "text/csv")}
                  ).status_code == 400


def test_only_csv_like_files_are_accepted(job):
    c, jid = job
    for name, ctype in (("run.exe", "application/x-msdownload"), ("photo.png", "image/png"), ("x.mp4", "video/mp4")):
        assert c.post(f"/api/jobs/{jid}/sales", files=csv_file(GOOD, name, ctype)).status_code == 415, name
    for name, ctype in (("sales.csv", "application/vnd.ms-excel"), ("sales.CSV", ""),
                        ("sales", "text/csv"), ("sales.txt", "text/plain")):
        assert c.post(f"/api/jobs/{jid}/sales", files=csv_file(GOOD, name, ctype)).status_code == 200, (name, ctype)


def test_utf8_bom_from_excel_is_accepted(job):
    c, jid = job
    body = ("\ufeff" + API_CSV).encode("utf-8")
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("sales.csv", body, "text/csv")})
    assert r.status_code == 200 and r.json()["sales"]["rows"] == 5


def test_over_one_megabyte_is_413(job, monkeypatch):
    import showroom_coach.app.main as main
    c, jid = job
    assert main.MAX_SALES_BYTES == 1 << 20
    big = "salesperson,timestamp,sold\n" + "Aziz,2026-03-02 09:41:00 padding padding padding,1\n" * 30_000
    assert len(big) > 1 << 20
    assert c.post(f"/api/jobs/{jid}/sales", files=csv_file(big)).status_code == 413
    monkeypatch.setattr(main, "MAX_SALES_BYTES", 50)                       # and a small cap is enforced on the file
    assert c.post(f"/api/jobs/{jid}/sales", files=csv_file(GOOD)).status_code == 413
    assert "sales" not in c.get(f"/api/jobs/{jid}/results").json()


def test_upload_without_content_length_is_411(job):
    c, jid = job
    req = c.build_request("POST", f"/api/jobs/{jid}/sales", files=csv_file(GOOD))
    del req.headers["Content-Length"]
    assert c.send(req).status_code == 411


def test_unknown_job_is_404(api):
    c, _ = api
    assert c.post("/api/jobs/000000000000/sales", files=csv_file(GOOD)).status_code == 404
    assert c.delete("/api/jobs/000000000000/sales").status_code == 404
    assert c.post("/api/jobs/zzz/sales", files=csv_file(GOOD)).status_code == 404


def test_before_needs_confirm_it_is_a_409(tmp_path, small_meet):  # noqa: F811
    import threading
    gate = threading.Event()
    with TestClient(create_app(stack_factory=_gated_factory(gate), jobs_root=tmp_path / "jobs")) as c:
        try:
            jid = upload(c, small_meet)
            assert c.post(f"/api/jobs/{jid}/sales", files=csv_file(GOOD)).status_code == 409
            assert c.delete(f"/api/jobs/{jid}/sales").status_code == 409
        finally:
            gate.set()
        wait_status(c, jid, {"needs_confirm", "failed"})


def test_sales_do_not_trigger_a_new_render(job):
    c, jid = job
    wait_status(c, jid, {"done"})
    before = c.get(f"/api/jobs/{jid}/results").json()["video_url"]
    c.post(f"/api/jobs/{jid}/sales", files=csv_file(API_CSV))
    s = c.get(f"/api/jobs/{jid}").json()
    assert s["status"] == "done" and s["video_status"] == "ready"
    assert c.get(f"/api/jobs/{jid}/results").json()["video_url"] == before
