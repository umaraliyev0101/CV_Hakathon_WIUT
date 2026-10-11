from pulse.datatypes import ZoneVisit
from pulse.eval.stops import precision_recall


def zv(zone, a, b, stopped=True, pid=1):
    return ZoneVisit(pid, zone, a, b, stopped)


def gt(zone, a, b, vid=1):
    return {"visit_id": vid, "zone_id": zone, "t_start": a, "t_end": b}


def test_perfect_match():
    r = precision_recall([zv("c1", 10, 20)], [gt("c1", 10, 20)])
    assert (r["n_pred"], r["n_gt"], r["pred_matched"], r["gt_matched"]) == (1, 1, 1, 1)
    assert r["precision"] == 1.0 and r["recall"] == 1.0


def test_unstopped_visits_are_not_predictions():
    r = precision_recall([zv("c1", 10, 20, stopped=False)], [gt("c1", 10, 20)])
    assert r["n_pred"] == 0 and r["precision"] is None and r["recall"] == 0.0


def test_wrong_zone_does_not_match():
    r = precision_recall([zv("c2", 10, 20)], [gt("c1", 10, 20)])
    assert r["precision"] == 0.0 and r["recall"] == 0.0


def test_tolerance_boundary():
    # prediction ends 2 s before the GT stop starts: inside +-2 s, outside +-1 s
    assert precision_recall([zv("c1", 0, 8)], [gt("c1", 10, 15)], tol_s=2.0)["recall"] == 1.0
    assert precision_recall([zv("c1", 0, 8)], [gt("c1", 10, 15)], tol_s=1.0)["recall"] == 0.0
    assert precision_recall([zv("c1", 0, 7.9)], [gt("c1", 10, 15)], tol_s=2.0)["recall"] == 0.0


def test_false_positive_and_miss():
    pred = [zv("c1", 10, 20), zv("c1", 40, 50, pid=2)]
    r = precision_recall(pred, [gt("c1", 11, 19), gt("c1", 80, 90, vid=2)])
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert (r["pred_matched"], r["gt_matched"]) == (1, 1)


def test_many_to_one_counts_each_side_once():
    # two predicted pieces of one GT stop: both are correct (precision 1), the GT stop is found once
    r = precision_recall([zv("c1", 10, 14), zv("c1", 15, 20, pid=2)], [gt("c1", 10, 20)])
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["gt_matched"] == 1


def test_empty_inputs():
    r = precision_recall([], [])
    assert r["precision"] is None and r["recall"] is None
    assert precision_recall([], [gt("c1", 1, 5)])["recall"] == 0.0
    assert precision_recall([zv("c1", 1, 5)], [])["precision"] == 0.0


def test_stops_section_states_when_nothing_is_labelled():
    from pulse.eval.run_eval import _stops_section
    assert "no labelled showroom clip" in " ".join(_stops_section([]))
