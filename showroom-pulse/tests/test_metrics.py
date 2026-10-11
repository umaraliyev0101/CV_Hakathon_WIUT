import pytest

from pulse.datatypes import Sample, ZoneVisit
from pulse.eval import metrics as M


def box(x, y=100.0, w=40.0, h=100.0):
    return (x, y, x + w, y + h)


def samples(pid, ts, xs, y=100.0):
    return [Sample(t, pid, (x / 960, (y + 100) / 540), box(x, y), None, None, "gt") for t, x in zip(ts, xs)]


# ---- tracking (motmetrics) -------------------------------------------------------------------

def test_mot_perfect_prediction():
    gt = {f: {1: box(10 + 5 * f), 2: box(300 + 5 * f)} for f in range(10)}
    r = M.mot(gt, {f: dict(v) for f, v in gt.items()}, list(range(10)))
    assert r["mota"] == pytest.approx(1.0) and r["idf1"] == pytest.approx(1.0)
    assert r["num_switches"] == 0 and r["precision"] == pytest.approx(1.0) and r["recall"] == pytest.approx(1.0)


def test_mot_counts_one_id_switch():
    gt = {f: {1: box(10 + 5 * f)} for f in range(10)}
    pred = {f: {(7 if f < 5 else 8): box(10 + 5 * f)} for f in range(10)}
    r = M.mot(gt, pred, list(range(10)))
    assert r["num_switches"] == 1
    assert r["mota"] == pytest.approx(1 - 1 / 10) and r["idf1"] < 1.0


def test_mot_missed_and_false_positive_lower_mota():
    gt = {f: {1: box(10)} for f in range(4)}
    pred = {0: {1: box(10)}, 1: {1: box(10)}, 2: {}, 3: {1: box(10), 2: box(600)}}
    r = M.mot(gt, pred, [0, 1, 2, 3])
    assert r["mota"] == pytest.approx(1 - 2 / 4)       # one miss + one false positive over 4 GT boxes
    assert r["num_objects"] == 4 and r["num_misses"] == 1 and r["num_false_positives"] == 1


def test_mot_low_iou_is_not_a_match():
    gt = {0: {1: box(0)}}
    pred = {0: {1: box(30)}}                            # IoU = 10*100 / (2*4000 - 1000) = 0.14 < 0.5
    r = M.mot(gt, pred, [0])
    assert r["num_misses"] == 1 and r["num_false_positives"] == 1


def test_mot_no_gt_is_none_not_a_crash():
    r = M.mot({0: {}}, {0: {1: box(0)}}, [0])
    assert r["mota"] is None and r["num_false_positives"] == 1


def test_pool_mot_sums_counts():
    a = M.mot({f: {1: box(0)} for f in range(10)}, {f: {1: box(0)} for f in range(10)}, list(range(10)))
    b = M.mot({f: {1: box(0)} for f in range(10)}, {f: {} for f in range(10)}, list(range(10)))
    p = M.pool_mot([a, b])
    assert p["num_objects"] == 20 and p["num_misses"] == 10
    assert p["mota"] == pytest.approx(0.5) and p["recall"] == pytest.approx(0.5)


# ---- counts, occupancy ----------------------------------------------------------------------

def test_count_error():
    assert M.count_error(9, 10) == pytest.approx(10.0)
    assert M.count_error(12, 10) == pytest.approx(20.0)
    assert M.count_error(0, 0) is None and M.count_error(3, 0) is None


def test_pooled_count_error_does_not_let_clips_cancel():
    assert M.pooled_count_error([9, 12], [10, 10]) == pytest.approx(15.0)     # (1 + 2) / 20
    assert M.pooled_count_error([10, 10], [10, 10]) == 0.0
    assert M.pooled_count_error([3], [0]) is None
    # entries and exits pooled together: 2 + 3 errors over 10 + 20 GT events
    assert M.pooled_count_error([8, 23], [10, 20]) == pytest.approx(5 / 30 * 100)


def test_occupancy_mae():
    assert M.occupancy_mae([1, 2], [1, 3]) == 0.5
    assert M.occupancy_mae([], []) is None


def test_occupancy_mae_pads_shorter_series_with_zero():
    assert M.occupancy_mae([1, 1, 1], [1]) == pytest.approx(2 / 3)


# ---- person matching ------------------------------------------------------------------------

def test_match_persons_pairs_overlapping_tracks_and_leaves_others_out():
    ts = [i * 0.2 for i in range(20)]
    gt = {1: samples(1, ts, [100] * 20), 2: samples(2, ts, [500] * 20), 3: samples(3, ts, [800] * 20)}
    pred = {10: samples(10, ts, [102] * 20), 20: samples(20, ts, [503] * 20),
            30: samples(30, ts, [300] * 20)}                      # overlaps no GT person
    assert M.match_persons(gt, pred) == {1: 10, 2: 20}


def test_match_persons_is_one_to_one():
    ts = [i * 0.2 for i in range(20)]
    gt = {1: samples(1, ts, [100] * 20), 2: samples(2, ts, [102] * 20)}
    pred = {10: samples(10, ts, [101] * 20)}
    m = M.match_persons(gt, pred)
    assert len(m) == 1 and list(m.values()) == [10]


def test_match_persons_many_to_one_lets_gt_snippets_share_a_long_person():
    ts = [i * 0.2 for i in range(60)]
    gt = {1: samples(1, ts[:10], [100] * 10), 2: samples(2, ts[30:40], [100] * 10),
          3: samples(3, ts[:10], [700] * 10)}
    pred = {10: samples(10, ts, [100] * 60)}
    assert M.match_persons(gt, pred, one_to_one=False) == {1: 10, 2: 10}
    assert len(M.match_persons(gt, pred)) == 1                    # one-to-one: only one snippet gets it


def test_match_persons_requires_min_overlap():
    ts = [i * 0.2 for i in range(20)]
    gt = {1: samples(1, ts, [100] * 20)}
    pred = {10: samples(10, ts, [125] * 20)}                      # IoU = 15/65 = 0.23 < 0.3
    assert M.match_persons(gt, pred) == {}


def test_match_persons_prefers_the_longer_overlap():
    ts = [i * 0.2 for i in range(40)]
    gt = {1: samples(1, ts, [100] * 40)}
    pred = {10: samples(10, ts[:30], [100] * 30), 20: samples(20, ts[30:34], [100] * 4)}
    assert M.match_persons(gt, pred) == {1: 10}


# ---- dwell, journeys ------------------------------------------------------------------------

def V(pid, zone, t_in, t_out):
    return ZoneVisit(pid, zone, t_in, t_out)


def test_dwell_errors_pairs_per_zone_and_sums_visits():
    gt_j = {1: [V(1, "a", 0, 10), V(1, "a", 20, 25), V(1, "b", 30, 40)], 2: [V(2, "a", 0, 5)]}
    pred_j = {10: [V(10, "a", 1, 12), V(10, "c", 40, 44)], 20: [V(20, "a", 0, 5)]}
    out = M.dwell_errors(gt_j, pred_j, {1: 10, 2: 20})
    assert sorted(out) == sorted([(15.0, 11.0), (10.0, 0.0), (0.0, 4.0), (5.0, 5.0)])


def test_dwell_errors_ignores_unmatched_persons():
    assert M.dwell_errors({1: [V(1, "a", 0, 5)]}, {10: [V(10, "a", 0, 5)]}, {}) == []


def test_dwell_summary_median_seconds_and_percent():
    s = M.dwell_summary([(10.0, 11.0), (10.0, 14.0), (20.0, 20.0), (0.0, 3.0)])
    assert s["n_pairs"] == 4 and s["n_gt_positive"] == 3 and s["n_pred_only"] == 1
    assert s["median_abs_s"] == pytest.approx(1.0)              # |errors| over GT>0 pairs: 1, 4, 0
    assert s["median_abs_pct"] == pytest.approx(10.0)           # 10%, 40%, 0%


def test_dwell_summary_empty():
    s = M.dwell_summary([])
    assert s["n_pairs"] == 0 and s["median_abs_s"] is None and s["median_abs_pct"] is None


def test_journey_accuracies_average_over_matched_persons():
    gt_j = {1: [V(1, "a", 0, 5), V(1, "b", 6, 9)], 2: [V(2, "a", 0, 5)], 3: [V(3, "a", 0, 5)]}
    pred_j = {10: [V(10, "a", 0, 5), V(10, "b", 6, 9)], 20: [V(20, "b", 0, 5)], 30: []}
    acc = M.journey_accuracies(gt_j, pred_j, {1: 10, 2: 20, 3: 30})
    assert sorted(acc) == [0.0, 0.0, 1.0]
    assert sum(acc) / len(acc) == pytest.approx(1 / 3)


def test_journey_accuracies_skip_pairs_where_both_sequences_are_empty():
    acc = M.journey_accuracies({1: []}, {10: []}, {1: 10})
    assert acc == []


def test_journey_accuracy_partial_credit():
    gt_j = {1: [V(1, "a", 0, 5), V(1, "b", 6, 9), V(1, "c", 10, 14), V(1, "d", 15, 19)]}
    pred_j = {10: [V(10, "a", 0, 5), V(10, "b", 6, 9), V(10, "c", 10, 14)]}
    assert M.journey_accuracies(gt_j, pred_j, {1: 10}) == [pytest.approx(0.75)]


# ---- entry / exit events --------------------------------------------------------------------

def test_match_events_nearest_within_window_one_to_one():
    gt = [10.0, 20.0, 40.0]
    pred = [11.0, 19.0, 21.0, 90.0]
    m = M.match_events(gt, pred, window_s=5.0)
    assert m == [(0, 0, 1.0), (1, 1, 1.0)]          # 40 s has nothing within 5 s; 21 is left over


def test_match_events_picks_globally_closest_pairs_first():
    m = M.match_events([10.0, 12.0], [11.5], window_s=5.0)
    assert m == [(1, 0, 0.5)]


def test_median_timing_error():
    assert M.median_timing_error([(0, 0, 1.0), (1, 1, 3.0), (2, 2, 2.0)]) == 2.0
    assert M.median_timing_error([]) is None
