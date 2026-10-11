"""MEVA evaluation: id mapping, MOT metrics, talk/engagement matching, enter/exit timing, aggregation, CLI glue.

The unit tests use synthetic ``GeomBox`` lists and ``SimpleNamespace`` stand-ins for the perception result and the
analysis (only the attributes the evaluation reads). One end-to-end test runs ``eval_clip`` on a synthetic video
with the model-free fake stack and hand-written KPF files.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from showroom_coach.config import Thresholds
from showroom_coach.evaluation import meva_eval as me
from showroom_coach.evaluation.kpf import Activity, GeomBox
from showroom_coach.events import PersonState, run_events

FPS = 30.0
TALK, ENTER, EXIT = me.GT_TALK, me.GT_ENTER, me.GT_EXIT


# ----------------------------------------------------------------------------------------------------- builders

def make_res(n_frames, obs, scale=0.5, sample_fps=5.0, src_fps=FPS):
    """Perception stand-in. ``obs`` = (sampled frame, track id, processed-px box) rows."""
    step = int(round(src_fps / sample_fps))
    src = np.arange(n_frames) * step
    return NS(
        video=NS(scale=scale, src_fps=src_fps, duration=n_frames / sample_fps),
        frame_times=src / src_fps, src_frame_idx=src,
        obs_frame=np.array([o[0] for o in obs], np.int64),
        obs_track=np.array([o[1] for o in obs], np.int64),
        obs_bbox=np.array([o[2] for o in obs], float).reshape(-1, 4),
        obs_xz=None)


def make_an(pids, xz=None, timelines=None):
    return NS(obs_pid=np.array(pids, np.int64), obs_xz=None if xz is None else np.array(xz, float),
              timelines=timelines or {})


def make_gt(boxes_by_actor, n_frames, step=6):
    """{actor: box or f(frame)->box} -> {source frame: [GeomBox]} for every source frame < n_frames * step."""
    gt = {}
    for f in range(n_frames * step):
        for actor, box in boxes_by_actor.items():
            b = box(f) if callable(box) else box
            if b is not None:
                gt.setdefault(f, []).append(GeomBox(actor, f, tuple(b)))
    return gt


def box_obs(frames, track, box):
    return [(f, track, box) for f in frames]


A_SRC, B_SRC, C_SRC = (100, 100, 200, 300), (600, 100, 700, 300), (1000, 100, 1100, 300)     # source px
A_OUT, B_OUT, C_OUT = (50, 50, 100, 150), (300, 50, 350, 150), (500, 50, 550, 150)           # at scale 0.5


def act(name, f0, f1, actors, spans=None, act_id=1):
    spans = spans if spans is not None else {a: (f0, f1) for a in actors}
    return Activity(name=name, act_id=act_id, f0=f0, f1=f1, actors=list(actors), actor_spans=dict(spans))


# ----------------------------------------------------------------------------------------------------- match_ids

def test_match_ids_maps_tracks_to_actors_with_scaled_boxes():
    res = make_res(20, box_obs(range(20), 10, A_OUT) + box_obs(range(20), 11, B_OUT))
    gt = make_gt({100: A_SRC, 200: B_SRC}, 20)
    assert me.match_ids(res, None, gt, use_persons=False) == {10: 100, 11: 200}


def test_match_ids_fragments_map_many_to_one():
    obs = box_obs(range(0, 10), 1, A_OUT) + box_obs(range(10, 20), 2, A_OUT)
    gt = make_gt({7: A_SRC}, 20)
    assert me.match_ids(make_res(20, obs), None, gt, False) == {1: 7, 2: 7}


def test_match_ids_needs_three_votes_and_thirty_percent():
    obs = (box_obs(range(2), 5, C_OUT)                                     # 2 obs, both matched: below 3 votes
           + box_obs(range(20), 6, B_OUT)                                  # 20 obs, actor 2 visible on 3 samples only
           + box_obs(range(10), 7, (800, 800, 900, 900))                   # never on anyone
           + box_obs(range(10), 8, A_OUT))                                 # always on actor 1
    gt = make_gt({1: A_SRC, 2: lambda f: B_SRC if f // 6 < 3 else None, 3: C_SRC}, 20)
    assert me.match_ids(make_res(20, obs), None, gt, False) == {8: 1}      # 6: 3 votes of 20 frames = 15% < 30%


def test_match_ids_exactly_thirty_percent_is_enough():
    obs = box_obs(range(10), 3, A_OUT)
    on = lambda keep: (lambda f: A_SRC if f // 6 < keep else None)       # noqa: E731 - actor visible for `keep` samples
    assert me.match_ids(make_res(10, obs), None, make_gt({9: on(3)}, 10), False) == {3: 9}
    assert me.match_ids(make_res(10, obs), None, make_gt({9: on(2)}, 10), False) == {}


def test_match_ids_majority_vote_and_tie_break():
    obs = box_obs(range(10), 1, A_OUT)
    gt = make_gt({1: lambda f: A_SRC if f // 6 < 4 else None, 2: lambda f: A_SRC if f // 6 >= 4 else None}, 10)
    assert me.match_ids(make_res(10, obs), None, gt, False) == {1: 2}    # 6 votes for actor 2 beat 4 for actor 1
    tie = make_gt({4: lambda f: A_SRC if f // 6 < 5 else None, 3: lambda f: A_SRC if f // 6 >= 5 else None}, 10)
    assert me.match_ids(make_res(10, obs), None, tie, False) == {1: 3}   # 5 : 5 -> the lower actor id


def test_match_ids_iou_threshold():
    shifted = (60, 50, 110, 150)                       # IoU with A_OUT = 40*100 / (2*5000 - 4000) = 0.667
    far = (90, 50, 140, 150)                           # IoU = 10*100 / 9000 = 0.11
    gt = make_gt({1: A_SRC}, 10)
    assert me.match_ids(make_res(10, box_obs(range(10), 1, shifted)), None, gt, False) == {1: 1}
    assert me.match_ids(make_res(10, box_obs(range(10), 1, far)), None, gt, False) == {}


def test_match_ids_one_to_one_per_frame():
    # two tracks on the same actor in the same frames: the better-overlapping one gets the vote, the other none
    obs = box_obs(range(10), 1, A_OUT) + box_obs(range(10), 2, (55, 50, 105, 150))
    assert me.match_ids(make_res(10, obs), None, make_gt({7: A_SRC}, 10), False) == {1: 7}


def test_match_ids_persons_use_obs_pid_and_skip_dropped():
    obs = box_obs(range(0, 10), 1, A_OUT) + box_obs(range(10, 20), 2, A_OUT) + box_obs(range(20), 3, B_OUT)
    an = make_an([5] * 20 + [0] * 20)                  # tracks 1 and 2 merged into person 5; track 3 dropped
    ids = me.match_ids(make_res(20, obs), an, make_gt({7: A_SRC, 8: B_SRC}, 20), use_persons=True)
    assert ids == {5: 7}


def test_match_ids_empty_inputs():
    res = make_res(5, [])
    assert me.match_ids(res, None, make_gt({1: A_SRC}, 5), False) == {}
    assert me.match_ids(make_res(5, box_obs(range(5), 1, A_OUT)), None, {}, False) == {}


# ------------------------------------------------------------------------------------------------ tracking_metrics

def test_tracking_perfect():
    res = make_res(10, box_obs(range(10), 1, A_OUT) + box_obs(range(10), 2, B_OUT))
    gt = make_gt({1: A_SRC, 2: B_SRC}, 10)
    m = me.tracking_metrics(res, None, gt, use_persons=False)
    assert m["mota"] == pytest.approx(1.0) and m["idf1"] == pytest.approx(1.0)
    assert m["num_switches"] == 0 and m["precision"] == pytest.approx(1.0) and m["recall"] == pytest.approx(1.0)
    assert m["num_frames"] == 10


def test_tracking_counts_id_switch_and_matches_motmetrics_formula():
    obs = box_obs(range(0, 5), 1, A_OUT) + box_obs(range(5, 10), 2, A_OUT)
    m = me.tracking_metrics(make_res(10, obs), None, make_gt({1: A_SRC}, 10), False)
    assert m["num_switches"] == 1
    assert m["mota"] == pytest.approx(1 - 1 / 10)                     # (FN + FP + IDSW) / GT = 1 / 10
    assert m["idf1"] == pytest.approx(0.5)                            # idtp 5, idfp 5, idfn 5 -> 10 / 20


def test_tracking_persons_undo_the_fragmentation():
    obs = box_obs(range(0, 5), 1, A_OUT) + box_obs(range(5, 10), 2, A_OUT)
    gt = make_gt({1: A_SRC}, 10)
    m = me.tracking_metrics(make_res(10, obs), make_an([4] * 10), gt, use_persons=True)
    assert m["num_switches"] == 0 and m["idf1"] == pytest.approx(1.0) and m["mota"] == pytest.approx(1.0)


def test_tracking_unannotated_people_count_as_false_positives_and_misses_lower_recall():
    obs = box_obs(range(10), 1, A_OUT) + box_obs(range(10), 9, (400, 300, 450, 400))   # 9: nobody annotated there
    m = me.tracking_metrics(make_res(10, obs), None, make_gt({1: A_SRC, 2: B_SRC}, 10), False)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["mota"] == pytest.approx(1 - (10 + 10) / 20)


def test_tracking_needs_iou_half():
    loose = (70, 50, 120, 150)                         # IoU with A_OUT = 30*100 / 7000 = 0.43 -> no match
    m = me.tracking_metrics(make_res(5, box_obs(range(5), 1, loose)), None, make_gt({1: A_SRC}, 5), False)
    assert m["recall"] == 0 and m["num_false_positives"] == 5 and m["num_misses"] == 5


def _scalar_iou_dist(a, b, max_dist=0.5):
    """1 - IoU of two x1y1x2y2 boxes, nan above ``max_dist`` (motmetrics' own helper needs numpy < 2)."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    d = 1.0 - inter / union
    return d if d <= max_dist else float("nan")


def test_tracking_agrees_with_the_motmetrics_reference_on_a_noisy_scenario():
    import motmetrics as mm
    rng = np.random.default_rng(3)
    n = 60
    gt_box = lambda a, f: (100 + 60 * a + 3 * f, 100 + 5 * a, 160 + 60 * a + 3 * f, 300 + 5 * a)   # noqa: E731
    gt = make_gt({a: (lambda f, a=a: gt_box(a, f // 6)) for a in (1, 2, 3)}, n)
    obs = []
    for f in range(n):
        for a in (1, 2, 3):
            if rng.random() < 0.15:
                continue                                                     # a missed detection
            x1, y1, x2, y2 = (np.array(gt_box(a, f)) + rng.uniform(-9, 9, 4)) * 0.5
            track = a * 10 + (f // 20 if a == 1 else 0) + (3 if (a in (2, 3) and f >= 40) else 0) * (a == 2)
            obs.append((f, track if a != 3 else 33, (x1, y1, x2, y2)))
        if f % 7 == 0:
            obs.append((f, 99, (700, 400, 760, 520)))                       # never annotated
    res = make_res(n, obs)
    mine = me.tracking_metrics(res, None, gt, use_persons=False)

    ref = mm.MOTAccumulator(auto_id=False)
    for f in range(n):
        g = [b for b in gt.get(int(res.src_frame_idx[f]), [])]
        h = [(o[1], o[2]) for o in obs if o[0] == f]
        d = np.array([[_scalar_iou_dist(tuple(0.5 * np.array(b.bbox)), hb) for _, hb in h] for b in g]).reshape(len(g), len(h))
        ref.update([b.actor for b in g], [t for t, _ in h], d, frameid=f)
    want = mm.metrics.create().compute(ref, metrics=["mota", "idf1", "num_switches", "precision", "recall"],
                                       name="r").loc["r"]
    for key in ("mota", "idf1", "precision", "recall"):
        assert mine[key] == pytest.approx(float(want[key])), key
    assert mine["num_switches"] == int(want["num_switches"]) >= 2


def test_tracking_empty_does_not_crash():
    m = me.tracking_metrics(make_res(5, []), None, {}, False)
    assert m["mota"] is None and m["idf1"] is None and m["precision"] is None and m["recall"] is None
    assert m["num_frames"] == 5
    m = me.tracking_metrics(make_res(5, []), None, make_gt({1: A_SRC}, 5), False)       # nobody found
    assert m["recall"] == 0 and m["mota"] == pytest.approx(0.0) and m["precision"] is None
    json.dumps(m)                                      # plain JSON types only


def test_gt_duplicate_share_counts_boxes_with_a_twin_in_the_same_frame():
    res = make_res(10, [])
    twin = (102, 100, 202, 300)                                        # IoU with A_SRC = 98*200 / (2*20000 - 19600) = 0.96
    gt = make_gt({1: A_SRC, 2: twin, 3: B_SRC}, 10)
    assert me.gt_duplicate_share(res, gt) == pytest.approx(2 / 3)
    assert me.gt_duplicate_share(res, make_gt({1: A_SRC, 3: B_SRC}, 10)) == 0.0
    assert me.gt_duplicate_share(res, {}) is None
    assert me.gt_duplicate_counts(res, gt) == (20, 30)


# ------------------------------------------------------------------------------------------------ pair_intervals

def pair_scene(n, sample_fps, positions):
    """positions: {pid: f(frame) -> (x, z) or None}; returns (an, res) with one observation per present person."""
    frames, pids, xz = [], [], []
    for f in range(n):
        for pid, pos in positions.items():
            p = pos(f)
            if p is not None:
                frames.append(f)
                pids.append(pid)
                xz.append(p)
    res = NS(frame_times=np.arange(n) / sample_fps, obs_frame=np.array(frames, np.int64))
    return make_an(pids, xz), res


def test_pair_intervals_basic_and_ordering():
    an, res = pair_scene(30, 1.0, {2: lambda f: (0.0, 0.0), 1: lambda f: (1.0, 0.0), 3: lambda f: (9.0, 0.0)})
    assert me.pair_intervals(an, res, dist=2.0, min_dur=8.0, gap_tol=1.5) == [(1, 2, 0.0, 29.0)]


def test_pair_intervals_distance_and_min_duration():
    an, res = pair_scene(30, 1.0, {1: lambda f: (0.0, 0.0), 2: lambda f: (3.0, 0.0)})
    assert me.pair_intervals(an, res, 2.0, 8.0, 1.5) == []
    an, res = pair_scene(30, 1.0, {1: lambda f: (0.0, 0.0), 2: lambda f: (1.0, 0.0) if f < 7 else (9.0, 0.0)})
    assert me.pair_intervals(an, res, 2.0, 8.0, 1.5) == []            # 6 s < 8 s
    assert me.pair_intervals(an, res, 2.0, 6.0, 1.5) == [(1, 2, 0.0, 6.0)]


def test_pair_intervals_bridge_small_gaps_and_split_large_ones():
    def near(missing):
        return lambda f: None if f in missing else (1.0, 0.0)
    # 5 fps: a 1.0 s hole between True samples at 4.0 s and 5.2 s is bridged (1.2 <= 1.5) ...
    an, res = pair_scene(100, 5.0, {1: lambda f: (0.0, 0.0), 2: near(set(range(21, 26)))})
    assert me.pair_intervals(an, res, 2.0, 8.0, 1.5) == [(1, 2, 0.0, 19.8)]
    # ... a 2.2 s hole (10 samples) splits it into two intervals
    an, res = pair_scene(100, 5.0, {1: lambda f: (0.0, 0.0), 2: near(set(range(41, 51)))})
    got = me.pair_intervals(an, res, 2.0, 8.0, 1.5)
    assert [(a, b) for a, b, _, _ in got] == [(1, 2), (1, 2)]
    assert got[0][3] == pytest.approx(8.0) and got[1][2] == pytest.approx(10.2)


def test_pair_intervals_ignore_dropped_persons():
    an, res = pair_scene(30, 1.0, {0: lambda f: (0.0, 0.0), 1: lambda f: (0.5, 0.0)})
    assert me.pair_intervals(an, res, 2.0, 8.0, 1.5) == []


def test_pair_intervals_match_the_event_engine_streaks():
    th = Thresholds(engage_dist=2.0, engage_time=8.0, gap_tol=1.5)
    xs = lambda f: (1.0, 0.0) if (f < 30 or 33 <= f < 60 or 66 <= f < 100) else (8.0, 0.0)    # noqa: E731
    an, res = pair_scene(100, 5.0, {1: lambda f: (0.0, 0.0), 2: xs})
    ours = me.pair_intervals(an, res, 2.0, 8.0, 1.5)
    timeline = [(f / 5.0, [PersonState(1, (0.0, 0.0), "staff"), PersonState(2, xs(f), "customer")])
                for f in range(100)]
    engaged = [(e.t, e.t_end) for e in run_events(timeline, th) if e.type == "engagement"]
    assert [(a, b, t0, t1) for a, b, t0, t1 in ours] == [(1, 2, t0, t1) for t0, t1 in engaged]
    assert len(ours) == 1 and ours[0][3] > 19.0        # the two short holes were bridged in both


# ----------------------------------------------------------------------------------------------------- talk_eval

def talk_act(f0=300, f1=900, actors=(1, 2), spans=None, act_id=1):
    return act(TALK, f0, f1, actors, spans, act_id)


def test_talk_eval_offset_by_two_seconds_is_a_match():
    r = me.talk_eval([(101, 102, 12.0, 32.0)], [talk_act()], {101: 1, 102: 2}, FPS)     # GT 10-30 s
    assert r["n_gt"] == 1 and r["recall_all"] == 1.0 and r["recall_tracked"] == 1.0
    assert r["precision_mapped"] == 1.0 and r["n_det"] == 1 and r["n_det_mapped"] == 1
    assert r["median_start_err_s"] == pytest.approx(2.0)
    assert r["tiou_0.1"]["recall_all"] == 1.0


def test_talk_eval_pair_may_be_reported_in_either_order():
    r = me.talk_eval([(102, 101, 10.0, 30.0)], [talk_act()], {101: 1, 102: 2}, FPS)
    assert r["recall_all"] == 1.0 and r["median_start_err_s"] == 0.0


def test_talk_eval_non_overlapping_is_a_miss():
    r = me.talk_eval([(101, 102, 40.0, 60.0)], [talk_act()], {101: 1, 102: 2}, FPS)
    assert r["recall_all"] == 0.0 and r["recall_tracked"] == 0.0 and r["precision_mapped"] == 0.0
    assert r["median_start_err_s"] is None
    assert r["tiou_0.1"]["recall_all"] == 0.0


def test_talk_eval_tiou_threshold_and_secondary_line():
    # GT 10-30 s (20 s); detection 26-46 s: intersection 4 s, union 36 s -> tIoU 0.11
    r = me.talk_eval([(101, 102, 26.0, 46.0)], [talk_act()], {101: 1, 102: 2}, FPS)
    assert r["recall_all"] == 0.0 and r["tiou_0.1"]["recall_all"] == 1.0
    assert r["tiou_0.1"]["median_start_err_s"] == pytest.approx(16.0)


def test_talk_eval_wrong_pair_does_not_match():
    r = me.talk_eval([(101, 103, 10.0, 30.0)], [talk_act()], {101: 1, 102: 2, 103: 3}, FPS)
    assert r["recall_all"] == 0.0 and r["precision_mapped"] == 0.0


def test_talk_eval_group_expands_to_all_pairs_and_intersects_actor_spans():
    group = talk_act(0, 900, (1, 2, 3), spans={1: (0, 900), 2: (0, 900), 3: (300, 600)})
    ids = {11: 1, 12: 2, 13: 3}
    dets = [(11, 12, 0.0, 30.0), (11, 13, 10.0, 20.0), (12, 13, 10.0, 20.0)]
    r = me.talk_eval(dets, [group], ids, FPS)
    assert r["n_gt"] == 3 and r["recall_all"] == 1.0 and r["precision_mapped"] == 1.0
    assert r["median_start_err_s"] == 0.0
    # actor 3 outside the activity's own span window of the others: the pairs with 3 disappear
    gone = talk_act(0, 900, (1, 2, 3), spans={1: (0, 900), 2: (0, 900), 3: (1000, 1100)})
    assert me.talk_eval([], [gone], ids, FPS)["n_gt"] == 1


def test_talk_eval_actor_without_span_is_skipped_and_empty_gt_is_none():
    r = me.talk_eval([], [talk_act(spans={1: (300, 900)})], {}, FPS)         # actor 2 absent
    assert r["n_gt"] == 0 and r["recall_all"] is None and r["recall_tracked"] is None


def test_talk_eval_fragmented_person_ids_still_match():
    r = me.talk_eval([(101, 202, 12.0, 32.0)], [talk_act()], {101: 1, 102: 1, 201: 2, 202: 2}, FPS)
    assert r["recall_all"] == 1.0


def test_talk_eval_recall_tracked_excludes_pairs_with_an_unmapped_actor():
    acts = [talk_act(300, 900, (1, 2), act_id=1), talk_act(1200, 1800, (3, 4), act_id=2)]
    ids = {101: 1, 102: 2, 103: 3}                                              # nobody maps to actor 4
    r = me.talk_eval([(101, 102, 10.0, 30.0)], acts, ids, FPS)
    assert r["n_gt"] == 2 and r["n_gt_tracked"] == 1
    assert r["recall_all"] == 0.5 and r["recall_tracked"] == 1.0


def test_talk_eval_precision_counts_only_detections_between_mapped_persons():
    ids = {101: 1, 102: 2}
    dets = [(101, 102, 10.0, 30.0), (101, 999, 10.0, 30.0), (101, 102, 100.0, 120.0)]
    r = me.talk_eval(dets, [talk_act()], ids, FPS)
    assert r["n_det"] == 3 and r["n_det_mapped"] == 2
    assert r["precision_mapped"] == pytest.approx(0.5)


def test_talk_eval_matching_is_one_to_one():
    dets = [(101, 102, 10.0, 30.0), (101, 102, 11.0, 31.0)]                     # both overlap the single GT pair
    r = me.talk_eval(dets, [talk_act()], {101: 1, 102: 2}, FPS)
    assert r["recall_all"] == 1.0 and r["precision_mapped"] == pytest.approx(0.5)
    two_gt = [talk_act(300, 600, (1, 2), act_id=1), talk_act(630, 900, (1, 2), act_id=2)]
    r = me.talk_eval([(101, 102, 10.0, 30.0)], two_gt, {101: 1, 102: 2}, FPS, tiou_thr=0.3)
    assert r["n_gt"] == 2 and r["recall_all"] == 0.5


def test_talk_eval_ignores_other_activities_and_clips_to_t_max():
    other = act("person_opens_facility_door", 0, 900, (1, 2))
    r = me.talk_eval([], [other], {}, FPS)
    assert r["n_gt"] == 0
    r = me.talk_eval([(101, 102, 10.0, 20.0)], [talk_act(300, 900)], {101: 1, 102: 2}, FPS, t_max=20.0)
    assert r["n_gt"] == 1 and r["recall_all"] == 1.0                            # GT clipped to 10-20 s
    r = me.talk_eval([], [talk_act(900, 1200)], {}, FPS, t_max=20.0)           # starts after the limit
    assert r["n_gt"] == 0


# -------------------------------------------------------------------------------------------- enter_exit_eval

def timelines(**present):
    return {int(k[1:]): {"present": v} for k, v in present.items()}


def enter(f0, f1, actor, act_id=1):
    return act(ENTER, f0, f1, [actor], act_id=act_id)


def leave(f0, f1, actor, act_id=1):
    return act(EXIT, f0, f1, [actor], act_id=act_id)


def test_enter_exit_matched_within_window_and_missed_outside():
    an = make_an([], timelines=timelines(p1=[[10.0, 40.0]], p2=[[16.0, 50.0]]))
    acts = [enter(270, 330, 5), leave(1110, 1230, 5, 2),                        # GT 9.0 s enter, 41.0 s exit: errs 1, 1
            enter(270, 330, 6, 3), leave(1110, 1230, 6, 4)]                    # GT 9.0 s enter vs 16: miss; 41 vs 50: miss
    r = me.enter_exit_eval(an, acts, {1: 5, 2: 6}, FPS, t_end=60.0)
    assert r["enter"]["n_gt"] == 2 and r["enter"]["n_matched"] == 1
    assert r["exit"]["n_gt"] == 2 and r["exit"]["n_matched"] == 1
    assert r["all"]["recall_all"] == 0.5 and r["all"]["recall_tracked"] == 0.5
    assert r["all"]["median_abs_err_s"] == pytest.approx(1.0)


def test_enter_exit_window_is_inclusive_at_five_seconds():
    an = make_an([], timelines=timelines(p1=[[14.0, 36.0]]))
    r = me.enter_exit_eval(an, [enter(270, 330, 5), leave(1110, 1230, 5, 2)], {1: 5}, FPS, t_end=60.0)
    assert r["all"]["n_matched"] == 2 and r["all"]["median_abs_err_s"] == pytest.approx(5.0)
    r = me.enter_exit_eval(an, [enter(264, 330, 5), leave(1110, 1236, 5, 2)], {1: 5}, FPS, t_end=60.0)
    assert r["all"]["n_matched"] == 0                                           # 5.2 s apart


def test_enter_exit_unmapped_actor_counts_in_recall_all_only():
    an = make_an([], timelines=timelines(p1=[[10.0, 40.0]]))
    acts = [enter(270, 330, 5), enter(270, 330, 6, 2)]
    r = me.enter_exit_eval(an, acts, {1: 5}, FPS, t_end=60.0)
    assert r["enter"]["n_gt"] == 2 and r["enter"]["n_gt_tracked"] == 1
    assert r["enter"]["recall_all"] == 0.5 and r["enter"]["recall_tracked"] == 1.0


def test_enter_exit_uses_the_presence_interval_nearest_the_event():
    an = make_an([], timelines=timelines(p1=[[10.0, 20.0], [50.0, 70.0]]))
    r = me.enter_exit_eval(an, [enter(1440, 1500, 5), leave(2040, 2130, 5, 2)], {1: 5}, FPS, t_end=100.0)
    # GT enter at 48.0 -> nearest start is 50.0 (err 2); GT exit at 71.0 -> nearest end is 70.0 (err 1)
    assert r["all"]["n_matched"] == 2 and r["enter"]["median_abs_err_s"] == pytest.approx(2.0)
    assert r["exit"]["median_abs_err_s"] == pytest.approx(1.0)


def test_enter_exit_fragments_of_one_actor_are_pooled():
    an = make_an([], timelines=timelines(p1=[[10.0, 20.0]], p2=[[21.0, 40.0]]))
    r = me.enter_exit_eval(an, [enter(270, 330, 5), leave(1110, 1230, 5, 2)], {1: 5, 2: 5}, FPS, t_end=60.0)
    assert r["enter"]["median_abs_err_s"] == pytest.approx(1.0) and r["exit"]["median_abs_err_s"] == pytest.approx(1.0)


def test_enter_exit_ignores_events_within_a_second_of_the_clip_ends():
    an = make_an([], timelines=timelines(p1=[[0.0, 59.0]]))
    acts = [enter(10, 40, 5),                    # enters at 0.33 s: already inside at the start
            leave(1700, 1790, 5, 2),             # exits at 59.67 s: the clip ends
            enter(150, 200, 5, 3)]               # 5.0 s: counted
    r = me.enter_exit_eval(an, acts, {1: 5}, FPS, t_end=60.0)
    assert r["all"]["n_gt"] == 1
    assert me.enter_exit_eval(an, acts, {1: 5}, FPS, t_end=60.0, edge_s=0.0)["all"]["n_gt"] == 3


def test_enter_exit_events_past_the_processed_part_are_dropped():
    an = make_an([], timelines=timelines(p1=[[10.0, 40.0]]))
    r = me.enter_exit_eval(an, [enter(270, 330, 5), enter(3000, 3060, 5, 2)], {1: 5}, FPS, t_end=60.0)
    assert r["enter"]["n_gt"] == 1


def test_enter_exit_empty_gives_none_not_zero_division():
    r = me.enter_exit_eval(make_an([]), [], {}, FPS, t_end=60.0)
    assert r["all"]["n_gt"] == 0 and r["all"]["recall_all"] is None and r["all"]["median_abs_err_s"] is None
    json.dumps(r)


# ------------------------------------------------------------------------------------------------- aggregation

def clip_result(name, talk, ee, trk_p, trk_t):
    return {"clip": name, "tracks": {"tracking": trk_t}, "persons": {"tracking": trk_p, "talk": talk,
                                                                     "enter_exit": ee}}


def test_aggregate_is_micro_averaged_and_pools_errors():
    a_talk = me.talk_eval([(1, 2, 12.0, 32.0)], [talk_act()], {1: 1, 2: 2}, FPS)                  # 1/1, err 2
    b_acts = [talk_act(300, 900, (1, 2), act_id=i) for i in range(1, 4)]
    b_talk = me.talk_eval([(1, 2, 10.0, 30.0)], b_acts, {1: 1, 2: 2}, FPS)                         # 1/3, err 0
    assert a_talk["recall_all"] == 1.0 and b_talk["recall_all"] == pytest.approx(1 / 3)
    ee_a = me.enter_exit_eval(make_an([], timelines=timelines(p1=[[10.0, 40.0]])), [enter(270, 330, 5)],
                              {1: 5}, FPS, t_end=60.0)
    ee_b = me.enter_exit_eval(make_an([], timelines=timelines(p1=[[11.0, 40.0]])),
                              [enter(270, 330, 5), enter(300, 330, 6, 2), enter(330, 360, 7, 3)], {1: 5}, FPS,
                              t_end=60.0)
    res = make_res(10, box_obs(range(10), 1, A_OUT))
    trk = me.tracking_metrics(res, None, make_gt({1: A_SRC}, 10), False)
    clips = [{**clip_result("a", a_talk, ee_a, trk, trk), "n_gt_boxes": 10, "n_gt_dup_boxes": 4},
             {**clip_result("b", b_talk, ee_b, trk, trk), "n_gt_boxes": 30, "n_gt_dup_boxes": 0}]
    total = me.aggregate(clips)
    assert total["gt_dup_share"] == pytest.approx(4 / 40)
    talk = total["persons"]["talk"]
    assert talk["n_gt"] == 4 and talk["recall_all"] == pytest.approx(2 / 4)                        # (1 + 1) / (1 + 3)
    assert talk["median_start_err_s"] == pytest.approx(1.0)                                        # median(2, 0)
    assert talk["n_det"] == 2 and talk["precision_mapped"] == 1.0
    ee = total["persons"]["enter_exit"]["all"]
    assert ee["n_gt"] == 4 and ee["n_matched"] == 2 and ee["recall_all"] == pytest.approx(0.5)
    assert ee["median_abs_err_s"] == pytest.approx(1.5)                                            # median(1, 2)
    t = total["persons"]["tracking"]
    assert t["num_frames"] == 20 and t["mota"] == pytest.approx(1.0) and t["idf1"] == pytest.approx(1.0)
    assert total["n_clips"] == 2


def test_aggregate_mota_is_computed_from_summed_counts():
    good = make_res(10, box_obs(range(10), 1, A_OUT))                            # 10 GT, perfect
    bad = make_res(10, box_obs(range(5), 1, A_OUT))                              # 10 GT, 5 misses
    gt = make_gt({1: A_SRC}, 10)
    t1, t2 = me.tracking_metrics(good, None, gt, False), me.tracking_metrics(bad, None, gt, False)
    total = me.aggregate([clip_result("a", {}, {}, t1, t1), clip_result("b", {}, {}, t2, t2)])
    assert total["tracks"]["tracking"]["mota"] == pytest.approx(1 - 5 / 20)


def test_aggregate_of_nothing():
    assert me.aggregate([])["n_clips"] == 0


# -------------------------------------------------------------------------------------------------- splits / CLI

CLIPS_CSV = """clip,camera,location,video_url,annotation_path,talk_count,enter_count,exit_count,video_bytes,fps_guess
c1.G1,G1,school,http://x/c1.avi,data/meva/annotations/c1.G1,5,1,1,10,30.0
c2.G2,G2,school,http://x/c2.avi,data/meva/annotations/c2.G2,6,0,0,10,30.0
c3.G3,G3,bus,http://x/c3.avi,data/meva/annotations/c3.G3,7,2,2,10,30.0
"""


@pytest.fixture
def csv_and_split(tmp_path):
    clips = tmp_path / "clips.csv"
    clips.write_text(CLIPS_CSV, encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["c1.G1", "c2.G2"], "test": ["c3.G3"]}), encoding="utf-8")
    return clips, split


def test_select_clips(csv_and_split):
    clips, split = csv_and_split
    rows = me.read_clips(clips)
    assert [r["clip"] for r in rows] == ["c1.G1", "c2.G2", "c3.G3"]
    pick = lambda which, f=split: [r["clip"] for r in me.select_clips(rows, which, f)]      # noqa: E731
    assert pick("dev") == ["c1.G1", "c2.G2"] and pick("test") == ["c3.G3"]
    assert pick("all") == ["c1.G1", "c2.G2", "c3.G3"]
    assert [r["clip"] for r in me.select_clips(rows, "all", None)] == ["c1.G1", "c2.G2", "c3.G3"]
    with pytest.raises(ValueError, match="split file"):
        me.select_clips(rows, "dev", None)
    split.write_text(json.dumps({"dev": ["nope"], "test": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        me.select_clips(rows, "dev", split)


def test_select_clips_all_with_a_split_file_means_the_clips_listed_in_it(csv_and_split):
    clips, split = csv_and_split
    split.write_text(json.dumps({"dev": ["c2.G2"], "test": []}), encoding="utf-8")
    assert [r["clip"] for r in me.select_clips(me.read_clips(clips), "all", split)] == ["c2.G2"]


def fake_clip_eval(clip, tag=1.0):
    trk = {"mota": tag, "idf1": tag, "num_switches": 0, "precision": tag, "recall": tag, "num_frames": 10,
           "num_objects": 10, "num_misses": 0, "num_false_positives": 0, "num_detections": 10, "idtp": 10,
           "idfp": 0, "idfn": 0}
    talk = me.talk_eval([(1, 2, 10.0, 30.0)], [talk_act()], {1: 1, 2: 2}, FPS)
    ee = me.enter_exit_eval(make_an([], timelines=timelines(p1=[[10.0, 40.0]])), [enter(270, 330, 5)], {1: 5}, FPS,
                            t_end=60.0)
    return {"clip": clip, "duration_s": 60.0, "cached": True, "timings": {"total": 1.0},
            **clip_result(clip, talk, ee, trk, trk)}


def test_main_writes_json_and_prints_table(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    seen = []

    def fake_eval(video, ann, stack, cache_root, **kw):
        seen.append((str(video), str(ann), kw))
        return fake_clip_eval(video.name.split(".r13")[0])

    monkeypatch.setattr(me, "eval_clip", fake_eval)
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    monkeypatch.setattr(me, "_files_exist", lambda *p: True)
    out = tmp_path / "out" / "meva_dev.json"
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--out", str(out),
                  "--preset", "short", "--dist", "2.5", "--min-dur", "6", "--gap-tol", "1.0", "--merge-thr", "0.4",
                  "--limit-seconds", "60"])
    assert rc == 0
    assert [Path_name(s[0]) for s in seen] == ["c1.G1.r13.avi", "c2.G2.r13.avi"]
    kw = seen[0][2]
    assert kw["dist"] == 2.5 and kw["min_dur"] == 6.0 and kw["gap_tol"] == 1.0 and kw["merge_thr"] == 0.4
    assert kw["preset"] == "short" and kw["limit_seconds"] == 60.0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [c["clip"] for c in data["clips"]] == ["c1.G1", "c2.G2"]
    assert data["total"]["n_clips"] == 2 and data["split"] == "dev"
    assert data["params"]["dist"] == 2.5 and data["notes"]
    text = capsys.readouterr().out
    assert "c1.G1" in text and "c2.G2" in text and "TOTAL" in text and "MOTA" in text


def Path_name(p):
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def test_main_usage_errors_return_2(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(tmp_path / "missing.json"),
                  "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "split file" in capsys.readouterr().err
    rc = me.main(["--clips", str(tmp_path / "none.csv"), "--split", "all", "--out", str(tmp_path / "o.json")])
    assert rc == 2


def test_main_skips_clips_with_missing_files_and_fails_if_nothing_ran(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    monkeypatch.setattr(me, "_files_exist", lambda *p: False)
    out = tmp_path / "o.json"
    assert me.main(["--clips", str(clips), "--split", "all", "--split-file", str(split), "--out", str(out)]) == 1
    assert "missing" in capsys.readouterr().err.lower()


def test_main_empty_split_is_a_usage_error(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    split.write_text(json.dumps({"dev": [], "test": ["c3.G3"]}), encoding="utf-8")
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    assert me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split),
                    "--out", str(tmp_path / "o.json")]) == 2
    assert "no clips" in capsys.readouterr().err


def test_analyze_passes_merge_thr_through_and_restores_build_persons(monkeypatch):
    from showroom_coach import analysis
    seen = {}

    def spy(res, merge_thr=0.3, **kw):
        seen["thr"] = merge_thr
        raise RuntimeError("stop here")

    monkeypatch.setattr(analysis, "build_persons", spy)
    with pytest.raises(RuntimeError, match="stop here"):
        me._analyze(NS(video=NS(duration=1.0)), "short", 0.41)
    assert seen["thr"] == 0.41 and analysis.build_persons is spy


# ------------------------------------------------------------------------------------------------ end to end

def _kpf_geom(path, boxes):
    lines = ["- {'meta': 'synthetic'}"]
    for frame, actor, (x1, y1, x2, y2) in boxes:
        lines.append(f"- {{'geom': {{'g0': '{x1} {y1} {x2} {y2}', 'id0': 1, 'id1': {actor}, 'keyframe': True, "
                     f"'ts0': {frame}}}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_eval_clip_end_to_end_on_the_synthetic_meet_video(meet_video, tmp_path):
    from showroom_coach.perception.stack import fake_stack
    # meet_video: 20 s at 30 fps, 1280x720. Blue (actor 1) stands at x 600..680; red (actor 2) walks in 4-8 s, stays
    # at x 470..550 until 18 s, then walks out. Both are 80x220 at y 250.
    from tests.conftest import _customer_x1
    boxes = []
    for f in range(600):
        t = f / 30.0
        boxes.append((f, 1, (600, 250, 680, 470)))
        x1 = _customer_x1(t)
        if x1 is not None:
            boxes.append((f, 2, (x1, 250, x1 + 80, 470)))
    prefix = tmp_path / "meet"
    _kpf_geom(tmp_path / "meet.geom.yml", boxes)
    (tmp_path / "meet.types.yml").write_text(
        "- {'types': {'cset3': {'person': 1.0}, 'id1': 1}}\n- {'types': {'cset3': {'person': 1.0}, 'id1': 2}}\n",
        encoding="utf-8")
    span = "{'tsr0': [240, 540]}"
    (tmp_path / "meet.activities.yml").write_text(
        "- {'act': {'act2': {'person_talks_to_person': 1.0}, 'actors': [{'id1': 1, 'timespan': [" + span + "]}, "
        "{'id1': 2, 'timespan': [" + span + "]}], 'id2': 4, 'src_status': 'good', 'timespan': [" + span + "]}}\n"
        "- {'act': {'act2': {'person_enters_scene_through_structure': 1.0}, 'actors': [{'id1': 2, 'timespan': "
        "[{'tsr0': [120, 150]}]}], 'id2': 5, 'src_status': 'good', 'timespan': [{'tsr0': [120, 150]}]}}\n",
        encoding="utf-8")

    r = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", preset="short", dist=2.0, min_dur=8.0,
                     gap_tol=1.5, merge_thr=0.30)
    json.dumps(r)
    assert r["clip"] == "meet"
    for variant in ("tracks", "persons"):
        t = r[variant]["tracking"]
        assert t["recall"] > 0.9 and t["mota"] > 0.8 and t["num_switches"] == 0, (variant, t)
    assert r["persons"]["id_map"]
    assert sorted(set(r["persons"]["id_map"].values())) == [1, 2]
    talk = r["persons"]["talk"]
    assert talk["n_gt"] == 1 and talk["n_gt_tracked"] == 1
    assert talk["recall_all"] == 1.0 and talk["precision_mapped"] == 1.0
    assert 0.0 <= talk["median_start_err_s"] <= 4.0
    ee = r["persons"]["enter_exit"]["enter"]
    assert ee["n_gt"] == 1 and ee["n_matched"] == 1                  # GT enter at 4.0 s; the red box appears at 4.0 s

    short = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", limit_seconds=10.0)
    assert short["duration_s"] == pytest.approx(10.0, abs=0.3)
    assert short["persons"]["talk"]["n_gt"] == 1                      # GT clipped to 8-10 s
