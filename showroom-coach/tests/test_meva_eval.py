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


def test_match_ids_strict_rule_needs_three_votes_and_thirty_percent():
    obs = (box_obs(range(2), 5, C_OUT)                                     # 2 obs, both matched: below 3 votes
           + box_obs(range(20), 6, B_OUT)                                  # 20 obs, actor 2 visible on 3 samples only
           + box_obs(range(10), 7, (800, 800, 900, 900))                   # never on anyone
           + box_obs(range(10), 8, A_OUT))                                 # always on actor 1
    gt = make_gt({1: A_SRC, 2: lambda f: B_SRC if f // 6 < 3 else None, 3: C_SRC}, 20)
    res = make_res(20, obs)
    strict = me.match_ids(res, None, gt, False, min_actor_share=None)
    assert strict == {8: 1}                                                # 6: 3 votes of 20 frames = 15% < 30%
    # the default rule also maps 6: actor 2 is annotated on 3 frames and the track is on it for all 3
    assert me.match_ids(res, None, gt, False) == {6: 2, 8: 1}


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


def actor_on(k0, overlap, annotated):
    """Actor annotated on samples k0 .. k0+annotated-1; on A_SRC for the first `overlap` of them, on B_SRC after."""
    def box(f):
        k = f // 6 - k0
        return (A_SRC if k < overlap else B_SRC) if 0 <= k < annotated else None
    return box


def test_match_ids_briefly_annotated_actor_maps_on_its_own_frames():
    # a person followed for 40 frames; actor 5 is annotated on 6 of them and the person is on it for all 6:
    # 6 votes are 15% of the person's frames (strict: no) but 100% of the actor's annotated frames
    res = make_res(40, box_obs(range(40), 1, A_OUT))
    gt = make_gt({5: actor_on(10, 6, 6)}, 40)
    assert me.match_ids(res, None, gt, False, min_actor_share=None) == {}
    assert me.match_ids(res, None, gt, False) == {1: 5}


def test_match_ids_actor_share_is_half_of_the_actors_annotated_frames_and_votes_floor_stays_three():
    res = make_res(40, box_obs(range(40), 1, A_OUT))
    assert me.match_ids(res, None, make_gt({5: actor_on(10, 3, 6)}, 40), False) == {1: 5}   # 3 / 6 = 50%
    assert me.match_ids(res, None, make_gt({5: actor_on(10, 3, 7)}, 40), False) == {}       # 3 / 7 = 43%
    assert me.match_ids(res, None, make_gt({5: actor_on(10, 2, 2)}, 40), False) == {}       # 100% but only 2 votes


def test_match_ids_uses_the_most_voted_actor_even_when_another_would_pass_the_actor_share():
    res = make_res(40, box_obs(range(40), 1, A_OUT))
    # actor 1: annotated on all 40 frames, overlapped on 5 (12% / 12%); actor 2: annotated on 4, overlapped on 4
    gt = make_gt({1: actor_on(0, 5, 40), 2: actor_on(10, 4, 4)}, 40)
    assert me.match_ids(res, None, gt, False) == {}


def test_match_ids_actor_frames_are_counted_inside_the_evaluated_window_only():
    # 20 sampled frames; the actor is annotated on samples 14-40, so 6 of them lie in the window and the person is on
    # it for 4: 4 / 6 passes, while 4 / 27 (the whole annotation) or 4 / 20 of the person's frames would not
    res = make_res(20, box_obs(range(20), 1, A_OUT))
    gt = make_gt({5: actor_on(14, 4, 27)}, 41)
    assert me.match_ids(res, None, gt, False, min_actor_share=None) == {}
    assert me.match_ids(res, None, gt, False) == {1: 5}


# ---------------------------------------------------------------------------------------------------- twin_clusters

TWIN_SRC = (102, 100, 202, 300)                          # IoU with A_SRC = 98*200 / (2*20000 - 19600) = 0.96


def test_twin_clusters_pair_up_actors_with_overlapping_boxes():
    res = make_res(10, [])
    gt = make_gt({1: A_SRC, 2: TWIN_SRC, 3: B_SRC}, 10)
    assert me.twin_clusters(res, gt) == {1: 1, 2: 1}


def test_twin_clusters_none_when_the_boxes_differ_or_no_actor_is_annotated_twice():
    res = make_res(10, [])
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: (150, 100, 250, 300)}, 10)) == {}      # IoU 0.33
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: B_SRC}, 10)) == {}
    assert me.twin_clusters(res, {}) == {}


def test_twin_clusters_need_half_of_the_co_annotated_frames_and_at_least_three_of_them():
    res = make_res(10, [])
    def twin_on(n_twin):                                  # twin on A for the first n samples, elsewhere afterwards
        return lambda f: TWIN_SRC if f // 6 < n_twin else B_SRC
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: twin_on(5)}, 10)) == {1: 1, 2: 1}      # 5 / 10
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: twin_on(4)}, 10)) == {}                # 4 / 10
    short = lambda f: TWIN_SRC if f // 6 < 2 else None                                       # noqa: E731
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: short}, 10)) == {}                     # only 2 co-annotated
    short3 = lambda f: TWIN_SRC if f // 6 < 3 else None                                      # noqa: E731
    assert me.twin_clusters(res, make_gt({1: A_SRC, 2: short3}, 10)) == {1: 1, 2: 1}


def test_twin_clusters_brief_twin_of_a_long_lived_actor():
    res = make_res(40, [])
    gt = make_gt({3: A_SRC, 9: lambda f: TWIN_SRC if 20 <= f // 6 < 24 else None}, 40)
    assert me.twin_clusters(res, gt) == {3: 3, 9: 3}


def test_twin_clusters_are_transitive_and_named_by_the_lowest_actor_id():
    a, b, c = (100, 100, 200, 300), (115, 100, 215, 300), (130, 100, 230, 300)   # a~b, b~c (0.74), a vs c 0.54
    gt = make_gt({7: a, 3: b, 5: c, 11: B_SRC}, 10)
    assert me.twin_clusters(make_res(10, []), gt) == {3: 3, 5: 3, 7: 3}


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


def test_talk_eval_twin_actor_gets_credit_only_when_twin_aware():
    dets, ids = [(101, 102, 10.0, 30.0)], {101: 11, 102: 2}          # our person follows actor 11, a twin of GT actor 1
    strict = me.talk_eval(dets, [talk_act()], ids, FPS)
    assert strict["recall_all"] == 0.0 and strict["n_gt_tracked"] == 0 and strict["precision_mapped"] == 0.0
    twin = me.talk_eval(dets, [talk_act()], ids, FPS, clusters={1: 1, 11: 1})
    assert twin["recall_all"] == 1.0 and twin["recall_tracked"] == 1.0 and twin["precision_mapped"] == 1.0
    assert twin["n_gt"] == 1 and twin["n_gt_tracked"] == 1 and twin["tiou_0.1"]["recall_all"] == 1.0


def test_talk_eval_twin_aware_merges_gt_pairs_that_become_identical():
    same_talk_twice = [talk_act(300, 900, (1, 2), act_id=1), talk_act(300, 900, (11, 2), act_id=2)]
    assert me.talk_eval([], same_talk_twice, {}, FPS)["n_gt"] == 2                                    # strict: untouched
    r = me.talk_eval([], same_talk_twice, {}, FPS, clusters={1: 1, 11: 1})
    assert r["n_gt"] == 1 and r["n_gt_collapsed"] == 1
    # a different conversation of the twin at another time stays a separate ground-truth interval
    other_time = [talk_act(300, 900, (1, 2), act_id=1), talk_act(1500, 1800, (11, 2), act_id=2)]
    assert me.talk_eval([], other_time, {}, FPS, clusters={1: 1, 11: 1})["n_gt"] == 2
    # an empty cluster map is still twin-aware mode: identical ground-truth pairs are one interval
    assert me.talk_eval([], [talk_act(act_id=1), talk_act(act_id=2)], {}, FPS, clusters={})["n_gt"] == 1


def test_talk_eval_twin_aware_dedupes_nearly_identical_spans_of_the_same_cluster_pair():
    # the same conversation annotated for two twins with spans a second apart: 28 s vs 29 s
    twice = [talk_act(300, 1140, (1, 2), act_id=1), talk_act(330, 1140, (11, 2), act_id=2)]
    assert me.talk_eval([], twice, {}, FPS, clusters={1: 1, 11: 1})["n_gt"] == 1


def test_talk_eval_twin_aware_drops_pairs_between_twins():
    talk = [talk_act(300, 900, (1, 11))]
    assert me.talk_eval([], talk, {}, FPS)["n_gt"] == 1
    r = me.talk_eval([], talk, {}, FPS, clusters={1: 1, 11: 1})
    assert r["n_gt"] == 0 and r["recall_all"] is None and r["n_gt_collapsed"] == 1


def test_talk_eval_twin_aware_detection_between_two_persons_of_one_person_is_a_false_positive():
    ids = {101: 1, 103: 11, 102: 2}
    r = me.talk_eval([(101, 103, 10.0, 30.0)], [talk_act()], ids, FPS, clusters={1: 1, 11: 1})
    assert r["n_det"] == 1 and r["n_det_mapped"] == 1 and r["n_matched"] == 0 and r["precision_mapped"] == 0.0


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


def test_enter_exit_twin_actor_event_is_credited_to_the_person_mapped_to_its_twin():
    an = make_an([], timelines=timelines(p1=[[10.0, 40.0]]))
    acts = [leave(1110, 1230, 293)]                                   # GT: actor 293 (a twin of actor 3) exits at 41 s
    strict = me.enter_exit_eval(an, acts, {1: 3}, FPS, t_end=60.0)
    assert strict["exit"]["n_gt"] == 1 and strict["exit"]["n_gt_tracked"] == 0 and strict["exit"]["n_matched"] == 0
    twin = me.enter_exit_eval(an, acts, {1: 3}, FPS, t_end=60.0, clusters={3: 3, 293: 3})
    assert twin["exit"]["n_gt_tracked"] == 1 and twin["exit"]["n_matched"] == 1
    assert twin["exit"]["median_abs_err_s"] == pytest.approx(1.0)
    # the other direction: the person follows the twin, the event is of the long-lived actor
    twin2 = me.enter_exit_eval(an, [leave(1110, 1230, 3)], {1: 293}, FPS, t_end=60.0, clusters={3: 3, 293: 3})
    assert twin2["exit"]["n_matched"] == 1


def test_enter_exit_first_last_variant_uses_the_first_and_last_time_the_person_was_seen():
    an = make_an([], timelines=timelines(p1=[[10.0, 20.0], [50.0, 70.0]], p2=[[75.0, 80.0]]))
    acts = [enter(1440, 1500, 5), leave(2040, 2130, 5, 2)]            # GT: enter 48.0 s, exit 71.0 s
    r = me.enter_exit_eval(an, acts, {1: 5}, FPS, t_end=100.0)
    assert r["enter"]["n_matched"] == 1 and r["exit"]["n_matched"] == 1          # nearest boundary: 50.0 and 70.0
    fl = r["first_last"]
    assert fl["enter"]["n_matched"] == 0 and fl["enter"]["errs_s"] == []         # first seen 10.0: 38 s off
    assert fl["exit"]["n_matched"] == 1 and fl["exit"]["median_abs_err_s"] == pytest.approx(1.0)   # last seen 70.0
    assert fl["enter"]["n_gt"] == 1 and fl["enter"]["n_gt_tracked"] == 1
    # persons that are fragments of one actor: first seen over all of them, last seen over all of them
    r2 = me.enter_exit_eval(an, acts, {1: 5, 2: 5}, FPS, t_end=100.0)
    assert r2["first_last"]["exit"]["n_matched"] == 0                            # last seen 80.0: 9 s off
    assert r2["exit"]["n_matched"] == 1                                          # nearest end is still 70.0
    assert r2["first_last"]["all"]["n_gt"] == 2


# ------------------------------------------------------------------------------------------------- aggregation

def clip_result(name, talk, ee, trk_p, trk_t, twin_talk=None, twin_ee=None):
    return {"clip": name, "tracks": {"tracking": trk_t}, "persons": {"tracking": trk_p},
            "strict": {"talk": talk, "enter_exit": ee},
            "twin_aware": {"talk": twin_talk or talk, "enter_exit": twin_ee or ee}}


def test_aggregate_is_micro_averaged_and_pools_errors_for_both_variants():
    a_talk = me.talk_eval([(1, 2, 12.0, 32.0)], [talk_act()], {1: 1, 2: 2}, FPS)                  # 1/1, err 2
    b_acts = [talk_act(300, 900, (1, 2), act_id=i) for i in range(1, 4)]
    b_talk = me.talk_eval([(1, 2, 10.0, 30.0)], b_acts, {1: 1, 2: 2}, FPS)                         # 1/3, err 0
    b_twin = me.talk_eval([(1, 2, 10.0, 30.0)], b_acts, {1: 1, 2: 2}, FPS, clusters={})            # 1/1: same talk x3
    assert a_talk["recall_all"] == 1.0 and b_talk["recall_all"] == pytest.approx(1 / 3) and b_twin["recall_all"] == 1.0
    ee_a = me.enter_exit_eval(make_an([], timelines=timelines(p1=[[10.0, 40.0]])), [enter(270, 330, 5)],
                              {1: 5}, FPS, t_end=60.0)
    ee_b_acts = [enter(270, 330, 5), enter(300, 330, 6, 2), enter(330, 360, 7, 3)]
    ee_b_an = make_an([], timelines=timelines(p1=[[11.0, 40.0]]))
    ee_b = me.enter_exit_eval(ee_b_an, ee_b_acts, {1: 5}, FPS, t_end=60.0)
    ee_b_twin = me.enter_exit_eval(ee_b_an, ee_b_acts, {1: 5}, FPS, t_end=60.0, clusters={5: 5, 6: 5})
    res = make_res(10, box_obs(range(10), 1, A_OUT))
    trk = me.tracking_metrics(res, None, make_gt({1: A_SRC}, 10), False)
    clips = [{**clip_result("a", a_talk, ee_a, trk, trk), "n_gt_boxes": 10, "n_gt_dup_boxes": 4},
             {**clip_result("b", b_talk, ee_b, trk, trk, b_twin, ee_b_twin), "n_gt_boxes": 30, "n_gt_dup_boxes": 0}]
    total = me.aggregate(clips)
    assert total["gt_dup_share"] == pytest.approx(4 / 40)
    talk = total["strict"]["talk"]
    assert talk["n_gt"] == 4 and talk["recall_all"] == pytest.approx(2 / 4)                        # (1 + 1) / (1 + 3)
    assert talk["median_start_err_s"] == pytest.approx(1.0)                                        # median(2, 0)
    assert talk["n_det"] == 2 and talk["n_det_mapped"] == 2 and talk["precision_mapped"] == 1.0
    ee = total["strict"]["enter_exit"]["all"]
    assert ee["n_gt"] == 4 and ee["n_matched"] == 2 and ee["recall_all"] == pytest.approx(0.5)
    assert ee["median_abs_err_s"] == pytest.approx(1.5)                                            # median(1, 2)
    assert total["strict"]["enter_exit"]["first_last"]["all"]["n_matched"] == 2
    twin_talk = total["twin_aware"]["talk"]                                                         # a's strict + b's twin
    assert twin_talk["n_gt"] == 2 and twin_talk["recall_all"] == 1.0 and twin_talk["n_gt_collapsed"] == 2
    twin_ee = total["twin_aware"]["enter_exit"]["all"]                                              # a + b (actor 6 credited)
    assert twin_ee["n_gt"] == 4 and twin_ee["n_gt_tracked"] == 3 and twin_ee["n_matched"] == 3
    t = total["persons"]["tracking"]
    assert t["num_frames"] == 20 and t["mota"] == pytest.approx(1.0) and t["idf1"] == pytest.approx(1.0)
    assert total["n_clips"] == 2 and total["n_failed"] == 0


def test_aggregate_mota_is_computed_from_summed_counts():
    good = make_res(10, box_obs(range(10), 1, A_OUT))                            # 10 GT, perfect
    bad = make_res(10, box_obs(range(5), 1, A_OUT))                              # 10 GT, 5 misses
    gt = make_gt({1: A_SRC}, 10)
    t1, t2 = me.tracking_metrics(good, None, gt, False), me.tracking_metrics(bad, None, gt, False)
    total = me.aggregate([clip_result("a", {}, {}, t1, t1), clip_result("b", {}, {}, t2, t2)])
    assert total["tracks"]["tracking"]["mota"] == pytest.approx(1 - 5 / 20)


def test_aggregate_skips_failed_clips_and_handles_nothing():
    assert me.aggregate([])["n_clips"] == 0
    t = me.tracking_metrics(make_res(5, []), None, {}, False)
    total = me.aggregate([{"clip": "x", "error": "ValueError: boom"}, clip_result("ok", {}, {}, t, t)])
    assert total["n_clips"] == 1 and total["n_failed"] == 1


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


BAD_SPLITS = ['["c1.G1"]', '"c1.G1"', "null", "not json at all", '{"dev": "c1.G1", "test": []}', '{"dev": ["c1.G1"]}',
              '{"dev": [], "test": {"a": 1}}', '{"dev": [1, 2], "test": []}', '{"dev": [null], "test": []}']


@pytest.mark.parametrize("content", BAD_SPLITS)
def test_a_split_file_that_is_not_an_object_of_dev_and_test_lists_is_rejected(csv_and_split, tmp_path, monkeypatch,
                                                                              capsys, content):
    clips, split = csv_and_split
    split.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="split file"):
        me.select_clips(me.read_clips(clips), "all", split)
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "split file" in capsys.readouterr().err


def fake_clip_eval(clip, tag=1.0):
    trk = {"mota": tag, "idf1": tag, "num_switches": 0, "precision": tag, "recall": tag, "num_frames": 10,
           "num_objects": 10, "num_misses": 0, "num_false_positives": 0, "num_detections": 10, "idtp": 10,
           "idfp": 0, "idfn": 0}
    talk_args = ([(1, 2, 10.0, 30.0)], [talk_act()], {1: 1, 2: 2}, FPS)
    ee_args = (make_an([], timelines=timelines(p1=[[10.0, 40.0]])), [enter(270, 330, 5)], {1: 5}, FPS)
    return {"clip": clip, "duration_s": 60.0, "cached": True, "timings": {"total": 1.0},
            "n_gt_boxes": 10, "n_gt_dup_boxes": 2, "gt_dup_share": 0.2,
            **clip_result(clip, me.talk_eval(*talk_args), me.enter_exit_eval(*ee_args, t_end=60.0), trk, trk,
                          me.talk_eval(*talk_args, clusters={}), me.enter_exit_eval(*ee_args, t_end=60.0, clusters={}))}


def Path_name(p):
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def test_main_writes_json_and_prints_both_variants(csv_and_split, tmp_path, monkeypatch, capsys):
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
    assert data["clips"][0]["strict"]["talk"]["recall_all"] == 1.0
    assert data["total"]["strict"]["talk"]["n_gt"] == 2 and data["total"]["twin_aware"]["talk"]["n_gt"] == 2
    assert data["total"]["twin_aware"]["enter_exit"]["first_last"]["all"]["n_gt"] == 2
    text = capsys.readouterr().out
    assert "c1.G1" in text and "c2.G2" in text and "TOTAL" in text and "MOTA" in text
    assert "STRICT" in text and "TWIN-AWARE" in text
    assert "1/1" in text                                              # n_det_mapped / n_det next to P_map


def test_main_records_a_failing_clip_and_continues_writing_the_json_after_each_clip(csv_and_split, tmp_path,
                                                                                    monkeypatch, capsys):
    clips, split = csv_and_split
    out = tmp_path / "o.json"
    on_disk_during_second = []

    def fake_eval(video, ann, stack, cache_root, **kw):
        name = video.name.split(".r13")[0]
        if name == "c1.G1":
            raise ValueError("boom")
        on_disk_during_second.append(json.loads(out.read_text(encoding="utf-8")))
        return fake_clip_eval(name)

    monkeypatch.setattr(me, "eval_clip", fake_eval)
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    monkeypatch.setattr(me, "_files_exist", lambda *p: True)
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--out", str(out)])
    assert rc == 1                                                    # a clip failed
    assert on_disk_during_second[0]["clips"] == [{"clip": "c1.G1", "error": "ValueError: boom"}]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["clips"][0] == {"clip": "c1.G1", "error": "ValueError: boom"}
    assert data["clips"][1]["clip"] == "c2.G2" and data["total"]["n_clips"] == 1 and data["total"]["n_failed"] == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.out and "boom" in captured.out and "boom" in captured.err


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


def test_motmetrics_is_imported_lazily_with_a_clear_message(csv_and_split, tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "motmetrics", None)              # `import motmetrics` now raises ImportError
    with pytest.raises(RuntimeError, match="motmetrics"):
        me.tracking_metrics(make_res(3, []), None, {}, False)
    clips, split = csv_and_split
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(me, "_default_stack", lambda: pytest.fail("must not build the stack"))
    monkeypatch.setattr(me, "_files_exist", lambda *p: True)
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "motmetrics" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ end to end

def _kpf_geom(path, boxes):
    lines = ["- {'meta': 'synthetic'}"]
    for frame, actor, (x1, y1, x2, y2) in boxes:
        lines.append(f"- {{'geom': {{'g0': '{x1} {y1} {x2} {y2}', 'id0': 1, 'id1': {actor}, 'keyframe': True, "
                     f"'ts0': {frame}}}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meet_annotations(tmp_path):
    """KPF files for ``meet_video`` (20 s at 30 fps, 1280x720); returns the annotation prefix. Blue (actor 1) stands at
    x 600..680; red (actor 2) walks in 4-8 s, stays at x 470..550 until 18 s, then walks out. Both are 80x220 at y 250.
    Actor 3 is a twin annotation of actor 2 for its last 4 s (a slightly larger box) and is the one that "exits"; actor 2
    itself enters at 4 s. Actors 1 and 2 talk from 8 s to 18 s."""
    from tests.conftest import _customer_x1
    boxes = []
    for f in range(600):
        t = f / 30.0
        boxes.append((f, 1, (600, 250, 680, 470)))
        x1 = _customer_x1(t)
        if x1 is not None:
            boxes.append((f, 2, (x1, 250, x1 + 80, 470)))
            if t >= 16.0:
                boxes.append((f, 3, (x1 - 2, 248, x1 + 82, 472)))
    prefix = tmp_path / "meet"
    _kpf_geom(tmp_path / "meet.geom.yml", boxes)
    (tmp_path / "meet.types.yml").write_text(
        "".join(f"- {{'types': {{'cset3': {{'person': 1.0}}, 'id1': {i}}}}}\n" for i in (1, 2, 3)), encoding="utf-8")
    span = "{'tsr0': [240, 540]}"
    (tmp_path / "meet.activities.yml").write_text(
        "- {'act': {'act2': {'person_talks_to_person': 1.0}, 'actors': [{'id1': 1, 'timespan': [" + span + "]}, "
        "{'id1': 2, 'timespan': [" + span + "]}], 'id2': 4, 'src_status': 'good', 'timespan': [" + span + "]}}\n"
        "- {'act': {'act2': {'person_enters_scene_through_structure': 1.0}, 'actors': [{'id1': 2, 'timespan': "
        "[{'tsr0': [120, 150]}]}], 'id2': 5, 'src_status': 'good', 'timespan': [{'tsr0': [120, 150]}]}}\n"
        "- {'act': {'act2': {'person_exits_scene_through_structure': 1.0}, 'actors': [{'id1': 3, 'timespan': "
        "[{'tsr0': [540, 570]}]}], 'id2': 6, 'src_status': 'good', 'timespan': [{'tsr0': [540, 570]}]}}\n",
        encoding="utf-8")
    return prefix


def test_eval_clip_end_to_end_on_the_synthetic_meet_video(meet_video, tmp_path, monkeypatch):
    from showroom_coach import analysis
    from showroom_coach.perception.stack import fake_stack
    prefix = write_meet_annotations(tmp_path)

    passed = []
    real_analyze = analysis.analyze
    monkeypatch.setattr(analysis, "analyze", lambda *a, **kw: (passed.append(kw), real_analyze(*a, **kw))[1])
    r = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", preset="short", dist=2.0, min_dur=8.0,
                     gap_tol=1.5, merge_thr=0.37)
    assert passed and passed[0]["merge_thr"] == 0.37 and passed[0]["preset"] == "short"
    json.dumps(r)
    assert r["clip"] == "meet"
    assert r["tracks"]["tracking"]["num_switches"] == 0 and r["persons"]["tracking"]["num_switches"] == 0
    assert r["persons"]["tracking"]["recall"] > 0.8
    strict, twin = r["strict"], r["twin_aware"]
    assert sorted(set(strict["id_map"].values())) == [1, 2]
    assert {int(k): v for k, v in twin["clusters"].items()} == {2: 2, 3: 2}
    for variant in (strict, twin):
        talk = variant["talk"]
        assert talk["n_gt"] == 1 and talk["n_gt_tracked"] == 1
        assert talk["recall_all"] == 1.0 and talk["precision_mapped"] == 1.0
        assert 0.0 <= talk["median_start_err_s"] <= 4.0
        enter_ev = variant["enter_exit"]["enter"]
        assert enter_ev["n_gt"] == 1 and enter_ev["n_matched"] == 1          # GT enter at 4.0 s; the red box appears then
        assert variant["enter_exit"]["first_last"]["enter"]["n_matched"] == 1
    # the exit belongs to actor 3, the twin of actor 2: only the twin-aware numbers credit it
    assert strict["enter_exit"]["exit"]["n_gt"] == 1 and strict["enter_exit"]["exit"]["n_gt_tracked"] == 0
    assert twin["enter_exit"]["exit"]["n_gt_tracked"] == 1 and twin["enter_exit"]["exit"]["n_matched"] == 1

    short = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", limit_seconds=10.0)
    assert short["duration_s"] == pytest.approx(10.0, abs=0.3)
    assert short["strict"]["talk"]["n_gt"] == 1                      # GT clipped to 8-10 s


# ------------------------------------------------------------------------------------------------ oracle ablation

def test_oracle_result_has_one_observation_per_gt_box_in_processed_pixels():
    res = make_res(10, [])
    gt = make_gt({100: A_SRC, 200: B_SRC}, 10)
    before = dict(vars(res))
    o = me.oracle_result(res, gt)
    assert len(o.obs_frame) == 20 and o.obs_frame.dtype == np.int64 and o.obs_track.dtype == np.int64
    assert list(o.obs_frame) == [f for f in range(10) for _ in (0, 1)]
    assert list(o.obs_track) == [100, 200] * 10                       # sorted by frame, then track id
    assert np.allclose(o.obs_bbox[0], A_OUT) and np.allclose(o.obs_bbox[1], B_OUT)   # scale 0.5
    assert np.all(o.obs_score == 1.0)
    assert o.obs_kpts.shape == (20, 17, 3) and not o.obs_kpts.any()   # confidence 0: geometry falls back to box height
    assert o.track_emb == {} and o.track_outfit == {}
    assert o.frame_times is res.frame_times and o.src_frame_idx is res.src_frame_idx and o.video is res.video
    assert all(vars(res)[k] is before[k] for k in before)             # the cached result is not touched
    assert o.stack_name == "oracle"


def test_oracle_result_uses_only_the_sampled_frames_of_the_window():
    res = make_res(5, [])                                              # source frames 0, 6, ..., 24
    gt = make_gt({1: A_SRC}, 20)                                       # annotated for 120 source frames
    o = me.oracle_result(res, gt)
    assert list(o.obs_frame) == [0, 1, 2, 3, 4]
    odd = {7: [GeomBox(1, 7, A_SRC)], 6: [GeomBox(2, 6, B_SRC)]}       # frame 7 is not sampled, frame 6 is
    o = me.oracle_result(res, odd)
    assert list(o.obs_frame) == [1] and list(o.obs_track) == [2]


def test_oracle_result_collapses_twin_actors_into_one_track_with_the_lowest_actors_box():
    twin = (102, 102, 202, 302)                                        # a slightly shifted second annotation
    gt = make_gt({8: twin, 7: A_SRC, 9: B_SRC}, 4)
    o = me.oracle_result(make_res(4, []), gt, clusters={7: 7, 8: 7})
    assert list(o.obs_track) == [7, 9] * 4                             # one observation per person and frame
    assert np.allclose(o.obs_bbox[0], A_OUT)                           # the box of actor 7, not of its twin 8
    assert list(me.oracle_result(make_res(4, []), gt, clusters={8: 7, 7: 7}).obs_track) == [7, 9] * 4


def test_oracle_result_follows_a_twin_when_its_representative_is_not_annotated():
    gt = make_gt({7: lambda f: A_SRC if f < 12 else None, 8: lambda f: (101, 101, 201, 301) if f >= 12 else None}, 4)
    o = me.oracle_result(make_res(4, []), gt, clusters={7: 7, 8: 7})
    assert list(o.obs_track) == [7, 7, 7, 7] and np.allclose(o.obs_bbox[3], np.array((101, 101, 201, 301)) * 0.5)


def test_oracle_result_without_ground_truth_is_empty():
    o = me.oracle_result(make_res(3, []), {})
    assert o.obs_frame.shape == (0,) and o.obs_bbox.shape == (0, 4) and o.obs_kpts.shape == (0, 17, 3)
    assert len(o.frame_times) == 3


def test_eval_clip_oracle_scores_the_engagement_logic_on_perfect_tracks(meet_video, tmp_path, monkeypatch):
    from showroom_coach import analysis
    from showroom_coach.perception.stack import fake_stack
    prefix = write_meet_annotations(tmp_path)
    seen = []
    real_analyze = analysis.analyze
    monkeypatch.setattr(analysis, "analyze", lambda res, *a, **kw: (seen.append(res.stack_name), real_analyze(res, *a, **kw))[1])
    r = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", dist=2.0, min_dur=8.0, oracle=True)
    assert seen == ["oracle"]                                          # the analysis ran on the oracle tracks
    json.dumps(r)
    assert r["mode"] == "oracle" and r["n_tracks"] == 2 and r["n_persons"] == 2   # actors 1 and 2 (3 is 2's twin)
    assert r["tracks"]["id_map"] == {1: 1, 2: 2} and r["strict"]["id_map"] == {1: 1, 2: 2}
    assert r["persons"]["tracking"]["num_switches"] == 0
    # the only ground-truth boxes the oracle misses are the second box of each twin pair (half of the boxes with a twin)
    assert (1 - r["persons"]["tracking"]["recall"]) == pytest.approx(r["gt_dup_share"] / 2, abs=0.01)
    assert r["persons"]["tracking"]["precision"] == 1.0
    for variant in me.VARIANTS:
        talk = r[variant]["talk"]
        assert talk["n_gt"] == 1 and talk["n_gt_tracked"] == 1 and talk["n_matched"] == 1
        assert talk["recall_all"] == 1.0 and talk["precision_mapped"] == 1.0
        assert r[variant]["enter_exit"]["enter"]["n_matched"] == 1
    # the exit belongs to actor 3, which the oracle folded into actor 2: only the twin-aware numbers credit it
    assert r["strict"]["enter_exit"]["exit"]["n_gt_tracked"] == 0
    assert r["twin_aware"]["enter_exit"]["exit"]["n_matched"] == 1
    normal = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache")
    assert normal["mode"] == "system" and normal["cached"] is True


def test_eval_clip_oracle_ignores_the_merge_threshold(meet_video, tmp_path):
    from showroom_coach.perception.stack import fake_stack
    prefix = write_meet_annotations(tmp_path)
    a = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", oracle=True, merge_thr=0.0)
    b = me.eval_clip(meet_video, prefix, fake_stack(), tmp_path / "cache", oracle=True, merge_thr=0.9)
    assert a["twin_aware"]["talk"] == b["twin_aware"]["talk"] and a["n_persons"] == b["n_persons"] == 2


def test_main_oracle_flag_reaches_eval_clip_and_marks_the_output(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    seen = []

    def fake_eval(video, ann, stack, cache_root, **kw):
        seen.append(kw)
        return {**fake_clip_eval(video.name.split(".r13")[0]), "mode": "oracle" if kw.get("oracle") else "system"}

    monkeypatch.setattr(me, "eval_clip", fake_eval)
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    monkeypatch.setattr(me, "_files_exist", lambda *p: True)
    monkeypatch.chdir(tmp_path)
    rc = me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--oracle"])
    assert rc == 0 and seen and all(kw["oracle"] is True for kw in seen)
    out = tmp_path / "runs" / "eval" / "meva_dev_oracle.json"             # a separate default file name
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "oracle" and data["params"]["oracle"] is True
    assert any("oracle" in n.lower() for n in data["notes"])
    text = capsys.readouterr().out
    assert "ORACLE" in text and "oracle=True" in text
    seen.clear()
    assert me.main(["--clips", str(clips), "--split", "dev", "--split-file", str(split), "--out",
                    str(tmp_path / "plain.json")]) == 0
    assert all(kw["oracle"] is False for kw in seen)
    assert json.loads((tmp_path / "plain.json").read_text(encoding="utf-8"))["mode"] == "system"


def test_a_clip_listed_in_both_dev_and_test_is_rejected(csv_and_split, tmp_path, monkeypatch, capsys):
    clips, split = csv_and_split
    split.write_text(json.dumps({"dev": ["c1.G1", "c2.G2"], "test": ["c2.G2", "c3.G3"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="both dev and test.*c2.G2"):
        me.select_clips(me.read_clips(clips), "test", split)
    monkeypatch.setattr(me, "eval_clip", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(me, "_default_stack", lambda: "STACK")
    assert me.main(["--clips", str(clips), "--split", "test", "--split-file", str(split), "--out", str(tmp_path / "o.json")]) == 2
    assert "both dev and test" in capsys.readouterr().err
