import numpy as np, pytest
from pulse.config import get_config
from pulse.datatypes import TrackObs
from pulse.identity import TrackSummary, merge, reattach, summarize

CFG = get_config("dealership")       # leave_after 10

def ts(tid, t0, t1, a=(0.5, 0.8), b=(0.5, 0.8), emb=(1.0, 0.0)):
    return TrackSummary(tid, t0, t1, a, b, np.array(emb, float), int((t1 - t0) * 5) + 1)

def groups(persons): return sorted(tuple(p.track_ids) for p in persons)

def test_merges_fragments_of_one_person():
    tr = {1: ts(1, 0, 10, b=(0.5, 0.8)), 2: ts(2, 11, 20, a=(0.52, 0.8))}
    assert groups(merge(tr, CFG)) == [(1, 2)]

def test_never_merges_overlapping_tracks():
    tr = {1: ts(1, 0, 10), 2: ts(2, 5, 15)}
    assert groups(merge(tr, CFG)) == [(1,), (2,)]

def test_respects_max_gap_and_space():
    far_time = {1: ts(1, 0, 10), 2: ts(2, 25, 30)}
    far_space = {1: ts(1, 0, 10, b=(0.1, 0.8)), 2: ts(2, 10.5, 20, a=(0.9, 0.8))}
    assert groups(merge(far_time, CFG)) == [(1,), (2,)]
    assert groups(merge(far_space, CFG)) == [(1,), (2,)]

def test_dissimilar_kept_apart_and_transitive_constraint():
    tr = {1: ts(1, 0, 10), 2: ts(2, 11, 20, emb=(0, 1)), 3: ts(3, 12, 18, emb=(1, 0.05))}
    persons = merge(tr, CFG)
    assert groups(persons) == [(1, 3), (2,)]
    assert [p.person_id for p in sorted(persons, key=lambda p: p.first_t)] == [1, 2]

def test_reattach_recently_lost():
    lost = [ts(7, 0, 10, b=(0.5, 0.8)), ts(8, 0, 3, b=(0.5, 0.8), emb=(0, 1))]
    new = ts(20, 12, 12, a=(0.52, 0.8))
    assert reattach(lost, new, now=12, cfg=CFG) == 7
    assert reattach(lost, ts(21, 30, 30), now=30, cfg=CFG) is None

def test_missing_embedding_never_merged():
    a = ts(1, 0, 10); b = ts(2, 11, 20); b.emb = None
    assert groups(merge({1: a, 2: b}, CFG)) == [(1,), (2,)]

def test_summarize_normalises_feet_and_skips_garbage():
    bb = lambda x: (x, 0.0, x + 10.0, 100.0)
    obs = [TrackObs(2.0, 1, 5, bb(180.0), 0.9), TrackObs(1.0, 0, 5, bb(90.0), 0.9),
           TrackObs(3.0, 2, 5, (float("nan"),) * 4, 0.9)]
    s = summarize(obs, {5: np.array([1.0, 0.0])}, (200, 100))
    assert s[5].t_start == 1.0 and s[5].t_end == 2.0 and s[5].n_obs == 2
    assert s[5].start_foot == pytest.approx((0.475, 1.0))
    assert s[5].end_foot == pytest.approx((0.925, 1.0))
    assert summarize(obs, {}, (200, 100))[5].emb is None


def test_three_fragment_chain_merges_into_one_person():
    tr = {1: ts(1, 0, 10), 2: ts(2, 11, 20), 3: ts(3, 21, 30)}
    persons = merge(tr, CFG)
    assert groups(persons) == [(1, 2, 3)]
    assert (persons[0].first_t, persons[0].last_t) == (0, 30)


def test_overlap_guard_rejects_join_via_other_member():
    # 1~3 are a valid pair on their own, but 2 (in 1's cluster) overlaps 3.
    tr = {1: ts(1, 0, 5), 2: ts(2, 3, 12), 3: ts(3, 6, 10)}
    assert groups(merge(tr, CFG)) == [(1, 3), (2,)]


def test_join_refused_when_a_is_not_latest_ending_of_its_cluster():
    # 1+2 merge first (sim .9). Then (1,3) has no time overlap but 1 is not the cluster end.
    tr = {1: ts(1, 0, 5, emb=(1.0, 0.0)),
          2: ts(2, 6, 10, emb=(0.9, float(np.sqrt(1 - 0.81)))),
          3: ts(3, 10.1, 15, emb=(0.8, -0.6))}
    assert groups(merge(tr, CFG)) == [(1, 2), (3,)]


def test_reattach_picks_highest_similarity_and_breaks_ties_by_id():
    new = ts(20, 12, 12, a=(0.52, 0.8))
    worse = ts(5, 0, 10, emb=(0.8, 0.6))      # cos .8
    better = ts(9, 0, 10, emb=(0.99, 0.141))  # cos ~.99
    assert reattach([worse, better], new, now=12, cfg=CFG) == 9
    t1, t2 = ts(9, 0, 10), ts(4, 0, 10)
    assert reattach([t1, t2], new, now=12, cfg=CFG) == 4


def test_reattach_rejects_track_ending_after_now():
    assert reattach([ts(7, 0, 13)], ts(20, 12, 12), now=12, cfg=CFG) is None


def test_non_finite_or_mismatched_embeddings_never_merge():
    nan = {1: ts(1, 0, 10), 2: ts(2, 11, 20, emb=(float("nan"), 0.0))}
    assert groups(merge(nan, CFG)) == [(1,), (2,)]
    dim = {1: ts(1, 0, 10), 2: ts(2, 11, 20, emb=(1.0, 0.0, 0.0))}
    assert groups(merge(dim, CFG)) == [(1,), (2,)]
    assert reattach([ts(7, 0, 10, emb=(float("inf"), 0.0))], ts(20, 12, 12), 12, CFG) is None
