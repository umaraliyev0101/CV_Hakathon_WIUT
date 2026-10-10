import time

import numpy as np

from showroom_coach.identity import LiveIdentity, TrackSummary, merge_tracks


def emb(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle_emb(deg):
    """Unit vector in the x-y plane at the given angle."""
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


def test_merge_fragments_of_same_person():
    m = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                      TrackSummary(2, 12.0, 20.0, emb([0.95, 0.1, 0])),
                      TrackSummary(3, 0.0, 20.0, emb([0, 1, 0]))], thr=0.3)
    assert m[1] == m[2] != m[3]
    assert sorted(set(m.values())) == [1, 2] and m[1] == 1


def test_time_overlap_forbids_merge_even_if_identical():
    m = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                      TrackSummary(2, 5.0, 15.0, emb([1, 0.01, 0]))], thr=0.3)
    assert m[1] != m[2]


def test_constraint_is_cluster_level():
    m = merge_tracks([TrackSummary(1, 0, 10, emb([1, 0, 0])),
                      TrackSummary(2, 11, 12, emb([1, 0.05, 0])),
                      TrackSummary(3, 5, 9, emb([1, 0.1, 0]))], thr=0.3)
    assert m[1] != m[3]
    assert m[2] in (m[1], m[3])


def test_dissimilar_tracks_stay_apart():
    m = merge_tracks([TrackSummary(1, 0, 5, emb([1, 0, 0])), TrackSummary(2, 6, 9, emb([0, 0, 1]))], thr=0.3)
    assert m[1] != m[2]


def test_live_reattach_recent_lost_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1})
    li.end_step(1.0, set())
    assert li.resolve(7, 3.0, emb([0.97, 0.1, 0])) == p1


def test_live_no_reattach_after_window():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    assert li.resolve(7, 30.0, emb([1, 0, 0])) != p1


def test_live_no_reattach_to_active_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1})
    assert li.resolve(2, 0.2, emb([1, 0, 0])) != p1


def test_known_track_keeps_pid_and_none_emb_gets_new_pid():
    li = LiveIdentity()
    p = li.resolve(1, 0.0, None)
    assert li.resolve(1, 0.2, emb([1, 0, 0])) == p
    assert li.resolve(2, 0.2, None) != p


# --- offline edge cases -------------------------------------------------------------------------


def test_merge_empty_and_single():
    assert merge_tracks([]) == {}
    assert merge_tracks([TrackSummary(9, 3.0, 4.0, emb([1, 0, 0]))]) == {9: 1}


def test_pids_ordered_by_first_appearance_then_track_id():
    m = merge_tracks([TrackSummary(5, 10.0, 12.0, emb([0, 1, 0])),
                      TrackSummary(9, 0.0, 2.0, emb([1, 0, 0])),
                      TrackSummary(2, 10.0, 12.0, emb([0, 0, 1])),
                      TrackSummary(7, 20.0, 22.0, emb([1, 0.02, 0]))], thr=0.3)
    # track 9 starts first; tracks 2 and 5 tie on t0 so track 2 comes first; track 7 joins track 9's person
    assert m == {9: 1, 7: 1, 2: 2, 5: 3}


def test_unnormalised_embeddings_are_handled():
    m = merge_tracks([TrackSummary(1, 0, 5, 7.0 * emb([1, 0, 0])),
                      TrackSummary(2, 6, 9, 0.2 * emb([1, 0.05, 0]))], thr=0.3)
    assert m[1] == m[2]


def test_zero_vector_does_not_crash_and_stays_apart():
    m = merge_tracks([TrackSummary(1, 0, 5, np.zeros(3)),
                      TrackSummary(2, 6, 9, np.zeros(3)),
                      TrackSummary(3, 10, 12, emb([1, 0, 0])),
                      TrackSummary(4, 13, 15, emb([1, 0.02, 0]))], thr=0.3)
    assert m[3] == m[4]
    assert len({m[1], m[2], m[3]}) == 3


def test_small_overlap_is_tolerated_but_large_is_not():
    near = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                         TrackSummary(2, 9.9, 15.0, emb([1, 0.01, 0]))], thr=0.3, max_overlap=0.2)
    far = merge_tracks([TrackSummary(1, 0.0, 10.0, emb([1, 0, 0])),
                        TrackSummary(2, 9.5, 15.0, emb([1, 0.01, 0]))], thr=0.3, max_overlap=0.2)
    assert near[1] == near[2]
    assert far[1] != far[2]


def test_cluster_mean_is_weighted_by_n_obs():
    # tracks 1 and 2 merge first; the merged mean then decides whether track 3 (on the other side) joins
    def tracks(n1):
        return [TrackSummary(1, 0, 5, angle_emb(0), n_obs=n1),
                TrackSummary(2, 6, 10, angle_emb(40), n_obs=1),
                TrackSummary(3, 11, 15, angle_emb(-40.5), n_obs=1)]

    heavy = merge_tracks(tracks(10), thr=0.3)
    even = merge_tracks(tracks(1), thr=0.3)
    assert heavy[1] == heavy[2] == heavy[3]
    assert even[1] == even[2] != even[3]


def test_merge_is_fast_for_busy_clip():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(40, 400))
    tracks = []
    for i in range(300):
        v = centres[rng.integers(0, 40)] + 0.35 * rng.normal(size=400)
        t0 = float(rng.uniform(0, 290))
        tracks.append(TrackSummary(i + 1, t0, t0 + float(rng.uniform(1, 15)), v / np.linalg.norm(v),
                                   n_obs=int(rng.integers(1, 50))))
    t = time.perf_counter()
    m = merge_tracks(tracks)
    elapsed = time.perf_counter() - t
    assert elapsed < 2.0
    assert set(m) == {tr.track_id for tr in tracks}
    pids = sorted(set(m.values()))
    assert pids == list(range(1, len(pids) + 1))
    assert len(pids) < 300  # something merged


# --- live edge cases ----------------------------------------------------------------------------


def test_live_picks_best_of_several_lost_persons():
    li = LiveIdentity(thr=0.3, window=10.0)
    pa = li.resolve(1, 0.0, angle_emb(0))
    pb = li.resolve(2, 0.0, angle_emb(30))
    li.end_step(0.0, {1, 2}); li.end_step(1.0, set())
    assert li.resolve(7, 2.0, angle_emb(25)) == pb
    assert pa != pb


def test_live_two_new_tracks_cannot_share_one_lost_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    a = li.resolve(7, 2.0, emb([1, 0, 0]))
    b = li.resolve(8, 2.0, emb([1, 0, 0]))
    assert a == p1 and b != p1


def test_live_reattached_track_keeps_pid_afterwards():
    li = LiveIdentity()
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    assert li.resolve(7, 2.0, emb([1, 0, 0])) == p1
    li.end_step(2.0, {7})
    assert li.resolve(7, 2.2, emb([0, 1, 0])) == p1


def test_live_person_without_embedding_is_never_a_candidate():
    li = LiveIdentity()
    p1 = li.resolve(1, 0.0, None); li.end_step(0.0, {1}); li.end_step(1.0, set())
    assert li.resolve(7, 2.0, emb([1, 0, 0])) != p1


def test_live_zero_embedding_gets_new_pid():
    li = LiveIdentity()
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    assert li.resolve(7, 2.0, np.zeros(3)) != p1


def test_update_emb_running_mean_drives_matching():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    li.update_emb(p1, emb([0, 1, 0]), alpha=0.8)  # mean is now ~(0.2, 0.8), re-normalised
    assert li.resolve(7, 2.0, emb([0, 1, 0])) == p1


def test_update_emb_ignores_unknown_pid_and_zero_vector():
    li = LiveIdentity(thr=0.3, window=10.0)
    p1 = li.resolve(1, 0.0, emb([1, 0, 0])); li.end_step(0.0, {1}); li.end_step(1.0, set())
    li.update_emb(99, emb([1, 0, 0]))
    li.update_emb(p1, np.zeros(3))
    assert li.resolve(7, 2.0, emb([1, 0, 0])) == p1
