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


# --- live call order: end_step(t, in view now) first, then resolve ------------------------------


def test_live_same_frame_id_switch_reattaches():
    li = LiveIdentity(thr=0.3, window=10.0)
    li.end_step(0.0, {1}); p1 = li.resolve(1, 0.0, emb([1, 0, 0]))
    li.end_step(0.2, {2})  # track 1 died and track 2 was born in the same frame
    assert li.resolve(2, 0.2, emb([0.98, 0.05, 0])) == p1
    li.end_step(0.4, {2})
    assert li.resolve(2, 0.4, emb([1, 0, 0])) == p1


def test_live_switch_in_a_crowd_takes_only_the_vanished_person():
    li = LiveIdentity(thr=0.3, window=10.0)
    li.end_step(0.0, {1, 2})
    pa = li.resolve(1, 0.0, emb([1, 0, 0]))
    pb = li.resolve(2, 0.0, emb([1, 0.02, 0]))  # looks the same as A but is a second person in view
    assert pa != pb
    li.end_step(0.2, {2, 3})  # A's track 1 died, track 3 was born
    assert li.resolve(2, 0.2, emb([1, 0.02, 0])) == pb
    assert li.resolve(3, 0.2, emb([1, 0, 0])) == pa  # B is still in view, so only A is available


def test_live_two_births_in_one_frame_share_one_vanished_person_once():
    li = LiveIdentity(thr=0.3, window=10.0)
    li.end_step(0.0, {1}); p1 = li.resolve(1, 0.0, emb([1, 0, 0]))
    li.end_step(0.2, {2, 3})
    a = li.resolve(2, 0.2, emb([1, 0, 0]))
    b = li.resolve(3, 0.2, emb([1, 0, 0]))
    assert a == p1 and b != p1


def test_live_stale_people_are_forgotten_but_their_tracks_keep_the_pid():
    li = LiveIdentity(thr=0.3, window=10.0)
    pids = []
    for k in range(1, 51):  # fifty people pass through one after another
        t = 100.0 * k
        li.end_step(t, {k}); pids.append(li.resolve(k, t, emb([1, k, 0])))
    li.end_step(5100.0, set())
    assert len(li._emb) == 0 and len(li._last_seen) == 0  # nothing older than the window is kept
    assert len(set(pids)) == 50
    # a long-lost track that does come back keeps its pid, re-learns its descriptor on resolve, and can be updated
    li.end_step(5101.0, {3})
    assert li.resolve(3, 5101.0, emb([0, 0, 1])) == pids[2]
    li.update_emb(pids[2], emb([0, 0, 1]))
    li.end_step(5102.0, set())
    assert li.resolve(900, 5103.0, emb([0, 0, 1])) == pids[2]


# --- brute-force reference for the incremental matrix updates -----------------------------------


def _reference_merge(tracks, thr=0.3, max_overlap=0.2):
    """Slow and obvious: recompute every pair from scratch each round, cannot-link checked on all member pairs."""
    clusters = [[s] for s in sorted(tracks, key=lambda s: (s.t0, s.track_id))]

    def mean(c):
        v = sum(s.n_obs * s.emb / np.linalg.norm(s.emb) for s in c)
        return v / np.linalg.norm(v)

    while True:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if any(min(x.t1, y.t1) - max(x.t0, y.t0) > max_overlap for x in clusters[a] for y in clusters[b]):
                    continue
                d = 1.0 - mean(clusters[a]) @ mean(clusters[b])
                if d < thr and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            first = lambda c: min((s.t0, s.track_id) for s in c)  # noqa: E731
            return {s.track_id: pid for pid, c in enumerate(sorted(clusters, key=first), 1) for s in c}
        clusters[best[1]] += clusters.pop(best[2])


def test_merge_matches_brute_force_reference():
    rng = np.random.default_rng(1234)
    merged = total = 0
    for _ in range(60):
        n, k = int(rng.integers(4, 26)), int(rng.integers(2, 6))
        centres = rng.normal(size=(k, 6))
        tracks = []
        for i in range(n):
            v = centres[rng.integers(k)] + 0.7 * rng.normal(size=6)
            t0 = float(rng.uniform(0, 50))
            tracks.append(TrackSummary(i + 1, t0, t0 + float(rng.uniform(1, 12)), v / np.linalg.norm(v),
                                       int(rng.integers(1, 20))))
        got = merge_tracks(tracks)
        assert got == _reference_merge(tracks)
        merged += n - len(set(got.values())); total += n
    assert 0 < merged < total  # the random cases really exercise both merging and cannot-link blocking


def test_update_emb_ignores_forgotten_person_and_reattach_survives():
    li = LiveIdentity(thr=0.3, window=10.0)
    li.end_step(0.0, {1}); p = li.resolve(1, 0.0, emb([1, 0, 0]))
    li.end_step(100.0, set())  # the person is forgotten
    li.update_emb(p, emb([1, 0, 0]))  # must not resurrect a half-forgotten entry
    assert len(li._emb) == 0
    li.end_step(101.0, {2})
    assert li.resolve(2, 101.0, emb([1, 0, 0])) != p  # too old to re-attach, and no KeyError


# --- one active track per pid: a tracker can revive a track whose person was meanwhile re-attached -------------------


def _revived_setup():
    """Track 1 is lost, track 2 is born looking the same and takes over the pid, then the tracker revives track 1."""
    li = LiveIdentity(thr=0.3, window=10.0)
    li.end_step(0.0, {1}); p1 = li.resolve(1, 0.0, emb([1, 0, 0]))
    li.end_step(0.2, {2})
    assert li.resolve(2, 0.2, emb([1, 0.02, 0])) == p1  # re-attached while track 1 was out of view
    return li, p1


def test_live_revived_track_whose_pid_is_held_by_another_active_track_gets_a_new_pid():
    li, p1 = _revived_setup()
    li.end_step(0.4, {1, 2})
    revived, holder = li.resolve(1, 0.4, None), li.resolve(2, 0.4, None)
    assert holder == p1 and revived != p1  # the track that holds the pid keeps it
    li.end_step(0.6, {1, 2})  # and it stays that way
    assert (li.resolve(1, 0.6, None), li.resolve(2, 0.6, None)) == (revived, p1)


def test_live_the_holder_keeps_the_pid_whatever_order_the_tracks_are_resolved_in():
    li, p1 = _revived_setup()
    li.end_step(0.4, {1, 2})
    holder, revived = li.resolve(2, 0.4, None), li.resolve(1, 0.4, None)
    assert holder == p1 and revived != p1


def test_live_revived_track_keeps_its_pid_when_the_holder_is_gone():
    li, p1 = _revived_setup()
    li.end_step(0.4, {1})  # track 2 died; track 1 is the only one left
    assert li.resolve(1, 0.4, None) == p1


def test_live_assign_new_gives_a_track_a_fresh_pid_and_keeps_it():
    li = LiveIdentity()
    li.end_step(0.0, {1, 2})
    p, q = li.resolve(1, 0.0, None), li.resolve(2, 0.0, None)
    fresh = li.assign_new(2, 0.0)
    assert fresh not in (p, q)
    li.end_step(0.2, {1, 2})
    assert li.resolve(2, 0.2, None) == fresh and li.resolve(1, 0.2, None) == p
