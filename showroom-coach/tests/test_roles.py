import numpy as np
import pytest
from showroom_coach.config import RoleWeights
from showroom_coach.roles import interaction_features, outfit_peers, presence, suggest_roles

W = RoleWeights(bias=-1.5, outfit_peers=1.2, contacts=1.6, presence=1.0, initiator=1.0,
                staff_thr=0.6, customer_thr=0.4)


def test_suggest_staff_and_customer():
    feats = {1: dict(outfit_peers=2, contacts=3, presence=0.95, initiated=2),
             2: dict(outfit_peers=0, contacts=1, presence=0.30, initiated=0)}
    out = {s.pid: s for s in suggest_roles(feats, clip_len=120.0, weights=W)}
    assert out[1].role == "staff" and out[1].confidence >= 0.6
    assert out[2].role == "customer" and out[2].confidence <= 0.4
    assert "talked to 3 people" in out[1].reasons
    assert 1 <= len(out[1].reasons) <= 3 and 1 <= len(out[2].reasons) <= 3
    assert out[1].features["contacts"] == 3


def test_presence_ignored_for_short_clips():
    a = suggest_roles({1: dict(outfit_peers=0, contacts=0, presence=1.0, initiated=0)}, 30.0, W)[0]
    b = suggest_roles({1: dict(outfit_peers=0, contacts=0, presence=0.0, initiated=0)}, 30.0, W)[0]
    assert a.confidence == pytest.approx(b.confidence)
    assert not any("on screen" in r for r in a.reasons)


def test_unsure_band():
    s = suggest_roles({1: dict(outfit_peers=0, contacts=2, presence=0.5, initiated=2)}, 120.0, W)[0]
    assert s.role == "unsure" and 0.4 < s.confidence < 0.6


def test_pinned_formula():
    s = suggest_roles({1: dict(outfit_peers=3, contacts=4, presence=1.0, initiated=3)}, 120.0, W)[0]
    z = -1.5 + 1.2 + 1.6 + 1.0 + 1.0
    assert s.confidence == pytest.approx(1 / (1 + np.exp(-z)))


def _traj():
    traj = {1: [], 2: [], 3: []}
    for i in range(101):                 # 20 s at 5 fps
        t = round(i * 0.2, 6)
        traj[1].append((t, max(1.0, 5.0 - t), 5.0))   # walks from x=5 to x=1, arrives t=4
        traj[2].append((t, 0.0, 5.0))
        traj[3].append((t, 10.0, 5.0))
    return traj


def test_interaction_features_contacts_and_initiator():
    f = interaction_features(_traj(), contact_dist=2.0, contact_min=5.0)
    assert (f[1]["contacts"], f[2]["contacts"], f[3]["contacts"]) == (1, 1, 0)
    assert (f[1]["initiated"], f[2]["initiated"]) == (1, 0)
    assert f[1]["partners"] == [2]


def test_short_contact_not_counted():
    f = interaction_features(_traj(), contact_dist=2.0, contact_min=30.0)
    assert f[1]["contacts"] == 0


def test_presence_fraction():
    times = [round(i * 0.2, 6) for i in range(101)]
    traj = {1: [(t, 0, 5) for t in times], 2: [(t, 0, 5) for t in times[:51]]}
    p = presence(traj, times)
    assert p[1] == pytest.approx(1.0) and p[2] == pytest.approx(51 / 101)


def test_outfit_peers_counts_similar_descriptors():
    a = np.array([1.0, 0, 0]); b = np.array([0.98, 0.2, 0]); c = np.array([0, 1.0, 0])
    assert outfit_peers({1: a, 2: b, 3: c}, thr=0.9) == {1: 1, 2: 1, 3: 0}


# ---- edge cases beyond the brief ------------------------------------------------------------

def _times(n=101):
    return [round(i * 0.2, 6) for i in range(n)]


def test_empty_inputs_do_not_crash():
    assert interaction_features({}) == {}
    assert presence({}, _times()) == {}
    assert presence({1: [(0.0, 0.0, 0.0)]}, []) == {1: 0.0}
    assert outfit_peers({}, 0.9) == {}
    assert suggest_roles({}, 120.0, W) == []
    f = interaction_features({1: [], 2: []})
    assert f == {1: {"contacts": 0, "initiated": 0, "partners": []},
                 2: {"contacts": 0, "initiated": 0, "partners": []}}
    assert presence({1: []}, _times()) == {1: 0.0}


def test_single_person_and_single_sample_people():
    one = interaction_features({7: [(float(i), 0.0, 0.0) for i in range(30)]}, contact_min=5.0)
    assert one == {7: {"contacts": 0, "initiated": 0, "partners": []}}
    # one sample each: no sampling step can be measured, so no contact time accrues
    two = interaction_features({1: [(0.0, 0.0, 0.0)], 2: [(0.0, 0.5, 0.0)]}, contact_min=5.0)
    assert two[1]["contacts"] == 0 and two[2]["contacts"] == 0


def test_contact_time_counts_only_shared_frames_with_gaps():
    # person 1 only visible for the first n frames; both stand 1 m apart, 0.2 s step
    def run(n_visible, contact_min=5.0):
        times = _times()
        traj = {1: [(t, 0.0, 0.0) for t in times[:n_visible]],
                2: [(t, 1.0, 0.0) for t in times]}
        return interaction_features(traj, contact_dist=2.0, contact_min=contact_min)

    assert run(20)[1]["contacts"] == 0          # 4.0 s shared < 5 s
    assert run(30)[1]["contacts"] == 1          # 6.0 s shared >= 5 s
    assert run(25)[1]["contacts"] == 1          # exactly 5.0 s counts


def test_contact_time_is_cumulative_across_separate_meetings():
    times = _times()
    traj = {1: [], 2: []}
    for t in times:
        near = t < 3.0 or 10.0 <= t < 13.0     # 15 + 15 close samples = 6 s in two meetings
        traj[1].append((t, 0.0, 0.0))
        traj[2].append((t, 1.0 if near else 8.0, 0.0))
    f = interaction_features(traj, contact_dist=2.0, contact_min=5.0)
    assert f[1]["contacts"] == 1 and f[2]["contacts"] == 1


def test_initiator_is_the_one_who_walked_up():
    traj = _traj()
    swapped = {1: traj[2], 2: traj[1], 3: traj[3]}   # now person 2 is the walker
    f = interaction_features(swapped, contact_dist=2.0, contact_min=5.0)
    assert (f[1]["initiated"], f[2]["initiated"]) == (0, 1)
    assert f[2]["partners"] == [1] and f[1]["partners"] == [2]


def test_nobody_initiates_when_both_stand_or_meet_halfway():
    times = _times()
    standing = {1: [(t, 0.0, 0.0) for t in times], 2: [(t, 1.0, 0.0) for t in times]}
    f = interaction_features(standing, contact_dist=2.0, contact_min=5.0)
    assert f[1]["contacts"] == 1 and (f[1]["initiated"], f[2]["initiated"]) == (0, 0)

    halfway = {1: [(t, min(4.0, t), 0.0) for t in times], 2: [(t, max(6.0, 10.0 - t), 0.0) for t in times]}
    g = interaction_features(halfway, contact_dist=2.0, contact_min=5.0)
    assert g[1]["contacts"] == 1 and (g[1]["initiated"], g[2]["initiated"]) == (0, 0)


def test_timestamps_match_within_tolerance():
    times = _times()
    traj = {1: [(t, 0.0, 0.0) for t in times], 2: [(t + 4e-7, 1.0, 0.0) for t in times]}
    f = interaction_features(traj, contact_dist=2.0, contact_min=5.0)
    assert f[1]["contacts"] == 1
    p = presence(traj, times)
    assert p[1] == pytest.approx(1.0) and p[2] == pytest.approx(1.0)


def test_many_people_long_clip_is_correct_and_quick():
    n_people, n_frames = 12, 3000
    times = [round(i * 0.2, 6) for i in range(n_frames)]
    traj = {pid: [(t, 3.0 * pid, 0.0) for t in times] for pid in range(n_people)}  # 3 m apart: never close
    traj[n_people] = [(t, 3.0 * 5 - 0.5, 0.0) for t in times]                      # 0.5 m from person 5 only
    f = interaction_features(traj, contact_dist=2.0, contact_min=5.0)
    assert f[5]["partners"] == [n_people] and f[n_people]["partners"] == [5]
    assert sum(v["contacts"] for v in f.values()) == 2


def test_presence_counts_only_given_frame_times():
    times = _times(51)
    traj = {1: [(t, 0, 0) for t in _times(101)], 2: [(t, 0, 0) for t in times[10:]]}
    p = presence(traj, times)
    assert p[1] == pytest.approx(1.0) and p[2] == pytest.approx(41 / 51)


def test_outfit_peers_edge_cases():
    assert outfit_peers({1: np.array([1.0, 0.0])}, 0.5) == {1: 0}
    same = {1: np.array([1.0, 1.0]), 2: np.array([2.0, 2.0]), 3: np.array([0.0, 0.0])}
    assert outfit_peers(same, 0.99) == {1: 1, 2: 1, 3: 0}       # zero vector matches nobody
    assert outfit_peers(same, 1.0 + 1e-3) == {1: 0, 2: 0, 3: 0}


def test_suggest_roles_missing_keys_and_extra_features():
    s = suggest_roles({4: dict(contacts=2, partners=[1, 2], note="x", extra=7)}, 120.0, W)[0]
    assert s.pid == 4 and 0.0 <= s.confidence <= 1.0
    assert s.features["contacts"] == 2 and s.features["extra"] == 7
    assert "partners" not in s.features and "note" not in s.features       # flat numeric dict only
    # missing presence is neutral, not penalised: same as presence = 0.5
    t = suggest_roles({4: dict(outfit_peers=0, contacts=2, presence=0.5, initiated=0)}, 120.0, W)[0]
    assert s.confidence == pytest.approx(t.confidence)


def test_reason_texts_and_limits():
    s = suggest_roles({1: dict(outfit_peers=1, contacts=1, presence=0.95, initiated=1)}, 120.0, W)[0]
    assert s.role == "staff"
    assert set(s.reasons) <= {"same outfit as 1 other", "talked to 1 person",
                              "on screen 95% of the clip", "approached 1 person first"}
    assert len(s.reasons) == 3
    c = suggest_roles({1: dict(outfit_peers=0, contacts=0, presence=0.1, initiated=0)}, 120.0, W)[0]
    assert c.role == "customer"
    assert "on screen 10% of the clip" in c.reasons
    assert len(c.reasons) == 3 and set(c.reasons) <= {
        "outfit unlike anyone else", "talked to 0 people", "on screen 10% of the clip",
        "never approached anyone first"}


def test_presence_never_a_reason_in_short_clips_for_any_role():
    for feats in (dict(outfit_peers=3, contacts=4, presence=0.9, initiated=3),
                  dict(outfit_peers=0, contacts=0, presence=0.1, initiated=0),
                  dict(outfit_peers=0, contacts=2, presence=0.5, initiated=2)):
        s = suggest_roles({1: feats}, 59.9, W)[0]
        assert not any("on screen" in r for r in s.reasons) and s.reasons


def test_numpy_scalars_are_accepted_and_returned_as_python_numbers():
    feats = {np.int64(3): dict(outfit_peers=np.int64(2), contacts=np.int64(3),
                               presence=np.float64(0.9), initiated=np.int64(1))}
    s = suggest_roles(feats, 120.0, W)[0]
    assert s.pid == 3 and type(s.pid) is int
    assert type(s.features["contacts"]) is int and type(s.features["presence"]) is float
    assert "talked to 3 people" in s.reasons


def test_results_sorted_by_pid():
    feats = {5: dict(contacts=0), 2: dict(contacts=4), 9: dict(contacts=1)}
    assert [s.pid for s in suggest_roles(feats, 120.0, W)] == [2, 5, 9]
