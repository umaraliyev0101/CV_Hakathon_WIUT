"""Offline analysis pass: perception -> persons -> geometry -> roles -> events -> metrics, and the fast re-score."""
import dataclasses
import json
import time

import numpy as np
import pytest

from showroom_coach.analysis import analyze, build_persons, compute_geometry, rescore
from showroom_coach.config import PRESETS
from showroom_coach.perception.pipeline import run_perception
from showroom_coach.perception.stack import fake_stack


@pytest.fixture(scope="module")
def meet_res(meet_video):
    # meet_video (conftest): 20 s, 30 fps, 1280x720; blue "staff" box (80x220) standing at x=600..680 the whole time;
    # red "customer" box (80x220) enters from the left edge at 4 s, walks until its box is x=470..550 at 8 s (a 50 px gap —
    # boxes must never touch or the fake detector merges them), stays until 18 s, then walks out left (gone by 20 s).
    return run_perception(meet_video, fake_stack())


def test_analyze_finds_two_people_and_greet(meet_res):
    an = analyze(meet_res, preset="short")
    assert len(an.persons) == 2
    blue = max(an.persons, key=lambda p: p.t_last - p.t_first).pid
    an2 = rescore(an, meet_res, roles={blue: "staff"})
    types = [e.type for e in an2.events]
    assert "greet" in types and "engagement" in types
    assert an2.scorecard.staff[0].staff == blue and an2.scorecard.staff[0].greets == 1


def test_rescore_is_fast_and_does_not_touch_perception(meet_res, monkeypatch):
    import showroom_coach.analysis as A
    an = analyze(meet_res, preset="short")
    monkeypatch.setattr(A, "merge_tracks", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-merged")))
    an2 = rescore(an, meet_res, roles={an.persons[0].pid: "staff"}, preset="dealership")
    assert an2.preset == "dealership" and an2.persons == an.persons


def test_analyze_no_people(tmp_path, make_video):
    v = make_video(tmp_path / "empty.avi", seconds=3, movers=[])
    an = analyze(run_perception(v, fake_stack()), preset="short")
    assert an.persons == [] and an.events == []
    assert "nobody detected — the camera may be too far or too dark" in an.warnings
    assert an.scorecard.store.customers == 0


def test_nobody_staff_warning(meet_res):
    an = analyze(meet_res, preset="short", roles={p: "customer" for p in [1, 2]})
    assert "nobody marked as staff — mark staff to get scorecards" in an.warnings


def test_approx_warning(meet_res):
    res = meet_res
    saved = res.obs_kpts.copy()
    try:
        res.obs_kpts[..., 2] = 0.0             # no keypoints → box fallback everywhere
        an = analyze(res, preset="short")
        assert an.approx_share == pytest.approx(1.0)
        assert any(w.startswith("distance is approximate") for w in an.warnings)
    finally:
        res.obs_kpts[:] = saved


def test_to_dict_is_json(meet_res):
    import json
    json.dumps(analyze(meet_res, preset="short").to_dict())


# --------------------------------------------------------------------------- persons from hand-built tracks

def _hand_res(base, tracks, n_frames=60, fps=5.0, emb=None):
    """A copy of `base` whose observations are `tracks`: {track_id: [(frame, x1, score), ...]} of 60x165 boxes."""
    rows = sorted((f, tid, x1, s) for tid, obs in tracks.items() for f, x1, s in obs)
    m = len(rows)
    frame = np.array([r[0] for r in rows], np.int64)
    bbox = np.array([(r[2], 187.0, r[2] + 60.0, 352.0) for r in rows], np.float64).reshape(-1, 4)
    duration = n_frames / fps
    video = dataclasses.replace(base.video, duration=duration, src_duration=duration, src_frames=n_frames * 6,
                                sample_fps=fps, trimmed=False)
    emb = {} if emb is None else emb
    return dataclasses.replace(
        base, video=video, frame_times=np.arange(n_frames) / fps, src_frame_idx=np.arange(n_frames) * 6,
        obs_frame=frame, obs_track=np.array([r[1] for r in rows], np.int64), obs_bbox=bbox,
        obs_score=np.array([r[3] for r in rows], np.float64), obs_kpts=np.zeros((m, 17, 3), np.float32),
        raw_frame=frame.copy(), raw_bbox=bbox.copy(), track_emb=emb, track_outfit={}, crops_dir=None)


def _unit(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


def test_build_persons_drops_noise_and_merges_fragments(meet_res):
    tracks = {
        1: [(f, 100.0, 0.9) for f in range(0, 10)],     # person A, first fragment
        2: [(f, 120.0, 0.9) for f in range(20, 35)],    # person A again after a gap (longer fragment)
        3: [(f, 700.0, 0.9) for f in range(0, 2)],      # 2 observations: noise
        4: [(f, 800.0, 0.9) for f in range(0, 5)],      # 5 observations over 0.8 s: noise
        5: [(f, 400.0, 0.9) for f in range(0, 30)],     # person B, no embedding at all
    }
    emb = {1: _unit(1, 0, 0), 2: _unit(1, 0.05, 0), 3: _unit(1, 0, 0), 4: _unit(1, 0, 0)}
    res = _hand_res(meet_res, tracks, emb=emb)
    persons, obs_pid = build_persons(res)
    assert [(p.pid, p.track_ids) for p in persons] == [(1, [1, 2]), (2, [5])]
    a, b = persons
    assert (a.n_obs, a.t_first, a.t_last, a.thumb) == (25, 0.0, pytest.approx(6.8), "2.jpg")
    assert (b.n_obs, b.thumb) == (30, "5.jpg")
    assert set(obs_pid[np.isin(res.obs_track, [3, 4])].tolist()) == {0}
    assert set(obs_pid[res.obs_track == 5].tolist()) == {2}


def test_one_observation_per_person_per_frame(meet_res):
    tracks = {1: [(f, 100.0, 0.9) for f in range(0, 11)],     # both see frame 10; the merge allows 0.2 s overlap
              2: [(f, 110.0, 0.5) for f in range(10, 21)]}
    res = _hand_res(meet_res, tracks, emb={1: _unit(0, 1, 0), 2: _unit(0, 1, 0)})
    persons, obs_pid = build_persons(res)
    assert len(persons) == 1 and persons[0].n_obs == 21
    at10 = np.flatnonzero(res.obs_frame == 10)
    kept = [int(i) for i in at10 if obs_pid[i] > 0]
    assert len(kept) == 1 and res.obs_track[kept[0]] == 1   # the higher-score observation wins


def test_person_without_embedding_is_its_own_person(meet_res):
    res = dataclasses.replace(meet_res, track_emb={}, track_outfit={})
    an = analyze(res, preset="short")
    assert [p.pid for p in an.persons] == [1, 2]
    assert all(s.features["outfit_peers"] == 0 for s in an.suggestions.values())


def test_single_person_warns_store_only(meet_res):
    res = _hand_res(meet_res, {1: [(f, 100.0, 0.9) for f in range(0, 30)]}, emb={1: _unit(1, 1, 0)})
    an = analyze(res, preset="short")
    assert len(an.persons) == 1
    assert "fewer than 2 people — store metrics only" in an.warnings
    assert "nobody detected — the camera may be too far or too dark" not in an.warnings


def test_trimmed_warning(meet_res):
    res = dataclasses.replace(meet_res, video=dataclasses.replace(meet_res.video, trimmed=True, max_seconds=15.0))
    assert "video trimmed to the first 15 s" in analyze(res, preset="short").warnings


# --------------------------------------------------------------------------- geometry

def test_geometry_arrays_align_with_observations(meet_res):
    xz, src, facing, edge = compute_geometry(meet_res)
    m = len(meet_res.obs_track)
    assert xz.shape == (m, 2) and src.shape == (m,) and facing.shape == (m, 2) and edge.shape == (m,)
    assert np.isfinite(xz).all() and (src == 0).all()               # fake pose: full standing skeletons
    assert np.isnan(facing).any(axis=1).sum() < m and edge.dtype == bool
    assert analyze(meet_res, preset="short").approx_share == pytest.approx(0.0)


def test_geometry_box_fallback_and_frame_edge(meet_res):
    res = _hand_res(meet_res, {1: [(f, 0.0, 0.9) for f in range(10)], 2: [(f, 400.0, 0.9) for f in range(10)]})
    xz, src, facing, edge = compute_geometry(res)
    assert (src == 2).all() and np.isnan(facing).all()               # no keypoints: box height, unknown facing
    assert edge[res.obs_track == 1].all() and not edge[res.obs_track == 2].any()
    assert (xz[res.obs_track == 1, 0] < xz[res.obs_track == 2, 0]).all()


# --------------------------------------------------------------------------- roles, names, re-score

def test_overrides_for_unknown_pids_are_ignored(meet_res):
    an = analyze(meet_res, preset="short", roles={1: "staff", 99: "staff"}, names={1: "Aziz", 42: "Ghost"})
    assert an.roles == {1: "staff", 2: "customer"} and an.overrides == {1: "staff"}
    assert an.names == {1: "Aziz"}
    assert an.scorecard.staff[0].name == "Aziz"


def test_final_role_is_suggestion_with_unsure_as_customer(meet_res):
    an = analyze(meet_res, preset="short")
    for pid, sug in an.suggestions.items():
        assert an.roles[pid] == ("staff" if sug.role == "staff" else "customer")
        assert sug.reasons


def test_rescore_keeps_earlier_overrides_and_switches_preset(meet_res):
    an = analyze(meet_res, preset="short", roles={1: "staff"})
    an2 = rescore(an, meet_res, roles={2: "customer"}, names={1: "Aziz"}, preset="dealership")
    assert an2.roles == {1: "staff", 2: "customer"} and an2.overrides == {1: "staff", 2: "customer"}
    assert an2.thresholds == PRESETS["dealership"] and an2.names == {1: "Aziz"}
    assert "engagement" not in [e.type for e in an2.events]       # 120 s engagement rule cannot fire in 20 s
    an3 = rescore(an2, meet_res, preset="short")
    assert [(e.type, e.t) for e in an3.events] == [(e.type, e.t) for e in an.events]
    assert an.preset == "short" and an.names == {}                 # the earlier analysis is untouched


def test_rescore_rejects_unknown_preset(meet_res):
    an = analyze(meet_res, preset="short")
    with pytest.raises(ValueError):
        rescore(an, meet_res, preset="nope")


# --------------------------------------------------------------------------- timelines and JSON

def test_timelines(meet_res):
    an = rescore(analyze(meet_res, preset="short"), meet_res, roles={1: "staff"})
    staff, cust = an.timelines[1], an.timelines[2]
    assert staff["present"] == [[0.0, pytest.approx(19.8)]] and staff["waiting"] == [] and staff["greeted"] is None
    (p0, p1), = cust["present"]
    assert p0 == pytest.approx(4.2, abs=0.3) and p1 == pytest.approx(18.0, abs=0.3)
    greet = next(e for e in an.events if e.type == "greet")
    assert cust["greeted"] == greet.t
    assert cust["waiting"] == [[p0, greet.t]]
    eng = next(e for e in an.events if e.type == "engagement")
    assert cust["engaged"] == [[eng.t, eng.t_end]] and staff["engaged"] == [[eng.t, eng.t_end]]


def test_to_dict_matches_the_api_contract(meet_res):
    d = analyze(meet_res, preset="short", roles={1: "staff"}, names={1: "Aziz"}).to_dict()
    assert list(d) == ["preset", "duration", "thresholds", "persons", "suggestions", "roles", "names", "events",
                       "scorecard", "timelines", "warnings"]
    assert set(d["roles"]) == {"1", "2"} and d["names"] == {"1": "Aziz"} and set(d["timelines"]) == {"1", "2"}
    assert d["thresholds"]["greet_dist"] == 1.5 and d["duration"] == 20.0
    assert set(d["suggestions"]["1"]) == {"pid", "role", "confidence", "reasons", "features"}
    assert set(d["persons"][0]) == {"pid", "track_ids", "t_first", "t_last", "n_obs", "thumb"}
    assert json.loads(json.dumps(d, allow_nan=False)) == d


# --------------------------------------------------------------------------- real clip (slow)

@pytest.mark.slow
def test_real_clip_end_to_end():
    from showroom_coach import models as M
    from showroom_coach.config import DEFAULT_DETECTOR, cache_dir, repo_root
    from showroom_coach.perception.pipeline import cached_perception
    from showroom_coach.perception.stack import default_stack

    clip = repo_root() / "data" / "samples" / "meva_a_60s.mp4"
    if not clip.is_file():
        pytest.skip("data/samples/meva_a_60s.mp4 is missing")
    for name in (DEFAULT_DETECTOR.partition("@")[0], "rtmpose_s", "dinov2_small"):
        if not (M.models_dir() / name / M.MODELS[name].filename).is_file():
            pytest.skip(f"model {name} not downloaded")
    # run_perception through the per-video cache (runs/cache, gitignored): about 2 min the first time, then seconds,
    # which keeps the whole suite inside its 3-minute budget.
    res, _ = cached_perception(clip, default_stack(), cache_dir())
    t0 = time.perf_counter()
    an = analyze(res, preset="short")
    t_analyze = time.perf_counter() - t0
    staff = [pid for pid, r in an.roles.items() if r == "staff"] or [an.persons[0].pid]
    t0 = time.perf_counter()
    an2 = rescore(an, res, roles={pid: "staff" for pid in staff})
    t_rescore = time.perf_counter() - t0
    print(f"\nreal clip: {len(an.persons)} persons, {len(staff)} staff, analyze {t_analyze:.2f} s, "
          f"rescore {t_rescore:.3f} s, events {sorted({e.type for e in an2.events})}")
    assert an.persons and an.suggestions
    assert all(s.reasons for s in an.suggestions.values())
    json.dumps(an2.to_dict(), allow_nan=False)
    assert t_rescore < 1.0
