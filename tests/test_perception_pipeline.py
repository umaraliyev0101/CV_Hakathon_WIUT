"""Perception pipeline: detect -> track -> pose -> embed over a video, the per-video cache, and the face blur
helpers that make the saved track crops safe to show."""
import inspect
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from showroom_coach import config
from showroom_coach.analysis import build_persons
from showroom_coach.perception import pipeline
from showroom_coach.perception.pipeline import (
    cache_key, cached_perception, carry_keypoints, load_result, perception_params, run_perception, save_result,
    video_key,
)
from showroom_coach.perception.stack import PerceptionStack, fake_stack
from showroom_coach.perception.tracking import Tracker
from showroom_coach.render import blur_faces, face_region
from showroom_coach.sources import VideoError
from showroom_coach.types import Det


def test_pipeline_on_synthetic_video(synth_video, tmp_path):
    seen = []
    res = run_perception(synth_video, fake_stack(), progress=lambda f, m: seen.append(f), crops_dir=tmp_path / "crops")
    assert len(res.frame_times) == 30
    ids = res.track_ids()
    assert len(ids) == 2                                    # red mover + blue stander, stable ids
    assert all(res.obs_idx(i).size >= 20 for i in ids)
    assert set(res.track_emb) == set(ids) and set(res.track_outfit) == set(ids)
    assert all((tmp_path / "crops" / f"{i}.jpg").exists() for i in ids)
    assert seen and seen[-1] <= 1.0


def test_save_load_roundtrip(synth_video, tmp_path):
    res = run_perception(synth_video, fake_stack())
    save_result(res, tmp_path / "c")
    back = load_result(tmp_path / "c")
    assert np.array_equal(back.obs_bbox, res.obs_bbox) and back.video == res.video
    assert set(back.track_emb) == set(res.track_emb)


def test_cache_hit_skips_detection(synth_video, tmp_path):
    st = fake_stack()
    calls = {"n": 0}
    det = st.detect
    st.detect = lambda f: (calls.__setitem__("n", calls["n"] + 1), det(f))[1]
    _, cached1 = cached_perception(synth_video, st, tmp_path)
    n1 = calls["n"]
    _, cached2 = cached_perception(synth_video, st, tmp_path)
    assert (cached1, cached2) == (False, True) and calls["n"] == n1


def test_face_region_from_keypoints_and_fallback():
    k = np.zeros((17, 3)); k[0] = (50, 30, 0.9); k[1] = (45, 25, 0.9); k[2] = (55, 25, 0.9)
    x1, y1, x2, y2 = face_region((20, 10, 80, 200), k)
    assert x1 <= 45 and x2 >= 55 and y1 <= 25 and y2 >= 30
    assert face_region((20, 10, 80, 210), None) == (20, 10, 80, 50)


def test_blur_changes_only_face():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)
    out = blur_faces(img, [(20, 10, 80, 210)], [None])
    assert out[10:50, 20:80].std() < img[10:50, 20:80].std() * 0.6
    assert np.array_equal(out[100:, :], img[100:, :])


# --------------------------------------------------------------------------- render helpers, edge cases


def test_face_region_min_side_and_confidence_gate():
    k = np.zeros((17, 3)); k[0] = (50, 30, 0.9)                    # only the nose is confident
    x1, y1, x2, y2 = face_region((20, 10, 80, 200), k)
    assert x2 - x1 >= 30 and y2 - y1 >= 30                         # min side 0.5 * box width (60)
    assert x1 <= 50 <= x2 and y1 <= 30 <= y2
    k[:, 2] = 0.1                                                  # nothing confident -> top 20 % of the box
    assert face_region((20, 10, 80, 210), k) == (20, 10, 80, 50)
    k2 = np.zeros((17, 3)); k2[5:] = (50, 100, 0.99)               # confident body points are not faces
    assert face_region((20, 10, 80, 210), k2) == (20, 10, 80, 50)


def test_face_region_distrusts_keypoints_outside_the_box():
    k = np.zeros((17, 3)); k[0] = (500, 300, 0.9)
    assert face_region((20, 10, 80, 210), k) == (20, 10, 80, 50)


def test_blur_faces_clips_to_frame_and_blurs_everyone():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    boxes = [(-30, -10, 40, 150), (60, 20, 160, 120), (200, 200, 300, 300)]   # partly / fully outside the frame
    out = blur_faces(img, boxes, [None])                           # kpts_list shorter than bboxes: still blur all
    assert out.shape == img.shape and out is not img
    assert out[0:30, 0:40].std() < img[0:30, 0:40].std() * 0.6
    assert out[20:40, 60:100].std() < img[20:40, 60:100].std() * 0.6
    assert np.array_equal(out[60:, :], img[60:, :])
    assert blur_faces(img, [], None).shape == img.shape


def test_blur_faces_leaves_input_untouched():
    img = np.random.default_rng(2).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    before = img.copy()
    blur_faces(img, [(10, 10, 60, 100)], None)
    assert np.array_equal(img, before)


# --------------------------------------------------------------------------- pipeline behaviour


def test_progress_is_monotonic_and_ends_at_one(synth_video):
    seen = []
    run_perception(synth_video, fake_stack(), progress=lambda f, m: seen.append((f, m)))
    fracs = [f for f, _ in seen]
    assert len(fracs) >= 3 and fracs[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fracs) and fracs == sorted(fracs)
    assert all(isinstance(m, str) and m for _, m in seen)


def test_progress_is_reported_while_describing_people(synth_video):
    """Embedding every real track happens after the frame loop and can take a while (one DINOv2 run per crop), so it
    reports progress too: one embedder call per track, a report before the first and after each, in the top
    1 - FRAMES_PROGRESS of the range; the frame loop stays below that."""
    events = []
    base = fake_stack()

    def embed(crops):
        events.append(("embed", len(crops)))
        return base.embed(crops)

    stack = PerceptionStack("fake", base.detect, base.pose, embed)
    plain = run_perception(synth_video, fake_stack())
    res = run_perception(synth_video, stack, progress=lambda f, m: events.append(("progress", f, m)))
    calls = [i for i, e in enumerate(events) if e[0] == "embed"]
    describing = [(i, e[1], e[2]) for i, e in enumerate(events) if e[0] == "progress" and e[2].startswith("describing")]
    assert len(calls) == 2                                              # synth_video: two real tracks
    assert [m for _, _, m in describing] == [f"describing people… {i} of 2" for i in range(3)]
    assert describing[0][0] < calls[0] < describing[1][0] < calls[1] < describing[2][0]     # during embedding
    fracs = [f for _, f, _ in describing]
    assert fracs[0] == pytest.approx(pipeline.FRAMES_PROGRESS) and fracs == sorted(fracs) and fracs[-1] == 1.0
    loop = [e[1] for e in events[:calls[0]] if e[0] == "progress" and not e[2].startswith("describing")]
    assert loop and max(loop) <= pipeline.FRAMES_PROGRESS
    for tid in plain.track_emb:                                         # per-track calls change no descriptor
        assert np.array_equal(res.track_emb[tid], plain.track_emb[tid])


def test_result_arrays_are_consistent(synth_video):
    res = run_perception(synth_video, fake_stack())
    m = len(res.obs_track)
    assert res.obs_frame.shape == (m,) and res.obs_score.shape == (m,)
    assert res.obs_bbox.shape == (m, 4) and res.obs_kpts.shape == (m, 17, 3)
    assert res.obs_frame.min() >= 0 and res.obs_frame.max() < len(res.frame_times)
    assert np.all(np.diff(res.frame_times) > 0) and res.src_frame_idx.shape == res.frame_times.shape
    assert res.src_frame_idx[1] - res.src_frame_idx[0] == 6          # 30 fps source sampled at 5 fps
    assert res.raw_bbox.shape == (len(res.raw_frame), 4) and len(res.raw_frame) >= m
    assert (res.obs_kpts[..., 2] > 0).any()                           # the fake stack's pose reached the result
    assert res.cars == [] and res.stack_name == "fake" and res.crops_dir is None
    assert set(res.timings) == {"decode", "detect", "track", "pose", "embed", "total"}
    assert all(v >= 0 for v in res.timings.values())
    assert res.timings["total"] >= max(v for k, v in res.timings.items() if k != "total")
    for i in res.track_ids():
        idx = res.obs_idx(i)
        assert np.all(res.obs_track[idx] == i) and np.all(np.diff(res.obs_frame[idx]) > 0)
        for d in (res.track_emb[i], res.track_outfit[i]):
            assert d.ndim == 1 and abs(float(np.linalg.norm(d)) - 1.0) < 1e-4


def test_descriptors_separate_the_two_people(synth_video):
    res = run_perception(synth_video, fake_stack())
    a, b = res.track_ids()
    assert float(res.track_emb[a] @ res.track_emb[b]) < 0.9
    assert float(res.track_outfit[a] @ res.track_outfit[b]) < 0.9


def test_without_pose_keypoints_are_zero(synth_video):
    st = fake_stack()
    st.pose = None
    res = run_perception(synth_video, st)
    assert res.obs_kpts.shape == (len(res.obs_track), 17, 3) and not res.obs_kpts.any()
    assert len(res.track_ids()) == 2


def test_without_embedder_uses_hsv_only(synth_video):
    st = fake_stack()
    st.embed = None
    res = run_perception(synth_video, st)
    a, b = res.track_ids()
    assert res.track_emb[a].shape == (128,) and res.track_outfit[a].shape == (128,)
    assert float(res.track_emb[a] @ res.track_emb[b]) < 0.9


def test_descriptor_shapes_with_embedder(synth_video):
    res = run_perception(synth_video, fake_stack())
    i = res.track_ids()[0]
    assert res.track_emb[i].shape == (131 + 128,) and res.track_outfit[i].shape == (128 + 131,)


def test_empty_video_has_no_tracks(make_video, tmp_path):
    path = make_video(tmp_path / "empty.avi", seconds=2.0, fps=30, size=(320, 240), movers=[])
    seen = []
    res = run_perception(path, fake_stack(), progress=lambda f, m: seen.append(f), crops_dir=tmp_path / "crops")
    assert res.track_ids() == [] and res.track_emb == {} and res.track_outfit == {}
    assert res.obs_bbox.shape == (0, 4) and res.obs_kpts.shape == (0, 17, 3) and res.raw_bbox.shape == (0, 4)
    assert len(res.frame_times) == 10 and seen[-1] == 1.0
    save_result(res, tmp_path / "c")
    back = load_result(tmp_path / "c")
    assert back.track_ids() == [] and back.obs_kpts.shape == (0, 17, 3) and len(back.frame_times) == 10


def test_cars_come_from_the_first_frame_only(make_video, tmp_path):
    path = make_video(tmp_path / "v.avi", seconds=1.0, fps=30, size=(320, 240), movers=[])
    calls = {"n": 0}

    def detect(frame):
        calls["n"] += 1
        if calls["n"] == 1:
            return [Det((10, 20, 110, 90), 0.8, 2), Det((150, 60, 250, 120), 0.6, 7), Det((0, 0, 30, 90), 0.9, 0)]
        return [Det((20, 30, 120, 100), 0.9, 2)]

    res = run_perception(path, PerceptionStack("cars", detect, None, None))
    assert res.cars == [[10.0, 20.0, 110.0, 90.0, 0.8], [150.0, 60.0, 250.0, 120.0, 0.6]]
    assert res.raw_frame.tolist() == [0] and res.raw_bbox.tolist() == [[0.0, 0.0, 30.0, 90.0]]   # persons only


def test_max_seconds_trims(synth_video):
    res = run_perception(synth_video, fake_stack(), max_seconds=3.0)
    assert len(res.frame_times) == 15 and res.video.trimmed and res.video.duration == pytest.approx(3.0)


def test_unreadable_input_raises_video_error(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_text("not a video")
    with pytest.raises(VideoError):
        run_perception(bad, fake_stack())
    with pytest.raises(VideoError):
        video_key(tmp_path / "missing.mp4", 5.0, 960, 300.0, "fake")


# --------------------------------------------------------------------------- pose stride


def _pose_logging_stack():
    st = fake_stack()
    seen = []
    real = st.pose

    def pose(frame, boxes):
        seen.append(len(boxes))
        return real(frame, boxes)

    st.pose = pose
    return st, seen


def test_carry_keypoints_follows_the_box():
    k = np.zeros((17, 3), np.float32)
    k[0] = (150, 200, 0.8)
    k[1] = (100, 100, 0.5)
    out = carry_keypoints(k, (100, 100, 200, 300), (120, 100, 220, 260))      # moved right 20, 20% shorter
    np.testing.assert_allclose(out[0], (170, 180, 0.8), atol=1e-4)            # x' = x1' + (x-x1) w'/w, conf kept
    np.testing.assert_allclose(out[1], (120, 100, 0.5), atol=1e-4)
    out = carry_keypoints(k, (100, 100, 200, 300), (100, 100, 150, 300))      # half as wide
    np.testing.assert_allclose(out[0], (125, 200, 0.8), atol=1e-4)
    assert k[0, 0] == 150                                                       # the input is not modified
    assert np.array_equal(carry_keypoints(k, (5, 5, 5, 9), (0, 0, 10, 10)), k)   # degenerate old box: unchanged


def test_pose_every_defaults_to_two():
    assert inspect.signature(run_perception).parameters["pose_every"].default == 2


def test_pose_stride_halves_pose_work(synth_video):
    st1, seen1 = _pose_logging_stack()
    every = run_perception(synth_video, st1, pose_every=1)
    st2, seen2 = _pose_logging_stack()
    strided = run_perception(synth_video, st2)
    assert sum(seen1) == len(every.obs_track)                       # stride 1: every tracked box gets pose
    assert 0.4 * sum(seen1) <= sum(seen2) <= 0.6 * sum(seen1)       # stride 2: about half
    assert len(strided.obs_track) == len(every.obs_track)
    assert (strided.obs_kpts[..., 2] > 0).all()                     # skipped frames still have keypoints


def test_carried_keypoints_follow_a_moving_box(synth_video):
    every = run_perception(synth_video, fake_stack(), pose_every=1)
    strided = run_perception(synth_video, fake_stack(), pose_every=2)
    # the fake skeleton is an affine function of its box, so a carry mapped by the box change must equal fresh pose,
    # while an unmapped (stale) carry would be off by the red walker's ~27 px per frame
    np.testing.assert_allclose(strided.obs_kpts, every.obs_kpts, atol=1e-2)
    red = max(every.track_ids(), key=lambda i: np.ptp(every.obs_bbox[every.obs_idx(i), 0]))
    assert np.ptp(every.obs_kpts[every.obs_idx(red), 0, 0]) > 300


def test_each_track_gets_fresh_pose_on_its_first_frame_and_every_other_frame_after(synth_video):
    calls = {"n": 0}
    real = fake_stack().pose

    def pose(frame, boxes):
        calls["n"] += 1
        out = real(frame, boxes)
        out[..., 2] = 0.1 + 0.01 * calls["n"]                       # every call leaves its own confidence
        return out

    st = fake_stack()
    st.pose = pose
    res = run_perception(synth_video, st, pose_every=2)
    for i in res.track_ids():
        conf = res.obs_kpts[res.obs_idx(i), 0, 2]
        assert np.array_equal(conf[1::2], conf[0:len(conf) - 1:2])      # odd observations carry the one before
        assert np.all(conf[2::2] != conf[0:-2:2])                       # even ones are fresh
        assert len(set(conf.tolist())) == (len(conf) + 1) // 2


def test_pose_every_one_runs_pose_for_every_box_and_zero_is_rejected(synth_video):
    st, seen = _pose_logging_stack()
    res = run_perception(synth_video, st, pose_every=1)
    assert sum(seen) == len(res.obs_track)
    with pytest.raises(ValueError):
        run_perception(synth_video, fake_stack(), pose_every=0)


def test_boxes_too_small_for_pose_are_retried_every_frame(make_video, tmp_path):
    path = make_video(tmp_path / "v.avi", seconds=1.2, fps=10, size=(320, 240), movers=[])
    n = {"i": -1}
    pose_frames = []

    def detect(frame):
        n["i"] += 1
        return [Det((20.0, 20.0, 60.0, 50.0 + 4.0 * n["i"]), 0.9, 0)]        # 30, 34, 38 px tall, then 42, 46, ...

    def pose(frame, boxes):
        pose_frames.append(n["i"])
        out = np.zeros((len(boxes), 17, 3), np.float32)
        for k, b in enumerate(boxes):
            if b[3] - b[1] >= 40:                                              # like PoseEstimator: tiny box -> zeros
                out[k, :, 2] = 0.9
        return out

    res = run_perception(path, PerceptionStack("small", detect, pose, None), sample_fps=10.0)
    (tid,) = res.track_ids()
    conf = res.obs_kpts[res.obs_idx(tid), 0, 2]
    assert (conf[:3] == 0).all() and np.allclose(conf[3:], 0.9)
    assert pose_frames[:6] == [0, 1, 2, 3, 5, 7]       # retried while too small, then every second frame


# --------------------------------------------------------------------------- only real tracks are described


def _flicker_video_stack(make_video, tmp_path):
    """30 frames at 10 fps. A: all frames. B: 2 frames. C: 4 frames (3 observations, 0.2 s). D: 13 frames (1.1 s)."""
    path = make_video(tmp_path / "v.avi", seconds=3.0, fps=10, size=(640, 240), movers=[])
    n = {"i": -1}

    def detect(frame):
        n["i"] += 1
        i = n["i"]
        out = [Det((20.0, 20.0, 80.0, 200.0), 0.9, 0)]
        if 5 <= i <= 6:
            out.append(Det((150.0, 20.0, 210.0, 200.0), 0.9, 0))
        if 10 <= i <= 13:
            out.append(Det((300.0, 20.0, 360.0, 200.0), 0.9, 0))
        if 15 <= i <= 27:
            out.append(Det((450.0, 20.0, 510.0, 200.0), 0.9, 0))
        return out

    return path, PerceptionStack("flicker", detect, None, None)


def test_min_track_constants_match_the_analysis_filter():
    assert config.MIN_TRACK_OBS == 3 and config.MIN_TRACK_S == 1.0
    sig = inspect.signature(build_persons).parameters          # analysis drops tracks by the very same constants
    assert sig["min_track_obs"].default == config.MIN_TRACK_OBS
    assert sig["min_track_s"].default == config.MIN_TRACK_S


def test_only_real_tracks_are_embedded_and_get_crops(make_video, tmp_path):
    path, st = _flicker_video_stack(make_video, tmp_path)
    res = run_perception(path, st, sample_fps=10.0, crops_dir=tmp_path / "crops")
    ids = res.track_ids()
    span = {i: float(np.ptp(res.frame_times[res.obs_frame[res.obs_idx(i)]])) for i in ids}
    n_obs = {i: len(res.obs_idx(i)) for i in ids}
    real = {i for i in ids if n_obs[i] >= 3 and span[i] >= 1.0}
    assert len(ids) == 4 and len(real) == 2 and sorted(n_obs.values()) == [1, 3, 12, 30]
    assert set(res.track_emb) == real and set(res.track_outfit) == real
    assert {int(p.stem) for p in (tmp_path / "crops").glob("*.jpg")} == real
    # the short tracks are still in the observations and their raw boxes are kept for face blurring
    assert set(res.obs_track.tolist()) == set(ids) and len(res.raw_frame) == 30 + 2 + 4 + 13


def test_real_track_boundary_is_three_observations_and_one_second(make_video, tmp_path):
    path = make_video(tmp_path / "v.avi", seconds=3.0, fps=10, size=(320, 240), movers=[])
    n = {"i": -1}

    def detect(frame):
        n["i"] += 1
        i = n["i"]
        out = []
        if 0 <= i <= 10:                                  # 11 sightings, 11 obs from frame 0: span exactly 1.0 s
            out.append(Det((20.0, 20.0, 80.0, 200.0), 0.9, 0))
        if 15 <= i <= 25:                                 # born later: first sighting is unconfirmed, 10 obs, 0.9 s
            out.append(Det((200.0, 20.0, 260.0, 200.0), 0.9, 0))
        return out

    res = run_perception(path, PerceptionStack("edge", detect, None, None), sample_fps=10.0)
    a, b = res.track_ids()
    assert len(res.obs_idx(a)) == 11 and len(res.obs_idx(b)) == 10
    assert float(np.ptp(res.frame_times[res.obs_frame[res.obs_idx(a)]])) == pytest.approx(1.0)
    assert float(np.ptp(res.frame_times[res.obs_frame[res.obs_idx(b)]])) == pytest.approx(0.9)
    assert set(res.track_emb) == {a}


def test_no_real_track_means_no_embedder_call(make_video, tmp_path):
    path = make_video(tmp_path / "v.avi", seconds=1.0, fps=10, size=(320, 240), movers=[])
    n = {"i": -1}

    def detect(frame):
        n["i"] += 1
        return [Det((20.0, 20.0, 80.0, 200.0), 0.9, 0)] if n["i"] in (2, 3) else []

    embedded = []
    st = PerceptionStack("none", detect, None, lambda crops: embedded.append(len(crops)) or np.zeros((len(crops), 4)))
    res = run_perception(path, st, sample_fps=10.0, crops_dir=tmp_path / "crops")
    assert res.track_ids() and res.track_emb == {} and embedded == []
    assert not list((tmp_path / "crops").glob("*.jpg"))


# --------------------------------------------------------------------------- crops are face-blurred


def test_saved_track_crops_have_the_face_blurred(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "noise.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(20):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    st = PerceptionStack("still", lambda f: [Det((50.0, 20.0, 150.0, 220.0), 0.9, 0)], None, None)
    res = run_perception(path, st, sample_fps=10.0, crops_dir=tmp_path / "crops")
    (tid,) = res.track_ids()
    img = cv2.imread(str(tmp_path / "crops" / f"{tid}.jpg"))
    assert img is not None and 100 <= img.shape[0] <= 160           # downscaled, not the full 220 px
    h, w = img.shape[:2]
    head = img[int(0.08 * h):int(0.19 * h), int(0.25 * w):int(0.75 * w)]
    body = img[int(0.5 * h):int(0.9 * h), int(0.25 * w):int(0.75 * w)]
    assert head.std() < body.std() * 0.35


# --------------------------------------------------------------------------- cache


def test_video_key_depends_on_content_params_and_stack(synth_video, tmp_path):
    k = video_key(synth_video, 5.0, 960, 300.0, "fake")
    assert len(k) == 16 and int(k, 16) >= 0
    assert k == video_key(synth_video, 5.0, 960, 300.0, "fake")
    assert k != video_key(synth_video, 10.0, 960, 300.0, "fake")
    assert k != video_key(synth_video, 5.0, 640, 300.0, "fake")
    assert k != video_key(synth_video, 5.0, 960, 60.0, "fake")
    assert k != video_key(synth_video, 5.0, 960, 300.0, "yolox_nano@2+rtmpose_s")
    copy = tmp_path / "copy.avi"
    copy.write_bytes(synth_video.read_bytes())
    assert video_key(copy, 5.0, 960, 300.0, "fake") == k            # same bytes, other path: same key
    copy.write_bytes(copy.read_bytes() + b"x")
    assert video_key(copy, 5.0, 960, 300.0, "fake") != k


def test_cache_files_load_without_pickle(synth_video, tmp_path):
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is False
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert d.name == cache_key(synth_video, perception_params(fake_stack()))
    with np.load(d / "perception.npz", allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)
    assert (d / "perception.json").is_file() and (d / "crops").is_dir()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp")]


def test_cached_result_matches_fresh_and_keeps_crops(synth_video, tmp_path):
    first, c1 = cached_perception(synth_video, fake_stack(), tmp_path)
    second, c2 = cached_perception(synth_video, fake_stack(), tmp_path)
    assert (c1, c2) == (False, True)
    for name in ("frame_times", "src_frame_idx", "obs_frame", "obs_track", "obs_bbox", "obs_score", "obs_kpts",
                 "raw_frame", "raw_bbox"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
    for i in first.track_ids():
        assert np.array_equal(first.track_emb[i], second.track_emb[i])
        assert np.array_equal(first.track_outfit[i], second.track_outfit[i])
        assert (Path(second.crops_dir) / f"{i}.jpg").is_file()
    assert Path(second.crops_dir).parent.parent == tmp_path        # crops live inside the cache entry
    assert second.timings == first.timings and second.cars == first.cars


def test_cache_hit_does_not_call_run_perception(synth_video, tmp_path, monkeypatch):
    cached_perception(synth_video, fake_stack(), tmp_path)

    def boom(*a, **k):
        raise AssertionError("perception re-ran on a cache hit")

    monkeypatch.setattr(pipeline, "run_perception", boom)
    res, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True and len(res.track_ids()) == 2


def test_cache_miss_calls_run_perception_with_params(synth_video, tmp_path, monkeypatch):
    seen = {}
    real = pipeline.run_perception

    def spy(path, stack, **kw):
        seen.update(kw)
        return real(path, stack, **kw)

    monkeypatch.setattr(pipeline, "run_perception", spy)
    msgs = []
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path, progress=lambda f, m: msgs.append(f),
                                  sample_fps=5.0, max_seconds=4.0)
    assert cached is False and seen["max_seconds"] == 4.0 and seen["sample_fps"] == 5.0
    assert msgs and msgs[-1] == 1.0
    msgs.clear()
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path, progress=lambda f, m: msgs.append(f),
                                  sample_fps=5.0, max_seconds=4.0)
    assert cached is True and msgs == [1.0]
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path, max_seconds=5.0)    # other params: other entry
    assert cached is False


def test_cached_result_points_at_the_current_video_path(synth_video, tmp_path):
    cached_perception(synth_video, fake_stack(), tmp_path)
    copy = tmp_path / "renamed.avi"
    copy.write_bytes(synth_video.read_bytes())
    res, cached = cached_perception(copy, fake_stack(), tmp_path)
    assert cached is True and res.video.path == str(copy)


def test_corrupt_cache_entry_is_recomputed(synth_video, tmp_path):
    cached_perception(synth_video, fake_stack(), tmp_path)
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    (d / "perception.npz").write_bytes(b"garbage")
    res, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is False and len(res.track_ids()) == 2
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True


# --- the key covers everything that changes what an entry contains


def _key(path, stack=None, **kw):
    return cache_key(path, perception_params(stack or fake_stack(), **kw))


def _tracker_default(monkeypatch, name, value):
    names = list(inspect.signature(Tracker.__init__).parameters)[1:]            # fps, lost_seconds, activation, ...
    defaults = list(Tracker.__init__.__defaults__)
    defaults[names.index(name)] = value
    monkeypatch.setattr(Tracker.__init__, "__defaults__", tuple(defaults))


def test_cache_key_covers_every_content_changing_parameter(synth_video, monkeypatch):
    base = _key(synth_video)
    assert base == _key(synth_video) and len(base) == 16
    keys = {
        "sample_fps": _key(synth_video, sample_fps=10.0),
        "max_width": _key(synth_video, max_width=640),
        "max_seconds": _key(synth_video, max_seconds=4.0),
        "pose_every arg": _key(synth_video, pose_every=1),
        "crops_per_track arg": _key(synth_video, crops_per_track=2),
        "stack name": _key(synth_video, PerceptionStack("other", fake_stack().detect, None, None)),
    }
    for name, value in (("DEFAULT_POSE_EVERY", 3), ("DEFAULT_CROPS_PER_TRACK", 3),    # defaults are effective values
                        ("MIN_TRACK_OBS", 4), ("MIN_TRACK_S", 2.0)):
        with monkeypatch.context() as m:
            m.setattr(pipeline, name, value)
            keys[name] = _key(synth_video)
    for name, value in (("activation", 0.4), ("lost_seconds", 3.0), ("match_thr", 0.7)):
        with monkeypatch.context() as m:
            _tracker_default(m, name, value)
            keys["tracker " + name] = _key(synth_video)
    assert len(set(keys.values()) | {base}) == len(keys) + 1, keys          # every one of them gives its own key
    assert base not in keys.values()


def test_explicit_values_equal_to_the_defaults_give_the_same_key(synth_video):
    assert _key(synth_video, pose_every=2, crops_per_track=4, sample_fps=5, max_width=960, max_seconds=300) == \
        _key(synth_video)


def test_a_tracker_that_exposes_its_settings_is_keyed_by_them(synth_video):
    def with_tracker(**settings):
        st = fake_stack()
        st.new_tracker = lambda fps: SimpleNamespace(**settings)
        return st

    a = _key(synth_video, with_tracker(activation=0.2, lost_seconds=2.0, match_thr=0.8))
    assert a == _key(synth_video, with_tracker(activation=0.2, lost_seconds=2.0, match_thr=0.8))
    assert a != _key(synth_video, with_tracker(activation=0.3, lost_seconds=2.0, match_thr=0.8))
    assert a != _key(synth_video, with_tracker(activation=0.2, lost_seconds=1.0, match_thr=0.8))
    assert a != _key(synth_video, with_tracker(activation=0.2, lost_seconds=2.0, match_thr=0.7))


def test_perception_params_lists_the_effective_settings(synth_video):
    p = perception_params(fake_stack())
    assert p["sample_fps"] == 5.0 and p["max_width"] == 960 and p["max_seconds"] == 300.0 and p["stack"] == "fake"
    assert p["pose_every"] == pipeline.DEFAULT_POSE_EVERY == 2 and p["crops_per_track"] == 4
    assert p["min_track_obs"] == config.MIN_TRACK_OBS and p["min_track_s"] == config.MIN_TRACK_S
    assert p["tracker"]["activation"] == 0.25 and p["tracker"]["lost_seconds"] == 2.0 and p["tracker"]["match_thr"] == 0.8
    assert p["tracker"]["class"].endswith("Tracker")
    assert perception_params(fake_stack(), crops_per_track=0)["crops_per_track"] == 1    # what run_perception uses
    with pytest.raises(ValueError):
        perception_params(fake_stack(), pose_every=0)
    with pytest.raises(TypeError):
        perception_params(fake_stack(), nonsense=1)


def test_the_params_are_stored_in_the_json_and_on_the_result(synth_video, tmp_path):
    res, _ = cached_perception(synth_video, fake_stack(), tmp_path)
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    meta = json.loads((d / "perception.json").read_text(encoding="utf-8"))
    assert meta["params"] == perception_params(fake_stack()) == res.params
    again, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True and again.params == res.params
    direct = run_perception(synth_video, fake_stack(), pose_every=1, crops_per_track=2)
    assert direct.params["pose_every"] == 1 and direct.params["crops_per_track"] == 2


def test_an_entry_whose_stored_params_disagree_is_recomputed(synth_video, tmp_path):
    cached_perception(synth_video, fake_stack(), tmp_path)
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    meta = json.loads((d / "perception.json").read_text(encoding="utf-8"))
    meta["params"]["min_track_obs"] = 99                                  # e.g. an entry written by other code
    (d / "perception.json").write_text(json.dumps(meta), encoding="utf-8")
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is False
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True


def test_changing_a_constant_does_not_serve_a_stale_entry(synth_video, tmp_path, monkeypatch):
    first, c1 = cached_perception(synth_video, fake_stack(), tmp_path)
    assert c1 is False and len(first.track_emb) == 2
    with monkeypatch.context() as m:
        m.setattr(pipeline, "MIN_TRACK_OBS", 1000)                        # no track is real any more
        second, c2 = cached_perception(synth_video, fake_stack(), tmp_path)
        assert c2 is False and second.track_emb == {} and second.params["min_track_obs"] == 1000
    third, c3 = cached_perception(synth_video, fake_stack(), tmp_path)    # constant restored: the first entry again
    assert c3 is True and len(third.track_emb) == 2
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 2
    with monkeypatch.context() as m:
        m.setattr(pipeline, "DEFAULT_POSE_EVERY", 1)
        _, c4 = cached_perception(synth_video, fake_stack(), tmp_path)
        assert c4 is False


def test_the_cache_dir_is_named_by_cache_key(synth_video, tmp_path):
    cached_perception(synth_video, fake_stack(), tmp_path, pose_every=1)
    (d,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert d.name == _key(synth_video, pose_every=1) != _key(synth_video)


# --- publish: lost race, stale temp dirs


def test_a_lost_race_still_returns_the_current_video_path(synth_video, tmp_path, monkeypatch):
    real_replace = os.replace

    def lose_the_race(src, dst):
        if Path(dst).name.startswith(".") or not Path(src).name.startswith(".tmp-"):
            return real_replace(src, dst)
        shutil.copytree(src, dst)                                         # the other run published first ...
        meta = json.loads((Path(dst) / "perception.json").read_text(encoding="utf-8"))
        meta["video"]["path"] = "elsewhere/winner.avi"                    # ... from another path
        (Path(dst) / "perception.json").write_text(json.dumps(meta), encoding="utf-8")
        raise FileExistsError(dst)

    monkeypatch.setattr(pipeline.os, "replace", lose_the_race)
    res, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is False and res.video.path == str(synth_video)
    assert len(res.track_ids()) == 2 and Path(res.crops_dir).is_dir()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    monkeypatch.undo()
    res, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True and res.video.path == str(synth_video)


def test_stale_tmp_dirs_are_swept_but_fresh_ones_and_entries_are_kept(synth_video, tmp_path):
    old, fresh, entry = tmp_path / ".tmp-abc-old", tmp_path / ".tmp-abc-fresh", tmp_path / "0123456789abcdef"
    for d in (old, fresh, entry):
        (d / "crops").mkdir(parents=True)
        (d / "crops" / "1.jpg").write_bytes(b"x")
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    os.utime(entry, (two_hours_ago, two_hours_ago))                       # an old entry is not a temp dir
    (tmp_path / ".tmp-file").write_text("not a dir")
    os.utime(tmp_path / ".tmp-file", (two_hours_ago, two_hours_ago))
    cached_perception(synth_video, fake_stack(), tmp_path)
    assert not old.exists() and fresh.exists() and entry.exists() and (tmp_path / ".tmp-file").exists()


def test_the_sweep_also_runs_on_a_cache_hit(synth_video, tmp_path):
    cached_perception(synth_video, fake_stack(), tmp_path)
    old = tmp_path / ".tmp-crashed"
    old.mkdir()
    three_hours_ago = time.time() - 10800
    os.utime(old, (three_hours_ago, three_hours_ago))
    _, cached = cached_perception(synth_video, fake_stack(), tmp_path)
    assert cached is True and not old.exists()


def test_bad_arguments_are_rejected(synth_video, tmp_path):
    with pytest.raises(TypeError):
        cached_perception(synth_video, fake_stack(), tmp_path, nonsense=1)


def test_failed_run_leaves_no_cache_entry(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_text("not a video")
    root = tmp_path / "cache"
    with pytest.raises(VideoError):
        cached_perception(bad, fake_stack(), root)
    assert not root.exists() or not any(root.iterdir())


# --------------------------------------------------------------------------- real models (slow)


@pytest.mark.slow
def test_real_stack_tracks_people_for_five_seconds_in_a_meva_clip():
    from showroom_coach import models as M
    from showroom_coach.config import DEFAULT_DETECTOR, repo_root
    from showroom_coach.perception.stack import default_stack

    clips = sorted((repo_root() / "data" / "meva" / "videos").glob("*.avi"))
    if not clips:
        pytest.skip("no MEVA clip in data/meva/videos")
    for name in (DEFAULT_DETECTOR.partition("@")[0], "rtmpose_s", "dinov2_small"):
        if not (M.models_dir() / name / M.MODELS[name].filename).is_file():
            pytest.skip(f"model {name} not downloaded")
    res = run_perception(clips[0], default_stack(), max_seconds=10.0)
    assert len(res.frame_times) >= 49
    ids = res.track_ids()
    assert ids, "no person was tracked in the first 10 s"
    spans = [float(np.ptp(res.frame_times[res.obs_frame[res.obs_idx(i)]])) for i in ids]
    assert max(spans) >= 5.0, f"longest track spans only {max(spans):.1f} s"
    real = [i for i in ids if len(res.obs_idx(i)) >= 3 and
            float(np.ptp(res.frame_times[res.obs_frame[res.obs_idx(i)]])) >= 1.0]
    assert real and set(res.track_emb) == set(real) == set(res.track_outfit)     # flicker tracks are not described
    assert all(res.track_emb[i].shape == (384 + 128,) for i in real)
    assert (res.obs_kpts[..., 2] > 0).any()
