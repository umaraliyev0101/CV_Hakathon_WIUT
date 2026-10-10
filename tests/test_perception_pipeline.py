"""Perception pipeline: detect -> track -> pose -> embed over a video, the per-video cache, and the face blur
helpers that make the saved track crops safe to show."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from showroom_coach.perception import pipeline
from showroom_coach.perception.pipeline import (
    cached_perception, load_result, run_perception, save_result, video_key,
)
from showroom_coach.perception.stack import PerceptionStack, fake_stack
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
    assert d.name == video_key(synth_video, 5.0, 960, 300.0, "fake")
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
    assert set(res.track_emb) == set(ids) and res.track_emb[ids[0]].shape == (384 + 128,)
    assert (res.obs_kpts[..., 2] > 0).any()
