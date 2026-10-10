"""Offline pass: perceive (models, cached per video) and build_people (cheap, re-runnable)."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import cache as cache_mod
from . import identity, sources
from .camera_motion import motion_flags, pick_key_idx, segments, to_gray_small
from .cache import PerceptionCache
from .config import Config
from .datatypes import Person, Timeline
from .engine import PerceptionEngine
from .perception.embed import Embedder, torso_crop
from .timeline import build_timeline

CROPS_PER_TRACK = 3
MIN_CROP_SCORE = 0.5
MIN_TRACK_S = 1.0
CROP_SPACING_S = 1.0
_MAX_KEPT = 12
_CROP_SIZE = (112, 112)


def _pick(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def perceive(video, cache_root: Path, cfg: Config, progress: Callable[[float], None] | None = None,
             engine: PerceptionEngine | None = None) -> PerceptionCache:
    video, cache_root = Path(video), Path(cache_root)
    sha1 = cache_mod.file_sha1(video)
    key = cache_mod.video_key(video, cfg, sha1)
    cdir = cache_root / key
    if cache_mod.exists(cdir):
        if progress:
            progress(1.0)
        return cache_mod.load(cdir)

    t_start = time.perf_counter()
    info = sources.probe(video)                       # VideoError propagates
    cache_mod.count_run()
    engine = engine or PerceptionEngine(cfg)
    n_est = max(1, round(min(info.duration or cfg.max_duration_s, cfg.max_duration_s) * cfg.sample_fps))
    step = max(1, n_est // 20)

    obs, times, grays = [], [], []
    crops: dict[int, list[tuple[float, np.ndarray]]] = {}
    t_span: dict[int, list[float]] = {}
    frame_w = frame_h = 0
    for k, (t, frame) in enumerate(sources.iter_frames(video, cfg.sample_fps, cfg.max_width,
                                                       cfg.max_duration_s)):
        if k == 0:
            frame_h, frame_w = frame.shape[:2]
        times.append(t)
        grays.append(to_gray_small(frame))
        for o in engine.process(t, frame, k):
            obs.append(o)
            sp = t_span.setdefault(o.track_id, [t, t])
            sp[1] = t
            lst = crops.setdefault(o.track_id, [])
            if o.score >= MIN_CROP_SCORE and (not lst or t - lst[-1][0] >= CROP_SPACING_S):
                c = torso_crop(frame, o, cfg.kp_thr)
                if c is not None:
                    lst.append((t, cv2.resize(c, _CROP_SIZE, interpolation=cv2.INTER_AREA)))
                    if len(lst) > _MAX_KEPT:
                        del lst[1::2]
        if progress and k % step == 0:
            progress(min(0.99, k / n_est))

    # camera motion -> keyframe -> segments (same_view relative to the keyframe)
    flags = motion_flags(grays)
    key_idx = pick_key_idx(flags) if times else 0
    segs = segments(times, grays, key_idx=key_idx, flags=flags) if times else []
    keyframe_t = times[key_idx] if times else 0.0
    keyframe, cars = None, []
    for t, frame in sources.iter_frames(video, cfg.sample_fps, cfg.max_width, cfg.max_duration_s):
        if t >= keyframe_t:
            keyframe = frame
            break
    if keyframe is not None:
        cars = engine.detect_vehicles(keyframe)

    # embeddings (timed separately; budget <= 20% of perception time)
    t_e = time.perf_counter()
    embedder = Embedder()
    groups = {tid: [c for _, c in _pick(lst, CROPS_PER_TRACK)] for tid, lst in crops.items() if lst}
    long_tracks = [tid for tid in groups if t_span[tid][1] - t_span[tid][0] >= MIN_TRACK_S]
    embeddings = embedder.embed_many(groups, dino_keys=long_tracks)
    embed_seconds = time.perf_counter() - t_e

    perception_seconds = time.perf_counter() - t_start
    last_t = times[-1] + 1.0 / cfg.sample_fps if times else 0.0
    analysed = min(last_t, info.duration) if info.duration > 0 else last_t
    meta = {
        "video": str(video), "sha1": sha1, "key": key,
        "src_w": info.width, "src_h": info.height, "src_fps": info.fps, "duration": info.duration,
        "analysed_duration": analysed,
        "trimmed": bool(info.duration > cfg.max_duration_s + 1.0 / cfg.sample_fps),
        "frame_w": frame_w, "frame_h": frame_h, "sample_fps": cfg.sample_fps, "n_samples": len(times),
        "detector": cfg.detector, "pose_model": cfg.pose_model,
        "providers": list(engine.detector.session.get_providers()),
        "dino": embedder.uses_dino,
        "perception_seconds": perception_seconds, "embed_seconds": embed_seconds,
        "rtf": analysed / perception_seconds if perception_seconds > 0 else 0.0,
    }
    pc = PerceptionCache(cdir, meta, obs, cars, keyframe_t, segs, embeddings, keyframe)
    cache_mod.save(pc)
    if progress:
        progress(1.0)
    return pc


def build_people(pc: PerceptionCache, cfg: Config,
                 floor_h=None) -> tuple[list[Person], Timeline, dict[int, int]]:
    size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
    summaries = identity.summarize(pc.obs, pc.embeddings, size)
    persons = identity.merge(summaries, cfg)
    track_to_person = {tid: p.person_id for p in persons for tid in p.track_ids}
    return persons, build_timeline(pc.obs, track_to_person, size, cfg, floor_h), track_to_person
