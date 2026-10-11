"""Offline pass: perceive (models, cached per video) and build_people (cheap, re-runnable)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import cache as cache_mod
from . import identity, insights as insights_mod, interest, journey, roles as roles_mod, sat_in_car, sources
from . import stats as stats_mod, zones as zones_mod
from .camera_motion import motion_flags, pick_key_idx, segments, to_gray_small
from .cache import PerceptionCache
from .config import Config, get_config
from .datatypes import (Event, InterestEpisode, Insight, Person, RoleSuggestion, Stats, Timeline, Zone,
                        ZoneVisit)
from .engine import PerceptionEngine
from .perception.embed import Embedder, torso_crop
from .timeline import build_timeline, filter_stable

CROPS_PER_TRACK = 3
MIN_CROP_SCORE = 0.5
CROP_SPACING_S = 1.0
_MAX_KEPT = 12
_CROP_SIZE = (112, 112)


def _pick(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def perceive(video, cache_root: Path, cfg: Config, progress: Callable[[float], None] | None = None,
             engine: PerceptionEngine | None = None, embedder: Embedder | None = None) -> PerceptionCache:
    video, cache_root = Path(video), Path(cache_root)
    if not video.is_file():
        raise sources.VideoError(f"Video not found: {video}")
    info = sources.probe(video)                       # VideoError propagates (before any hashing/caching)
    sha1 = cache_mod.file_sha1(video)
    key = cache_mod.video_key(video, cfg, sha1)
    cdir = cache_root / key
    if cache_mod.exists(cdir):
        if progress:
            progress(1.0)
        return cache_mod.load(cdir)

    t_start = time.perf_counter()
    cache_mod.count_run()
    engine = engine or PerceptionEngine(cfg)
    engine.reset()
    n_est = max(1, round(min(info.duration or cfg.max_duration_s, cfg.max_duration_s) * cfg.sample_fps))
    step = max(1, n_est // 20)

    obs, times, grays, raw_dets = [], [], [], []
    crops: dict[int, list[tuple[float, np.ndarray]]] = {}
    best: dict[int, tuple[float, np.ndarray]] = {}   # best sub-threshold crop per track
    frame_w = frame_h = 0
    for k, (t, frame) in enumerate(sources.iter_frames(video, cfg.sample_fps, cfg.max_width,
                                                       cfg.max_duration_s)):
        if k == 0:
            frame_h, frame_w = frame.shape[:2]
        times.append(t)
        grays.append(to_gray_small(frame))
        tracked = engine.process(t, frame, k)
        raw_dets.extend((t, tuple(d.bbox), float(d.score)) for d in getattr(engine, "last_person_dets", []))
        for o in tracked:
            obs.append(o)
            lst = crops.setdefault(o.track_id, [])
            if o.score >= MIN_CROP_SCORE and (not lst or t - lst[-1][0] >= CROP_SPACING_S):
                c = torso_crop(frame, o, cfg.kp_thr)
                if c is not None:
                    lst.append((t, cv2.resize(c, _CROP_SIZE, interpolation=cv2.INTER_AREA)))
                    if len(lst) > _MAX_KEPT:
                        del lst[1::2]
            elif o.score < MIN_CROP_SCORE and not lst and o.score > best.get(o.track_id, (-1,))[0]:
                c = torso_crop(frame, o, cfg.kp_thr)       # fallback when no crop ever reaches 0.5
                if c is not None:
                    best[o.track_id] = (o.score, cv2.resize(c, _CROP_SIZE, interpolation=cv2.INTER_AREA))
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
    embedder = embedder or Embedder(providers=engine.providers)
    groups = {tid: [c for _, c in _pick(lst, CROPS_PER_TRACK)] for tid, lst in crops.items() if lst}
    for tid, (_, c) in best.items():
        if not groups.get(tid):
            groups[tid] = [c]
    embeddings = embedder.embed_many(groups)
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
    pc = PerceptionCache(cdir, meta, obs, cars, keyframe_t, segs, embeddings, keyframe, raw_dets)
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


def sat_in_car_events(pc: PerceptionCache, stable: Timeline, track_to_person: dict[int, int],
                      zones: list[Zone], cfg: Config) -> list[Event]:
    """Sat-in-car events (spec 4.5). Keyframe car boxes only hold while the camera is stable, so only
    observations that survive the stability filter are used."""
    size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
    stable_t = {pid: {s.t for s in ss} for pid, ss in stable.items()}
    obs_by_person: dict[int, list] = {}
    for o in pc.obs:
        pid = track_to_person.get(o.track_id)
        if pid is not None and o.t in stable_t.get(pid, ()):
            obs_by_person.setdefault(pid, []).append(o)
    return sat_in_car.assign_zones(sat_in_car.detect(stable, obs_by_person, pc.cars, cfg, frame_size=size),
                                   pc.cars, zones, size)


@dataclass
class Analysis:
    persons: list[Person]
    timeline: Timeline
    suggestions: list[RoleSuggestion]
    roles: dict[int, str]
    journeys: dict[int, list[ZoneVisit]]
    episodes: list[InterestEpisode]
    assist: dict
    stats: Stats
    insights: list[Insight]
    zones: list[Zone]
    meta: dict = field(default_factory=dict)


def score(pc: PerceptionCache, cfg: Config, zones: list[Zone] | None = None,
          overrides: dict[int, str] | None = None,
          imported_roles: dict[int, str] | None = None) -> Analysis:
    """Everything after perception. Pure: reads only the cache, never runs a model."""
    size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
    duration = float(pc.meta.get("analysed_duration") or pc.meta.get("duration") or 0.0)
    persons, timeline, t2p = build_people(pc, cfg)
    if zones is None:
        zones = zones_mod.auto_zones(pc.cars, (size[1], size[0]))
    inter = roles_mod.interactions(persons, timeline, pc.embeddings, duration, cfg)
    suggestions = roles_mod.suggest(persons, inter, cfg)
    roles = roles_mod.final_roles(suggestions, imported_roles, overrides)

    stable = filter_stable(timeline, pc.segments)
    staff_tracks = {pid: ss for pid, ss in stable.items() if roles.get(pid) == "staff"}
    journeys = journey.build(stable, zones, cfg.preset)
    assist = interest.assistance(stable, staff_tracks, cfg.preset)
    episodes = interest.episodes(journeys, staff_tracks, cfg.preset, stable, zones)
    sat = sat_in_car_events(pc, stable, t2p, zones, cfg)
    same_view = sum(max(0.0, min(s.t1, duration) - s.t0) for s in pc.segments if s.stable and s.same_view)
    share = min(1.0, same_view / duration) if duration > 0 and pc.segments else 1.0
    extra = []
    if pc.meta.get("trimmed"):
        extra.append(f"Video trimmed to the first {int(cfg.max_duration_s)} s")
    st = stats_mod.compute(persons, journeys, roles, cfg, timeline=timeline, episodes=episodes,
                           assist=assist, zones=zones, duration=duration, frame_size=size,
                           analysed_share=share, extra_warnings=extra, sat_events=sat)
    ins = insights_mod.recommend(st, episodes, cfg)
    meta = {"video": Path(str(pc.meta.get("video", ""))).name, "preset": cfg.preset.name,
            "duration": float(pc.meta.get("duration") or 0.0), "analysed_duration": duration,
            "trimmed": bool(pc.meta.get("trimmed")), "sample_fps": cfg.sample_fps,
            "frame_w": size[0], "frame_h": size[1], "n_samples": int(pc.meta.get("n_samples", 0)),
            "providers": list(pc.meta.get("providers", [])),
            "perception_seconds": float(pc.meta.get("perception_seconds", 0.0)),
            "rtf": float(pc.meta.get("rtf", 0.0)), "demo_only": cfg.preset.demo_only}
    return Analysis(persons, timeline, suggestions, roles, journeys, episodes, assist, st, ins,
                    zones, meta)


def write_outputs(pc: PerceptionCache, analysis: Analysis, out_dir, cfg: Config, video=None,
                  render_video: bool = True) -> list[Path]:
    from . import report
    out_dir = Path(out_dir)
    report.build_report(analysis, pc, out_dir, cfg, video=video,
                        render_video=render_video and video is not None)
    return sorted(p for p in out_dir.rglob("*") if p.is_file())


def analyze(video, out_dir, preset: str = "short", zones_path=None, roles_path=None,
            cache_root=Path("runs/cache"), render_video: bool = True, cfg: Config | None = None,
            progress: Callable[[float], None] | None = None) -> Analysis:
    cfg = cfg or get_config(preset)
    pc = perceive(video, Path(cache_root), cfg, progress)
    zones = zones_mod.load_zones(zones_path) if zones_path else None
    imported = roles_mod.load_roles_json(roles_path) if roles_path else None
    analysis = score(pc, cfg, zones=zones, imported_roles=imported)
    write_outputs(pc, analysis, out_dir, cfg, video=video, render_video=render_video)
    return analysis
