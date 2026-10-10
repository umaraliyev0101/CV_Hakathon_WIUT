"""The offline perception pass: video -> sampled frames -> detect -> track -> pose -> per-track appearance, plus a
per-video cache so roles and thresholds can change without running a single model again.

`run_perception` makes one pass over the (sampled, resized) frames and returns a `PerceptionResult`: flat arrays of
tracked observations (frame, track, box, score, 17 keypoints), every raw person detection (so rendering can blur
faces of people that were never tracked), the car boxes of the first frame, and one identity and one outfit
descriptor per track. While streaming it keeps, per track, the `crops_per_track` best observations (largest
box area x detector score) as small crops; after the pass those are embedded and averaged. Nothing here recognises
faces: descriptors come from whole-body and torso crops, and the saved track thumbnails are face-blurred.

Cache layout, one directory per `video_key`:
    perception.npz    plain numeric arrays only (never pickled; loaded with allow_pickle=False)
    perception.json   format version, stack name, VideoInfo as a dict, timings, crops dir, and the descriptor
                      dicts as {"<track id>": [floats]} (float32 values survive the JSON round trip exactly)
    crops/<id>.jpg    face-blurred best crop of each track
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import MAX_SECONDS, MAX_WIDTH, SAMPLE_FPS
from ..render import blur_faces
from ..sources import VideoError, VideoInfo, iter_frames, probe
from ..types import to_jsonable
from .embed import hsv_hist, person_crop, torso_crop, usable_crop
from .stack import PerceptionStack
from .tracking import PERSON

log = logging.getLogger(__name__)

CAR_CLASSES = (2, 5, 7)          # COCO: car, bus, truck
CROP_MAX_H = 160                 # stored crops are downscaled to at most this many pixels tall
DEFAULT_CROPS_PER_TRACK = 4
PROGRESS_EVERY = 10              # sampled frames between progress callbacks
HSV_WEIGHT = 0.5                 # weight of the secondary half of each descriptor
FORMAT_VERSION = 1
NPZ_NAME, JSON_NAME, CROPS_NAME = "perception.npz", "perception.json", "crops"
JPEG_QUALITY = 85
_EPS = 1e-12
_ARRAYS = ("frame_times", "src_frame_idx", "obs_frame", "obs_track", "obs_bbox", "obs_score", "obs_kpts",
           "raw_frame", "raw_bbox")


@dataclass(eq=False)
class PerceptionResult:
    video: VideoInfo
    stack_name: str
    frame_times: np.ndarray        # (F,) seconds of each sampled frame
    src_frame_idx: np.ndarray      # (F,) source frame index (for MEVA ground truth)
    obs_frame: np.ndarray          # (M,) sampled-frame index of each tracked observation
    obs_track: np.ndarray          # (M,) track id
    obs_bbox: np.ndarray           # (M,4) processed-frame pixels
    obs_score: np.ndarray          # (M,)
    obs_kpts: np.ndarray           # (M,17,3); conf 0 when pose is unavailable
    raw_frame: np.ndarray          # (R,) frame index of every person detection (for face blur)
    raw_bbox: np.ndarray           # (R,4)
    track_emb: dict[int, np.ndarray]     # identity descriptor: L2norm(concat(dino_mean, 0.5*hsv_full_mean))
    track_outfit: dict[int, np.ndarray]  # outfit descriptor: L2norm(concat(hsv_torso_mean, 0.5*dino_mean))
    crops_dir: str | None                # <dir>/<track_id>.jpg = face-blurred best crop of each track
    cars: list[list[float]]              # car boxes [x1,y1,x2,y2,score] from the first frame (used in M8)
    timings: dict[str, float]            # decode/detect/track/pose/embed/total seconds

    def track_ids(self) -> list[int]:
        return [int(i) for i in np.unique(self.obs_track)]

    def obs_idx(self, track_id: int) -> np.ndarray:
        """Indices into the obs_* arrays of one track's observations, in frame order."""
        return np.flatnonzero(self.obs_track == track_id)


# --------------------------------------------------------------------------- the pass


def _shrink(img: np.ndarray | None, max_h: int = CROP_MAX_H) -> np.ndarray | None:
    if img is None or img.shape[0] <= max_h:
        return img
    scale = max_h / img.shape[0]
    return cv2.resize(img, (max(1, round(img.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > _EPS else np.zeros(v.shape, np.float32)


def _mean_nonzero(rows: list[np.ndarray], dim: int) -> np.ndarray:
    """Mean of the rows that carry information (an unusable crop gives a zero row); zeros when there are none."""
    kept = [r for r in rows if float(np.linalg.norm(r)) > _EPS]
    return np.mean(kept, axis=0, dtype=np.float64) if kept else np.zeros(dim, np.float64)


def _describe(stack: PerceptionStack, heaps: dict[int, list]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Identity and outfit descriptors per track from its kept crops (see the module docstring)."""
    dino: dict[tuple[int, int], np.ndarray] = {}
    dino_dim = 0
    if stack.embed is not None:
        keys = [(tid, k) for tid, h in heaps.items() for k, item in enumerate(h) if usable_crop(item[2])]
        if keys:
            out = np.asarray(stack.embed([heaps[tid][k][2] for tid, k in keys]), np.float32)
            if out.ndim != 2 or len(out) != len(keys):
                raise ValueError(f"embedder returned {out.shape} for {len(keys)} crops")
            dino = {key: out[n] for n, key in enumerate(keys)}
            dino_dim = out.shape[1]
    hsv_dim = len(hsv_hist(None))
    emb: dict[int, np.ndarray] = {}
    outfit: dict[int, np.ndarray] = {}
    for tid, h in heaps.items():
        hsv_full = _mean_nonzero([hsv_hist(item[2]) for item in h], hsv_dim)
        hsv_torso = _mean_nonzero([hsv_hist(item[3]) for item in h], hsv_dim)
        if dino_dim:
            dino_mean = _mean_nonzero([dino[(tid, k)] for k in range(len(h)) if (tid, k) in dino], dino_dim)
            emb[tid] = _unit(np.concatenate([dino_mean, HSV_WEIGHT * hsv_full]))
            outfit[tid] = _unit(np.concatenate([hsv_torso, HSV_WEIGHT * dino_mean]))
        else:
            emb[tid], outfit[tid] = _unit(hsv_full), _unit(hsv_torso)
    return emb, outfit


def _write_jpeg(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"cannot encode {path.name}")
    path.write_bytes(buf.tobytes())


def run_perception(path, stack: PerceptionStack, sample_fps: float = SAMPLE_FPS, max_width: int = MAX_WIDTH,
                   max_seconds: float = MAX_SECONDS, progress: Callable[[float, str], None] | None = None,
                   crops_dir: str | Path | None = None,
                   crops_per_track: int = DEFAULT_CROPS_PER_TRACK) -> PerceptionResult:
    """Run the whole perception stack over `path` (sampled at `sample_fps`, resized to `max_width`, cut at
    `max_seconds`). `progress(fraction, message)` is called about every 10 sampled frames and finally with 1.0.
    With `crops_dir`, each track's face-blurred best crop is written there as `<track_id>.jpg`.
    Raises `VideoError` when the file cannot be decoded.

    Timings: `decode` is probing plus frame decoding; `embed` is everything appearance-related (cutting crops,
    descriptors, thumbnails); `total` is the wall time of the whole call.
    """
    t_start = time.perf_counter()
    keep = max(1, int(crops_per_track))
    tm = {"decode": 0.0, "detect": 0.0, "track": 0.0, "pose": 0.0, "embed": 0.0}

    t0 = time.perf_counter()
    info = probe(path, max_width, sample_fps, max_seconds)
    tm["decode"] += time.perf_counter() - t0
    tracker = stack.new_tracker(sample_fps)

    frame_times: list[float] = []
    src_idx: list[int] = []
    obs_frame: list[int] = []
    obs_track: list[int] = []
    obs_bbox: list[tuple[float, float, float, float]] = []
    obs_score: list[float] = []
    obs_kpts: list[np.ndarray] = []
    raw_frame: list[int] = []
    raw_bbox: list[tuple[float, float, float, float]] = []
    cars: list[list[float]] = []
    heaps: dict[int, list] = {}                          # track -> min-heap of (priority, seq, person crop, torso crop)
    best: dict[int, tuple[float, np.ndarray]] = {}       # track -> (priority, face-blurred person crop)
    seq = itertools.count()
    out_dir = Path(crops_dir) if crops_dir is not None else None

    frames = iter_frames(path, sample_fps, max_width, max_seconds)
    try:
        while True:
            t0 = time.perf_counter()
            item = next(frames, None)
            tm["decode"] += time.perf_counter() - t0
            if item is None:
                break
            src_i, t, frame = item
            fi = len(frame_times)
            frame_times.append(float(t))
            src_idx.append(int(src_i))

            t0 = time.perf_counter()
            dets = stack.detect(frame)
            tm["detect"] += time.perf_counter() - t0
            persons = [d for d in dets if d.cls == PERSON]
            if fi == 0:
                cars = [[*(float(v) for v in d.bbox), float(d.score)] for d in dets if d.cls in CAR_CLASSES]
            for d in persons:
                raw_frame.append(fi)
                raw_bbox.append(tuple(float(v) for v in d.bbox))

            t0 = time.perf_counter()
            tracked = tracker.update(persons)
            tm["track"] += time.perf_counter() - t0

            kpts = np.zeros((len(tracked), 17, 3), np.float32)
            if tracked and stack.pose is not None:
                t0 = time.perf_counter()
                boxes = np.asarray([d.bbox for _, d in tracked], np.float64)
                kpts = np.asarray(stack.pose(frame, boxes), np.float32).reshape(len(tracked), 17, 3)
                tm["pose"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            blurred = None
            for j, (tid, d) in enumerate(tracked):
                x1, y1, x2, y2 = (float(v) for v in d.bbox)
                score = float(d.score)
                obs_frame.append(fi)
                obs_track.append(tid)
                obs_bbox.append((x1, y1, x2, y2))
                obs_score.append(score)
                obs_kpts.append(kpts[j])
                prio = max(0.0, x2 - x1) * max(0.0, y2 - y1) * score
                heap = heaps.setdefault(tid, [])
                if len(heap) < keep or prio > heap[0][0]:
                    entry = (prio, next(seq), _shrink(person_crop(frame, d.bbox)),
                             _shrink(torso_crop(frame, d.bbox, kpts[j])))
                    if len(heap) < keep:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heapreplace(heap, entry)
                if out_dir is not None and prio > best.get(tid, (-1.0, None))[0]:
                    if blurred is None:       # blur every person in the frame: others may overlap this crop
                        by_det = {id(det): kpts[k] for k, (_, det) in enumerate(tracked)}
                        blurred = blur_faces(frame, [p.bbox for p in persons], [by_det.get(id(p)) for p in persons])
                    thumb = _shrink(person_crop(blurred, d.bbox))
                    if thumb is not None:
                        best[tid] = (prio, thumb)
            tm["embed"] += time.perf_counter() - t0

            if progress is not None and fi % PROGRESS_EVERY == 0:
                frac = min(1.0, max(0.0, float(t) / info.duration)) if info.duration > 0 else 0.0
                progress(frac, f"Finding and tracking people: {t:.0f} of {info.duration:.0f} s")
    finally:
        frames.close()

    t0 = time.perf_counter()
    track_emb, track_outfit = _describe(stack, heaps)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid, (_, thumb) in best.items():
            _write_jpeg(out_dir / f"{tid}.jpg", thumb)
    tm["embed"] += time.perf_counter() - t0

    tm["total"] = time.perf_counter() - t_start
    result = PerceptionResult(
        video=info, stack_name=stack.name,
        frame_times=np.asarray(frame_times, np.float64), src_frame_idx=np.asarray(src_idx, np.int64),
        obs_frame=np.asarray(obs_frame, np.int64), obs_track=np.asarray(obs_track, np.int64),
        obs_bbox=np.asarray(obs_bbox, np.float64).reshape(-1, 4), obs_score=np.asarray(obs_score, np.float64),
        obs_kpts=np.stack(obs_kpts).astype(np.float32) if obs_kpts else np.zeros((0, 17, 3), np.float32),
        raw_frame=np.asarray(raw_frame, np.int64), raw_bbox=np.asarray(raw_bbox, np.float64).reshape(-1, 4),
        track_emb=track_emb, track_outfit=track_outfit,
        crops_dir=str(out_dir) if out_dir is not None else None,
        cars=cars, timings=tm,
    )
    if progress is not None:
        progress(1.0, "Analysis finished")
    return result


# --------------------------------------------------------------------------- the cache


def video_key(path, sample_fps, max_width, max_seconds, stack_name) -> str:
    """16 hex chars identifying one (video content, processing parameters, perception stack) combination:
    sha1 over the file's bytes (read in 1 MB chunks) followed by the parameters."""
    h = hashlib.sha1(usedforsecurity=False)
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    except OSError as exc:
        raise VideoError(f"cannot read video: {Path(str(path)).name}") from exc
    h.update(f"|fps={float(sample_fps)!r}|width={int(max_width)}|seconds={float(max_seconds)!r}"
             f"|stack={stack_name}".encode())
    return h.hexdigest()[:16]


def _atomic_write(path: Path, write: Callable) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def save_result(res: PerceptionResult, d: Path) -> None:
    """Write `res` into directory `d` (created): numeric arrays to perception.npz, the rest to perception.json.
    The JSON goes last, so its presence means the entry is complete."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(res, name)) for name in _ARRAYS}
    arrays["cars"] = np.asarray(res.cars, np.float64).reshape(-1, 5)
    _atomic_write(d / NPZ_NAME, lambda f: np.savez_compressed(f, **arrays))
    crops = None
    if res.crops_dir is not None:
        try:
            crops = Path(res.crops_dir).resolve().relative_to(d.resolve()).as_posix()
        except ValueError:
            crops = str(res.crops_dir)
    meta = to_jsonable({
        "format": FORMAT_VERSION, "stack_name": res.stack_name, "video": res.video, "timings": res.timings,
        "crops_dir": crops,
        "track_emb": {int(k): np.asarray(v, np.float32).tolist() for k, v in res.track_emb.items()},
        "track_outfit": {int(k): np.asarray(v, np.float32).tolist() for k, v in res.track_outfit.items()},
    })
    _atomic_write(d / JSON_NAME, lambda f: f.write(json.dumps(meta, allow_nan=False).encode("utf-8")))


def load_result(d: Path) -> PerceptionResult:
    """Read what `save_result` wrote. Raises (OSError, ValueError, KeyError, TypeError) on a missing, truncated,
    corrupt or other-version entry; arrays are loaded with pickle disabled."""
    d = Path(d)
    meta = json.loads((d / JSON_NAME).read_text(encoding="utf-8"))
    if meta["format"] != FORMAT_VERSION:
        raise ValueError(f"unsupported perception cache format {meta['format']!r}")
    with np.load(d / NPZ_NAME, allow_pickle=False) as z:
        arrays = {name: z[name] for name in _ARRAYS}
        cars = z["cars"].reshape(-1, 5).tolist()
    m, r, f = len(arrays["obs_track"]), len(arrays["raw_frame"]), len(arrays["frame_times"])
    if not (all(len(arrays[n]) == m for n in ("obs_frame", "obs_bbox", "obs_score", "obs_kpts"))
            and len(arrays["raw_bbox"]) == r and len(arrays["src_frame_idx"]) == f):
        raise ValueError("perception arrays disagree in length")
    crops = None
    if meta["crops_dir"] is not None:
        p = Path(meta["crops_dir"])
        p = p if p.is_absolute() else d / p
        crops = str(p) if p.is_dir() else None
    return PerceptionResult(
        video=VideoInfo(**meta["video"]), stack_name=meta["stack_name"], **arrays,
        track_emb={int(k): np.asarray(v, np.float32) for k, v in meta["track_emb"].items()},
        track_outfit={int(k): np.asarray(v, np.float32) for k, v in meta["track_outfit"].items()},
        crops_dir=crops, cars=cars, timings={k: float(v) for k, v in meta["timings"].items()},
    )


def cached_perception(path, stack: PerceptionStack, cache_root: Path, progress=None,
                      **kw) -> tuple[PerceptionResult, bool]:
    """`(result, was_cached)`: the saved perception of this exact video + parameters + stack when there is one,
    otherwise run it (via the module-level `run_perception`) and save it under `cache_root/<video_key>/`.

    `kw` may carry `sample_fps`, `max_width`, `max_seconds` and `crops_per_track`; anything else is a TypeError.
    A cache entry that cannot be read is discarded and recomputed. New entries are built in a temporary
    directory and moved into place, so a crash or a concurrent run never leaves a half-written entry. The returned
    result always describes the video at `path` (the stored path is replaced, since the same bytes may have
    been cached under another name).
    """
    sample_fps = kw.pop("sample_fps", SAMPLE_FPS)
    max_width = kw.pop("max_width", MAX_WIDTH)
    max_seconds = kw.pop("max_seconds", MAX_SECONDS)
    crops_per_track = kw.pop("crops_per_track", DEFAULT_CROPS_PER_TRACK)
    if kw:
        raise TypeError(f"cached_perception() got unexpected keyword arguments: {sorted(kw)}")
    cache_root = Path(cache_root)
    name = stack.name if crops_per_track == DEFAULT_CROPS_PER_TRACK else f"{stack.name}|crops={crops_per_track}"
    d = cache_root / video_key(path, sample_fps, max_width, max_seconds, name)

    if d.is_dir():
        try:
            res = load_result(d)
        except Exception as exc:                                        # any unreadable entry is just a miss
            log.warning("discarding unreadable perception cache %s (%s: %s)", d.name, type(exc).__name__, exc)
            shutil.rmtree(d, ignore_errors=True)
        else:
            res.video = replace(res.video, path=str(path))
            if progress is not None:
                progress(1.0, "Using the saved analysis of this video")
            return res, True

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = cache_root / f".tmp-{d.name}-{uuid.uuid4().hex[:8]}"
    try:
        res = run_perception(path, stack, sample_fps=sample_fps, max_width=max_width, max_seconds=max_seconds,
                             progress=progress, crops_dir=tmp / CROPS_NAME, crops_per_track=crops_per_track)
        save_result(res, tmp)
        try:
            os.replace(tmp, d)
        except OSError:
            if not d.is_dir():                                          # not a lost race: a real failure
                raise
            log.info("perception cache %s was written concurrently; using that entry", d.name)
        return load_result(d), False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
