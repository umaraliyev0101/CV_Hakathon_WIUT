"""The offline perception pass: video -> sampled frames -> detect -> track -> pose -> per-track appearance, plus a
per-video cache so roles and thresholds can change without running a single model again.

`run_perception` makes one pass over the (sampled, resized) frames and returns a `PerceptionResult`: flat arrays of
tracked observations (frame, track, box, score, 17 keypoints), every raw person detection (so rendering can blur
faces of people that were never tracked), the car boxes of the first frame, and one identity and one outfit
descriptor per real track. While streaming it keeps, per track, the `crops_per_track` best observations (largest
box area x detector score) as small crops; after the pass those of the real tracks are embedded and averaged.
Nothing here recognises faces: descriptors come from whole-body and torso crops, and the saved track thumbnails
are face-blurred.

Two cost savers, both visible in the result. Pose runs for a track only every `pose_every`-th sampled frame (all
due boxes of a frame in one batched model call); in between, the track's last fresh keypoints ride along with its
box (`carry_keypoints`). And only real tracks (`config.MIN_TRACK_OBS` observations spanning `config.MIN_TRACK_S`
seconds, the same bar analysis applies) get descriptors and a crop file; shorter ones are flicker, but their
observations and raw boxes stay in the arrays.

Cache layout, one directory per `video_key`:
    perception.npz    plain numeric arrays only (never pickled; loaded with allow_pickle=False)
    perception.json   format version, stack name, VideoInfo as a dict, timings, crops dir, and the descriptor
                      dicts as {"<track id>": [floats]} (float32 values survive the JSON round trip exactly)
    crops/<id>.jpg    face-blurred best crop of each track
"""

from __future__ import annotations

import hashlib
import heapq
import inspect
import itertools
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import MAX_SECONDS, MAX_WIDTH, MIN_TRACK_OBS, MIN_TRACK_S, SAMPLE_FPS
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
DEFAULT_POSE_EVERY = 2
PROGRESS_EVERY = 10              # sampled frames between progress callbacks
FRAMES_PROGRESS = 0.85           # progress share of the frame loop; describing people (embedding) takes the rest
HSV_WEIGHT = 0.5                 # weight of the secondary half of each descriptor
FORMAT_VERSION = 3               # 3: the effective parameters are stored (and checked) next to the arrays
STALE_TMP_SECONDS = 3600.0       # a `.tmp-*` build dir older than this is a crashed run and gets swept
TRACKER_KNOBS = ("activation", "lost_seconds", "match_thr")
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
    # Descriptors and crops exist for REAL tracks only (config.MIN_TRACK_OBS observations spanning
    # config.MIN_TRACK_S seconds); use .get(track_id) / `in`. Flicker tracks keep their obs_* rows and raw boxes.
    track_emb: dict[int, np.ndarray]     # real tracks: identity descriptor L2norm(concat(dino_mean, 0.5*hsv_full_mean))
    track_outfit: dict[int, np.ndarray]  # real tracks: outfit descriptor L2norm(concat(hsv_torso_mean, 0.5*dino_mean))
    crops_dir: str | None                # <dir>/<track_id>.jpg = face-blurred best crop of each REAL track
    cars: list[list[float]]              # car boxes [x1,y1,x2,y2,score] from the first frame (used in M8)
    timings: dict[str, float]            # decode/detect/track/pose/embed/total seconds
    params: dict = field(default_factory=dict)   # the effective settings of this run (see `perception_params`)

    def track_ids(self) -> list[int]:
        """Every track that has observations, flicker included (a subset of these has descriptors and crops)."""
        return [int(i) for i in np.unique(self.obs_track)]

    def obs_idx(self, track_id: int) -> np.ndarray:
        """Indices into the obs_* arrays of one track's observations, in frame order."""
        return np.flatnonzero(self.obs_track == track_id)


# --------------------------------------------------------------------------- effective parameters


def tracker_signature(tracker) -> dict:
    """The tracker settings that shape the tracks: its class and `activation`, `lost_seconds`, `match_thr`.

    A tracker that exposes those as attributes is read exactly. Otherwise they are taken from its constructor's
    defaults, which is what a change of the class's tuning shows up as; a stack that builds its tracker with
    non-default arguments but does not expose them must say so in its `name`.
    """
    cls = type(tracker)
    try:
        defaults = {n: prm.default for n, prm in inspect.signature(cls.__init__).parameters.items()
                    if prm.default is not inspect.Parameter.empty}
    except (TypeError, ValueError):
        defaults = {}
    return {"class": f"{cls.__module__}.{cls.__qualname__}",
            **{k: getattr(tracker, k, defaults.get(k)) for k in TRACKER_KNOBS}}


def _effective_params(stack: PerceptionStack, tracker, sample_fps, max_width, max_seconds, crops_per_track,
                      pose_every) -> dict:
    """Everything besides the video bytes that decides what a perception run stores, with the values the run
    really uses (defaults filled in, constants read now). It is hashed into the cache key and stored in the JSON."""
    pose_every = int(pose_every)
    if pose_every < 1:
        raise ValueError("pose_every must be >= 1")
    return {
        "sample_fps": float(sample_fps), "max_width": int(max_width), "max_seconds": float(max_seconds),
        "stack": stack.name, "crops_per_track": max(1, int(crops_per_track)), "pose_every": pose_every,
        "min_track_obs": int(MIN_TRACK_OBS), "min_track_s": float(MIN_TRACK_S),
        "tracker": tracker_signature(tracker),
    }


def perception_params(stack: PerceptionStack, **kw) -> dict:
    """The effective parameters of `cached_perception(path, stack, root, **kw)`: `kw` may carry `sample_fps`,
    `max_width`, `max_seconds`, `crops_per_track` and `pose_every` (anything else is a TypeError); what is left out
    takes its current default."""
    sample_fps = kw.pop("sample_fps", SAMPLE_FPS)
    max_width = kw.pop("max_width", MAX_WIDTH)
    max_seconds = kw.pop("max_seconds", MAX_SECONDS)
    crops_per_track = kw.pop("crops_per_track", DEFAULT_CROPS_PER_TRACK)
    pose_every = kw.pop("pose_every", DEFAULT_POSE_EVERY)
    if kw:
        raise TypeError(f"unexpected keyword arguments: {sorted(kw)}")
    return _effective_params(stack, stack.new_tracker(sample_fps), sample_fps, max_width, max_seconds,
                             crops_per_track, pose_every)


# --------------------------------------------------------------------------- the pass


def _shrink(img: np.ndarray | None, max_h: int = CROP_MAX_H) -> np.ndarray | None:
    if img is None or img.shape[0] <= max_h:
        return img
    scale = max_h / img.shape[0]
    return cv2.resize(img, (max(1, round(img.shape[1] * scale)), max_h), interpolation=cv2.INTER_AREA)


def carry_keypoints(kpts: np.ndarray, old_box, new_box) -> np.ndarray:
    """`kpts` (17, 3) found for `old_box`, moved to `new_box` by the box change: x' = x1' + (x - x1) w'/w and
    y' = y1' + (y - y1) h'/h; confidences are kept. A degenerate old box leaves them unchanged."""
    ox1, oy1, ox2, oy2 = (float(v) for v in old_box)
    nx1, ny1, nx2, ny2 = (float(v) for v in new_box)
    ow, oh = ox2 - ox1, oy2 - oy1
    out = np.array(kpts, np.float32)
    if ow > 0 and oh > 0:
        out[:, 0] = nx1 + (out[:, 0] - ox1) * ((nx2 - nx1) / ow)
        out[:, 1] = ny1 + (out[:, 1] - oy1) * ((ny2 - ny1) / oh)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > _EPS else np.zeros(v.shape, np.float32)


def _mean_nonzero(rows: list[np.ndarray], dim: int) -> np.ndarray:
    """Mean of the rows that carry information (an unusable crop gives a zero row); zeros when there are none."""
    kept = [r for r in rows if float(np.linalg.norm(r)) > _EPS]
    return np.mean(kept, axis=0, dtype=np.float64) if kept else np.zeros(dim, np.float64)


def _describe(stack: PerceptionStack, heaps: dict[int, list],
              progress: Callable[[int, int], None] | None = None) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Identity and outfit descriptors per track from its kept crops (see the module docstring).

    The embedder gets one call per track (its usable crops), so `progress(i, n)` can report after each of the `n`
    tracks that have usable crops (and once with i = 0 before the first). Production embeds one crop per model run
    (`stack.EMBED_BATCH`), so how the crops are grouped into calls does not change a single value."""
    dino: dict[tuple[int, int], np.ndarray] = {}
    dino_dim = 0
    if stack.embed is not None:
        groups = [(tid, [k for k, item in enumerate(h) if usable_crop(item[2])]) for tid, h in heaps.items()]
        groups = [(tid, ks) for tid, ks in groups if ks]
        if groups and progress is not None:
            progress(0, len(groups))
        for i, (tid, ks) in enumerate(groups, start=1):
            out = np.asarray(stack.embed([heaps[tid][k][2] for k in ks]), np.float32)
            if out.ndim != 2 or len(out) != len(ks) or (dino_dim and out.shape[1] != dino_dim):
                raise ValueError(f"embedder returned {out.shape} for {len(ks)} crops")
            dino.update({(tid, k): out[n] for n, k in enumerate(ks)})
            dino_dim = out.shape[1]
            if progress is not None:
                progress(i, len(groups))
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
                   crops_dir: str | Path | None = None, crops_per_track: int = DEFAULT_CROPS_PER_TRACK,
                   pose_every: int = DEFAULT_POSE_EVERY) -> PerceptionResult:
    """Run the whole perception stack over `path` (sampled at `sample_fps`, resized to `max_width`, cut at
    `max_seconds`). `progress(fraction, message)` is called about every 10 sampled frames (fractions up to
    `FRAMES_PROGRESS`), then while describing people ("describing people… i of n", up to 1.0) and finally with 1.0.
    With `crops_dir`, each real track's face-blurred best crop is written there as `<track_id>.jpg`.
    `pose_every` is the pose stride: a track gets fresh keypoints on its first frame and then whenever
    `pose_every` sampled frames have passed since its last fresh ones (1 = every frame); in between they are
    carried along with its box. A box too small for pose (no keypoints came back) is retried every frame.
    Only real tracks (see the module docstring) get `track_emb` / `track_outfit` entries and crop files.
    Raises `VideoError` when the file cannot be decoded, ValueError for `pose_every < 1`.

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
    params = _effective_params(stack, tracker, sample_fps, max_width, max_seconds, keep, pose_every)
    pose_every = params["pose_every"]

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
    pose_state: dict[int, tuple[int, np.ndarray, tuple]] = {}   # track -> (frame, keypoints, box) of last fresh pose
    n_obs: dict[int, int] = {}                           # track -> observations so far
    t_span: dict[int, list[float]] = {}                  # track -> [first, last] observation time
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
                due = []                                 # tracked boxes that need fresh keypoints this frame
                for j, (tid, d) in enumerate(tracked):
                    last = pose_state.get(tid)
                    if last is None or fi - last[0] >= pose_every:
                        due.append(j)
                    else:
                        kpts[j] = carry_keypoints(last[1], last[2], d.bbox)
                if due:
                    boxes = np.asarray([tracked[j][1].bbox for j in due], np.float64)
                    fresh = np.asarray(stack.pose(frame, boxes), np.float32).reshape(len(due), 17, 3)
                    for n, j in enumerate(due):
                        tid, d = tracked[j]
                        kpts[j] = fresh[n]
                        x1, y1, x2, y2 = d.bbox
                        if fresh[n, :, 2].max() > 0 and x2 > x1 and y2 > y1:
                            pose_state[tid] = (fi, fresh[n].copy(), (x1, y1, x2, y2))
                        else:                            # nothing came back (too small a box): try again next frame
                            pose_state.pop(tid, None)
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
                n_obs[tid] = n_obs.get(tid, 0) + 1
                span = t_span.setdefault(tid, [float(t), float(t)])
                span[1] = float(t)
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
                frac *= FRAMES_PROGRESS
                progress(frac, f"Finding and tracking people: {t:.0f} of {info.duration:.0f} s")
    finally:
        frames.close()

    t0 = time.perf_counter()
    real = {tid for tid, (first, last) in t_span.items()
            if n_obs[tid] >= MIN_TRACK_OBS and last - first >= MIN_TRACK_S - 1e-6}
    heaps = {tid: h for tid, h in heaps.items() if tid in real}
    described = None
    if progress is not None:
        def described(i: int, n: int) -> None:
            progress(FRAMES_PROGRESS + (1.0 - FRAMES_PROGRESS) * i / n, f"describing people… {i} of {n}")
    track_emb, track_outfit = _describe(stack, heaps, described)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid, (_, thumb) in best.items():
            if tid in real:
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
        cars=cars, timings=tm, params=params,
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
        "params": res.params, "crops_dir": crops,
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
        params=dict(meta["params"]),
    )


def cache_key(path, params: dict) -> str:
    """The cache directory name for `path` under `params` (from `perception_params`): `video_key` over the file's
    bytes, the three video-shaping parameters and a canonical JSON of every other parameter."""
    rest = {k: v for k, v in params.items() if k not in ("sample_fps", "max_width", "max_seconds")}
    return video_key(path, params["sample_fps"], params["max_width"], params["max_seconds"],
                     json.dumps(to_jsonable(rest), sort_keys=True, separators=(",", ":")))


def _sweep_stale_tmp(cache_root: Path, max_age: float = STALE_TMP_SECONDS) -> None:
    """Remove `.tmp-*` build directories older than `max_age` seconds: what a crashed or killed run left behind.
    A live build has no such directory until its very last step, so an hour-old one is never in use."""
    now = time.time()
    try:
        candidates = [p for p in cache_root.glob(".tmp-*") if p.is_dir()]
    except OSError:
        return
    for p in candidates:
        try:
            if now - p.stat().st_mtime > max_age:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def cached_perception(path, stack: PerceptionStack, cache_root: Path, progress=None,
                      **kw) -> tuple[PerceptionResult, bool]:
    """`(result, was_cached)`: the saved perception of this exact video + parameters + stack when there is one,
    otherwise run it (via the module-level `run_perception`) and save it under `cache_root/<cache_key>/`.

    `kw` may carry `sample_fps`, `max_width`, `max_seconds`, `crops_per_track` and `pose_every`; anything else
    is a TypeError. The key always covers the effective value of each of those plus `MIN_TRACK_OBS`, `MIN_TRACK_S`
    and the tracker settings (see `perception_params`), and the same dict is stored in perception.json: changing
    a default, a constant or the tracker's tuning never serves an entry built under the old ones. An entry that
    cannot be read, or whose stored parameters disagree with the expected ones, is discarded and recomputed.
    New entries are built in a temporary directory and moved into place, so a crash or a concurrent run never
    leaves a half-written entry; temp directories older than an hour are swept first. The returned result
    always describes the video at `path` (the stored path is replaced, since the same bytes may have been cached
    under another name, or by a run that won a race).
    """
    params = perception_params(stack, **kw)
    cache_root = Path(cache_root)
    if cache_root.is_dir():
        _sweep_stale_tmp(cache_root)
    d = cache_root / cache_key(path, params)

    if d.is_dir():
        try:
            res = load_result(d)
            if res.params != to_jsonable(params):
                raise ValueError("stored parameters differ from the requested ones")
        except Exception as exc:                                        # any unreadable entry is just a miss
            log.warning("discarding unusable perception cache %s (%s: %s)", d.name, type(exc).__name__, exc)
            shutil.rmtree(d, ignore_errors=True)
        else:
            res.video = replace(res.video, path=str(path))
            if progress is not None:
                progress(1.0, "Using the saved analysis of this video")
            return res, True

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = cache_root / f".tmp-{d.name}-{uuid.uuid4().hex[:8]}"
    try:
        res = run_perception(path, stack, sample_fps=params["sample_fps"], max_width=params["max_width"],
                             max_seconds=params["max_seconds"], progress=progress, crops_dir=tmp / CROPS_NAME,
                             crops_per_track=params["crops_per_track"], pose_every=params["pose_every"])
        save_result(res, tmp)
        try:
            os.replace(tmp, d)
        except OSError:
            if not d.is_dir():                                          # not a lost race: a real failure
                raise
            log.info("perception cache %s was written concurrently; using that entry", d.name)
        res = load_result(d)
        res.video = replace(res.video, path=str(path))                  # also right when another run's entry won
        return res, False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
