"""Per-video perception cache: models run once; zones, roles and thresholds can change afterwards.

Layout ``<cache_root>/<key>/``: meta.json, tracks.npz, cars.json, segments.json, embeddings.npz and
keyframe.jpg. The keyframe is the raw (unblurred) frame and stays only inside this directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .datatypes import Det, StableSegment, TrackObs
from .perception import finetuned

PERCEPTION_RUNS = 0
_FILES = ("meta.json", "tracks.npz", "cars.json", "segments.json", "embeddings.npz", "raw_dets.npz")
CACHE_VERSION = 2   # v2: raw person detections stored (privacy blur); older caches are rebuilt


def perception_runs() -> int:
    return PERCEPTION_RUNS


def count_run() -> None:
    global PERCEPTION_RUNS
    PERCEPTION_RUNS += 1


@dataclass
class PerceptionCache:
    dir: Path
    meta: dict
    obs: list[TrackObs]
    cars: list[Det]
    keyframe_t: float
    segments: list[StableSegment]
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    keyframe: np.ndarray | None = None      # raw BGR frame; written on save, not loaded
    # every raw person detection (t, bbox, score), tracked or not; used only to blur faces
    raw_dets: list[tuple[float, tuple, float]] = field(default_factory=list)

    @property
    def keyframe_path(self) -> Path:
        return Path(self.dir) / "keyframe.jpg"


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_key(path, cfg: Config, sha1: str | None = None) -> str:
    params = f"|v{CACHE_VERSION}|{cfg.sample_fps}|{cfg.max_width}|{cfg.max_duration_s}|{cfg.detector}|{cfg.pose_model}"
    if cfg.vehicle_detector != cfg.detector:
        params += f"|cars={cfg.vehicle_detector}"
    ft = finetuned.tag_from_env()          # PULSE_RFDETR_FT: person detector weights (absent: key unchanged)
    if ft is not None:
        params += f"|person={ft}"
    return hashlib.sha1(((sha1 or file_sha1(path)) + params).encode()).hexdigest()


def exists(d) -> bool:
    d = Path(d)
    return all((d / n).is_file() for n in _FILES)


def save(pc: PerceptionCache) -> None:
    d = Path(pc.dir)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{d.name}.", suffix=".tmp", dir=d.parent))
    try:
        n = len(pc.obs)
        kps = np.full((n, 17, 3), np.nan)
        for i, o in enumerate(pc.obs):
            if o.keypoints is not None:
                kps[i] = np.asarray(o.keypoints, float)
        np.savez(tmp / "tracks.npz",
                 t=np.array([o.t for o in pc.obs], float),
                 frame_idx=np.array([o.frame_idx for o in pc.obs], np.int64),
                 track_id=np.array([o.track_id for o in pc.obs], np.int64),
                 bbox=np.array([o.bbox for o in pc.obs], float).reshape(n, 4),
                 score=np.array([o.score for o in pc.obs], float), kps=kps)
        rd = pc.raw_dets
        np.savez(tmp / "raw_dets.npz", t=np.array([r[0] for r in rd], float),
                 bbox=np.array([r[1] for r in rd], float).reshape(len(rd), 4),
                 score=np.array([r[2] for r in rd], float))
        ids = sorted(pc.embeddings)
        vecs = (np.stack([np.asarray(pc.embeddings[i], float) for i in ids])
                if ids else np.zeros((0, 0)))
        np.savez(tmp / "embeddings.npz", ids=np.array(ids, np.int64), vecs=vecs)
        (tmp / "cars.json").write_text(json.dumps([c.to_dict() for c in pc.cars]))
        (tmp / "segments.json").write_text(json.dumps([s.to_dict() for s in pc.segments]))
        if pc.keyframe is not None:
            cv2.imwrite(str(tmp / "keyframe.jpg"), pc.keyframe, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # meta.json last: it carries keyframe_t and is the final file written before the rename
        (tmp / "meta.json").write_text(json.dumps({**pc.meta, "keyframe_t": pc.keyframe_t}))
        if d.exists():
            shutil.rmtree(d)
        os.replace(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load(d) -> PerceptionCache:
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    keyframe_t = float(meta.pop("keyframe_t", 0.0))
    z = np.load(d / "tracks.npz")
    obs = []
    for i in range(len(z["t"])):
        kp = z["kps"][i]
        obs.append(TrackObs(float(z["t"][i]), int(z["frame_idx"][i]), int(z["track_id"][i]),
                            tuple(float(v) for v in z["bbox"][i]), float(z["score"][i]),
                            None if np.isnan(kp).all() else kp.tolist()))
    e = np.load(d / "embeddings.npz")
    emb = {int(i): v for i, v in zip(e["ids"], e["vecs"])}
    cars = [Det(tuple(c["bbox"]), c["score"], c["cls"]) for c in json.loads((d / "cars.json").read_text())]
    segs = [StableSegment.from_dict(s) for s in json.loads((d / "segments.json").read_text())]
    r = np.load(d / "raw_dets.npz")
    raw = [(float(t), tuple(float(v) for v in b), float(sc)) for t, b, sc in zip(r["t"], r["bbox"], r["score"])]
    return PerceptionCache(d, meta, obs, cars, keyframe_t, segs, emb, raw_dets=raw)
