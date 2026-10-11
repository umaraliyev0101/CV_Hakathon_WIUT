"""MEVA evaluation (spec section 6): tracking, engagement against talks, and arrival/leaving against enter/exit events.

MEVA (Corona et al., WACV 2021; Kitware/IARPA; CC BY 4.0) gives person boxes for the actors that take part in an
annotated activity, plus the activities themselves. This module scores the offline pipeline against them on the sampled
frames, with the ground truth taken at the matching source frame and scaled to the processed frames:

* ``match_ids`` maps our track or person ids to ground-truth actors by a vote of per-frame IoU matches; ``twin_clusters``
  finds the actors that MEVA annotated twice for one physical person.
* ``tracking_metrics`` is MOTA / IDF1 (motmetrics), once for raw tracks and once for the merged persons.
* ``pair_intervals`` finds every period in which two people stood within ``dist`` metres for ``min_dur`` seconds (roles
  ignored, same streak rule as the event engine's engagement); ``talk_eval`` compares them with
  ``person_talks_to_person``.
* ``enter_exit_eval`` compares the presence of a mapped person with the enter/exit-through-structure events, as the
  nearest presence boundary and as first/last seen.
* ``oracle_result`` replaces the cached detections with the ground-truth boxes (``--oracle``): an ablation that scores the
  geometry and the engagement rule alone, with perfect detection and tracking.
* ``eval_clip`` runs all of it for one clip. Talks and enter/exit are scored twice and reported side by side: ``strict``
  (an id maps to one actor if it is on it for >= 3 frames and >= 30% of its own frames; twin actors are separate) and
  ``twin_aware`` (an id also maps when it is on >= 50% of a briefly annotated actor's frames, and twin actors count as
  one person, on both sides). ``aggregate`` pools clips (counts are summed first, so the averages are micro-averages),
  ``main`` is the command line:

      python -m showroom_coach.evaluation.meva_eval --clips data/meva/clips.csv --split dev           --split-file data/meva/split.json --out runs/eval/meva_dev.json

Only part of each scene is annotated (the actors of the annotated activities). A person we track who was never
annotated therefore counts as a false positive; tracking precision and MOTA are pessimistic for that reason. Talk
metrics have two recalls to separate the two ways of missing a conversation: ``recall_all`` counts every ground-truth pair
interval, ``recall_tracked`` only those whose two actors we followed as persons (so it isolates the engagement logic from
detection and tracking misses).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import analysis, config
from ..events import EPS, _Streak
from ..geometry import distance_m
from ..perception.pipeline import cached_perception
from .kpf import Activity, GeomBox, load_activities, load_geom, load_types, person_ids

GT_TALK = "person_talks_to_person"
GT_ENTER = "person_enters_scene_through_structure"
GT_EXIT = "person_exits_scene_through_structure"
VARIANTS = ("strict", "twin_aware")  # the two ways of crediting talks and enter/exit events to our persons

ID_IOU = 0.3          # IoU a detection needs with a ground-truth box to vote for that actor
ID_MIN_VOTES = 3      # votes (matched frames) an id needs for the actor that won ...
ID_MIN_SHARE = 0.30   # ... and the share of the id's observed frames those votes must make up ...
ID_ACTOR_SHARE = 0.50  # ... or the share of the actor's own annotated frames (in the evaluated window) they make up
TWIN_IOU = 0.70       # two actors are twins when their boxes overlap this much ...
TWIN_SHARE = 0.50     # ... on this share of their co-annotated sampled frames ...
TWIN_MIN_FRAMES = 3   # ... and they are co-annotated on at least this many
TWIN_DEDUPE_TIOU = 0.80  # twin-aware talks: ground-truth intervals of one cluster pair this similar are one conversation
MOT_IOU = 0.5         # IoU for a tracking match (distance 1 - IoU <= 0.5)
TALK_TIOU = 0.30      # temporal IoU for a detected interval to match a ground-truth pair interval
TALK_TIOU_2 = 0.10    # the secondary, looser line
ENTER_EXIT_WINDOW_S = 5.0
EDGE_S = 1.0          # enter/exit events this close to the start or end of the clip are not scored

ORACLE_NOTE = (
    "ORACLE MODE (ablation, not the system): the cached detections are replaced by the ground-truth person boxes at the "
    "same sampled frames, one track per person (twin actors collapsed into the lowest actor id of their cluster), score "
    "1.0, no keypoints (distance comes from the box height) and no embeddings (persons are the tracks, nothing is "
    "merged). The analysis, pair detector and scoring are the same as in a normal run; our ids map to themselves. It "
    "isolates the geometry and the engagement rule from detection and tracking. Ground-truth boxes exist only while an "
    "actor takes part in an annotated activity, so presence and enter/exit timing are partly circular here.")

NOTES = [
    "MEVA annotates only the actors of its activities: people we track who are not annotated count as false "
    "positives, so tracking precision and MOTA are pessimistic.",
    "MEVA often annotates one person as two actors (gt_dup_share = share of ground-truth boxes with a near-identical "
    "twin in the same frame); a twin cannot be matched one-to-one, which caps recall in the tracking table.",
    "Talk ground truth: groups of k actors are expanded to all k*(k-1)/2 pairs, each pair's interval is the activity "
    "span intersected with both actors' own spans. Our intervals are every person pair within --dist metres for "
    "--min-dur seconds (roles ignored). A match needs the same mapped pair and temporal IoU >= 0.3 (secondary line 0.1).",
    "recall_all counts every ground-truth pair interval; recall_tracked only those whose two actors both map from a "
    "person of ours; precision_mapped only counts our intervals whose two persons both map to annotated actors "
    "(n_det_mapped of n_det).",
    "strict: an id maps to the actor it votes for most if the votes are >= 3 and >= 30% of the id's observed frames; "
    "twin actors are separate. twin_aware: the id also maps when its votes are >= 50% of that actor's annotated sampled "
    "frames (a briefly annotated actor can only collect a few votes), and actors whose boxes have IoU >= 0.7 on >= 50% "
    "of >= 3 co-annotated sampled frames are one person: an event or talk pair is credited to any member of the "
    "cluster, GT talk pairs that become identical (same cluster pair, interval tIoU >= 0.8) count once, and pairs "
    "between two members of one cluster are dropped.",
    "Enter/exit: the ground-truth time is the start (enter) or end (exit) of the event; ours is the start or end of the "
    "nearest presence interval of any person mapped to that actor (main numbers), or the earliest start / latest end "
    "of them (first_last); a match is within 5 s. Events in the first and last second of the clip are ignored.",
    "Averages over clips are micro-averages (counts summed first); median errors are pooled over all matches.",
]


# ------------------------------------------------------------------------------------------------ small helpers

def _ratio(num, den):
    return float(num) / den if den else None


def _median(values):
    return float(statistics.median(values)) if len(values) else None


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (A, 4) with every box in ``b`` (B, 4), all ``x1 y1 x2 y2``."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _greedy(score: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """One-to-one pairs (row, col) with ``score >= thr``, best score first (ties: lower row, then lower column)."""
    cand = np.argwhere(score >= thr - 1e-12)
    order = sorted(((-float(score[i, j]), int(i), int(j)) for i, j in cand))
    used_r, used_c, out = set(), set(), []
    for _, i, j in order:
        if i not in used_r and j not in used_c:
            used_r.add(i)
            used_c.add(j)
            out.append((i, j))
    return out


def _hyp_ids(res, an, use_persons: bool) -> tuple[np.ndarray, np.ndarray]:
    """(ids, keep): the id of each observation (track id, or person id) and which observations count."""
    if use_persons:
        ids = np.asarray(an.obs_pid, np.int64)
        return ids, ids > 0
    ids = np.asarray(res.obs_track, np.int64)
    return ids, np.ones(len(ids), bool)


def _group_by_frame(obs_frame, keep) -> dict[int, np.ndarray]:
    """Sampled frame -> indices of the kept observations in it."""
    obs_frame = np.asarray(obs_frame)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return {}
    order = idx[np.argsort(obs_frame[idx], kind="stable")]
    frames = obs_frame[order]
    cuts = np.flatnonzero(np.diff(frames)) + 1
    return {int(obs_frame[g[0]]): g for g in np.split(order, cuts)}


def _gt_at(gt_boxes, res, f: int, scale: float) -> tuple[list[int], np.ndarray]:
    """Actor ids and processed-pixel boxes of the ground truth at sampled frame ``f``."""
    boxes = gt_boxes.get(int(res.src_frame_idx[f]), ())
    return [b.actor for b in boxes], np.array([b.bbox for b in boxes], float).reshape(-1, 4) * scale


# ---------------------------------------------------------------------------------------------------- id mapping

def match_ids(res, an_or_none, gt_boxes: dict[int, list[GeomBox]], use_persons: bool, iou_thr: float = ID_IOU,
              min_votes: int = ID_MIN_VOTES, min_share: float = ID_MIN_SHARE,
              min_actor_share: float | None = ID_ACTOR_SHARE) -> dict[int, int]:
    """Our id (a track id, or a person id with ``use_persons``) -> the ground-truth actor it follows.

    On every sampled frame the ground-truth person boxes of the matching source frame (scaled by ``res.video.scale``)
    are matched one-to-one, greedily by IoU (>= ``iou_thr``), with our observations of that frame; each match is a vote
    of that id for that actor. An id maps to the actor with the most votes (ties: the lower actor id) when the votes are
    at least ``min_votes`` and either at least ``min_share`` of the id's observed frames or at least ``min_actor_share``
    of the actor's own annotated frames among the sampled frames (an actor annotated for a few seconds can only collect
    a few votes). ``min_actor_share=None`` drops the second condition: the strict rule of the first version. Several of
    our ids may map to one actor (fragments of one person).
    """
    ids, keep = _hyp_ids(res, an_or_none, use_persons)
    scale = float(res.video.scale)
    votes: dict[int, Counter] = defaultdict(Counter)
    annotated: Counter = Counter()  # actor -> sampled frames on which it has a box
    for f in range(len(res.frame_times)):
        annotated.update(b.actor for b in gt_boxes.get(int(res.src_frame_idx[f]), ()))
    for f, obs in _group_by_frame(res.obs_frame, keep).items():
        actors, g_boxes = _gt_at(gt_boxes, res, f, scale)
        if not actors:
            continue
        for gi, hi in _greedy(_iou_matrix(g_boxes, res.obs_bbox[obs]), iou_thr):
            votes[int(ids[obs[hi]])][actors[gi]] += 1
    seen = Counter(int(i) for i in ids[keep])
    out: dict[int, int] = {}
    for hid, counter in votes.items():
        actor, n = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))
        if n < min_votes:
            continue
        if n >= min_share * seen[hid] - 1e-9 or (min_actor_share is not None
                                                 and n >= min_actor_share * annotated[actor] - 1e-9):
            out[hid] = actor
    return dict(sorted(out.items()))


def twin_clusters(res, gt_boxes: dict[int, list[GeomBox]], iou_thr: float = TWIN_IOU, min_share: float = TWIN_SHARE,
                  min_frames: int = TWIN_MIN_FRAMES) -> dict[int, int]:
    """Ground-truth actors that are the same physical person: ``{actor: representative}`` for every actor in a cluster
    of two or more (the representative is the cluster's lowest actor id and maps to itself).

    MEVA often annotates one person as several actors. Two actors are twins when their boxes have IoU >= ``iou_thr`` on
    at least ``min_share`` of the sampled frames on which both are annotated, and both are annotated together on at
    least ``min_frames`` of them. Twin-ness is transitive.
    """
    both: Counter = Counter()
    hit: Counter = Counter()
    for f in range(len(res.frame_times)):
        boxes = gt_boxes.get(int(res.src_frame_idx[f]), ())
        if len(boxes) < 2:
            continue
        arr = np.array([b.bbox for b in boxes], float)
        iou = _iou_matrix(arr, arr)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                key = tuple(sorted((boxes[i].actor, boxes[j].actor)))
                both[key] += 1
                hit[key] += iou[i, j] >= iou_thr - 1e-12
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), n in both.items():
        if n >= min_frames and hit[(a, b)] >= min_share * n - 1e-9:
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    return {actor: find(actor) for actor in sorted(parent)}


def gt_duplicate_counts(res, gt_boxes: dict[int, list[GeomBox]], iou_thr: float = 0.7) -> tuple[int, int]:
    """``(boxes with a twin, all boxes)`` over the ground-truth person boxes on the sampled frames. A twin is another
    ground-truth box of the same frame with IoU >= ``iou_thr``: MEVA often annotates one physical person as two actors,
    and the twin cannot be matched one-to-one, which lowers the recall any tracker can reach."""
    total = twins = 0
    for f in range(len(res.frame_times)):
        boxes = gt_boxes.get(int(res.src_frame_idx[f]), ())
        if not boxes:
            continue
        arr = np.array([b.bbox for b in boxes], float)
        iou = _iou_matrix(arr, arr)
        np.fill_diagonal(iou, 0.0)
        total += len(arr)
        twins += int((iou.max(axis=1) >= iou_thr).sum())
    return twins, total


def gt_duplicate_share(res, gt_boxes: dict[int, list[GeomBox]], iou_thr: float = 0.7) -> float | None:
    """``gt_duplicate_counts`` as a share (None when there are no boxes)."""
    twins, total = gt_duplicate_counts(res, gt_boxes, iou_thr)
    return _ratio(twins, total)


# ---------------------------------------------------------------------------------------------------- tracking

_MOT_COUNTS = ("num_objects", "num_misses", "num_false_positives", "num_switches", "num_detections", "idtp", "idfp",
               "idfn")


def _motmetrics():
    """The motmetrics module, imported on first use (it is an optional dependency, the ``eval`` extra)."""
    try:
        import motmetrics
    except ImportError as exc:
        raise RuntimeError('the MEVA tracking metrics need the "motmetrics" package: pip install -e ".[eval]"') from exc
    return motmetrics


def _mot_summary(c: dict) -> dict:
    """MOTA, IDF1, precision and recall from the raw counts (None where the denominator is 0)."""
    objects, det = c.get("num_objects", 0), c.get("num_detections", 0)
    idtp = c.get("idtp", 0)
    errors = c.get("num_misses", 0) + c.get("num_false_positives", 0) + c.get("num_switches", 0)
    return {
        "mota": None if not objects else 1.0 - errors / objects,
        "idf1": _ratio(2 * idtp, 2 * idtp + c.get("idfp", 0) + c.get("idfn", 0)),
        "precision": _ratio(det, det + c.get("num_false_positives", 0)),
        "recall": _ratio(det, objects),
    }


def tracking_metrics(res, an, gt_boxes: dict[int, list[GeomBox]], use_persons: bool) -> dict:
    """MOTA / IDF1 on the sampled frames (motmetrics). Ground truth = the annotated person actors present at the
    frame's source frame; hypotheses = our tracks (``use_persons=False``) or persons. A pair may match when its box
    IoU is >= 0.5 (distance 1 - IoU <= 0.5).

    Returns the metrics, ``num_frames`` and the raw counts they come from (so clips can be pooled): ``num_objects``,
    ``num_misses``, ``num_false_positives``, ``num_switches``, ``num_detections``, ``idtp``, ``idfp``, ``idfn``.
    Unannotated people we track are false positives (see the module docstring).
    """
    mm = _motmetrics()
    ids, keep = _hyp_ids(res, an, use_persons)
    groups = _group_by_frame(res.obs_frame, keep)
    scale = float(res.video.scale)
    acc = mm.MOTAccumulator(auto_id=False)
    n_frames = len(res.frame_times)
    empty = np.array([], np.int64)
    for f in range(n_frames):
        actors, g_boxes = _gt_at(gt_boxes, res, f, scale)
        obs = groups.get(f, empty)
        iou = _iou_matrix(g_boxes, res.obs_bbox[obs] if len(obs) else np.zeros((0, 4)))
        dist = 1.0 - iou
        dist[iou < MOT_IOU - 1e-12] = np.nan
        acc.update(actors, [int(i) for i in ids[obs]], dist, frameid=f)
    counts = dict.fromkeys(_MOT_COUNTS, 0)
    if n_frames:
        summary = mm.metrics.create().compute(acc, metrics=list(_MOT_COUNTS), name="m")
        counts = {k: int(summary.loc["m", k]) for k in _MOT_COUNTS}
    return {**_mot_summary(counts), "num_frames": n_frames, **counts}


# ----------------------------------------------------------------------------------------------- engagement pairs

def pair_intervals(an, res, dist: float, min_dur: float, gap_tol: float) -> list[tuple[int, int, float, float]]:
    """``(pid_a, pid_b, t_start, t_end)`` (a < b) for every period in which two persons stood within ``dist`` metres.

    Roles are ignored: every pair of persons counts. A period is a streak of the event engine (``events._Streak``): it
    starts at the first sample within ``dist``, bridges gaps of up to ``gap_tol`` seconds (including samples where one
    of the two is not seen), ends at its last sample within ``dist``, and is kept when it lasted at least ``min_dur``
    seconds. Sorted by start time.
    """
    pid = np.asarray(an.obs_pid, np.int64)
    xz = np.asarray(an.obs_xz, float)
    groups = _group_by_frame(res.obs_frame, pid > 0)
    streaks: dict[tuple[int, int], _Streak] = {}
    out: list[tuple[int, int, float, float]] = []

    def keep(key, run):
        if run is not None and run.end - run.start >= min_dur - EPS:
            out.append((key[0], key[1], float(run.start), float(run.end)))

    for f, t in enumerate(np.asarray(res.frame_times, float)):
        near: dict[tuple[int, int], bool] = {}
        obs = groups.get(f)
        if obs is not None:
            for x in range(len(obs)):
                for y in range(x + 1, len(obs)):
                    a, b = sorted((int(pid[obs[x]]), int(pid[obs[y]])))
                    near[(a, b)] = distance_m(tuple(xz[obs[x]]), tuple(xz[obs[y]])) <= dist
        for key in sorted(set(streaks) | set(near)):
            streak = streaks.get(key) or _Streak(gap_tol)
            keep(key, streak.update(float(t), near.get(key, False)))
            if streak.is_open:
                streaks[key] = streak
            else:
                streaks.pop(key, None)
    for key in sorted(streaks):
        keep(key, streaks[key].close())
    out.sort(key=lambda r: (r[2], r[0], r[1]))
    return out


@dataclass(frozen=True)
class GtPair:
    a: int
    b: int  # a < b
    t0: float
    t1: float
    act_id: int


def gt_talk_pairs(talks: list[Activity], fps: float, t_max: float | None = None,
                  persons: set[int] | None = None, clusters: dict[int, int] | None = None) -> list[GtPair]:
    """Ground-truth pair intervals of the ``person_talks_to_person`` activities in ``talks``.

    A group of k actors becomes all k*(k-1)/2 pairs. A pair's span is the activity span intersected with both actors'
    own spans; a pair is skipped when that is empty, when an actor has no span of its own, when it is shorter than a
    frame, or (with ``persons``) when an actor is not a person. Times are frames / ``fps`` (end = last frame);
    ``t_max`` cuts the interval at that time and drops pairs that start after it.

    With ``clusters`` (twin-aware mode, see ``twin_clusters``; an empty dict still counts) every actor is replaced by
    its cluster's representative; pairs between two actors of one cluster are dropped, and pairs of one cluster pair
    whose intervals overlap with temporal IoU >= ``TWIN_DEDUPE_TIOU`` are one conversation annotated twice: the first
    is kept.
    """
    canon = (lambda x: x) if clusters is None else (lambda x: clusters.get(x, x))
    out: list[GtPair] = []
    for act in talks:
        if act.name != GT_TALK:
            continue
        actors = sorted({x for x in act.actors if persons is None or x in persons})
        for i, a in enumerate(actors):
            for b in actors[i + 1:]:
                if a not in act.actor_spans or b not in act.actor_spans:
                    continue
                lo = max(act.f0, act.actor_spans[a][0], act.actor_spans[b][0])
                hi = min(act.f1, act.actor_spans[a][1], act.actor_spans[b][1])
                if lo > hi:
                    continue
                t0, t1 = lo / fps, hi / fps
                if t_max is not None:
                    t1 = min(t1, t_max)
                if t1 <= t0:
                    continue
                ca, cb = sorted((canon(a), canon(b)))
                if ca == cb:
                    continue  # a talk between two annotations of one person
                if clusters is not None and any(g.a == ca and g.b == cb and _tiou(t0, t1, g.t0, g.t1) >= TWIN_DEDUPE_TIOU
                                                for g in out):
                    continue
                out.append(GtPair(ca, cb, t0, t1, act.act_id))
    return out


def _tiou(a0, a1, b0, b1) -> float:
    inter = min(a1, b1) - max(a0, b0)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if inter > 0 and union > 0 else 0.0


def _talk_summary(c: dict, errs: list[float]) -> dict:
    """Ratios and median errors from the counts of ``_match_talk`` (also used to pool clips)."""
    return {
        "n_gt": c["n_gt"], "n_gt_tracked": c["n_gt_tracked"], "n_det": c["n_det"], "n_det_mapped": c["n_det_mapped"],
        "n_matched": c["n_matched"], "n_gt_collapsed": c.get("n_gt_collapsed", 0),
        "recall_all": _ratio(c["n_matched"], c["n_gt"]),
        "recall_tracked": _ratio(c["n_matched"], c["n_gt_tracked"]),
        "precision_mapped": _ratio(c["n_matched"], c["n_det_mapped"]),
        "median_start_err_s": _median([abs(e) for e in errs]),
        "median_start_bias_s": _median(errs),
        "start_errs_s": [float(e) for e in errs],
    }


_TALK_COUNTS = ("n_gt", "n_gt_tracked", "n_det", "n_det_mapped", "n_matched", "n_gt_collapsed")


def talk_eval(intervals, talks: list[Activity], id_map: dict[int, int], fps: float, tiou_thr: float = TALK_TIOU,
              t_max: float | None = None, persons: set[int] | None = None,
              secondary_tiou: float | None = TALK_TIOU_2, clusters: dict[int, int] | None = None) -> dict:
    """Detected pair intervals against the ground-truth talks (see the module docstring and ``gt_talk_pairs``).

    ``intervals`` are ``(pid_a, pid_b, t0, t1)``; ``id_map`` maps our person ids to actors. A detection matches a
    ground-truth pair interval when its two persons map to exactly that pair of actors (either order) and the temporal
    IoU is >= ``tiou_thr``; matching is one-to-one, best IoU first.

    Returns n_gt, n_gt_tracked (ground-truth pairs whose two actors both map from some person of ours), n_det,
    n_det_mapped (detections whose two persons both map), n_matched, recall_all, recall_tracked, precision_mapped,
    median_start_err_s (median |our start - GT start| over the matches), median_start_bias_s (signed; positive = we
    start late) and the signed ``start_errs_s``. The same under ``"tiou_<secondary>"`` for the looser threshold.

    ``clusters`` switches on twin-aware mode: actors that are twins (``twin_clusters``) count as one, on the
    ground-truth side (``gt_talk_pairs``) and for the actors our persons map to; ``n_gt_collapsed`` is the number of
    ground-truth pair intervals folded away by that. Without it the mapping is strict: one actor per id.
    """
    gt = gt_talk_pairs(talks, fps, t_max, persons, clusters)
    n_collapsed = len(gt_talk_pairs(talks, fps, t_max, persons)) - len(gt) if clusters is not None else 0
    canon = (lambda x: x) if clusters is None else (lambda x: clusters.get(x, x))
    mapped = {canon(v) for v in id_map.values()}
    dets = []  # (mapped pair or None, t0, t1)
    for a, b, t0, t1 in intervals:
        ma, mb = id_map.get(a), id_map.get(b)
        pair = tuple(sorted((canon(ma), canon(mb)))) if ma is not None and mb is not None else None
        dets.append((pair, t0, t1))
    base = {"n_gt": len(gt), "n_gt_tracked": sum(1 for g in gt if g.a in mapped and g.b in mapped),
            "n_det": len(dets), "n_det_mapped": sum(1 for d in dets if d[0] is not None),
            "n_gt_collapsed": n_collapsed}

    def at(thr: float) -> dict:
        by_pair = defaultdict(list)
        for j, g in enumerate(gt):
            by_pair[(g.a, g.b)].append(j)
        score = np.zeros((len(dets), len(gt)))
        for i, (pair, t0, t1) in enumerate(dets):
            for j in by_pair.get(pair, ()):
                score[i, j] = _tiou(t0, t1, gt[j].t0, gt[j].t1)
        pairs = _greedy(score, thr) if score.size else []
        errs = [dets[i][1] - gt[j].t0 for i, j in pairs]
        return _talk_summary({**base, "n_matched": len(pairs)}, errs)

    out = {**at(tiou_thr), "tiou": tiou_thr}
    if secondary_tiou is not None:
        out[f"tiou_{secondary_tiou:g}"] = {**at(secondary_tiou), "tiou": secondary_tiou}
    return out


# --------------------------------------------------------------------------------------------------- enter / exit

def _ee_summary(n_gt: int, n_tracked: int, n_matched: int, errs: list[float]) -> dict:
    return {"n_gt": n_gt, "n_gt_tracked": n_tracked, "n_matched": n_matched,
            "recall_all": _ratio(n_matched, n_gt), "recall_tracked": _ratio(n_matched, n_tracked),
            "median_abs_err_s": _median([abs(e) for e in errs]), "median_bias_s": _median(errs),
            "errs_s": [float(e) for e in errs]}


def _ee_blocks(tally: dict) -> dict:
    """``{"enter", "exit", "all"}`` summaries from ``{"enter": [n_gt, n_tracked, n_matched, errs], "exit": ...}``."""
    out = {k: _ee_summary(*v) for k, v in tally.items()}
    n_gt, n_tracked, n_matched = (sum(row[i] for row in tally.values()) for i in range(3))
    out["all"] = _ee_summary(n_gt, n_tracked, n_matched, tally["enter"][3] + tally["exit"][3])
    return out


def enter_exit_eval(an, acts: list[Activity], id_map: dict[int, int], fps: float, window_s: float = ENTER_EXIT_WINDOW_S,
                    t_end: float | None = None, edge_s: float = EDGE_S, persons: set[int] | None = None,
                    clusters: dict[int, int] | None = None) -> dict:
    """Recall and timing of our presence intervals against the enter / exit-through-structure events.

    The ground-truth time of an enter is the start of the actor's span in the activity (``f0`` for a lone actor), of an
    exit its end (``f1``), in seconds. Two timings of ours are scored against it, both over the presence intervals
    (``an.timelines[pid]["present"]``) of every person mapped to that actor (so fragments of one person are pooled):

    * nearest boundary (the main ``enter`` / ``exit`` / ``all`` blocks): the start (enter) or end (exit) of the presence
      interval nearest to the ground-truth time, so a person who leaves and re-enters is handled;
    * first / last seen (the same blocks under ``"first_last"``): the earliest start (enter) or latest end (exit) of any
      of them, the brief's definition.

    A match is within ``window_s`` seconds (inclusive). Events in the first or last ``edge_s`` seconds of the clip
    (``t_end`` = its processed length) are ignored, and any event after ``t_end``. With ``clusters`` (twin-aware mode,
    ``twin_clusters``) an actor counts as tracked when a person maps to any actor of its cluster.

    Each block has n_gt, n_gt_tracked (the actor maps from one of our persons), n_matched, recall_all, recall_tracked,
    median_abs_err_s, median_bias_s (signed; ours minus GT) and errs_s.
    """
    canon = (lambda x: x) if clusters is None else (lambda x: clusters.get(x, x))
    pids_of: dict[int, list[int]] = defaultdict(list)
    for pid, actor in id_map.items():
        pids_of[canon(actor)].append(pid)
    # per timing, per kind: n_gt, n_tracked, n_matched, signed errors
    tally = {mode: {"enter": [0, 0, 0, []], "exit": [0, 0, 0, []]} for mode in ("nearest", "first_last")}
    for act in acts:
        kind = {GT_ENTER: "enter", GT_EXIT: "exit"}.get(act.name)
        if kind is None:
            continue
        for actor in dict.fromkeys(act.actors):
            if persons is not None and actor not in persons:
                continue
            span = act.actor_spans.get(actor, (act.f0, act.f1))
            t_gt = (span[0] if kind == "enter" else span[1]) / fps
            if t_end is not None and (t_gt < edge_s or t_gt > t_end - edge_s):
                continue
            times = [iv[0 if kind == "enter" else 1] for pid in pids_of.get(canon(actor), ())
                     for iv in an.timelines.get(pid, {}).get("present", ())]
            ours = {"nearest": min(times, key=lambda t: abs(t - t_gt)) if times else None,
                    "first_last": (min(times) if kind == "enter" else max(times)) if times else None}
            for mode, t_ours in ours.items():
                row = tally[mode][kind]
                row[0] += 1
                if t_ours is None:
                    continue
                row[1] += 1
                if abs(t_ours - t_gt) <= window_s + 1e-9:
                    row[2] += 1
                    row[3].append(t_ours - t_gt)
    return {**_ee_blocks(tally["nearest"]), "first_last": _ee_blocks(tally["first_last"])}


# ---------------------------------------------------------------------------------------------------- aggregation

def _sum_tracking(parts: list[dict]) -> dict:
    counts = {k: sum(int(p.get(k, 0)) for p in parts) for k in _MOT_COUNTS}
    return {**_mot_summary(counts), "num_frames": sum(int(p.get("num_frames", 0)) for p in parts), **counts}


def _sum_talk(parts: list[dict]) -> dict:
    def pool(sub):
        picked = [(p.get(sub) if sub else p) or {} for p in parts]
        counts = {k: sum(int(p.get(k, 0)) for p in picked) for k in _TALK_COUNTS}
        return _talk_summary(counts, [e for p in picked for e in p.get("start_errs_s", ())])

    out = pool(None)
    out["tiou"] = next((p["tiou"] for p in parts if "tiou" in p), TALK_TIOU)
    for key in sorted({k for p in parts for k in p if k.startswith("tiou_")}):
        out[key] = {**pool(key), "tiou": next(p[key]["tiou"] for p in parts if key in p)}
    return out


def _sum_ee(parts: list[dict]) -> dict:
    n = lambda k: sum(int(p.get(k, 0)) for p in parts)  # noqa: E731
    return _ee_summary(n("n_gt"), n("n_gt_tracked"), n("n_matched"), [e for p in parts for e in p.get("errs_s", ())])


def _sum_ee_blocks(parts: list[dict]) -> dict:
    keys = ("enter", "exit", "all")
    out = {k: _sum_ee([p.get(k) or {} for p in parts]) for k in keys}
    out["first_last"] = {k: _sum_ee([(p.get("first_last") or {}).get(k) or {} for p in parts]) for k in keys}
    return out


def aggregate(clips: list[dict]) -> dict:
    """Pool the per-clip results of ``eval_clip``: counts are summed and ratios recomputed (micro-average); median
    errors are taken over the pooled matches. Clips that failed (``{"clip", "error"}``) are left out and counted in
    ``n_failed``. The ``strict`` and ``twin_aware`` blocks are pooled separately."""
    ok = [c for c in clips if "error" not in c]

    def per(variant, key):
        return [(c.get(variant) or {}).get(key) or {} for c in ok]

    boxes = sum(int(c.get("n_gt_boxes", 0)) for c in ok)
    twins = sum(int(c.get("n_gt_dup_boxes", 0)) for c in ok)
    return {
        "n_clips": len(ok), "n_failed": len(clips) - len(ok),
        "n_gt_boxes": boxes, "n_gt_dup_boxes": twins, "gt_dup_share": _ratio(twins, boxes),
        "tracks": {"tracking": _sum_tracking(per("tracks", "tracking"))},
        "persons": {"tracking": _sum_tracking(per("persons", "tracking"))},
        **{variant: {"talk": _sum_talk(per(variant, "talk")),
                     "enter_exit": _sum_ee_blocks(per(variant, "enter_exit"))} for variant in VARIANTS},
    }


# ------------------------------------------------------------------------------------------------------- oracle

def oracle_result(res, gt_boxes: dict[int, list[GeomBox]], clusters: dict[int, int] | None = None):
    """A copy of the perception result ``res`` whose observations are the ground-truth person boxes (perfect detection
    and tracking), for the oracle ablation.

    The sampled frames (``frame_times``, ``src_frame_idx``) and the video info are those of ``res``. On every sampled frame
    there is one observation per ground-truth person: the box of the source frame it matches, scaled to processed pixels,
    score 1.0, keypoints with confidence 0 (geometry then falls back to the box height and flags the distance as
    approximate). The track id is the actor id, or with ``clusters`` (``twin_clusters``) the representative of the
    actor's twin cluster, so the several annotations of one physical person become one track; where several members of
    a cluster are annotated in a frame, the box of the lowest actor id is kept. There are no embeddings or outfit
    descriptors, so the identity merge leaves every track a person of its own. Observations are ordered by frame, then
    track id. ``res`` is not modified.
    """
    clusters = clusters or {}
    scale = float(res.video.scale)
    frames: list[int] = []
    tracks: list[int] = []
    boxes: list[tuple] = []
    for f in range(len(res.frame_times)):
        per_track: dict[int, tuple] = {}
        for b in sorted(gt_boxes.get(int(res.src_frame_idx[f]), ()), key=lambda b: b.actor):
            per_track.setdefault(clusters.get(b.actor, b.actor), b.bbox)
        for track in sorted(per_track):
            frames.append(f)
            tracks.append(int(track))
            boxes.append(per_track[track])
    out = copy.copy(res)
    out.obs_frame = np.asarray(frames, np.int64)
    out.obs_track = np.asarray(tracks, np.int64)
    out.obs_bbox = np.asarray(boxes, float).reshape(-1, 4) * scale
    out.obs_score = np.ones(len(frames))
    out.obs_kpts = np.zeros((len(frames), 17, 3), np.float32)
    out.raw_frame, out.raw_bbox = out.obs_frame.copy(), out.obs_bbox.copy()
    out.track_emb, out.track_outfit = {}, {}
    out.crops_dir, out.cars, out.timings = None, [], {}
    out.stack_name = "oracle"
    return out


def oracle_id_map(an) -> dict[int, int]:
    """Person id -> actor for an analysis of ``oracle_result`` tracks: a person is its track, and the track id is the
    actor id (or the representative of a twin cluster)."""
    return {p.pid: p.track_ids[0] for p in an.persons}


# ---------------------------------------------------------------------------------------------------- one clip

def eval_clip(video, ann_prefix, stack, cache_root, preset: str = "short", dist: float = 2.0, min_dur: float = 8.0,
              gap_tol: float = 1.5, merge_thr: float = 0.30, limit_seconds: float | None = None,
              oracle: bool = False) -> dict:
    """Run perception (cached), the offline analysis and every metric for one clip.

    ``ann_prefix`` is the annotation path without ``.geom.yml`` / ``.activities.yml`` / ``.types.yml``. With
    ``limit_seconds`` only the first N seconds are processed, and the ground truth is cut to match. Tracking metrics
    are computed for the raw tracks (``tracks``) and the merged persons (``persons``). Engagement and enter/exit are
    scored for the persons, twice: ``strict`` (each of our ids maps to one actor by the first version's share rule, no
    twins) and ``twin_aware`` (the actor-share rule of ``match_ids`` and twin actors credited as one person).

    With ``oracle`` the cached perception only supplies the sampled frames; the observations are the ground-truth boxes
    (``oracle_result``), the analysis and scoring are unchanged, and every track maps to its own actor.
    ``merge_thr`` then has no effect (there are no embeddings). The output says ``"mode": "oracle"``.
    """
    started = time.perf_counter()
    prefix = str(ann_prefix)
    persons_gt = person_ids(load_types(prefix + ".types.yml"))
    gt_boxes = load_geom(prefix + ".geom.yml", persons_gt)
    acts = load_activities(prefix + ".activities.yml")
    res, cached = cached_perception(video, stack, cache_root,
                                    **({} if limit_seconds is None else {"max_seconds": float(limit_seconds)}))
    clusters = twin_clusters(res, gt_boxes)
    if oracle:
        res = oracle_result(res, gt_boxes, clusters)
    an = analysis.analyze(res, preset=preset, merge_thr=merge_thr)
    fps, t_end = float(res.video.src_fps), float(res.video.duration)

    if oracle:      # the tracks are the actors: every person maps to its track id, which is its actor id
        strict_map = twin_map = oracle_id_map(an)
        track_map = {int(t): int(t) for t in res.track_ids()}
    else:
        strict_map = match_ids(res, an, gt_boxes, True, min_actor_share=None)
        twin_map = match_ids(res, an, gt_boxes, True)
        track_map = match_ids(res, None, gt_boxes, False, min_actor_share=None)
    intervals = pair_intervals(an, res, dist, min_dur, gap_tol)
    talks = [a for a in acts if a.name == GT_TALK]
    annotated = {b.actor for f in res.src_frame_idx for b in gt_boxes.get(int(f), ())}
    dup_counts = gt_duplicate_counts(res, gt_boxes)

    def pairs(**kw):
        return [[g.a, g.b, g.t0, g.t1] for g in gt_talk_pairs(talks, fps, t_end, persons_gt, **kw)]

    return {
        "clip": Path(prefix).name, "mode": "oracle" if oracle else "system", "video": str(video), "duration_s": t_end, "src_fps": fps,
        "limit_s": limit_seconds, "cached": bool(cached), "n_sampled_frames": int(len(res.frame_times)),
        "n_gt_persons": len(annotated), "n_tracks": len(res.track_ids()), "n_persons": len(an.persons),
        "n_gt_boxes": dup_counts[1], "n_gt_dup_boxes": dup_counts[0], "gt_dup_share": _ratio(*dup_counts),
        "timings": {"perception_s": float(res.timings.get("total", 0.0)),
                    "eval_s": time.perf_counter() - started},
        "tracks": {"id_map": track_map, "tracking": tracking_metrics(res, None, gt_boxes, False)},
        "persons": {"tracking": tracking_metrics(res, an, gt_boxes, True),
                    "intervals": [[a, b, t0, t1] for a, b, t0, t1 in intervals]},
        "strict": {
            "id_map": strict_map, "gt_pairs": pairs(),
            "talk": talk_eval(intervals, talks, strict_map, fps, t_max=t_end, persons=persons_gt),
            "enter_exit": enter_exit_eval(an, acts, strict_map, fps, t_end=t_end, persons=persons_gt),
        },
        "twin_aware": {
            "id_map": twin_map, "clusters": clusters, "n_twin_actors": len(clusters),
            "n_twin_clusters": len(set(clusters.values())), "gt_pairs": pairs(clusters=clusters),
            "talk": talk_eval(intervals, talks, twin_map, fps, t_max=t_end, persons=persons_gt, clusters=clusters),
            "enter_exit": enter_exit_eval(an, acts, twin_map, fps, t_end=t_end, persons=persons_gt,
                                          clusters=clusters),
        },
    }


# --------------------------------------------------------------------------------------------------- command line

def read_clips(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_split(path: Path) -> dict:
    """The split file: a JSON object whose ``dev`` and ``test`` are lists of clip names. ValueError otherwise."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"split file {path} cannot be read as JSON: {exc}") from exc
    if not (isinstance(data, dict) and all(isinstance(data.get(k), list) and all(isinstance(n, str) for n in data[k])
                                           for k in ("dev", "test"))):
        raise ValueError(f"split file {path} must be a JSON object with 'dev' and 'test' lists of clip names")
    return data


def select_clips(rows: list[dict], which: str, split_file) -> list[dict]:
    """The rows of one split. ``split_file`` is JSON ``{"dev": [clip, ...], "test": [clip, ...]}`` (anything else is a
    ValueError). ``dev`` and ``test`` need it; ``all`` is every clip listed in it (dev and test), or every row when there
    is no split file."""
    path = Path(split_file) if split_file else None
    if path is None or not path.is_file():
        if which != "all":
            raise ValueError(f"--split {which} needs a split file ({path or 'none given'} not found)")
        return list(rows)
    split = _load_split(path)
    names = split["dev"] + split["test"] if which == "all" else split[which]
    known = {r["clip"] for r in rows}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"split file {path} lists clips that are not in the clips table: {unknown}")
    wanted = set(names)
    return [r for r in rows if r["clip"] in wanted]


def _default_stack():
    from ..perception.stack import default_stack
    return default_stack(config.DEFAULT_DETECTOR)


def _files_exist(*paths) -> bool:
    return all(Path(p).is_file() for p in paths)


def _annotation_prefix(row: dict, clips_path: Path) -> Path:
    """The annotation prefix of a clip: next to the clips table first, then the CSV's own path (as given, then from the
    repository root). The first candidate whose geom file exists wins; otherwise the first one."""
    clip = row["clip"]
    candidates = [clips_path.parent / "annotations" / clip]
    given = row.get("annotation_path")
    if given:
        candidates += [Path(given), config.repo_root() / given]
    return next((c for c in candidates if Path(str(c) + ".geom.yml").is_file()), candidates[0])


def _pct(x) -> str:
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def _num(x, spec=".2f", unit="") -> str:
    return "n/a" if x is None else f"{x:{spec}}{unit}"


def _label(clip: str) -> str:
    parts = clip.split(".")
    return ".".join([parts[0], parts[1], parts[-1]]) if len(parts) >= 4 else clip


def _talk_cols(t: dict) -> list[str]:
    return [f"{t.get('n_gt', 0):>5}", _pct(t.get("recall_all")), _pct(t.get("recall_tracked")),
            _pct(t.get("precision_mapped")), f"{t.get('n_det_mapped', 0)}/{t.get('n_det', 0)}".rjust(7),
            f"{_num(t.get('median_start_err_s'), '.1f', 's'):>8}"]


def _ee_cols(e: dict) -> list[str]:
    a, fl = e.get("all") or {}, (e.get("first_last") or {}).get("all") or {}
    return [f"{a.get('n_gt', 0):>4}", _pct(a.get("recall_all")), _pct(a.get("recall_tracked")),
            f"{_num(a.get('median_abs_err_s'), '.1f', 's'):>6}", _pct(fl.get("recall_all")),
            f"{_num(fl.get('median_abs_err_s'), '.1f', 's'):>8}"]


def _trk_cols(t: dict) -> list[str]:
    return [_pct(t.get("mota")), _pct(t.get("idf1")), f"{t.get('num_switches', 0):>5}"]


VARIANT_TITLES = {
    "strict": "STRICT: an id maps to one actor if it is on it for >= 3 and >= 30% of the id's frames; twin actors are separate",
    "twin_aware": "TWIN-AWARE: also maps on >= 50% of a briefly annotated actor's frames; twin actors count as one person",
}


def format_tables(clips: list[dict], total: dict, params: dict | None = None) -> str:
    """The per-clip and total tables as text: tracking, then engagement and enter/exit for each variant."""
    w = max([len("clip")] + [len(_label(c["clip"])) for c in clips])
    ok = [(_label(c["clip"]), c) for c in clips if "error" not in c]
    failed = [(_label(c["clip"]), c) for c in clips if "error" in c]
    rows = ok + [("TOTAL", total)]
    lines = [""]
    if params:
        lines.append("settings: " + " ".join(f"{k}={v}" for k, v in params.items()))
        if params.get("oracle"):
            lines.append("ORACLE MODE: ground-truth boxes instead of detections (perfect detection and tracking); "
                         "an ablation of the geometry and the engagement rule, not the system")
    lines += ["Tracking (sampled frames; MEVA annotates only some people, so precision/MOTA are pessimistic;"
              " dup = GT boxes with a twin box, which caps recall)",
              f"{'clip':<{w}}  {'MOTA trk':>8} {'IDF1 trk':>8} {'IDSW':>5}  {'MOTA per':>8} {'IDF1 per':>8} {'IDSW':>5}"
              f"  {'prec':>6} {'recall':>6} {'dup':>6}"]
    for name, c in rows:
        p = c["persons"]["tracking"]
        lines.append(f"{name:<{w}}  " + " ".join(_trk_cols(c["tracks"]["tracking"])) + "  " + " ".join(_trk_cols(p))
                     + f"  {_pct(p.get('precision'))} {_pct(p.get('recall'))} {_pct(c.get('gt_dup_share'))}")
    lines += [f"{name:<{w}}  ERROR: {c['error']}" for name, c in failed]
    for variant in VARIANTS:
        lines += ["", f"Engagement vs person_talks_to_person (roles ignored, tIoU >= 0.3) and enter/exit within 5 s, "
                      f"{VARIANT_TITLES[variant]}",
                  f"{'clip':<{w}}  {'talks':>5} {'R_all':>6} {'R_trk':>6} {'P_map':>6} {'map/det':>7} {'startErr':>8}"
                  f"  {'ev':>4} {'R_all':>6} {'R_trk':>6} {'|err|':>6}  {'fl_R_all':>8} {'fl_|err|':>8}"]
        for name, c in rows:
            v = c[variant]
            lines.append(f"{name:<{w}}  " + " ".join(_talk_cols(v["talk"])) + "  " + " ".join(_ee_cols(v["enter_exit"])))
        lines += [f"{name:<{w}}  ERROR" for name, _ in failed]
        t = total[variant]["talk"]
        loose = next((v for k, v in t.items() if k.startswith("tiou_") and isinstance(v, dict)), None)
        if loose:
            lines.append(f"  talk at tIoU >= {loose['tiou']:g}: recall_all {_pct(loose['recall_all']).strip()}, "
                         f"recall_tracked {_pct(loose['recall_tracked']).strip()}, precision_mapped "
                         f"{_pct(loose['precision_mapped']).strip()}, median start error "
                         f"{_num(loose['median_start_err_s'], '.1f', 's')}")
        e = total[variant]["enter_exit"]
        for label, block in (("nearest boundary", e), ("first/last seen", e.get("first_last") or {})):
            en, ex = block.get("enter") or {}, block.get("exit") or {}
            lines.append(f"  enter/exit, {label}: enter {_pct(en.get('recall_all')).strip()} of {en.get('n_gt', 0)} "
                         f"(|err| {_num(en.get('median_abs_err_s'), '.1f', 's')}), exit "
                         f"{_pct(ex.get('recall_all')).strip()} of {ex.get('n_gt', 0)} "
                         f"(|err| {_num(ex.get('median_abs_err_s'), '.1f', 's')})")
    twin_actors = sum(int(c.get("twin_aware", {}).get("n_twin_actors", 0)) for _, c in ok)
    collapsed = total["twin_aware"]["talk"].get("n_gt_collapsed", 0)
    lines += ["", f"twin clusters: {twin_actors} actors in clusters over {len(ok)} clip(s); {collapsed} ground-truth "
                  "talk pair interval(s) folded into another by the twin-aware count", ""]
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _write_results(path: Path, payload: dict) -> None:
    """Write the JSON next to its final name first, then move it into place, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m showroom_coach.evaluation.meva_eval",
                                description="Evaluate tracking, engagement and enter/exit against MEVA annotations.")
    p.add_argument("--clips", default="data/meva/clips.csv", help="the clips table (default: %(default)s)")
    p.add_argument("--split", choices=("dev", "test", "all"), default="all",
                   help="dev or test: the clips listed under that key of the split file; all: every clip listed in it "
                        "(or every row of the clips table when there is no split file)")
    p.add_argument("--split-file", default="data/meva/split.json",
                   help='JSON {"dev": [...], "test": [...]} of clip names (default: %(default)s)')
    p.add_argument("--out", default=None, help="results JSON (default: runs/eval/meva_<split>.json)")
    p.add_argument("--preset", default="short", help="threshold preset for the analysis (default: %(default)s)")
    p.add_argument("--dist", type=float, default=2.0, help="engagement distance in metres")
    p.add_argument("--min-dur", type=float, default=8.0, help="shortest engagement in seconds")
    p.add_argument("--gap-tol", type=float, default=1.5, help="gap bridged inside an engagement, seconds")
    p.add_argument("--merge-thr", type=float, default=0.30, help="identity-merge threshold of the analysis")
    p.add_argument("--limit-seconds", type=float, default=None, help="only the first N seconds of every clip")
    p.add_argument("--cache", default=None, help="perception cache directory (default: the app's)")
    p.add_argument("--oracle", action="store_true",
                   help="ablation: use the ground-truth person boxes instead of the detections (perfect detection and "
                        "tracking); writes runs/eval/meva_<split>_oracle.json by default")
    return p


def main(argv=None) -> int:
    """Exit codes: 0 all clips evaluated, 1 a clip failed or nothing could be evaluated, 2 usage error."""
    args = _parser().parse_args(argv)
    clips_path = Path(args.clips)
    suffix = "_oracle" if args.oracle else ""
    out_path = Path(args.out) if args.out else Path("runs") / "eval" / f"meva_{args.split}{suffix}.json"
    try:
        rows = select_clips(read_clips(clips_path), args.split, args.split_file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"error: the {args.split!r} split selects no clips", file=sys.stderr)
        return 2
    try:
        _motmetrics()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    jobs, skipped = [], []
    for row in rows:
        video = clips_path.parent / "videos" / f"{row['clip']}.r13.avi"
        prefix = _annotation_prefix(row, clips_path)
        needed = [video] + [Path(str(prefix) + s) for s in (".geom.yml", ".activities.yml", ".types.yml")]
        if _files_exist(*needed):
            jobs.append((row["clip"], video, prefix))
        else:
            skipped.append(row["clip"])
            print(f"warning: skipping {row['clip']}: video or annotation files missing", file=sys.stderr)
    if not jobs:
        print("error: no clip could be evaluated (files missing)", file=sys.stderr)
        return 1

    params = {"preset": args.preset, "dist": args.dist, "min_dur": args.min_dur, "gap_tol": args.gap_tol,
              "merge_thr": args.merge_thr, "limit_seconds": args.limit_seconds, "oracle": args.oracle}
    notes = NOTES + ([ORACLE_NOTE] if args.oracle else [])
    mode = "oracle" if args.oracle else "system"
    stack = _default_stack()
    cache_root = Path(args.cache) if args.cache else config.cache_dir()
    results: list[dict] = []
    for clip, video, prefix in jobs:
        print(f"evaluating {clip} ...", flush=True)
        try:
            results.append(eval_clip(video, prefix, stack, cache_root, **params))
        except Exception as exc:  # one bad clip must not lose the others; KeyboardInterrupt still stops the run
            message = f"{type(exc).__name__}: {exc}"
            print(f"error: {clip}: {message}", file=sys.stderr)
            results.append({"clip": clip, "error": message})
        _write_results(out_path, {"split": args.split, "mode": mode, "params": params, "notes": notes,
                                  "skipped": skipped, "clips": results, "total": aggregate(results)})
    total = aggregate(results)
    print(format_tables(results, total, params))
    print(f"wrote {out_path}")
    return 1 if total["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
