"""Person identity from appearance: merge track fragments offline, re-attach lost people live (spec §4.3).

A tracker loses people (occlusion, leaving the frame, ID switches) and the same person comes back under a new
track id. This module turns tracks back into people. Embeddings are body and clothing appearance descriptors
(DINOv2 plus HSV histograms), never faces, and nothing here recognises or stores who anyone is.

Offline, ``merge_tracks`` does greedy constrained average-linkage clustering over whole-track summaries. Time is
a hard constraint: one person cannot be in two places at once, so two clusters whose tracks overlap in time by
more than ``max_overlap`` seconds are never merged, however alike they look. The check is on clusters, not on
the pair being merged, so merging A with B and B with C cannot sneak A and C together.

Live, ``LiveIdentity`` hands out person ids frame by frame. A track keeps its pid; a new track may take over the
pid of someone who just vanished (no active track, seen within ``window`` seconds) if they look alike. Per frame
call ``end_step(t, ids_in_view_now)`` first, then ``resolve`` for each track in that frame (see ``LiveIdentity``).

Pure numpy, no I/O and no clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LiveIdentity", "TrackSummary", "merge_tracks"]

_MIN_NORM = 1e-12


@dataclass
class TrackSummary:
    """One track as the merge sees it: its time span and one appearance descriptor for the whole track."""

    track_id: int
    t0: float
    t1: float
    emb: np.ndarray  # L2-normalised appearance descriptor (normalised again defensively)
    n_obs: int = 1  # how many observations the descriptor averages; weights the cluster mean


def _unit(v: np.ndarray | None) -> np.ndarray | None:
    """Return ``v`` as a 1-D unit float vector, or None if it is missing, non-finite or (near) zero."""
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    if n < _MIN_NORM:
        return None
    return a / n


def merge_tracks(tracks: list[TrackSummary], thr: float = 0.30, max_overlap: float = 0.2) -> dict[int, int]:
    """Group tracks into people. Returns ``track_id -> pid`` with pids 1..N.

    Greedy average-linkage: start with one cluster per track and repeatedly merge the pair of clusters with the
    smallest cosine distance between their mean embeddings (members weighted by ``n_obs``, re-normalised). A
    merge happens only if that distance is below ``thr`` and no track in one cluster overlaps any track in the
    other by more than ``max_overlap`` seconds (overlap = ``min(t1) - max(t0)``). It stops when no allowed pair
    is below ``thr``. Pids are numbered by each person's first appearance (earliest ``t0``, then ``track_id``).

    The pair distances live in one matrix. A merge only recomputes the merged cluster's row and column, so a
    busy clip (hundreds of tracks) costs milliseconds. A missing or zero embedding has distance 1.0 to everything.
    """
    if not tracks:
        return {}
    ordered = sorted(tracks, key=lambda s: (s.t0, s.track_id))
    n = len(ordered)

    units = [_unit(s.emb) for s in ordered]
    dim = next((u.size for u in units if u is not None), 1)
    sums = np.zeros((n, dim))  # per cluster: n_obs-weighted sum of member embeddings
    for i, (s, u) in enumerate(zip(ordered, units)):
        if u is not None:
            sums[i] = max(int(s.n_obs), 1) * u
    means = _unit_rows(sums)

    t0 = np.array([s.t0 for s in ordered], dtype=float)
    t1 = np.array([s.t1 for s in ordered], dtype=float)
    overlap = np.minimum(t1[:, None], t1[None, :]) - np.maximum(t0[:, None], t0[None, :])
    blocked = overlap > max_overlap  # cluster-level cannot-link, kept up to date on every merge

    dist = 1.0 - means @ means.T  # a zero mean gives exactly 1.0
    dist = (dist + dist.T) / 2.0  # exactly symmetric, whatever the BLAS did
    dist[blocked] = np.inf
    np.fill_diagonal(dist, np.inf)

    alive = np.ones(n, dtype=bool)
    members: list[list[int]] = [[i] for i in range(n)]
    while True:
        i, j = divmod(int(np.argmin(dist)), n)
        if not dist[i, j] < thr:  # also stops on inf and NaN
            break
        sums[i] += sums[j]
        members[i].extend(members[j])
        members[j] = []
        alive[j] = False
        blocked[i] |= blocked[j]
        blocked[:, i] = blocked[i]
        means[i] = _unit_rows(sums[i : i + 1])[0]
        row = 1.0 - means @ means[i]
        row[blocked[i] | ~alive] = np.inf
        row[i] = np.inf
        dist[i, :] = row
        dist[:, i] = row
        dist[j, :] = np.inf
        dist[:, j] = np.inf

    # ``ordered`` is sorted by (t0, track_id), so a cluster's first appearance is its smallest member index
    clusters = sorted((m for m in members if m), key=min)
    return {ordered[k].track_id: pid for pid, m in enumerate(clusters, start=1) for k in m}


def _unit_rows(a: np.ndarray) -> np.ndarray:
    """Normalise each row to unit length; rows that are (near) zero stay zero."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, norms, out=np.zeros_like(a), where=norms >= _MIN_NORM)


class LiveIdentity:
    """Frame-by-frame person ids with re-attach of recently lost people.

    Call order, once per frame at time ``t``:

    1. ``end_step(t, ids_in_view_now)`` with the ids of every track in view in this frame, new ones included;
    2. ``resolve(track_id, t, emb)`` for each of those tracks.

    End the step first. A person is *active* while any of their tracks is in the latest ``end_step`` set or has
    been resolved since, so two tracks in view together never share a pid, and someone still in view is never
    taken over. With this order a track that dies in the same frame another is born is already lost when the new
    track resolves, so the common tracker ID switch re-attaches. Resolving first would leave the dead track's
    person looking active and the new track would get a fresh pid that is never repaired.
    """

    def __init__(self, thr: float = 0.30, window: float = 10.0):
        self.thr = thr
        self.window = window
        self._next_pid = 1
        self._pid_of: dict[int, int] = {}  # track id -> pid, kept for the whole session (one small int per track)
        self._emb: dict[int, np.ndarray] = {}  # pid -> unit appearance descriptor, if the person has one
        self._last_seen: dict[int, float] = {}  # pid -> last time any track of theirs was active
        self._active: set[int] = set()  # track ids in view at the last end_step, plus those resolved since

    def resolve(self, track_id: int, t: float, emb: np.ndarray | None) -> int:
        """Return the pid for ``track_id`` at time ``t``. Call it after this frame's ``end_step``.

        A known track keeps its pid. A new track with an embedding takes the pid of the best-matching person who
        has no active track, was last seen within ``window`` seconds, and is closer than ``thr`` (cosine
        distance); otherwise it gets a fresh pid. A new track without an embedding always gets a fresh pid.
        """
        unit = _unit(emb)
        pid = self._pid_of.get(track_id)
        if pid is None:
            pid = self._reattach(t, unit) if unit is not None else None
            if pid is None:
                pid = self._next_pid
                self._next_pid += 1
            self._pid_of[track_id] = pid
        if unit is not None and pid not in self._emb:
            self._emb[pid] = unit  # first descriptor for this person; later ones go through update_emb
        self._last_seen[pid] = t
        self._active.add(track_id)
        return pid

    def end_step(self, t: float, active_track_ids: set[int]) -> None:
        """Start frame ``t``: ``active_track_ids`` are all the tracks in view now, including ones born this frame.

        Call this before ``resolve`` so that people whose tracks died are already lost when this frame's new
        tracks look for someone to re-attach to. People in view are marked as seen at ``t``. People not seen for
        more than ``window`` seconds are forgotten (their descriptors are dropped), which keeps a long session
        bounded; a known track that comes back later keeps its pid and re-learns its descriptor on its next ``resolve``.
        """
        self._active = set(active_track_ids)
        for track_id in self._active:
            pid = self._pid_of.get(track_id)
            if pid is not None:
                self._last_seen[pid] = t
        stale = [pid for pid, seen in self._last_seen.items() if t - seen > self.window]
        for pid in stale:
            del self._last_seen[pid]
            self._emb.pop(pid, None)

    def update_emb(self, pid: int, emb: np.ndarray, alpha: float = 0.2) -> None:
        """Blend a new descriptor into a person's stored one: ``(1 - alpha) * old + alpha * new``, re-normalised.

        An unknown or forgotten pid, or a missing or zero descriptor, is ignored. A person who is known but has
        no descriptor adopts it.
        """
        if pid not in self._last_seen:
            return
        unit = _unit(emb)
        if unit is None:
            return
        old = self._emb.get(pid)
        if old is None:
            self._emb[pid] = unit
            return
        a = min(max(float(alpha), 0.0), 1.0)
        mixed = (1.0 - a) * old + a * unit
        norm = float(np.linalg.norm(mixed))
        if norm >= _MIN_NORM:
            self._emb[pid] = mixed / norm

    def _reattach(self, t: float, unit: np.ndarray) -> int | None:
        """Best lost, recently seen person closer than ``thr`` to ``unit``, or None. Ties go to the lowest pid."""
        active = {self._pid_of[k] for k in self._active if k in self._pid_of}
        best: int | None = None
        best_dist = self.thr
        for pid in sorted(self._emb):
            seen = self._last_seen.get(pid)
            if seen is None or pid in active or t - seen > self.window:
                continue
            d = 1.0 - float(self._emb[pid] @ unit)
            if d < best_dist:
                best, best_dist = pid, d
        return best
