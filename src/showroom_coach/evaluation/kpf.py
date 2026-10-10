"""Parsing of MEVA KPF annotations (``*.geom.yml``, ``*.activities.yml``, ``*.types.yml``).

MEVA (Corona et al., WACV 2021; Kitware/IARPA; CC BY 4.0) ships its annotations as YAML, but every
record sits on one line in Python-dict style, e.g.::

    - {'geom': {'g0': '353 325 428 491', 'id0': 1, 'id1': 654, 'keyframe': True, 'ts0': 1369}}
    - {'types': {'cset3': {'person': 1.0}, 'id1': 654}}
    - {'act': {'act2': {'person_talks_to_person': 1.0}, 'actors': [{'id1': 5, 'timespan': [{'tsr0': [5445, 5964]}]}],
               'id2': 4, 'src_status': 'good', 'timespan': [{'tsr0': [5445, 5964]}]}}

The files are downloaded, so they are treated as untrusted data: nothing is evaluated and no YAML
loader is used. Each line is read with small, bounded regular expressions that do not depend on key
order, and any line that does not match is skipped (headers, comments, truncated or garbled records).

Frame numbers are 0-based indices into the source video; every frame of a track has a box
(``keyframe: False`` boxes are interpolated). Boxes are ``x1 y1 x2 y2`` in source pixels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

# Records are ~100 B (geom) to ~1.5 KB (a 16-actor talk). Anything longer than this is not a record.
_MAX_LINE = 1 << 20

# Digit runs are length-bounded so every number converts to a finite float / small int.
_COORD = r"-?\d{1,9}(?:\.\d{1,9})?"
_INT_END = r"(\d{1,9})\s*[,}]"

_G0 = re.compile(rf"'g0':\s*'\s*({_COORD})\s+({_COORD})\s+({_COORD})\s+({_COORD})\s*'")
_ID1 = re.compile(rf"'id1':\s*{_INT_END}")
_ID2 = re.compile(rf"'id2':\s*{_INT_END}")
_TS0 = re.compile(rf"'ts0':\s*{_INT_END}")
_ACT2 = re.compile(r"'act2':\s*\{\s*'([^']{1,200})'")
_ACTORS_KEY = re.compile(r"'actors':\s*\[")
_TSR0 = re.compile(r"'tsr0':\s*\[\s*(\d{1,9})\s*,\s*(\d{1,9})\s*\]")
# an innermost {'class': score, ...} dict; the outer dicts hold dicts or an id, so they never match
_SCORES = re.compile(rf"\{{\s*((?:'[^']{{1,100}}'\s*:\s*{_COORD}\s*,?\s*)+)\}}")
_SCORE_ITEM = re.compile(rf"'([^']+)'\s*:\s*({_COORD})")
_NOT_A_CLASS = re.compile(r"(?:id|ts)\d+")


@dataclass(slots=True)
class GeomBox:
    actor: int
    frame: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels


@dataclass(slots=True)
class Activity:
    name: str
    act_id: int
    f0: int  # first frame, inclusive
    f1: int  # last frame, inclusive
    actors: list[int]
    # each actor's own (first, last) frame inside the activity; an actor without one is absent
    actor_spans: dict[int, tuple[int, int]] = field(default_factory=dict)


def _lines(path) -> Iterator[str]:
    """Yield the lines of ``path`` that could be records (decoding errors are replaced, not raised)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) <= _MAX_LINE:
                yield line


def _hull(text: str) -> tuple[int, int] | None:
    """Smallest (first, last) frame covering every ``tsr0`` range in ``text``; None if there is none."""
    spans = [(int(a), int(b)) for a, b in _TSR0.findall(text)]
    spans = [s for s in spans if s[0] <= s[1]]
    if not spans:
        return None
    return min(s[0] for s in spans), max(s[1] for s in spans)


def load_types(path) -> dict[int, str]:
    """Actor id -> class name as written in the file (``person``, ``other``, ``bag``, ``vehicle``, ...).

    When a record lists several classes the one with the highest score wins (the first on a tie).
    If an actor appears more than once the last record wins.
    """
    types: dict[int, str] = {}
    for line in _lines(path):
        m_id = _ID1.search(line)
        if not m_id or "'types'" not in line:
            continue
        best_name, best_score = None, None
        for block in _SCORES.findall(line):
            for name, score in _SCORE_ITEM.findall(block):
                if _NOT_A_CLASS.fullmatch(name):
                    continue
                if best_score is None or float(score) > best_score:
                    best_name, best_score = name.strip(), float(score)
        if best_name:
            types[int(m_id.group(1))] = best_name
    return types


def person_ids(types: dict[int, str]) -> set[int]:
    """Ids whose class is ``person`` (case-insensitive)."""
    return {actor for actor, name in types.items() if name.strip().lower() == "person"}


def load_geom(path, actors: set[int] | None = None) -> dict[int, list[GeomBox]]:
    """Frame -> boxes of every actor on that frame, frames in ascending order.

    ``actors`` limits the result to those ids (``None`` keeps all; an empty set keeps nothing). Files are
    4-30 MB, so the filter is applied before any box is built.
    """
    wanted = None if actors is None else {int(a) for a in actors}
    by_frame: dict[int, list[GeomBox]] = {}
    for line in _lines(path):
        m_id = _ID1.search(line)
        if not m_id:
            continue
        actor = int(m_id.group(1))
        if wanted is not None and actor not in wanted:
            continue
        m_ts = _TS0.search(line)
        m_box = _G0.search(line)
        if not m_ts or not m_box:
            continue
        frame = int(m_ts.group(1))
        x1, y1, x2, y2 = (float(v) for v in m_box.groups())
        by_frame.setdefault(frame, []).append(GeomBox(actor=actor, frame=frame, bbox=(x1, y1, x2, y2)))
    return {f: by_frame[f] for f in sorted(by_frame)}


def _split_actors(line: str) -> tuple[str, str]:
    """Return (text of the ``actors`` list, the line without it). Brackets are balanced, quotes ignored."""
    m = _ACTORS_KEY.search(line)
    if not m:
        return "", line
    start, depth = m.end(), 1
    for i in range(start, len(line)):
        ch = line[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return line[start:i], line[: m.start()] + line[i + 1:]
    return "", ""  # unbalanced: a truncated record


def _actor_chunks(text: str) -> Iterator[str]:
    """The top-level ``{...}`` items of an actors list."""
    depth, begin = 0, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[begin:i + 1]


def load_activities(path) -> list[Activity]:
    """Activities in file order. Records missing a name, an id or an activity-level span are skipped.

    ``f0``/``f1`` come from the activity-level ``tsr0`` (inclusive). Should a record list several
    ranges, the hull is used. Each actor's own range goes to ``actor_spans``.
    """
    out: list[Activity] = []
    for line in _lines(path):
        if "'act'" not in line:
            continue
        actors_text, rest = _split_actors(line)
        m_name, m_id = _ACT2.search(rest), _ID2.search(rest)
        span = _hull(rest)
        if not (m_name and m_id and span):
            continue
        actors: list[int] = []
        actor_spans: dict[int, tuple[int, int]] = {}
        for chunk in _actor_chunks(actors_text):
            m_actor = _ID1.search(chunk)
            if not m_actor:
                continue
            actor = int(m_actor.group(1))
            if actor not in actors:
                actors.append(actor)
            actor_span = _hull(chunk)
            if actor_span:
                old = actor_spans.get(actor)
                actor_spans[actor] = actor_span if old is None else (min(old[0], actor_span[0]), max(old[1], actor_span[1]))
        out.append(Activity(name=m_name.group(1), act_id=int(m_id.group(1)), f0=span[0], f1=span[1],
                            actors=actors, actor_spans=actor_spans))
    return out
