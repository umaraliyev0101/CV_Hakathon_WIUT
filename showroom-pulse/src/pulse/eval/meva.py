"""MEVA KPF annotation parser (two YAML dialects: flow and quoted python-dict).
Geometry is parsed by regex (fast on 100k-250k rows); activities/types by yaml."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pulse.datatypes import BBox

try:
    _Loader = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    _Loader = yaml.SafeLoader

_ID1 = re.compile(r"""['"]?id1['"]?:\s*(\d+)""")
_TS0 = re.compile(r"""['"]?ts0['"]?:\s*(\d+)""")
_NUM = r"(\d+(?:\.\d+)?)"
_G0 = re.compile(rf"""['"]?g0['"]?:\s*['"]?{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}""")


@dataclass
class MevaActivity:
    act_id: int
    name: str
    frame_start: int
    frame_end: int
    actors: list[int]
    actor_spans: dict[int, tuple[int, int]] = field(default_factory=dict)


@dataclass
class MevaClip:
    clip: str
    tracks: dict[int, dict[int, BBox]]
    track_types: dict[int, str]
    activities: list[MevaActivity]

    def person_tracks(self) -> dict[int, dict[int, BBox]]:
        return {t: b for t, b in self.tracks.items() if self.track_types.get(t) == "person"}


def _parse_geom(path: Path) -> dict[int, dict[int, BBox]]:
    tracks: dict[int, dict[int, BBox]] = {}
    with open(path) as f:
        for line in f:
            if "g0" not in line:
                continue
            a, b, g = _ID1.search(line), _TS0.search(line), _G0.search(line)
            if not (a and b and g):
                continue
            tracks.setdefault(int(a.group(1)), {})[int(b.group(1))] = tuple(
                float(v) for v in g.groups())  # type: ignore[assignment]
    return tracks


def _entries(path: Path, key: str):
    with open(path) as f:
        for line in f:
            if key not in line:
                continue
            try:
                d = yaml.load(line.strip()[2:] if line.lstrip().startswith("- ") else line,
                              Loader=_Loader)
            except yaml.YAMLError:
                continue
            if isinstance(d, dict) and key in d:
                yield d[key]


def _span(timespan) -> tuple[int, int]:
    r = timespan[0]["tsr0"]
    return int(r[0]), int(r[1])


def _parse_types(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for t in _entries(path, "types"):
        cset = t.get("cset3") or {}
        if cset:
            out[int(t["id1"])] = max(cset, key=cset.get)
    return out


def _parse_activities(path: Path) -> list[MevaActivity]:
    acts = []
    for a in _entries(path, "act"):
        act2 = a.get("act2") or {}
        if not act2:
            continue
        name = max(act2, key=act2.get)
        s, e = _span(a["timespan"])
        actors, spans = [], {}
        for ac in a.get("actors", []):
            i = int(ac["id1"])
            actors.append(i)
            spans[i] = _span(ac["timespan"]) if ac.get("timespan") else (s, e)
        acts.append(MevaActivity(int(a["id2"]), name, s, e, actors, spans))
    return acts


def load_kpf(annotation_dir: Path, clip: str) -> MevaClip:
    d = Path(annotation_dir)
    return MevaClip(
        clip=clip,
        tracks=_parse_geom(d / f"{clip}.geom.yml"),
        track_types=_parse_types(d / f"{clip}.types.yml"),
        activities=_parse_activities(d / f"{clip}.activities.yml"),
    )
