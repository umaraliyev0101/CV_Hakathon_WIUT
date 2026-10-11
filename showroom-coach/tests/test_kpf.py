"""MEVA KPF annotation parsing (geom / activities / types).

The mini files in tests/data are excerpts adapted from MEVA (CC BY 4.0, Kitware/IARPA) with a few
malformed lines added on purpose: the parser must skip them rather than crash.
"""
import time
import tracemalloc
from pathlib import Path

import pytest

from showroom_coach.evaluation import kpf
from showroom_coach.evaluation.kpf import (
    Activity,
    GeomBox,
    load_activities,
    load_geom,
    load_types,
    person_ids,
)

DATA = Path(__file__).resolve().parent / "data"
MINI_GEOM = DATA / "mini.geom.yml"
MINI_ACTS = DATA / "mini.activities.yml"
MINI_TYPES = DATA / "mini.types.yml"

REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "meva" / "annotations"
REAL_STEM = "2018-03-13.15-50-00.15-55-00.school.G421"


# --------------------------------------------------------------------------- types

def test_load_types_mini():
    types = load_types(MINI_TYPES)
    assert types == {8: "other", 43: "person", 66: "person", 106: "person", 107: "person",
                     108: "person", 70: "vehicle"}


def test_load_types_picks_highest_score():
    # actor 70 is listed as {'person': 0.4, 'vehicle': 0.6} with the id1 key first
    assert load_types(MINI_TYPES)[70] == "vehicle"


def test_person_ids_is_case_insensitive():
    types = {1: "person", 2: "Person", 3: "PERSON", 4: "other", 5: "vehicle", 6: "bag", 7: " person "}
    assert person_ids(types) == {1, 2, 3, 7}
    assert person_ids({}) == set()


def test_person_ids_mini():
    assert person_ids(load_types(MINI_TYPES)) == {43, 66, 106, 107, 108}


# --------------------------------------------------------------------------- geom

def test_load_geom_mini_counts():
    geom = load_geom(MINI_GEOM)
    assert sorted(geom) == [273, 274, 1577, 1578, 3736, 3737, 6745, 6746, 6809, 6810]
    assert sum(len(boxes) for boxes in geom.values()) == 15
    assert {len(geom[f]) for f in (6745, 6746)} == {2}
    assert len(geom[6809]) == 3 and len(geom[6810]) == 2


def test_load_geom_mini_exact_box():
    boxes = {b.actor: b for b in load_geom(MINI_GEOM)[6809]}
    assert boxes[108] == GeomBox(actor=108, frame=6809, bbox=(1407.0, 262.0, 1550.0, 671.0))
    assert boxes[106].bbox == (1460.0, 254.0, 1578.0, 580.0)
    assert all(isinstance(v, float) for v in boxes[108].bbox)


def test_load_geom_ignores_key_order():
    # the 107@6810 record lists ts0, keyframe, id1, id0, g0 (reverse of the usual order)
    boxes = {b.actor: b for b in load_geom(MINI_GEOM)[6810]}
    assert boxes[107] == GeomBox(actor=107, frame=6810, bbox=(1222.0, 213.0, 1333.0, 485.0))


def test_load_geom_skips_malformed_lines():
    geom = load_geom(MINI_GEOM)
    # 106@6810 has a non-numeric box; the truncated and free-text lines carry no frame at all
    assert 106 not in {b.actor for b in geom[6810]}
    assert all(f >= 0 for f in geom)


def test_load_geom_actor_filter():
    only = load_geom(MINI_GEOM, actors={108})
    assert sorted(only) == [6809, 6810]
    assert all(b.actor == 108 for boxes in only.values() for b in boxes)
    two = load_geom(MINI_GEOM, actors={8, 43})
    assert sorted(two) == [273, 274, 1577, 1578]


def test_load_geom_empty_actor_set_is_not_none():
    assert load_geom(MINI_GEOM, actors=set()) == {}
    assert load_geom(MINI_GEOM, actors={999}) == {}


def test_load_geom_returns_plain_dict_with_sorted_frames():
    geom = load_geom(MINI_GEOM)
    assert type(geom) is dict
    assert list(geom) == sorted(geom)


# --------------------------------------------------------------------------- activities

def test_load_activities_mini_counts():
    acts = load_activities(MINI_ACTS)
    assert [a.act_id for a in acts] == [19, 28, 64, 7, 99]
    assert [a.name for a in acts] == [
        "person_exits_scene_through_structure",
        "person_enters_scene_through_structure",
        "person_talks_to_person",
        "person_picks_up_object",
        "person_opens_facility_door",
    ]


def test_load_activities_three_actor_talk_exact():
    talk = next(a for a in load_activities(MINI_ACTS) if a.name == "person_talks_to_person")
    assert talk == Activity(
        name="person_talks_to_person",
        act_id=64,
        f0=6745,
        f1=9000,
        actors=[108, 106, 107],
        actor_spans={108: (6809, 9000), 106: (6745, 9000), 107: (6745, 9000)},
    )


def test_load_activities_enter_and_exit():
    acts = {a.act_id: a for a in load_activities(MINI_ACTS)}
    assert (acts[28].f0, acts[28].f1, acts[28].actors) == (3736, 3799, [66])
    assert (acts[19].f0, acts[19].f1, acts[19].actors) == (1577, 1654, [43])
    assert acts[28].actor_spans == {66: (3736, 3799)}


def test_load_activities_ignores_key_order():
    # id2 first, timespan before actors, and the actor lists timespan before id1
    shuffled = next(a for a in load_activities(MINI_ACTS) if a.act_id == 99)
    assert shuffled.name == "person_opens_facility_door"
    assert (shuffled.f0, shuffled.f1) == (100, 120)  # activity-level span, not the actor's 101..119
    assert shuffled.actors == [66]
    assert shuffled.actor_spans == {66: (101, 119)}


def test_activity_actor_spans_defaults_to_empty():
    a = Activity(name="x", act_id=1, f0=0, f1=1, actors=[3])
    assert a.actor_spans == {}
    b = Activity(name="x", act_id=2, f0=0, f1=1, actors=[3])
    assert a.actor_spans is not b.actor_spans  # not a shared default


def test_load_activities_skips_garbage(tmp_path):
    path = tmp_path / "bad.activities.yml"
    path.write_text(
        "- { meta: header }\n"
        "garbage\n"
        "- {'act': {'act2': {'a': 1.0}, 'actors': [], 'id2': 1, 'timespan': [{'tsr0': [3, 9]}]}}\n"
        "- {'act': {'act2': {'b': 1.0}, 'actors': [{'id1': 4}], 'id2': 2}}\n"           # no activity span
        "- {'act': {'actors': [{'id1': 4}], 'id2': 3, 'timespan': [{'tsr0': [3, 9]}]}}\n"  # no name
        "- {'act': {'act2': {'c': 1.0}, 'actors': [{'id1': 4}], 'timespan': [{'tsr0': [3, 9]}]}}\n"  # no id
        "- {'act': {'act2': {'d': 1.0}, 'actors': [{'id1': 4, 'timespan': [{'tsr0': [5, 6]}]}, {'id1': 7}],"
        " 'id2': 5, 'timespan': [{'tsr0': [3, 9]}]}}\n",
        encoding="utf-8",
    )
    acts = load_activities(path)
    assert [a.act_id for a in acts] == [1, 5]
    assert acts[0].actors == [] and acts[0].actor_spans == {}
    assert acts[1].actors == [4, 7]
    assert acts[1].actor_spans == {4: (5, 6)}  # actor 7 has no span of its own


def test_loaders_never_evaluate_content(tmp_path):
    """Each payload is a record the loader parses successfully, so its regexes really run on the code-like text."""
    marker = tmp_path / "pwned"
    code = f"__import__('pathlib').Path('{marker.as_posix()}').write_text('x')"
    cases = [
        (
            "e.geom.yml", load_geom,
            f"- {{'geom': {{'g0': '1 2 3 4', 'id0': {code}, 'id1': 7, 'keyframe': True, 'ts0': 5}}}}\n",
            lambda got: [(f, b.actor, b.bbox) for f, bs in got.items() for b in bs] == [(5, 7, (1.0, 2.0, 3.0, 4.0))],
        ),
        (
            "e.types.yml", load_types,
            f"- {{'types': {{'cset3': {{'person': 1.0}}, 'id1': 7, 'x': {code}}}}}\n",
            lambda got: got == {7: "person"},
        ),
        (
            "e.activities.yml", load_activities,
            f"- {{'act': {{'act2': {{'person_talks_to_person': 1.0}}, 'actors': [{{'id1': 7, 'timespan': [{{'tsr0': [1, 2]}}]}}],"
            f" 'id2': 3, 'timespan': [{{'tsr0': [1, 2]}}], 'x': {code}}}}}\n",
            lambda got: [(a.name, a.act_id, a.f0, a.f1, a.actors) for a in got] == [("person_talks_to_person", 3, 1, 2, [7])],
        ),
    ]
    for name, loader, line, parsed_as_expected in cases:
        path = tmp_path / name
        path.write_bytes(line.encode("utf-8"))
        assert parsed_as_expected(loader(path)), name
    assert not marker.exists()


def test_loaders_handle_empty_and_binary_files(tmp_path):
    empty = tmp_path / "empty.yml"
    empty.write_text("", encoding="utf-8")
    assert load_geom(empty) == {} and load_types(empty) == {} and load_activities(empty) == []
    binary = tmp_path / "bin.yml"
    binary.write_bytes(b"\xff\xfe\x00\x01- {'geom': \xc3\x28 \n" * 3)
    assert load_geom(binary) == {} and load_types(binary) == {} and load_activities(binary) == []


# --------------------------------------------------------------------------- hostile input: time and memory

def _pad_geom(total_chars: int, frame: int = 2) -> str:
    """A valid geom record padded with spaces to exactly ``total_chars`` characters, newline included."""
    head, tail = f"- {{'geom': {{'g0': '1 2 3 4', 'id1': 5, 'ts0': {frame}", "}}\n"
    return head + " " * (total_chars - len(head) - len(tail)) + tail


def _spaces_geom(n: int) -> str:
    return f"- {{'geom': {{'g0': '1 2 3 4{' ' * n}x', 'id1': 1{' ' * n}x, 'ts0': 1{' ' * n}x}}}}\n"


def _spaces_types(n: int) -> str:
    return f"- {{'types': {{'id1': 1, 'cset3': {{'a': 1{' ' * n}x}}}}}}\n"


def _spaces_activities(n: int) -> str:
    return (f"- {{'act': {{'act2': {{'a': 1}}, 'actors': [{{'id1': 1{' ' * n}x}}], 'id2': 1{' ' * n}x,"
            f" 'timespan': [{{'tsr0': [1, 2{' ' * n}x]}}]}}}}\n")


def test_max_line_is_small():
    assert kpf._MAX_LINE <= 64 * 1024  # the longest real record is under 1 KB


def test_overlong_lines_are_skipped_and_reading_resumes(tmp_path):
    cap = kpf._MAX_LINE
    path = tmp_path / "long.geom.yml"
    path.write_bytes((
        _pad_geom(100, frame=2)
        + _pad_geom(cap + 1, frame=3)            # one char over the cap: dropped
        + _pad_geom(100, frame=4)
        + _pad_geom(cap, frame=5)                # exactly the cap: kept
        + _pad_geom(3 * cap + 17, frame=6)       # several reads long: dropped
        + _pad_geom(100, frame=7)
        + _pad_geom(5 * cap, frame=8).rstrip("\n")  # over-long and no trailing newline: dropped
    ).encode("ascii"))
    assert list(load_geom(path)) == [2, 4, 5, 7]


def test_overlong_line_is_not_held_in_memory(tmp_path):
    path = tmp_path / "huge.geom.yml"
    with open(path, "wb") as fh:
        fh.write(_pad_geom(100, frame=2).encode("ascii"))
        fh.write(b"- {'geom': " + b"x" * (6 * 1024 * 1024) + b"}\n")
        fh.write(_pad_geom(100, frame=3).encode("ascii"))
    tracemalloc.start()
    try:
        kept = list(kpf._lines(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(kept) == 2
    assert peak < 1024 * 1024, f"peak {peak} B while skipping a 6 MB line"


@pytest.mark.parametrize("n", [20_000, 100_000])  # three 20k runs fit under the line cap, 100k does not
@pytest.mark.parametrize("loader, build, empty", [
    (load_geom, _spaces_geom, {}),
    (load_types, _spaces_types, {}),
    (load_activities, _spaces_activities, []),
])
def test_long_whitespace_runs_are_linear_time(tmp_path, loader, build, empty, n):
    path = tmp_path / "ws.yml"
    path.write_bytes(build(n).encode("ascii"))
    t0 = time.perf_counter()
    result = loader(path)
    elapsed = time.perf_counter() - t0
    assert result == empty
    assert elapsed < 1.0, f"{loader.__name__} took {elapsed:.2f}s on a {path.stat().st_size} B line"


def test_every_regex_is_linear_on_a_million_spaces():
    """ReDoS guard: each pattern gets its own worst-case whitespace run, far beyond any allowed line."""
    ws = " " * 1_000_000
    cases = [
        (kpf._SCORES.findall, "{'types': {'id1': 1, 'cset3': {'a': 1" + ws + "x}}}"),
        (kpf._SCORES.findall, "{'a': 1," + ws + "'b': 2" + ws + "x"),
        (kpf._SCORE_ITEM.findall, "'a'" + ws + ":" + ws + "1" + ws),
        (kpf._G0.search, "'g0': '1 2 3 4" + ws + "x"),
        (kpf._G0.search, "'g0': '1" + ws + "2" + ws + "3" + ws + "x"),
        (kpf._ID1.search, "'id1': 1" + ws + "x"),
        (kpf._ID2.search, "'id2':" + ws + "x"),
        (kpf._TS0.search, "'ts0': 1" + ws + "x"),
        (kpf._ACT2.search, "'act2': {" + ws + "x"),
        (kpf._ACTORS_KEY.search, "'actors':" + ws + "x"),
        (kpf._TSR0.findall, "'tsr0': [1, 2" + ws + "x"),
        (kpf._TSR0.findall, "'tsr0': [1" + ws + "," + ws + "x"),
    ]
    for fn, text in cases:
        t0 = time.perf_counter()
        fn(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"{fn.__self__.pattern[:40]!r} took {elapsed:.2f}s"


def test_actor_list_scanners_are_linear_on_nesting_runs():
    n = 200_000
    for text in ("'actors': [" + "[" * n, "'actors': [" + "{" * n, "'actors': [{" + "{}" * n):
        t0 = time.perf_counter()
        actors_text, _rest = kpf._split_actors(text)
        list(kpf._actor_chunks(actors_text))
        assert time.perf_counter() - t0 < 1.0


# --------------------------------------------------------------------------- real clip (slow)

@pytest.mark.slow
def test_real_g421_annotations():
    geom_p = REAL_DIR / f"{REAL_STEM}.geom.yml"
    acts_p = REAL_DIR / f"{REAL_STEM}.activities.yml"
    types_p = REAL_DIR / f"{REAL_STEM}.types.yml"
    for p in (geom_p, acts_p, types_p):
        if not p.exists():
            pytest.skip(f"missing {p}")

    types = load_types(types_p)
    persons = person_ids(types)
    assert persons and persons <= set(types)

    acts = load_activities(acts_p)
    talks = [a for a in acts if a.name == "person_talks_to_person"]
    assert len(acts) > 20 and talks
    assert all(0 <= a.f0 <= a.f1 for a in acts)
    assert all(a.actors for a in talks)
    assert all(set(a.actor_spans) <= set(a.actors) for a in acts)

    t0 = time.perf_counter()
    geom = load_geom(geom_p)
    elapsed = time.perf_counter() - t0
    assert geom and all(f >= 0 for f in geom)
    assert max(geom) <= 9000 + 60  # a 5-minute 30 fps clip
    assert elapsed < 20, f"load_geom took {elapsed:.1f}s"

    boxes = [b for bs in geom.values() for b in bs]
    assert boxes
    assert all(b.bbox[2] > b.bbox[0] and b.bbox[3] > b.bbox[1] for b in boxes[:5000])
    geom_actors = {b.actor for b in boxes}
    assert persons & geom_actors  # person tracks have boxes

    only = load_geom(geom_p, actors=persons)
    assert {b.actor for bs in only.values() for b in bs} <= persons
    assert sum(len(bs) for bs in only.values()) <= len(boxes)
