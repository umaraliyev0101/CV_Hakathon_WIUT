"""MEVA KPF annotation parsing (geom / activities / types).

The mini files in tests/data are excerpts adapted from MEVA (CC BY 4.0, Kitware/IARPA) with a few
malformed lines added on purpose: the parser must skip them rather than crash.
"""
import time
from pathlib import Path

import pytest

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
    marker = tmp_path / "pwned"
    evil = f"- {{'geom': {{'g0': '1 2 3 4', 'id1': __import__('pathlib').Path(r'{marker}').write_text('x'), 'ts0': 1}}}}\n"
    for name, loader in (("e.geom.yml", load_geom), ("e.types.yml", load_types), ("e.activities.yml", load_activities)):
        p = tmp_path / name
        p.write_text(evil, encoding="utf-8")
        loader(p)
    assert not marker.exists()


def test_loaders_handle_empty_and_binary_files(tmp_path):
    empty = tmp_path / "empty.yml"
    empty.write_text("", encoding="utf-8")
    assert load_geom(empty) == {} and load_types(empty) == {} and load_activities(empty) == []
    binary = tmp_path / "bin.yml"
    binary.write_bytes(b"\xff\xfe\x00\x01- {'geom': \xc3\x28 \n" * 3)
    assert load_geom(binary) == {} and load_types(binary) == {} and load_activities(binary) == []


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
