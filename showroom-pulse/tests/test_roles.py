import numpy as np, json, pytest
from pulse.config import get_config
from pulse.datatypes import Person
from pulse import roles as R
from tests.conftest import stay

CFG = get_config("dealership")

def test_suggest_staff_customer_unsure_with_reasons():
    persons = [Person(i, [i], 0, 100) for i in (1, 2, 3)]
    feats = {1: {"outfit_peers": 3, "contacts": 5, "presence": 0.95, "initiator": 0.8},
             2: {"outfit_peers": 0, "contacts": 0, "presence": 0.1, "initiator": 0.0},
             3: {"outfit_peers": 1, "contacts": 2, "presence": 0.5, "initiator": 0.5}}
    sug = {s.person_id: s for s in R.suggest(persons, feats, CFG)}
    assert sug[1].role == "staff" and sug[2].role == "customer" and sug[3].role == "unsure"
    assert "same outfit as 3 others" in sug[1].reasons and len(sug[1].reasons) <= 3
    assert sug[2].reasons == []

def test_presence_ignored_for_short_clips():
    persons = [Person(1, [1], 0, 30), Person(2, [2], 0, 30)]
    tl = {1: stay(1, 0.5, 0.5, 0, 30, pos=(0, 4)), 2: stay(2, 0.1, 0.5, 0, 2, pos=(8, 4))}
    emb = {1: np.array([1.0, 0]), 2: np.array([0.0, 1])}
    f = R.interactions(persons, tl, emb, 30.0, get_config("short"))
    assert f[1]["presence"] is None

def test_contacts_and_initiator():
    # person 1 walks up to person 2 (standing) and stays 6 s within 1 m
    from tests.conftest import track
    approach = [(0.0 + 0.1 * i, 4.0) for i in range(30)]          # 0 → 2.9 m along X
    p1 = track(1, [(0.3, 0.5)] * 30, pos=approach) + stay(1, 0.3, 0.5, 6.0, 6, pos=(3.0, 4.0))
    p2 = stay(2, 0.35, 0.5, 0, 12, pos=(3.8, 4.0))
    persons = [Person(1, [1], 0, 12), Person(2, [2], 0, 12)]
    f = R.interactions(persons, {1: p1, 2: p2}, {1: np.ones(2), 2: -np.ones(2)}, 12.0, get_config("short"))
    assert f[1]["contacts"] == 1 and f[2]["contacts"] == 1
    assert f[1]["initiator"] == 1.0 and f[2]["initiator"] == 0.0

def test_outfit_peers_cosine():
    persons = [Person(i, [i], 0, 10) for i in (1, 2, 3)]
    emb = {1: np.array([1.0, 0.1]), 2: np.array([1.0, 0.12]), 3: np.array([0.0, 1.0])}
    tl = {i: stay(i, 0.1 * i, 0.5, 0, 1, pos=(10.0 * i, 4)) for i in (1, 2, 3)}
    f = R.interactions(persons, tl, emb, 10.0, get_config("short"))
    assert (f[1]["outfit_peers"], f[2]["outfit_peers"], f[3]["outfit_peers"]) == (1, 1, 0)

def test_roles_json_formats_and_precedence(tmp_path):
    p = tmp_path / "roles.json"
    p.write_text(json.dumps({"persons": [{"person_id": 1, "role": "staff", "name": "Aziz"}]}))
    assert R.load_roles_json(p) == {1: "staff"}
    p.write_text(json.dumps({"2": "customer"}))
    assert R.load_roles_json(p) == {2: "customer"}
    p.write_text(json.dumps({"2": "boss"}))
    with pytest.raises(ValueError):
        R.load_roles_json(p)
    from pulse.datatypes import RoleSuggestion
    sug = [RoleSuggestion(1, "customer", 0.9, []), RoleSuggestion(2, "unsure", 0.5, []),
           RoleSuggestion(3, "staff", 0.8, [])]
    assert R.final_roles(sug, {1: "staff"}, {3: "customer"}) == {1: "staff", 2: "unsure", 3: "customer"}
    assert R.export_roles_json({1: "staff"}, {1: "Aziz"})["persons"][0] == {"person_id": 1, "role": "staff", "name": "Aziz"}


# ---- fix round 1: focused tests -------------------------------------------------
from tests.conftest import track
from pulse.datatypes import RoleSuggestion

SHORT = get_config("short")


def _pair(p1_times, gap_pos=(1.0, 4.0), far=(20.0, 4.0), near_pos=(1.0, 4.0)):
    """Person 1 at origin-ish; person 2 near at p1_times, far otherwise."""
    ts = [round(0.2 * i, 2) for i in range(int(round(max(p1_times) / 0.2)) + 8)]
    from pulse.datatypes import Sample
    p1 = [Sample(t, 1, (0.5, 0.5), (0, 0, 1, 1), (0.0, 4.0), 0.0) for t in ts]
    p2 = [Sample(t, 2, (0.5, 0.5), (0, 0, 1, 1), near_pos if any(abs(t - u) < 1e-6 for u in p1_times) else far, 0.0)
          for t in ts]
    return [Person(1, [1], 0, 20), Person(2, [2], 0, 20)], {1: p1, 2: p2}


def _near_times(a, b):
    n = int(round((b - a) / 0.2))
    return [round(a + 0.2 * i, 2) for i in range(n + 1)]


def _contacts(times):
    persons, tl = _pair(times)
    return R.interactions(persons, tl, {}, 30.0, SHORT)[1]["contacts"]


def test_contact_bridge_one_second_gap_joins():
    # 0..2.6 near, gap of exactly 1.0 s (2.6 -> 3.6), 3.6..6.0 near: bridged, span 6.0 >= 5
    assert _contacts(_near_times(0, 2.6) + _near_times(3.6, 6.0)) == 1


def test_contact_bridge_larger_gap_splits():
    # gap 1.4 s splits into runs of 2.6 and 2.2 s, neither reaches 5 s
    assert _contacts(_near_times(0, 2.6) + _near_times(4.0, 6.2)) == 0


def test_contact_qualify_boundary():
    assert _contacts(_near_times(0, 5.0)) == 1
    assert _contacts(_near_times(0, 4.8)) == 0


def test_contact_ignores_samples_without_position():
    persons, tl = _pair(_near_times(0, 8.0))
    for s in tl[2]:
        s.pos_m = None
    f = R.interactions(persons, tl, {}, 30.0, SHORT)
    assert f[1]["contacts"] == 0 and f[1]["initiator"] == 0.0


def test_missing_embedding_gives_zero_peers():
    persons = [Person(i, [i], 0, 10) for i in (1, 2, 3)]
    emb = {1: np.array([1.0, 0.1]), 2: np.array([1.0, 0.12])}
    tl = {i: stay(i, 0.1 * i, 0.5, 0, 1, pos=(10.0 * i, 4)) for i in (1, 2, 3)}
    f = R.interactions(persons, tl, emb, 10.0, SHORT)
    assert f[3]["outfit_peers"] == 0 and f[1]["outfit_peers"] == 1


def test_presence_value_and_cap():
    persons = [Person(1, [1], 0, 100), Person(2, [2], 0, 100)]
    n_half = int(0.5 * 100 * SHORT.sample_fps)
    tl = {1: track(1, [(0.5, 0.5)] * n_half, pos=[(0, 4)] * n_half),
          2: track(2, [(0.5, 0.5)] * (3 * n_half), pos=[(30, 4)] * (3 * n_half))}
    f = R.interactions(persons, tl, {}, 100.0, SHORT)
    assert f[1]["presence"] == pytest.approx(0.5)
    assert f[2]["presence"] == 1.0


def test_reason_omitted_below_quarter_norm():
    persons = [Person(1, [1], 0, 100)]
    # presence 0.2 (<0.25) omitted; contacts 1 -> norm 0.25 kept
    feats = {1: {"outfit_peers": 0, "contacts": 1, "presence": 0.2, "initiator": 0.0}}
    s = R.suggest(persons, feats, CFG)[0]
    assert s.reasons == ["close to 1 people for 5 s or more"]


def test_reasons_capped_at_three():
    persons = [Person(1, [1], 0, 100)]
    feats = {1: {"outfit_peers": 3, "contacts": 4, "presence": 0.9, "initiator": 0.9}}
    s = R.suggest(persons, feats, CFG)[0]
    assert len(s.reasons) == 3
    assert not any(r.startswith("approached") for r in s.reasons)  # lowest w*norm dropped


def test_initiator_tie_goes_to_neither():
    # both walk toward each other at equal speed, then stand 1 m apart for several seconds
    m = 70
    p1 = track(1, [(0.3, 0.5)] * m, pos=[(min(0.1 * i, 2.5), 4.0) for i in range(m)])
    p2 = track(2, [(0.3, 0.5)] * m, pos=[(max(6.0 - 0.1 * i, 3.5), 4.0) for i in range(m)])
    persons = [Person(1, [1], 0, 14), Person(2, [2], 0, 14)]
    f = R.interactions(persons, {1: p1, 2: p2}, {}, 14.0, SHORT)
    assert f[1]["contacts"] == 1
    assert f[1]["initiator"] == 0.0 and f[2]["initiator"] == 0.0


def test_suggest_cfg_none_and_inactive_presence():
    persons = [Person(1, [1], 0, 100)]
    feats = {1: {"outfit_peers": 3, "contacts": 4, "presence": None, "initiator": 1.0}}
    s = R.suggest(persons, feats)[0]  # cfg=None -> dealership defaults
    assert s.role == "staff" and s.confidence == pytest.approx(1.0)
    assert "presence" not in s.features
    assert not any(r.startswith("on screen") for r in s.reasons)


@pytest.mark.parametrize("payload", [
    [1, 2],                                           # top level not a dict
    {"persons": [{"person_id": 1}]},                  # missing role
    {"persons": [3]},                                 # entry not a dict
    {"persons": {}},                                  # persons not a list
    {"abc": "staff"},                                 # key not an int
    {"1.5": "staff"},                                 # key not an int
    {"persons": [{"person_id": True, "role": "staff"}]},
    {"persons": [{"person_id": 1.5, "role": "staff"}]},
    {"persons": [{"person_id": 1, "role": "boss"}]},
])
def test_load_roles_json_rejects_malformed(tmp_path, payload):
    p = tmp_path / "roles.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="roles.json"):
        R.load_roles_json(p)


def test_load_roles_json_invalid_json(tmp_path):
    p = tmp_path / "roles.json"
    p.write_text("{nope")
    with pytest.raises(ValueError, match="roles.json"):
        R.load_roles_json(p)
