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
