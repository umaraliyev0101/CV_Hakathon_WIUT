from pathlib import Path

from pulse.eval.meva import load_kpf

FIX = Path(__file__).parent / "fixtures" / "meva_mini"


def test_flow_dialect_tracks_and_types():
    c = load_kpf(FIX, "mini")
    assert c.clip == "mini"
    assert c.tracks[718954285527769780][674] == (908.0, 337.0, 943.0, 458.0)
    assert sorted(c.tracks[718954285527769780]) == [674, 675, 676]
    assert c.track_types[718954285527769780] == "person"
    assert c.track_types[5555555555555555555] == "vehicle"


def test_person_filter_via_types():
    c = load_kpf(FIX, "mini")
    assert set(c.person_tracks()) == {718954285527769780, 9143980220328846590}


def test_dict_dialect():
    c = load_kpf(FIX, "minidict")
    assert c.tracks[241][4399] == (922.0, 471.0, 972.0, 550.0)
    assert c.tracks[16][669] == (1257.0, 477.0, 1411.0, 815.0)
    assert c.track_types[241] == "person"
    assert set(c.person_tracks()) == {241, 17}


def test_meta_lines_ignored():
    c = load_kpf(FIX, "mini")
    assert all(isinstance(k, int) for k in c.tracks)
    assert len(c.tracks) == 3


def test_activities_flow():
    c = load_kpf(FIX, "mini")
    assert [a.name for a in c.activities] == [
        "person_exits_scene_through_structure", "person_enters_scene_through_structure"]
    a = c.activities[0]
    assert (a.act_id, a.frame_start, a.frame_end) == (813, 674, 723)
    assert a.actors == [718954285527769780]
    assert c.activities[1].frame_start == 675 and c.activities[1].frame_end == 700


def test_activities_dict_per_actor_spans():
    c = load_kpf(FIX, "minidict")
    (a,) = c.activities
    assert a.name == "person_enters_vehicle" and a.actors == [17, 16]
    assert a.actor_spans == {17: (1626, 1676), 16: (1630, 1676)}
