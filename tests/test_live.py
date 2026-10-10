"""Live session tests. The model layer is a scripted stub engine (injected dependency); the session
logic (re-attach, timeline, journeys, alerts, roles, trimming) and the WebSocket plumbing are real."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pulse.app.main import create_app
from pulse.config import get_config
from pulse.datatypes import Det, TrackObs, Zone
from pulse.live import LiveSession

W, H = 320, 240
CAR = Zone("car-a", "Car A", "car", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], "user")
BOX = (140.0, 60.0, 180.0, 200.0)


class StubEngine:
    """script(t) -> list of (track_id, bbox)."""

    def __init__(self, script):
        self.script = script
        self.last_person_dets = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def process(self, t, frame, k):
        obs = [TrackObs(t, k, tid, bb, 0.9, None) for tid, bb in self.script(t)]
        self.last_person_dets = [Det(o.bbox, o.score, "person") for o in obs]
        return obs


def frame():
    f = np.zeros((H, W, 3), np.uint8)
    f[:] = (30, 90, 160)
    return f


def run(session, seconds, dt=0.2, t0=0.0):
    out = []
    for i in range(int(round(seconds / dt)) + 1):
        out.append(session.step(round(t0 + i * dt, 3), frame()))
    return out


def test_alert_after_eight_seconds_and_not_before():
    s = LiveSession(get_config("short"), StubEngine(lambda t: [(1, BOX)]), zones=[CAR])
    msgs = run(s, 9.0)
    for m in msgs:
        assert m["type"] == "overlay" and m["frame_size"] == [W, H]
        if m["t"] < 8.0 - 1e-6:
            assert m["alerts"] == [], m["t"]
    last = msgs[-1]
    assert [a["zone_id"] for a in last["alerts"]] == ["car-a"]
    assert last["alerts"][0]["message"].startswith("Customer at Car A for 9 s")
    assert "no staff nearby" in last["alerts"][0]["message"]
    p = last["persons"][0]
    assert p["alert"] is True and p["zone_id"] == "car-a" and p["dwell"] > 8.0
    assert p["role"] == "customer"
    assert last["occupancy"] == 1
    z = last["zones"][0]
    assert z["now"] == 1 and z["total"] == 1
    first_alert = next(m["t"] for m in msgs if m["alerts"])
    assert 8.0 - 1e-6 <= first_alert <= 8.2 + 1e-6


def test_toggle_role_clears_alert():
    s = LiveSession(get_config("short"), StubEngine(lambda t: [(1, BOX)]), zones=[CAR])
    msgs = run(s, 9.0)
    assert msgs[-1]["alerts"]
    pid = s.toggle_role_at(160 / W, 120 / H)
    assert pid == msgs[-1]["persons"][0]["person_id"]
    assert s.roles[pid] == "staff"
    nxt = s.step(9.2, frame())
    assert nxt["alerts"] == []
    assert nxt["persons"][0]["role"] == "staff" and nxt["persons"][0]["alert"] is False
    assert nxt["occupancy"] == 0                      # staff are not customers
    assert s.toggle_role_at(0.01, 0.01) is None       # empty space
    assert s.toggle_role_at(160 / W, 120 / H) == pid and s.roles[pid] == "customer"


def test_old_samples_trimmed():
    s = LiveSession(get_config("short"), StubEngine(lambda t: [(1, BOX)]), zones=[CAR], window_s=5.0)
    run(s, 12.0)
    ts = [x.t for ss in s.timeline.values() for x in ss]
    assert ts and min(ts) >= 12.0 - 5.0 - 1e-6
    assert all(v.t_out >= 12.0 - 5.0 - 1e-6 for vs in s.journeys.values() for v in vs)


def test_new_track_reattached_to_recent_person():
    def script(t):
        if t < 2.0:
            return [(1, BOX)]
        if 2.0 <= t < 2.6:
            return []
        return [(7, BOX)]                              # tracker lost id 1, issued id 7
    s = LiveSession(get_config("short"), StubEngine(script), zones=[CAR])
    msgs = run(s, 4.0)
    ids = {p["person_id"] for m in msgs for p in m["persons"]}
    assert len(ids) == 1


def test_far_apart_people_are_distinct_and_engine_reset():
    eng = StubEngine(lambda t: [(1, BOX), (2, (10.0, 60.0, 50.0, 200.0))])
    s = LiveSession(get_config("short"), eng, zones=[CAR])
    assert eng.resets == 1
    m = run(s, 1.0)[-1]
    assert len({p["person_id"] for p in m["persons"]}) == 2 and m["occupancy"] == 2


# ------------------------------------------------------------------ websocket
def jpeg():
    ok, buf = cv2.imencode(".jpg", frame())
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_path):
    eng = StubEngine(lambda t: [(1, BOX)])
    app = create_app(jobs_dir=tmp_path / "jobs", engine_factory=lambda: eng)
    return TestClient(app)


def test_ws_garbage_then_valid_frame(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"this is not a jpeg")
        err = ws.receive_json()
        assert err["type"] == "error" and err["message"]
        ws.send_bytes(jpeg())
        m = ws.receive_json()
        assert m["type"] == "overlay" and len(m["persons"]) == 1
        assert m["fps"] >= 0 and m["dropped"] == 0


def test_ws_zones_toggle_preset(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "zones", "zones": [CAR.to_dict()]}))
        ws.send_text(json.dumps({"type": "preset", "name": "short"}))
        ws.send_bytes(jpeg())
        m = ws.receive_json()
        assert m["type"] == "overlay" and m["zones"][0]["zone_id"] == "car-a"
        ws.send_text(json.dumps({"type": "toggle", "x": 0.5, "y": 0.5}))
        ws.send_bytes(jpeg())
        m = ws.receive_json()
        assert m["persons"][0]["role"] == "staff"


def test_ws_bad_control_messages(client):
    with client.websocket_connect("/ws/live") as ws:
        for bad in ({"type": "zones", "zones": [{"zone_id": "x"}]},
                    {"type": "preset", "name": "nope"},
                    {"type": "load_zones", "path": "../../../etc/passwd"},
                    {"type": "load_zones", "path": "/etc/passwd"},
                    {"type": "load_zones", "path": "labels/zones/../../pyproject.toml"},
                    {"type": "wat"}):
            ws.send_text(json.dumps(bad))
            assert ws.receive_json()["type"] == "error", bad
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(jpeg())
        assert ws.receive_json()["type"] == "overlay"          # still alive


def test_ws_load_zones_from_allowed_dir(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labels" / "zones").mkdir(parents=True)
    doc = {"version": 1, "zones": [CAR.to_dict()]}
    (tmp_path / "labels" / "zones" / "a.json").write_text(json.dumps(doc))
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "load_zones", "path": "labels/zones/a.json"}))
        m = ws.receive_json()
        assert m["type"] == "zones" and m["zones"][0]["zone_id"] == "car-a"


def test_ws_drops_stale_frames_when_behind(tmp_path):
    import time as _t

    class Slow(StubEngine):
        def process(self, t, frame, k):
            _t.sleep(0.3)
            return super().process(t, frame, k)

    eng = Slow(lambda t: [(1, BOX)])
    c = TestClient(create_app(jobs_dir=tmp_path / "jobs", engine_factory=lambda: eng))
    with c.websocket_connect("/ws/live") as ws:
        for _ in range(8):
            ws.send_bytes(jpeg())
        first = ws.receive_json()
        last = ws.receive_json()
        assert first["type"] == last["type"] == "overlay"
        assert last["dropped"] >= 5 and last["t"] > first["t"]
