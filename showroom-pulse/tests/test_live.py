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
    from pulse.app import main as _m
    monkeypatch.setattr(_m, "REPO_ROOT", tmp_path)
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


# ------------------------------------------------------------------ fix round 1: security
import struct

from starlette.websockets import WebSocketDisconnect

from pulse import sources
from pulse.app import main as main_mod


def fake_jpeg_header(w, h):
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + sof + b"\xff\xd9"


def test_decode_rejects_bomb_and_non_jpeg():
    with pytest.raises(ValueError):
        sources.decode_jpeg(fake_jpeg_header(60000, 60000))
    with pytest.raises(ValueError):
        sources.decode_jpeg(fake_jpeg_header(9000, 100))
    ok, png = cv2.imencode(".png", frame())
    with pytest.raises(ValueError):
        sources.decode_jpeg(png.tobytes())            # only JPEG is accepted on the live path


def test_ws_foreign_origin_rejected_and_same_origin_ok(client):
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/live", headers={"origin": "http://evil.example"}):
            pass
    assert e.value.code == 1008
    with client.websocket_connect("/ws/live", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(jpeg())
        assert ws.receive_json()["type"] == "overlay"


def test_foreign_host_rejected_http_and_ws(client):
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code in (400, 403)
    assert client.get("/api/health", headers={"host": "localhost:8000"}).status_code == 200
    assert client.get("/api/health", headers={"host": "[::1]:8000"}).status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live", headers={"host": "evil.example"}):
            pass


def test_cross_origin_post_rejected(client):
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"x", "video/mp4")},
                    headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.get("/api/health", headers={"origin": "http://evil.example"})
    assert r.status_code == 200                        # reads of other origins stay blocked by the browser


def test_allowed_hosts_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSE_ALLOWED_HOSTS", "192.168.1.20, demo.local")
    c = TestClient(create_app(jobs_dir=tmp_path / "j", engine_factory=lambda: StubEngine(lambda t: [])))
    assert c.get("/api/health", headers={"host": "192.168.1.20:8000"}).status_code == 200
    assert c.get("/api/health", headers={"host": "other.example"}).status_code in (400, 403)


def test_ws_oversize_messages_rejected_connection_kept(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"\xff\xd8" + b"0" * (2 * 1024 * 1024 + 1))
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "zones", "zones": [], "pad": "x" * 70000}))
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(jpeg())
        assert ws.receive_json()["type"] == "overlay"


def test_zone_caps():
    from pulse.app.main import ZoneError, validate_zones
    z = CAR.to_dict()
    with pytest.raises(ZoneError):
        validate_zones({"zones": [dict(z, zone_id=f"z{i}") for i in range(51)]})
    with pytest.raises(ZoneError):
        validate_zones({"zones": [dict(z, polygon=[[0.1 + 0.0001 * (i % 2), 0.1 + 0.8 * i / 300]
                                                   for i in range(201)])]})
    assert len(validate_zones({"zones": [dict(z, zone_id=f"z{i}") for i in range(50)]})) == 50


def test_load_zones_json_only_size_and_repo_root(client, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "labels" / "zones").mkdir(parents=True)
    monkeypatch.setattr(main_mod, "REPO_ROOT", root)
    monkeypatch.chdir(tmp_path)                        # CWD must not matter
    doc = json.dumps({"version": 1, "zones": [CAR.to_dict()]})
    (root / "labels" / "zones" / "ok.json").write_text(doc)
    (root / "labels" / "zones" / "a.txt").write_text(doc)
    (root / "labels" / "zones" / "big.json").write_text(" " * 1_100_000 + doc)
    with client.websocket_connect("/ws/live") as ws:
        for path, good in (("labels/zones/ok.json", True), ("labels/zones/a.txt", False),
                           ("labels/zones/big.json", False)):
            ws.send_text(json.dumps({"type": "load_zones", "path": path}))
            assert ws.receive_json()["type"] == ("zones" if good else "error"), path


def test_engine_failure_reports_error(tmp_path):
    def boom():
        raise RuntimeError("model missing")
    c = TestClient(create_app(jobs_dir=tmp_path / "j", engine_factory=boom))
    with c.websocket_connect("/ws/live") as ws:
        m = ws.receive_json()
        assert m == {"type": "error", "message": "Live engine unavailable"}
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1011


def test_stale_track_bookkeeping_pruned():
    ids = iter(range(1, 1000))
    cur = {"id": 1}

    def script(t):
        if int(t * 5) % 10 == 0:                       # new track id every 2 s, same person
            cur["id"] = next(ids) + 100
        return [(cur["id"], BOX)]
    s = LiveSession(get_config("short"), StubEngine(script), zones=[CAR], window_s=5.0)
    run(s, 30.0)
    assert len(s._track_person) <= 5
