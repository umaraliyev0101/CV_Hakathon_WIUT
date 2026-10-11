"""API tests. The model layer is injected (stub engine + HSV-only embedder); everything else is real:
upload, background worker, perceive/cache, score, write_outputs, report files."""
import time

import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pulse import cache
from pulse.app.main import create_app
from pulse.datatypes import Det, TrackObs
from pulse.perception.embed import Embedder

W, H, FPS, SECONDS = 320, 240, 10, 12


def write_video(path, seconds=SECONDS):
    gen = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                      macro_block_size=1, output_params=["-loglevel", "error"])
    gen.send(None)
    rng = np.random.default_rng(0)
    base = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    for _ in range(int(FPS * seconds)):
        gen.send(base)
    gen.close()


class StubEngine:
    """Scripted people: track 1 stands near the car from t=1 s to 10 s; track 2 does the same elsewhere."""
    providers = ("CPUExecutionProvider",)

    class _Det:
        class session:
            @staticmethod
            def get_providers():
                return ["CPUExecutionProvider"]
    detector = _Det()

    def __init__(self, people=True):
        self.people = people
        self.process_calls = 0
        self.last_person_dets = []
        self.embedder = Embedder(use_dino=False)

    def reset(self):
        self.last_person_dets = []

    def process(self, t, frame, k):
        self.process_calls += 1
        if not self.people or not (1.0 <= t <= 10.0):
            self.last_person_dets = []
            return []
        obs = [TrackObs(t, k, 1, (150, 120, 170, 190), 0.9, None),
               TrackObs(t, k, 2, (40, 30, 60, 100), 0.9, None)]
        self.last_person_dets = [Det(o.bbox, o.score, "person") for o in obs]
        return obs

    def detect_vehicles(self, frame):
        return [Det((100.0, 100.0, 220.0, 200.0), 0.9, "car")]


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(people=True):
        engine = StubEngine(people)
        app = create_app(jobs_dir=tmp_path / "jobs", engine_factory=lambda: engine)
        c = TestClient(app)
        clients.append(c)
        c.engine = engine
        return c
    yield make


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    write_video(p)
    return p


def upload(c, path, name="clip.mp4", preset="short"):
    with open(path, "rb") as f:
        r = c.post("/api/jobs", files={"file": (name, f, "video/mp4")}, data={"preset": preset})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def wait_state(c, jid, states, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/jobs/{jid}")
        assert r.status_code == 200
        if r.json()["state"] in states:
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"timeout, last: {r.json()}")


def rect(zid, name, kind, x0, y0, x1, y1):
    return {"zone_id": zid, "name": name, "kind": kind,
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "source": "user"}


def test_upload_garbage_fails_cleanly(make_client, tmp_path):
    c = make_client()
    g = tmp_path / "x.mp4"
    g.write_bytes(b"not a video")
    jid = upload(c, g, "x.mp4")
    st = wait_state(c, jid, {"failed"})
    assert "Unreadable" in st["message"]
    assert str(tmp_path) not in st["message"]
    assert c.get(f"/api/jobs/{jid}").status_code == 200
    assert c.get(f"/api/jobs/{jid}/results").status_code == 404


def test_bad_extension_400(make_client, tmp_path):
    c = make_client()
    g = tmp_path / "x.exe"
    g.write_bytes(b"MZ")
    with open(g, "rb") as f:
        r = c.post("/api/jobs", files={"file": ("x.exe", f, "application/octet-stream")},
                   data={"preset": "short"})
    assert r.status_code == 400 and "error" in r.json()


def test_unknown_job_404_and_bad_id(make_client):
    c = make_client()
    assert c.get("/api/jobs/" + "0" * 32).status_code == 404
    assert "error" in c.get("/api/jobs/" + "0" * 32).json()
    assert c.get("/api/jobs/not-an-id").status_code == 404
    assert c.get("/api/health").json()["ok"] is True


def test_job_with_no_people(make_client, video):
    c = make_client(people=False)
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    assert c.get(f"/api/jobs/{jid}/people").json() == []
    r = c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    assert r.status_code == 200, r.text
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert any("Nobody detected" in w for w in res["warnings"])
    assert c.get(f"/api/jobs/{jid}").json()["state"] == "scored"


@pytest.mark.parametrize("zones", [
    {"zones": [{"zone_id": "a", "name": "A", "kind": "car", "polygon": [[0, 0], [1, 1]]}]},
    {"zones": [rect("a", "A", "car", 0.1, 0.1, 1.5, 0.5)]},
    {"zones": [{"zone_id": "a", "name": "A", "kind": "car", "polygon": [[0, 0], [0.5, 0.5], [1, 1]]}]},
    {"zones": [rect("a", "A", "garage", 0.1, 0.1, 0.5, 0.5)]},
    {"zones": [rect("a", "A", "car", 0.1, 0.1, 0.5, 0.5), rect("a", "B", "desk", 0.6, 0.6, 0.9, 0.9)]},
    {"zones": [rect("a", "", "car", 0.1, 0.1, 0.5, 0.5)]},
    {"zones": [rect("a", "x" * 81, "car", 0.1, 0.1, 0.5, 0.5)]},
    {"zones": [{"zone_id": "a", "name": "A", "kind": "car", "polygon": [[0, 0], [0.5, "nan"], [1, 1]]}]},
    {"nope": 1},
])
def test_zone_validation_422(make_client, video, zones):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    r = c.put(f"/api/jobs/{jid}/zones", json=zones)
    assert r.status_code == 422 and r.json()["error"]


def test_full_flow_rescore_no_redetect(make_client, video):
    c = make_client()
    eng = c.engine
    jid = upload(c, video)
    st = wait_state(c, jid, {"ready_for_zones"})
    assert st["progress"] == 1.0 and st["preset"] == "short"
    calls, runs = eng.process_calls, cache.perception_runs()
    assert calls > 0
    assert c.get(f"/api/jobs/{jid}/keyframe.jpg").headers["content-type"] == "image/jpeg"
    zones = c.get(f"/api/jobs/{jid}/zones").json()
    assert zones["version"] == 1 and zones["zones"][0]["kind"] == "car"
    car = zones["zones"][0]["polygon"]
    new = {"version": 1, "frame_size": zones["frame_size"], "zones": [
        {"zone_id": "car-1", "name": "Car: Demo", "kind": "car", "polygon": car, "source": "user"},
        rect("desk", "Desk", "desk", 0.05, 0.05, 0.3, 0.3)]}
    assert c.put(f"/api/jobs/{jid}/zones", json=new).status_code == 200
    people = c.get(f"/api/jobs/{jid}/people").json()
    assert len(people) == 2 and {"person_id", "role", "suggested_role", "confidence", "reasons",
                                 "thumb_url", "first_t", "last_t"} <= set(people[0])
    assert c.get(people[0]["thumb_url"]).status_code == 200
    pid = people[1]["person_id"]     # the person away from the car
    r = c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(pid): "staff"}, "skip": False})
    assert r.status_code == 200
    assert {p["person_id"]: p["role"] for p in c.get(f"/api/jobs/{jid}/people").json()}[pid] == "staff"
    assert c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(pid): "boss"}}).status_code == 422
    tracks = (cache.load(next((c.app.state.jobs_dir / jid / "cache").iterdir())).dir / "tracks.npz")
    m0 = tracks.stat().st_mtime_ns
    r1 = c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    assert r1.status_code == 200
    res1 = c.get(f"/api/jobs/{jid}/results").json()
    assert res1["store"]["visitors"] >= 1 and res1["per_car"][0]["visitors"] == 1
    assert c.get(f"/api/jobs/{jid}/export.json").status_code == 200
    assert "zone_id" in c.get(f"/api/jobs/{jid}/export.csv").text
    new["zones"][0] = {**new["zones"][0], "polygon": [[0.7, 0.1], [0.95, 0.1], [0.95, 0.3], [0.7, 0.3]]}
    assert c.put(f"/api/jobs/{jid}/zones", json=new).status_code == 200
    assert c.post(f"/api/jobs/{jid}/score", json={"render_video": False}).status_code == 200
    res2 = c.get(f"/api/jobs/{jid}/results").json()
    assert res2["per_car"][0]["visitors"] == 0 and res2["per_car"] != res1["per_car"]
    assert res2["per_zone"] != res1["per_zone"]
    assert eng.process_calls == calls
    assert cache.perception_runs() == runs
    assert tracks.stat().st_mtime_ns == m0
    assert c.get(f"/api/jobs/{jid}/files/report.json").status_code == 200


def test_roles_import(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    bad = c.post(f"/api/jobs/{jid}/roles/import", files={"file": ("roles.json", b"[1,2", "application/json")})
    assert bad.status_code == 422 and bad.json()["error"]
    good = b'{"persons": [{"person_id": 1, "role": "staff"}]}'
    ok = c.post(f"/api/jobs/{jid}/roles/import", files={"file": ("roles.json", good, "application/json")})
    assert ok.status_code == 200
    assert {p["person_id"]: p["role"] for p in c.get(f"/api/jobs/{jid}/people").json()}[1] == "staff"


def test_files_path_traversal_blocked(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    for bad in ("../job.json", "%2e%2e/job.json", "..%2fjob.json", "%2fetc/passwd", "thumbs/../../job.json"):
        r = c.get(f"/api/jobs/{jid}/files/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
    r = c.get(f"/api/jobs/{jid}/files/../job.json")
    assert r.status_code in (400, 404)
    out = c.app.state.jobs_dir / jid / "out"
    (out / "evil").symlink_to(c.app.state.jobs_dir / jid / "job.json")
    assert c.get(f"/api/jobs/{jid}/files/evil").status_code in (400, 404)


def test_cleanup_old_jobs(tmp_path):
    import os
    from pulse.app import jobs
    old, new = "a" * 32, "b" * 32
    for n in (old, new, "keepme"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "job.json").write_text("{}")
    past = time.time() - 90000
    for p in ((tmp_path / old), (tmp_path / old / "job.json"), (tmp_path / "keepme"),
              (tmp_path / "keepme" / "job.json")):
        os.utime(p, (past, past))
    jobs.cleanup(tmp_path, max_age_s=86400)
    assert not (tmp_path / old).exists() and (tmp_path / new).exists() and (tmp_path / "keepme").exists()


def test_startup_runs_cleanup(tmp_path):
    import os
    old = tmp_path / "jobs" / ("c" * 32)
    old.mkdir(parents=True)
    past = time.time() - 90000
    os.utime(old, (past, past))
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", engine_factory=lambda: StubEngine())):
        pass
    assert not old.exists()


def test_static_pages_and_no_model_at_import():
    c = TestClient(create_app(jobs_dir=__import__("pathlib").Path("/nonexistent-jobs")))
    assert c.get("/").status_code == 200 and c.get("/live").status_code == 200


# ---------------------------------------------------------------- fix round 1
def test_status_surfaces_analysis_warnings(make_client, video):
    c = make_client(people=False)
    jid = upload(c, video)
    st = wait_state(c, jid, {"ready_for_zones"})
    assert any("Nobody detected" in w for w in st["warnings"])
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    st = c.get(f"/api/jobs/{jid}").json()
    assert sum("Nobody detected" in w for w in st["warnings"]) == 1


def test_warnings_refresh_after_score(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    doc = {"version": 1, "frame_size": [W, H], "zones": []}
    assert c.put(f"/api/jobs/{jid}/zones", json=doc).status_code == 200
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    assert any("No zones" in w for w in c.get(f"/api/jobs/{jid}").json()["warnings"])
    doc["zones"] = [rect("a", "A", "car", 0.1, 0.1, 0.5, 0.5)]
    c.put(f"/api/jobs/{jid}/zones", json=doc)
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    assert not any("No zones" in w for w in c.get(f"/api/jobs/{jid}").json()["warnings"])


def test_atomic_write_never_exposes_partial_file(tmp_path):
    import json
    import threading
    from pulse.app.jobs import atomic_write_text
    p = tmp_path / "report.json"
    atomic_write_text(p, json.dumps({"x": 0}))
    stop, bad = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                bad.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(300):
        atomic_write_text(p, json.dumps({"x": i, "pad": "y" * 200000}))
    stop.set()
    t.join()
    assert not bad


def test_removing_override_restores_suggested_role(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    people = c.get(f"/api/jobs/{jid}/people").json()
    pid, suggested = people[0]["person_id"], people[0]["suggested_role"]
    flipped = "staff" if suggested != "staff" else "customer"
    c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(pid): flipped}})
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    now = {p["person_id"]: p["role"] for p in c.get(f"/api/jobs/{jid}/people").json()}
    assert now[pid] == flipped
    c.put(f"/api/jobs/{jid}/roles", json={"overrides": {}})
    now = {p["person_id"]: p["role"] for p in c.get(f"/api/jobs/{jid}/people").json()}
    assert now[pid] == suggested


def test_upload_too_large_413(make_client, video, monkeypatch):
    from pulse.app import main as main_mod
    monkeypatch.setattr(main_mod, "MAX_UPLOAD", 100)
    c = make_client()
    with open(video, "rb") as f:
        r = c.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")}, data={"preset": "short"})
    assert r.status_code == 413 and "error" in r.json()
    assert not list((c.app.state.jobs_dir).glob("*")) or not any(c.app.state.jobs_dir.iterdir())


def test_409_while_perceiving_and_when_failed(tmp_path, video):
    import threading
    gate = threading.Event()

    class Blocking(StubEngine):
        def process(self, t, frame, k):
            gate.wait(30)
            return super().process(t, frame, k)

    eng = Blocking()
    c = TestClient(create_app(jobs_dir=tmp_path / "jobs", engine_factory=lambda: eng))
    try:
        jid = upload(c, video)
        wait_state(c, jid, {"perceiving"})
        body = {"zones": [rect("a", "A", "car", 0.1, 0.1, 0.5, 0.5)]}
        assert c.get(f"/api/jobs/{jid}/zones").status_code == 409
        assert c.put(f"/api/jobs/{jid}/zones", json=body).status_code == 409
        assert c.get(f"/api/jobs/{jid}/people").status_code == 409
        assert c.put(f"/api/jobs/{jid}/roles", json={"overrides": {}}).status_code == 409
        assert c.post(f"/api/jobs/{jid}/score", json={}).status_code == 409
    finally:
        gate.set()
    wait_state(c, jid, {"ready_for_zones"})
    g = tmp_path / "x.mp4"
    g.write_bytes(b"junk")
    bad = upload(c, g, "x.mp4")
    wait_state(c, bad, {"failed"})
    assert c.get(f"/api/jobs/{bad}/zones").status_code == 409
    assert c.put(f"/api/jobs/{bad}/zones", json=body).status_code == 409
    assert c.get(f"/api/jobs/{bad}/people").status_code == 409
    assert c.post(f"/api/jobs/{bad}/score", json={}).status_code == 409


def test_missing_cache_dir_is_409(make_client, video):
    import shutil
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    shutil.rmtree(c.app.state.jobs_dir / jid / "cache")
    c.app.state.store.get(jid).pc = None
    body = {"zones": [rect("a", "A", "car", 0.1, 0.1, 0.5, 0.5)]}
    assert c.put(f"/api/jobs/{jid}/zones", json=body).status_code == 409


def test_nan_literal_zone_422(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    raw = (b'{"zones":[{"zone_id":"a","name":"A","kind":"car",'
           b'"polygon":[[0,0],[0.5,NaN],[1,1]]}]}')
    r = c.put(f"/api/jobs/{jid}/zones", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_rescore_without_render_removes_stale_video(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    c.post(f"/api/jobs/{jid}/score", json={"render_video": True})
    wait_state_video = time.time() + 60
    while c.get(f"/api/jobs/{jid}").json()["video_state"] == "rendering" and time.time() < wait_state_video:
        time.sleep(0.1)
    assert c.get(f"/api/jobs/{jid}").json()["video_state"] == "ready"
    assert c.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 200
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    assert c.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 404


def _roles_of(c, jid):
    return {p["person_id"]: p["role"] for p in c.get(f"/api/jobs/{jid}/people").json()}


def test_put_roles_replaces_so_full_set_applies(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    a, b = (p["person_id"] for p in c.get(f"/api/jobs/{jid}/people").json())
    c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(a): "staff"}})
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    c.put(f"/api/jobs/{jid}/roles", json={"overrides": {str(a): "staff", str(b): "staff"}})
    assert c.post(f"/api/jobs/{jid}/score", json={"render_video": False}).status_code == 200
    assert _roles_of(c, jid) == {a: "staff", b: "staff"}
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert {p["person_id"]: p["role"] for p in res["people"]} == {a: "staff", b: "staff"}
    assert res["store"]["visitors"] == 0


SKIP_WARNING = "Roles skipped: everyone counted as a customer"


def test_skip_roles_makes_everyone_customer_and_warns(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    a = c.get(f"/api/jobs/{jid}/people").json()[0]["person_id"]
    imported = ('{"persons": [{"person_id": %d, "role": "staff"}]}' % a).encode()
    c.post(f"/api/jobs/{jid}/roles/import", files={"file": ("roles.json", imported, "application/json")})
    assert c.put(f"/api/jobs/{jid}/roles", json={"overrides": {}, "skip": True}).status_code == 200
    assert c.get(f"/api/jobs/{jid}").json()["roles_skipped"] is True
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert res["store"]["staff"] == 0 and res["store"]["visitors"] == 2
    assert all(p["role"] == "customer" for p in res["people"])
    assert SKIP_WARNING in res["warnings"]
    assert SKIP_WARNING in c.get(f"/api/jobs/{jid}").json()["warnings"]
    assert SKIP_WARNING in c.get(f"/api/jobs/{jid}/files/report.json").text
    # any non-skip PUT clears it; imported role applies again
    c.put(f"/api/jobs/{jid}/roles", json={"overrides": {}})
    assert c.get(f"/api/jobs/{jid}").json()["roles_skipped"] is False
    c.post(f"/api/jobs/{jid}/score", json={"render_video": False})
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert res["store"]["staff"] == 1 and SKIP_WARNING not in res["warnings"]


def test_floor_calibration_api(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    doc = c.get(f"/api/jobs/{jid}/zones").json()
    floor = {"img": [[0.1, 0.9], [0.9, 0.9], [0.8, 0.5], [0.2, 0.5]],
             "world": [[0, 0], [6, 0], [6, 8], [0, 8]]}
    bad = c.put(f"/api/jobs/{jid}/zones", json={**doc, "floor": {**floor, "img": floor["img"][:3]}})
    assert bad.status_code == 422 and "floor" in bad.json()["error"]
    ok = c.put(f"/api/jobs/{jid}/zones", json={**doc, "floor": floor})
    assert ok.status_code == 200 and ok.json()["floor"] == floor
    assert c.get(f"/api/jobs/{jid}/zones").json()["floor"] == floor
    assert c.post(f"/api/jobs/{jid}/score", json={"render_video": False}).status_code == 200
    assert c.get(f"/api/jobs/{jid}/files/heatmap_topdown.png").status_code == 200
    assert "floor" not in c.put(f"/api/jobs/{jid}/zones", json=doc).json()


def test_sales_upload_api(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    zs = [rect("car-1", "Car: Hatch", "car", 0.3, 0.3, 0.7, 0.95)]
    assert c.put(f"/api/jobs/{jid}/zones", json={"zones": zs}).status_code == 200
    ok = c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", b"car_model,period,sold\nhatch,q1,4\nGhost,q1,1\n", "text/csv")})
    assert ok.status_code == 200 and ok.json()["rows"] == 2 and ok.json()["rescored"] is False
    assert c.post(f"/api/jobs/{jid}/score", json={"render_video": False}).status_code == 200
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert res["per_car"][0]["sold"] == 4
    assert any("Ghost" in w for w in res["warnings"])
    # upload after scoring re-scores without detection
    n = c.engine.process_calls
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", b"car_model,period,sold\nHatch,q1,2\n", "text/csv")})
    assert r.status_code == 200 and r.json()["rescored"] is True
    assert c.engine.process_calls == n
    assert c.get(f"/api/jobs/{jid}/results").json()["per_car"][0]["sold"] == 2
    assert "attention_to_sales" in c.get(f"/api/jobs/{jid}/export.csv").text


def test_sales_upload_rejections(make_client, video):
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "header" in r.json()["error"]
    big = b"car_model,period,sold\n" + b"A,x,1\n" * 300000
    assert c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", big, "text/csv")}).status_code in (413, 422)
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", b"car_model,period,sold\nA,x,1\n", "text/csv")},
               headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert c.post("/api/jobs/" + "0" * 32 + "/sales", files={"file": ("s.csv", b"x", "text/csv")}).status_code == 404


def test_sales_rescore_failure_is_reported(make_client, video, monkeypatch):
    from pulse.app import main as app_main
    c = make_client()
    jid = upload(c, video)
    wait_state(c, jid, {"ready_for_zones"})
    assert c.post(f"/api/jobs/{jid}/score", json={"render_video": False}).status_code == 200

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(app_main.pipeline, "score", boom)
    r = c.post(f"/api/jobs/{jid}/sales", files={"file": ("s.csv", b"car_model,period,sold\nA,x,1\n", "text/csv")})
    assert r.status_code == 500 and "Scoring failed" in r.json()["error"]
