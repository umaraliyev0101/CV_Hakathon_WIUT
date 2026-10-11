"""The web app: job API with background perception, staff confirm, instant re-score, files, static pages and /ws/live.

Everything runs on the fake perception stack and the synthetic `meet_video` (pid 1 = the blue stander = staff), with a
`tmp_path` jobs root. The first six tests are the Task 4.1 brief's, verbatim.
"""
import itertools
import json
import threading
import time

import cv2
import pytest
from fastapi.testclient import TestClient

from showroom_coach.app.main import create_app
from showroom_coach.perception.stack import PerceptionStack, fake_stack
from showroom_coach.sources import iter_frames


@pytest.fixture
def client(tmp_path):
    app = create_app(stack_factory=fake_stack, jobs_root=tmp_path / "jobs")
    with TestClient(app) as c:
        yield c


def wait_status(c, jid, want, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get(f"/api/jobs/{jid}").json()
        if s["status"] in want:
            return s
        time.sleep(0.2)
    raise AssertionError(f"timeout, last={s}")


def upload(c, path, preset="short"):
    with open(path, "rb") as f:
        r = c.post("/api/jobs", files={"file": (path.name, f, "video/x-msvideo")}, data={"preset": preset})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_upload_confirm_results(client, meet_video):
    jid = upload(client, meet_video)
    s = wait_status(client, jid, {"needs_confirm", "failed"})
    assert s["status"] == "needs_confirm"
    people = client.get(f"/api/jobs/{jid}/people").json()["people"]
    assert len(people) == 2 and all(p["suggestion"]["reasons"] for p in people)
    assert client.get(people[0]["thumb_url"]).status_code == 200
    staff = max(people, key=lambda p: p["t_last"] - p["t_first"])["pid"]
    r = client.post(f"/api/jobs/{jid}/roles", json={"roles": {str(staff): "staff"}, "names": {str(staff): "Aziz"}})
    assert r.status_code == 200
    res = r.json()
    assert res["scorecard"]["staff"][0]["name"] == "Aziz"
    wait_status(client, jid, {"done"})
    assert client.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/events.csv").status_code == 200


def test_results_list_the_contact_sheets_once_the_video_is_ready(client, meet_video, tmp_path):
    import re
    jid = upload(client, meet_video)
    wait_status(client, jid, {"needs_confirm"})
    assert client.get(f"/api/jobs/{jid}/results").json()["contact_sheets"] == []          # nothing rendered yet
    r = client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff"}})
    assert r.json()["contact_sheets"] == []                                             # still drawing
    wait_status(client, jid, {"done"})
    sheets = client.get(f"/api/jobs/{jid}/results").json()["contact_sheets"]
    on_disk = sorted(p.name for p in (tmp_path / "jobs" / jid / "contact_sheets").glob("*.jpg"))
    assert sheets and len(sheets) == len(on_disk)
    pattern = re.compile(rf"/api/jobs/{jid}/files/contact_sheets/(\d{{2}})_(greet|engagement|wait_alert|walkout)_(\d+)s"
                         r"\.jpg\?v=\d+")
    for sheet, name in zip(sheets, on_disk):
        assert set(sheet) == {"url", "type", "t", "caption"}
        m = pattern.fullmatch(sheet["url"])
        assert m and sheet["url"].split("/")[-1].split("?")[0] == name                  # in file (= time) order
        assert sheet["type"] == m.group(2) and sheet["t"] == float(m.group(3))
        assert isinstance(sheet["caption"], str) and sheet["caption"]
        got = client.get(sheet["url"])
        assert got.status_code == 200 and got.headers["content-type"] == "image/jpeg"


def test_rescore_does_not_redetect(client, meet_video, monkeypatch):
    jid = upload(client, meet_video)
    wait_status(client, jid, {"needs_confirm"})
    import showroom_coach.perception.pipeline as P
    monkeypatch.setattr(P, "run_perception", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-detect")))
    t0 = time.time()
    r1 = client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff"}})
    r2 = client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "customer", "2": "staff"}, "preset": "dealership"})
    assert r1.status_code == r2.status_code == 200
    assert time.time() - t0 < 5
    assert r2.json()["roles"]["2"] == "staff" and r2.json()["preset"] == "dealership"


def test_upload_garbage_fails_cleanly(client, tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_text("definitely not a video")
    jid = upload(client, bad)
    s = wait_status(client, jid, {"failed", "needs_confirm"})
    assert s["status"] == "failed" and "video" in s["error"].lower()
    assert client.get("/api/presets").status_code == 200        # server still alive


def test_rejects_wrong_extension_and_unknown_preset(client, tmp_path):
    p = tmp_path / "x.txt"; p.write_text("x")
    with open(p, "rb") as f:
        assert client.post("/api/jobs", files={"file": ("x.txt", f)}, data={"preset": "short"}).status_code == 415
    with open(p, "rb") as f:
        assert client.post("/api/jobs", files={"file": ("x.mp4", f)}, data={"preset": "zzz"}).status_code == 400


def test_file_whitelist(client, meet_video):
    jid = upload(client, meet_video)
    wait_status(client, jid, {"needs_confirm"})
    assert client.get(f"/api/jobs/{jid}/files/../../pyproject.toml").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/cache/perception.npz").status_code == 404


def test_cleanup_removes_old_jobs(tmp_path):
    import os
    from showroom_coach.app.jobs import JobManager
    old = tmp_path / "jobs" / "old"; old.mkdir(parents=True)
    os.utime(old, (time.time() - 30 * 3600,) * 2)
    assert JobManager(tmp_path / "jobs", fake_stack).cleanup() == 1 and not old.exists()


# --------------------------------------------------------------------------- added: pages (Tasks 4.2 and 5.2)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ZRU-1115" in r.text and "app.js" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_live_page_served(client):
    r = client.get("/live")
    assert r.status_code == 200 and "live.js" in r.text and "ZRU-1115" in r.text


# --------------------------------------------------------------------------- added: job lifecycle and API shapes
#
# These use `small_meet`, the meet_video scene at half size and 10 fps (3-4x cheaper to decode and render), and the
# read-only checks share one module-scoped job that has reached needs_confirm.


@pytest.fixture(scope="module")
def small_meet(tmp_path_factory):
    """meet_video at 640x360, 10 fps: blue staff stands at x 300 all along; red customer walks in, waits, leaves."""
    from tests.conftest import BLUE, RED, _customer_x1, _write_video

    def customer(t):
        x1 = _customer_x1(t)
        return None if x1 is None else (x1 / 2, 125)

    movers = [{"color": BLUE, "w": 40, "h": 110, "pos": lambda t: (300, 125)},
              {"color": RED, "w": 40, "h": 110, "pos": customer}]
    return _write_video(tmp_path_factory.mktemp("small") / "small.avi", 20.0, 10, (640, 360), movers)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """One app for the tests below that neither monkeypatch nor need a fresh server: (client, jobs root)."""
    root = tmp_path_factory.mktemp("shared") / "jobs"
    with TestClient(create_app(stack_factory=fake_stack, jobs_root=root)) as c:
        yield c, root


@pytest.fixture(scope="module")
def confirmed(shared, small_meet):
    """A job uploaded under a non-ASCII, path-like client name and analysed to needs_confirm: (client, root, id)."""
    c, root = shared
    with open(small_meet, "rb") as f:
        r = c.post("/api/jobs", files={"file": ("..\\..\\Шоурум клип.AVI", f)}, data={"preset": "short"})
    jid = r.json()["job_id"]
    assert wait_status(c, jid, {"needs_confirm", "failed"})["status"] == "needs_confirm"
    return c, root, jid


def _gated_factory(gate, at=12):
    """A fake stack whose detector blocks on `gate` at its `at`-th call, so a job can be watched mid-perception
    (perception reports progress every 10 sampled frames, so frames 0 and 10 have been reported by then)."""
    def factory():
        base, calls = fake_stack(), itertools.count()

        def detect(frame):
            if next(calls) == at:
                gate.wait(30)
            return base.detect(frame)
        return PerceptionStack("gated", detect, base.pose, base.embed)
    return factory


def test_progress_conflicts_and_one_perception_at_a_time(tmp_path, small_meet):
    gate = threading.Event()
    app = create_app(stack_factory=_gated_factory(gate), jobs_root=tmp_path / "jobs")
    with TestClient(app) as c:
        try:
            first = upload(c, small_meet)
            t0 = time.time()
            while (s := c.get(f"/api/jobs/{first}").json())["progress"] <= 0 and time.time() - t0 < 30:
                time.sleep(0.05)
            assert s["status"] == "perceiving" and 0 < s["progress"] < 0.9
            assert "detecting people" in s["message"] and "of 20 s" in s["message"]
            second = upload(c, small_meet)
            time.sleep(0.5)
            assert c.get(f"/api/jobs/{second}").json()["status"] == "queued"      # waits for the only perception slot
            bad = tmp_path / "bad.mp4"
            bad.write_text("not a video")
            garbage = upload(c, bad)                       # probed before queueing: fails while job 1 still runs
            s = wait_status(c, garbage, {"failed"}, timeout=10)
            assert "cannot read video: bad.mp4" == s["error"] and s["message"] == "failed"
            assert c.get(f"/api/jobs/{first}").json()["status"] == "perceiving"
            for path in ("people", "results"):
                assert c.get(f"/api/jobs/{first}/{path}").status_code == 409
            assert c.post(f"/api/jobs/{first}/roles", json={"roles": {"1": "staff"}}).status_code == 409
        finally:
            gate.set()
        for jid in (first, second):
            s = wait_status(c, jid, {"needs_confirm", "failed"})
            assert s["status"] == "needs_confirm" and s["progress"] == 1.0 and s["video_status"] == "none"


def test_job_status_shape_and_unknown_ids(confirmed):
    c, _, jid = confirmed
    s = c.get(f"/api/jobs/{jid}").json()
    assert len(jid) == 12 and int(jid, 16) >= 0
    assert set(s) == {"id", "filename", "preset", "status", "progress", "message", "error", "warnings",
                      "video_status"}
    assert s["id"] == jid and s["filename"] == "Шоурум клип.AVI" and s["preset"] == "short" and s["error"] is None
    for url in ("/api/jobs/000000000000", "/api/jobs/zzz/people", "/api/jobs/000000000000/results",
                "/api/jobs/000000000000/files/events.csv"):
        assert c.get(url).status_code == 404, url
    assert c.post("/api/jobs/000000000000/roles", json={}).status_code == 404


def test_initial_results_people_and_exports_at_needs_confirm(confirmed):
    c, _, jid = confirmed
    people = c.get(f"/api/jobs/{jid}/people").json()["people"]
    assert [p["pid"] for p in people] == [1, 2]
    for p in people:
        assert set(p) == {"pid", "thumb_url", "suggestion", "role", "name", "t_first", "t_last"}
        assert p["thumb_url"] == f"/api/jobs/{jid}/files/people/{p['pid']}.jpg"
        assert set(p["suggestion"]) == {"role", "confidence", "reasons"} and p["role"] in ("staff", "customer")
        assert c.get(p["thumb_url"]).headers["content-type"] == "image/jpeg"
    r = c.get(f"/api/jobs/{jid}/results")
    assert r.status_code == 200
    res = r.json()
    assert [p["thumb"] for p in res["persons"]] == ["1.jpg", "2.jpg"]       # person thumbnails, not track crops
    assert res["video_status"] == "none" and res["video_url"] is None
    assert res["timeline_url"] == f"/api/jobs/{jid}/files/timeline.png"
    assert res["export"] == {"json": f"/api/jobs/{jid}/files/results.json",
                             "events_csv": f"/api/jobs/{jid}/files/events.csv",
                             "scorecard_csv": f"/api/jobs/{jid}/files/scorecard.csv"}
    for url in [res["timeline_url"], *res["export"].values()]:
        assert c.get(url).status_code == 200, url
    assert json.loads(c.get(res["export"]["json"]).content)["persons"] == res["persons"]
    assert c.get(f"/api/jobs/{jid}/files/annotated.mp4").status_code == 404       # nothing rendered yet


def test_upload_is_stored_under_an_ascii_name_and_only_outputs_are_served(confirmed):
    c, root, jid = confirmed
    assert sorted(p.name for p in (root / jid).glob("upload*")) == ["upload.avi"]
    for name in ("upload.avi", "cache", "people/..%2F..%2Fupload.avi", "people%2F1.jpg%2F..%2F..%2Fupload.avi",
                 "people/1.png", "contact_sheets/../upload.avi", "annotated-x.mp4", "people/1.jpg/"):
        assert c.get(f"/api/jobs/{jid}/files/{name}").status_code == 404, name


def test_presets(client):
    p = client.get("/api/presets").json()
    assert set(p) == {"dealership", "short"}
    assert p["dealership"]["label"].startswith("Dealership")
    assert p["short"]["thresholds"]["greet_time"] == 2.0 and p["dealership"]["thresholds"]["engage_time"] == 120.0


def test_roles_merge_names_and_validation(shared, small_meet):
    c, _ = shared
    jid = upload(c, small_meet)
    wait_status(c, jid, {"needs_confirm"})
    r = c.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff", "2": "customer"}, "names": {"1": "Aziz"}})
    assert r.json()["roles"] == {"1": "staff", "2": "customer"} and r.json()["names"] == {"1": "Aziz"}
    r = c.post(f"/api/jobs/{jid}/roles", json={"roles": {"2": "staff"}})            # pid 1 keeps its override
    assert r.json()["roles"] == {"1": "staff", "2": "staff"} and r.json()["names"] == {"1": "Aziz"}
    r = c.post(f"/api/jobs/{jid}/roles", json={"names": {"1": ""}})                  # "" removes a name
    assert r.status_code == 200 and r.json()["names"] == {}
    for body in ({"roles": {"1": "manager"}}, {"roles": {"one": "staff"}}, {"preset": "zzz"}, {"roles": ["x"]}):
        assert c.post(f"/api/jobs/{jid}/roles", json=body).status_code in (400, 422), body
    assert c.get(f"/api/jobs/{jid}/results").json()["roles"] == {"1": "staff", "2": "staff"}


def test_render_supersede_versioned_video_and_range(shared, small_meet):
    c, root = shared
    jid = upload(c, small_meet)
    wait_status(c, jid, {"needs_confirm"})
    r1 = c.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff"}})
    assert r1.json()["video_status"] == "rendering" and r1.json()["video_url"] is None
    assert c.get(f"/api/jobs/{jid}").json()["status"] == "rendering"
    c.post(f"/api/jobs/{jid}/roles", json={"names": {"1": "Aziz"}})                  # supersedes render 1
    s = wait_status(c, jid, {"done"})
    assert s["video_status"] == "ready" and s["error"] is None
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert res["video_status"] == "ready" and res["video_url"] == f"/api/jobs/{jid}/files/annotated-2.mp4"
    assert res["names"] == {"1": "Aziz"}
    job_dir = root / jid
    assert (job_dir / "annotated-2.mp4").is_file() and not (job_dir / "annotated-1.mp4").exists()
    assert c.get(f"/api/jobs/{jid}/files/annotated-1.mp4").status_code == 404
    full = c.get(res["video_url"])
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    part = c.get(res["video_url"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.content == full.content[:100]
    assert part.headers["content-range"] == f"bytes 0-99/{len(full.content)}"
    assert c.get(f"/api/jobs/{jid}/files/annotated.mp4").content == full.content      # alias of the latest
    sheets = [p.name for p in (job_dir / "contact_sheets").glob("*.jpg")]
    assert sheets and all(c.get(f"/api/jobs/{jid}/files/contact_sheets/{n}").status_code == 200 for n in sheets)


def test_render_failure_keeps_scores(client, small_meet, monkeypatch):
    import showroom_coach.app.jobs as jobs

    def boom(*a, **k):
        raise RuntimeError("encoder exploded")
    monkeypatch.setattr(jobs, "render_video", boom)
    jid = upload(client, small_meet)
    wait_status(client, jid, {"needs_confirm"})
    assert client.post(f"/api/jobs/{jid}/roles", json={"roles": {"1": "staff"}}).status_code == 200
    t0 = time.time()
    while (s := client.get(f"/api/jobs/{jid}").json())["video_status"] == "rendering" and time.time() - t0 < 30:
        time.sleep(0.1)
    assert s["status"] == "done" and s["video_status"] == "failed" and "encoder exploded" in s["error"]
    res = client.get(f"/api/jobs/{jid}/results").json()
    assert res["video_status"] == "failed" and res["video_url"] is None and res["scorecard"]["staff"]


def test_cut_off_video_gives_partial_results_with_a_warning(shared, small_meet, tmp_path):
    c, _ = shared
    data = small_meet.read_bytes()
    bad = tmp_path / "cut.avi"
    bad.write_bytes(data[: int(len(data) * 0.4)])        # header intact: probe passes, decoding stops at ~8 s
    jid = upload(c, bad)
    s = wait_status(c, jid, {"failed", "needs_confirm"})
    assert s["status"] == "needs_confirm" and s["error"] is None
    (warning,) = [w for w in s["warnings"] if w.startswith("video ended at")]
    res = c.get(f"/api/jobs/{jid}/results").json()
    assert 0 < res["duration"] < 0.9 * 20
    assert warning == (f"video ended at {res['duration']:.0f} s of an expected 20 s — the file may be damaged or have "
                       f"a longer audio track; results cover the first {res['duration']:.0f} s")
    assert res["warnings"].count(warning) == 1
    assert c.get(f"/api/jobs/{jid}/people").status_code == 200


def test_unreadable_video_still_fails_without_results(shared, tmp_path):
    c, _ = shared
    zero = tmp_path / "zero.avi"
    cv2.VideoWriter(str(zero), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)).release()   # a header, no frames
    jid = upload(c, zero)
    s = wait_status(c, jid, {"failed", "needs_confirm"})
    assert s["status"] == "failed" and s["error"] == "cannot read video: zero.avi" and s["message"] == "failed"
    assert c.get(f"/api/jobs/{jid}/people").status_code == 409
    assert c.get(f"/api/jobs/{jid}/results").status_code == 409


def test_trimmed_video_warns(client, small_meet, monkeypatch):
    import showroom_coach.config as config
    monkeypatch.setattr(config, "MAX_SECONDS", 5.0)
    jid = upload(client, small_meet)
    s = wait_status(client, jid, {"needs_confirm", "failed"})
    assert s["status"] == "needs_confirm" and s["warnings"] == ["video trimmed to the first 5 s"]
    res = client.get(f"/api/jobs/{jid}/results").json()
    assert res["duration"] == pytest.approx(5.0) and res["warnings"].count("video trimmed to the first 5 s") == 1


def test_upload_without_content_length_is_411(client, small_meet, tmp_path):
    """No declared length means no bound on what Starlette would spool to disk: refused before the body is read."""
    req = client.build_request("POST", "/api/jobs", files={"file": ("a.avi", small_meet.read_bytes())},
                               data={"preset": "short"})
    del req.headers["Content-Length"]
    assert client.send(req).status_code == 411

    def body():                                                   # a generator body is sent chunked
        yield b'--zz\r\nContent-Disposition: form-data; name="preset"\r\n\r\nshort\r\n--zz--\r\n'
    r = client.post("/api/jobs", content=body(), headers={"Content-Type": "multipart/form-data; boundary=zz"})
    assert r.status_code == 411
    r = client.post("/api/jobs", content=b"--zz--\r\n", headers={"Content-Type": "multipart/form-data; boundary=zz",
                                                                 "Transfer-Encoding": "chunked"})
    assert r.status_code == 411
    jobs_root = tmp_path / "jobs"
    assert not jobs_root.exists() or not any(jobs_root.iterdir())


def test_upload_over_the_cap_is_413_and_leaves_nothing(client, small_meet, tmp_path, monkeypatch):
    import showroom_coach.app.main as main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    with open(small_meet, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("big.avi", f)}, data={"preset": "short"})
    assert r.status_code == 413
    jobs_root = tmp_path / "jobs"
    assert not jobs_root.exists() or not any(jobs_root.iterdir())          # no job dir, no partial upload
    # A declared Content-Length over the cap is refused before the body is parsed (this one is not even multipart).
    monkeypatch.setattr(main, "MULTIPART_SLACK", 0)
    r = client.post("/api/jobs", content=b"x" * 5000, headers={"Content-Type": "multipart/form-data; boundary=zz"})
    assert r.status_code == 413


# --------------------------------------------------------------------------- added: /ws/live on the real app


def _jpeg(frame):
    return cv2.imencode(".jpg", frame)[1].tobytes()


def test_ws_live_file_fed(shared, meet_video):
    client, _ = shared
    with client.websocket_connect("/ws/live?preset=short") as ws:
        n = 0
        for _, t, frame in iter_frames(meet_video, sample_fps=5):
            ws.send_text(json.dumps({"type": "time", "t": t}))
            ws.send_bytes(_jpeg(frame))
            msg = ws.receive_json()
            n += 1
            assert msg["type"] == "overlay" and "people" in msg and "fps" in msg
            assert msg["t"] == pytest.approx(t)
        assert n == 100
        ws.send_bytes(b"garbage")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "summary"}))
        summary = ws.receive_json()
        assert summary["type"] == "summary" and summary["events"] and "scorecard" in summary


def test_ws_live_disconnect_and_reconnect(shared, meet_video):
    client, _ = shared
    frame = _jpeg(next(iter_frames(meet_video))[2])
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(frame)
        assert ws.receive_json()["type"] == "overlay"
    with client.websocket_connect("/ws/live") as ws:              # a new, working session after the drop
        ws.send_bytes(frame)
        first = ws.receive_json()
        assert first["type"] == "overlay" and first["people"]
        ws.send_text(json.dumps({"type": "summary"}))
        assert ws.receive_json()["type"] == "summary"
    assert client.get("/api/presets").status_code == 200
