import hashlib, http.server, json, threading, zipfile
from pathlib import Path
import pytest
from showroom_coach import models as M
from showroom_coach.cli import main


@pytest.fixture
def http_dir(tmp_path):
    root = tmp_path / "srv"; root.mkdir()
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_ensure_model_downloads_and_hashes(http_dir, tmp_path, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    p = M.ensure_model("fake")
    assert p.read_bytes() == b"fake-onnx"
    man = json.loads(M.write_manifest().read_text())
    entry = [e for e in man["models"] if e["name"] == "fake"][0]
    assert entry["sha256"] == hashlib.sha256(b"fake-onnx").hexdigest()


def test_ensure_model_extracts_zip_member(http_dir, tmp_path, monkeypatch):
    root, base = http_dir
    with zipfile.ZipFile(root / "a.zip", "w") as z:
        z.writestr("dir/end2end.onnx", b"inner")
        z.writestr("dir/readme.txt", b"x")
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "z", M.ModelSpec("z", f"{base}/a.zip", "z.onnx", "Apache-2.0", "none", "x", archive_member="end2end.onnx"))
    assert M.ensure_model("z").read_bytes() == b"inner"


def test_ensure_model_raises_after_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    monkeypatch.setitem(M.MODELS, "bad", M.ModelSpec("bad", "http://127.0.0.1:9/nope.onnx", "n.onnx", "MIT", "none", "x"))
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("bad", retries=2, backoff=0)


# ---- behaviours the brief asks for beyond the three tests above ----

@pytest.fixture
def models_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "models"))
    return tmp_path / "models"


def test_zip_keeps_only_the_onnx_member(http_dir, models_env, monkeypatch):
    root, base = http_dir
    with zipfile.ZipFile(root / "a.zip", "w") as z:
        z.writestr("dir/end2end.onnx", b"inner")
        z.writestr("dir/other.onnx", b"nope")
    monkeypatch.setitem(M.MODELS, "z", M.ModelSpec("z", f"{base}/a.zip", "z.onnx", "Apache-2.0", "none", "x", archive_member="end2end.onnx"))
    M.ensure_model("z")
    assert sorted(p.name for p in (models_env / "z").iterdir()) == ["z.onnx"]


def test_zip_without_member_fails_cleanly(http_dir, models_env, monkeypatch):
    root, base = http_dir
    with zipfile.ZipFile(root / "a.zip", "w") as z:
        z.writestr("dir/readme.txt", b"x")
    monkeypatch.setitem(M.MODELS, "z", M.ModelSpec("z", f"{base}/a.zip", "z.onnx", "Apache-2.0", "none", "x", archive_member="end2end.onnx"))
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("z", retries=1, backoff=0)
    assert not list((models_env / "z").iterdir())  # no partial files left behind


def test_sha256_mismatch_is_rejected(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x", sha256="0" * 64))
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("fake", retries=1, backoff=0)
    assert not list((models_env / "fake").iterdir())


def test_sha256_match_is_accepted(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    good = hashlib.sha256(b"fake-onnx").hexdigest()
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x", sha256=good))
    assert M.ensure_model("fake").read_bytes() == b"fake-onnx"


def test_existing_file_is_not_downloaded_again(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    first = M.ensure_model("fake")
    (root / "m.onnx").unlink()  # the server would now answer 404
    assert M.ensure_model("fake") == first


def test_retry_then_success(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    real, calls = M._download, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("boom")
        return real(*a, **k)

    monkeypatch.setattr(M, "_download", flaky)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    assert M.ensure_model("fake", retries=3).read_bytes() == b"fake-onnx"
    assert len(calls) == 3


def test_free_disk_guard_stops_without_retrying(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    monkeypatch.setattr(M, "MIN_FREE_BYTES", 10**18)
    with pytest.raises(M.ModelUnavailable, match="free disk"):
        M.ensure_model("fake", retries=3, backoff=0)
    assert not (models_env / "fake" / "m.onnx").exists()


def test_unknown_model_name():
    with pytest.raises(ValueError):
        M.ensure_model("no-such-model")


def test_registry_matches_brief():
    expected = {"yolox_nano", "yolox_tiny", "yolox_s", "rtmpose_s", "dinov2_small", "rfdetr_nano"}
    assert set(M.MODELS) == expected
    for spec in M.MODELS.values():
        assert spec.url.startswith("https://") and spec.licence == "Apache-2.0"
        assert spec.trained_on and spec.source_page and spec.filename.endswith(".onnx")
        assert spec.sha256 and len(spec.sha256) == 64 and set(spec.sha256) <= set("0123456789abcdef")
    assert M.MODELS["rtmpose_s"].archive_member == "end2end.onnx"
    assert M.MODELS["rtmpose_s"].url.endswith(".zip")


def test_models_dir_env_and_default(monkeypatch, tmp_path):
    from showroom_coach.config import repo_root
    monkeypatch.delenv("SHOWROOM_MODELS", raising=False)
    assert M.models_dir() == repo_root() / "models"
    monkeypatch.setenv("SHOWROOM_MODELS", str(tmp_path / "m"))
    assert M.models_dir() == tmp_path / "m"


def test_manifest_lists_every_registry_entry(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "src", note="a note"))
    M.ensure_model("fake")
    man = json.loads(M.write_manifest().read_text(encoding="utf-8"))
    by_name = {e["name"]: e for e in man["models"]}
    # one row per registry entry, in registry order, whether or not the file is on disk
    assert [e["name"] for e in man["models"]] == list(M.MODELS)
    assert by_name["fake"] == {"name": "fake", "file": "fake/m.onnx", "url": f"{base}/m.onnx",
                               "sha256": hashlib.sha256(b"fake-onnx").hexdigest(), "bytes": 9,
                               "licence": "MIT", "trained_on": "none", "source_page": "src", "note": "a note"}
    for name, spec in M.MODELS.items():
        if name == "fake":
            continue
        e = by_name[name]
        assert e["present"] is False and e["bytes"] is None
        assert e["sha256"] == spec.sha256  # the pinned hash stands in for the missing file
        assert e["file"] == f"{name}/{spec.filename}" and e["url"] == spec.url
        assert e["licence"] == spec.licence and e["trained_on"] == spec.trained_on


def test_manifest_with_nothing_on_disk_still_lists_all_rows(models_env):
    man = json.loads(M.write_manifest().read_text(encoding="utf-8"))
    assert [e["name"] for e in man["models"]] == list(M.MODELS)
    assert all(e["present"] is False and e["bytes"] is None for e in man["models"])


def test_cli_fetch_models(http_dir, models_env, monkeypatch, capsys):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    monkeypatch.setattr(M, "MODELS", {"fake": M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x")})
    assert main(["fetch-models", "fake"]) == 0
    assert (models_env / "fake" / "m.onnx").read_bytes() == b"fake-onnx"
    assert (models_env / "MANIFEST.json").is_file()
    assert "fake" in capsys.readouterr().out


def test_cli_fetch_models_unknown_name(models_env, capsys):
    assert main(["fetch-models", "no-such-model"]) == 2
    assert "no-such-model" in capsys.readouterr().err


def test_cli_fetch_models_reports_failure(models_env, monkeypatch, capsys):
    def refuse(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(M, "MODELS", {"bad": M.ModelSpec("bad", "http://127.0.0.1:9/x.onnx", "x.onnx", "MIT", "none", "x")})
    monkeypatch.setattr(M, "_download", refuse)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    assert main(["fetch-models", "bad"]) == 1
    assert "bad" in capsys.readouterr().err


def test_cli_partial_fetch_keeps_every_manifest_row(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "a.onnx").write_bytes(b"aaa")
    monkeypatch.setattr(M, "MODELS", {
        "a": M.ModelSpec("a", f"{base}/a.onnx", "a.onnx", "MIT", "none", "x"),
        "b": M.ModelSpec("b", f"{base}/b.onnx", "b.onnx", "MIT", "none", "x", sha256="1" * 64),
    })
    assert main(["fetch-models", "a"]) == 0
    man = json.loads((models_env / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = {e["name"]: e for e in man["models"]}
    assert set(rows) == {"a", "b"}
    assert rows["a"]["bytes"] == 3 and "present" not in rows["a"]
    assert rows["b"] == {"name": "b", "file": "b/b.onnx", "url": f"{base}/b.onnx", "sha256": "1" * 64,
                         "bytes": None, "licence": "MIT", "trained_on": "none", "source_page": "x",
                         "present": False}


def test_cli_failed_fetch_still_writes_all_rows(models_env, monkeypatch, capsys):
    def refuse(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(M, "MODELS", {"bad": M.ModelSpec("bad", "http://127.0.0.1:9/x.onnx", "x.onnx", "MIT", "none", "x")})
    monkeypatch.setattr(M, "_download", refuse)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    assert main(["fetch-models", "bad"]) == 1
    man = json.loads((models_env / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [(e["name"], e["present"], e["sha256"], e["bytes"]) for e in man["models"]] == [("bad", False, None, None)]


# ---- truncated downloads ----

@pytest.fixture
def short_server():
    """Promises 1000 bytes but sends only 10, then closes the connection."""
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"0123456789")

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", hits
    srv.shutdown()
    srv.server_close()


def test_truncated_download_is_rejected_without_pinned_hash(short_server, models_env, monkeypatch):
    base, hits = short_server
    monkeypatch.setitem(M.MODELS, "t", M.ModelSpec("t", f"{base}/t.onnx", "t.onnx", "MIT", "none", "x"))  # sha256=None
    with pytest.raises(M.ModelUnavailable, match="Content-Length"):
        M.ensure_model("t", retries=1, backoff=0)
    assert not list((models_env / "t").iterdir())  # neither the model nor a .part file survives


def test_truncated_download_is_retried(short_server, models_env, monkeypatch):
    base, hits = short_server
    monkeypatch.setitem(M.MODELS, "t", M.ModelSpec("t", f"{base}/t.onnx", "t.onnx", "MIT", "none", "x"))
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("t", retries=3, backoff=0)
    assert len(hits) == 3


def test_truncated_archive_is_rejected(short_server, models_env, monkeypatch):
    base, hits = short_server
    monkeypatch.setitem(M.MODELS, "t", M.ModelSpec("t", f"{base}/t.zip", "t.onnx", "MIT", "none", "x", archive_member="end2end.onnx"))
    with pytest.raises(M.ModelUnavailable, match="Content-Length"):
        M.ensure_model("t", retries=1, backoff=0)
    assert not list((models_env / "t").iterdir())


def test_full_download_with_content_length_still_accepted(http_dir, models_env, monkeypatch):
    root, base = http_dir  # SimpleHTTPRequestHandler sends Content-Length
    (root / "m.onnx").write_bytes(b"x" * 3_000_000)  # several 1 MiB chunks
    monkeypatch.setitem(M.MODELS, "big", M.ModelSpec("big", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x"))
    assert M.ensure_model("big").stat().st_size == 3_000_000


# ---- a file the user placed is only replaced by a verified download ----

def test_failed_download_keeps_existing_file_with_wrong_hash(http_dir, models_env, monkeypatch):
    root, base = http_dir  # server has no m.onnx, so the download answers 404
    mine = models_env / "fake" / "m.onnx"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"user-placed")
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x", sha256="2" * 64))
    with pytest.raises(M.ModelUnavailable):
        M.ensure_model("fake", retries=1, backoff=0)
    assert mine.read_bytes() == b"user-placed"


def test_successful_download_replaces_file_with_wrong_hash(http_dir, models_env, monkeypatch):
    root, base = http_dir
    (root / "m.onnx").write_bytes(b"fake-onnx")
    mine = models_env / "fake" / "m.onnx"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"stale")
    good = hashlib.sha256(b"fake-onnx").hexdigest()
    monkeypatch.setitem(M.MODELS, "fake", M.ModelSpec("fake", f"{base}/m.onnx", "m.onnx", "MIT", "none", "x", sha256=good))
    assert M.ensure_model("fake").read_bytes() == b"fake-onnx"
