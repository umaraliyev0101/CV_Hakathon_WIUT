"""Static checks on the Docker and deploy files (nothing is built or executed)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DEPLOY = ROOT / "scripts" / "deploy.sh"


def _instructions():
    return [ln.strip() for ln in DOCKERFILE.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_dockerfile_required_instructions():
    ins = _instructions()
    assert "FROM python:3.12-slim" in ins
    assert any(i.startswith("USER ") and i != "USER root" for i in ins)
    assert "EXPOSE 8000" in ins
    assert any(i.startswith("HEALTHCHECK") for i in ins)
    assert "/api/health" in DOCKERFILE
    assert "pip install --no-cache-dir" in DOCKERFILE
    cmd = next(i for i in ins if i.startswith("CMD ["))
    assert "0.0.0.0" in cmd and "--ws-max-size" in cmd and "4194304" in cmd
    assert "PULSE_ALLOWED_HOSTS" in DOCKERFILE
    assert any(i.startswith("VOLUME") and "/app/models" in i and "/app/jobs" in i for i in ins)


def test_dockerfile_does_not_bake_models():
    for i in _instructions():
        if i.startswith(("COPY", "ADD")):
            assert "models" not in i, i
        assert "ensure_model" not in i and "curl" not in i and "wget" not in i
    ignore = (ROOT / ".dockerignore").read_text().split()
    for entry in (".venv", "data", "models", "runs", "jobs", ".git", ".superpowers"):
        assert entry in ignore


def test_deploy_script_header_and_guards():
    text = DEPLOY.read_text()
    lines = text.splitlines()
    assert lines[0].startswith("#!") and lines[1] == "# NOT RUN — local-only hackathon rule"
    assert "--i-know-this-deploys" in text
    assert '"$(uname -s)" != "Linux"' in text
    # guards come before the first state-changing command
    assert text.index("uname -s") < text.index("useradd") < text.index("systemctl")
    for needle in ("Restart=on-failure", "NoNewPrivileges=true", "ProtectSystem=strict",
                   "ReadWritePaths=", "Upgrade", "client_max_body_size", "24 h", "Signage"):
        assert needle in text, needle


def test_deploy_script_syntax():
    assert subprocess.run(["bash", "-n", str(DEPLOY)]).returncode == 0
