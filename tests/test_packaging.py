"""A non-editable install (the Dockerfile's `pip install .`) must carry the frontend and its demo data."""
import glob
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "showroom_coach"


def test_package_data_ships_every_frontend_file_including_the_demo_mock_folder():
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["setuptools"]["package-data"]["showroom_coach"]
    shipped = {p.replace("\\", "/") for pat in patterns for p in glob.glob(pat, root_dir=PKG, recursive=True)}
    static = {p.relative_to(PKG).as_posix() for p in (PKG / "app" / "static").rglob("*")
              if p.is_file() and p.suffix != ".mp4" and "__pycache__" not in p.parts}
    assert {"app/static/app.js", "app/static/index.html", "app/static/mock/results.json",
            "app/static/mock/thumbs/1.svg"} <= static                       # the files this test is about exist
    assert static - shipped == set(), "not in package-data: " + ", ".join(sorted(static - shipped))
    assert not any(p.endswith(".mp4") for p in shipped)                      # videos are never shipped
