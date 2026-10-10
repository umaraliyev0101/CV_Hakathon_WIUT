"""qa_assets: every file under assets/ (and every asset the configs reference) has a MANIFEST.csv row
with a source and a licence; every manifest row points at an existing file."""
import csv
import glob
import json
import sys

from sim.qa.common import REPO, load_qa, report

META_SUFFIXES = (".md", ".csv")


def referenced_assets():
    out = set()
    cars = json.loads((REPO / "sim" / "configs" / "assets_cars.json").read_text())["cars"]
    by_model = {c["model"]: c["file"] for c in cars}
    for path in sorted(glob.glob(str(REPO / "sim" / "configs" / "scene_*.json"))):
        cfg = json.loads(open(path).read())
        for p in cfg["persons"]:
            if p.get("character"):
                out.add(f"assets/people/characters/{p['character']}/Export/{p['character']}.fbx")
        for c in cfg["cars"]:
            if c.get("model"):
                out.add(by_model[c["model"]])
    out.add("assets/people/anims/rocketbox_actions_retargeted.blend")
    return out


def run(_clip, qa):
    q = qa["assets"]
    fails, info = [], []
    man = REPO / q["manifest"]
    if not man.exists():
        return [f"missing {q['manifest']}"], info
    rows = list(csv.DictReader(man.open()))
    for col in q["required_columns"]:
        if rows and col not in rows[0]:
            fails.append(f"manifest has no column {col}")
    listed = {}
    for r in rows:
        f = r["file"].strip()
        listed[f] = r
        if not r.get("licence", "").strip():
            fails.append(f"{f}: no licence")
        if not r.get("source_url", "").strip():
            fails.append(f"{f}: no source_url")
        if not (REPO / f).exists():
            fails.append(f"{f}: listed but missing on disk")
    on_disk = sorted(str(p.relative_to(REPO)) for p in (REPO / "assets").rglob("*")
                     if p.is_file() and p.suffix.lower() not in META_SUFFIXES)
    for f in on_disk:
        if f not in listed:
            fails.append(f"{f}: on disk but not in MANIFEST.csv")
    for f in sorted(referenced_assets()):
        if f not in listed:
            fails.append(f"{f}: referenced by a config but not in MANIFEST.csv")
    lic = {}
    for r in rows:
        lic[r["licence"].strip()] = lic.get(r["licence"].strip(), 0) + 1
    info.append(f"{len(rows)} manifest rows, {len(on_disk)} asset files on disk, "
                f"{len(referenced_assets())} referenced by configs; licences: {lic}")
    return fails, info


def main(argv=sys.argv):
    fails, info = run(None, load_qa())
    return report("qa_assets", fails, info)


if __name__ == "__main__":
    sys.exit(main())
