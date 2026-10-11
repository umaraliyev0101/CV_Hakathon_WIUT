# Showroom Coach: WIUT Hackathon 2026 CV final (Jira YAK-44)

This is camera-based coaching for car-showroom salespeople: time to greet, waits while staff are free,
walk-outs and engagement time. The design is in `docs/superpowers/specs/2026-10-11-showroom-coach-design.md`.
Progress and decisions go in `PROGRESS.md`. Read both before doing anything.

## Rules
- **Local only.** Never add a git remote, push, deploy, or create accounts.
- **Commits:** no `Co-Authored-By` trailers. This matches the owner's AP-862 hackathon repo.
- **Licences: permissive only.**
  - Banned: ultralytics, boxmot (AGPL), and re-ID weights trained on Market-1501, DukeMTMC or MSMT17.
  - Non-commercial data lives only in `data/eval_only/` and is never committed or shipped.
  - Every model or dataset you add gets a row in the README licence table and `models/MANIFEST.json`.
- `data/`, `models/`, `runs/` and `.venv/` are gitignored. The only files committed from them are `data/sources.csv`,
  `data/TODO_DOWNLOADS.md` and `models/MANIFEST.json`. Never commit videos or weights.
- **Disk:** about 9 GB free. Keep the venv, models and data at ≤ 4 GB combined, and stop downloading if free disk drops below 3 GB.
- **Privacy:** never add face recognition. Every rendered frame or thumbnail blurs faces.
- **No questions.** Nobody is awake to answer. When blocked, pick the safest option in the spec, log it under "Decisions" in `PROGRESS.md`, and continue.
- **Honest numbers only.** If a target is missed, report the measured value.

## Setup (this machine: i5-10210U, 16 GB, no GPU, Windows 11)
```bash
"/c/Users/USER/AppData/Local/Programs/Python/Python312/python.exe" -m venv .venv   # not the default 3.14
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```
There is no ffmpeg on PATH; use `imageio-ffmpeg`. Docker isn't running, so don't depend on it.
