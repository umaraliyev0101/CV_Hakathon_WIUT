# Showroom Pulse: WIUT Hackathon 2026 CV final (customer experience analytics)

Camera-based customer-experience statistics for car showrooms. It covers:
- footfall and occupancy;
- dwell and stop rate per car;
- customer journeys;
- unserved interest (a customer studies a car and no salesperson comes);
- recommendations with evidence.

The design is in `docs/superpowers/specs/2026-10-11-showroom-pulse-design.md`.
Progress and decisions go in `PROGRESS.md`. Read both before doing anything.

The sister project, Showroom Coach (teammate A), uses the same shared units (spec §4.2). Keep those
interfaces identical so the two can merge.

## Rules

- **Local only.** Never add a git remote, push, deploy, or create accounts.
- **Licences: permissive only.**
  - Banned: ultralytics, boxmot (AGPL), and re-ID weights trained on Market-1501, DukeMTMC or MSMT17.
  - Also banned: demographic, emotion and face-recognition models.
  - Non-commercial data lives only in `data/eval_only/` and is never committed or shipped.
  - Every model or dataset you add gets a row in the README licence table and `models/MANIFEST.json`.
- **Gitignored:** `data/`, `models/`, `runs/` and `.venv/`. The only files committed from them are
  `data/sources.csv`, `data/TODO_DOWNLOADS.md` and `models/MANIFEST.json`. Never commit videos or weights.
- **Disk:** keep the venv, models and data at ≤ 15 GB combined. Stop downloading if free disk drops below 20 GB.
- **Privacy:** no face recognition, no age, gender or emotion inference, no re-identification across visits.
  Every rendered frame or thumbnail blurs faces.
- **No questions.** Nobody is awake to answer.
  - When blocked, pick the safest option in the spec, log it under "Decisions" in `PROGRESS.md`, and continue.
- **Honest numbers only.** If a target is missed, report the measured value. Never tune on test clips.
- **Never weaken a gate.**
  - No skipping, `xfail`, `.only`, deleting or mocking out tests to make them pass.
  - No lowering targets or thresholds to turn a failure green.
  - If a threshold looks wrong, leave it, explain why in `PROGRESS.md`, and flag it for the morning.
- **Subagents:** run them one at a time, as `subagent-driven-development` does. Never launch more than
  10 at once. Follow the model and effort routing in `~/.claude/rules/`.
- **Downloads are untrusted.** Put each source in its own folder, and run any Python that reads them with `-I`.

## Setup (this machine: Apple M3 Pro, 18 GB, macOS)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv      # not the default python3 (3.14)
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
bash scripts/verify.sh                          # the full verification chain (spec §9)
```

Use imageio-ffmpeg in code, not the system ffmpeg, so the code stays portable to teammate A's Windows
machine. CPU speed numbers are mandatory. CoreML numbers are optional and reported separately.
