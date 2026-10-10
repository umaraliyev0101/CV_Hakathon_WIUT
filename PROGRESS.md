# Progress

## Status

Task 1 (skeleton, datatypes, config, verify.sh) done. See Milestones.

## Milestones

### M0 / Task 1: skeleton
- Added package `pulse` (src layout), datatypes, presets/config, minimal CLI, `scripts/verify.sh`,
  data records. Tests: config, datatypes, slow ffmpeg smoke marker.

## Decisions

- 2026-10-11: Pre-flight was not run by the owner; the agent created the repo, copied the spec/CLAUDE.md/prompt; `data/stock/` and `data/synthetic/share/` are empty.
- 2026-10-11: rtmlib is vendored (subset) instead of pip-installed, because its wheel requires `opencv-python` + `opencv-contrib-python`, which clash with `opencv-python-headless`, and its multiclass YOLOX path drops scores (ByteTrack needs them).
- 2026-10-11: `supervision` pinned `<0.31`: `ByteTrack` is deprecated since 0.28 and removed in 0.31.
- 2026-10-11: No `docs/journal.md` logging gate (optional pre-flight step not run); PROGRESS.md is the log.
- Owner instruction 2026-10-11 02:45: never ask questions; pick the recommended/spec-supported option and log it here.
- Dev/test split fixed at plan time: dev = 2018-03-11.16-35-01.16-40-00.school.G423, 2018-03-11.12-00-00.12-05-00.school.G420; test = 2018-03-12.11-05-01.11-10-01.school.G423, 2018-03-13.17-10-00.17-15-00.school.G421, 2018-03-09.10-40-00.10-45-00.school.G419, 2018-03-09.10-15-00.10-20-00.school.G299.
- MEVA: 6 indoor clips downloaded to data/meva/ (CC BY 4.0, Corona et al., WACV 2021); see data/meva/CLIPS.md.
- data/stock/ and data/synthetic/share/ empty at start; agent will look for a commercially licensed showroom clip by script (not Pexels).

## Open problems

- None yet.
