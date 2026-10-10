# Showroom Coach: progress log (Jira YAK-44)

Design: `docs/superpowers/specs/2026-10-11-showroom-coach-design.md`. Plan: `docs/superpowers/plans/2026-10-11-showroom-coach.md`.

## Log

- 2026-10-11 M0 / Task 0.1: package skeleton, `pyproject.toml` and pytest harness. Environment: Python 3.12.6 venv at `.venv` (not the default 3.14), Windows 11, i5-10210U, 16 GB, no GPU. Free disk on C: 21G; `.venv` is 708M. `pip install -e ".[dev]"` succeeded and `pytest -q` passes.
- 2026-10-11 M0 / Task 1.0: core dataclasses (`types.py`), `to_jsonable`, `Thresholds`/presets, `CameraConfig`, `RoleWeights`, `repo_root()`/`jobs_dir()`/`cache_dir()` (`config.py`). Presets match spec section 4.6.
- 2026-10-11 M0 / Task 0.2: model registry, retrying download with sha256 check, `showroom-coach fetch-models`, `models/MANIFEST.json`. Fetched 6 models (yolox_nano 3.7 MB, yolox_tiny 20.2 MB, yolox_s 35.9 MB, rtmpose_s 21.9 MB, dinov2_small 24.5 MB, rfdetr_nano 112.6 MB): `models/` is 209M, `.venv` is 708M, free disk on C: 21G. All six load in onnxruntime; input shapes match the registry notes.

## Decisions

- Python 3.12 venv, because the machine default (3.14) is outside the `requires-python` range `>=3.11,<3.14`.
- Registry sha256 values are pinned to the files fetched on 2026-10-11, so later fetches fail loudly if an upstream file changes (for rtmpose_s the hash is of the extracted `end2end.onnx`, not the zip).
- Every model is saved as `models/<name>/<name>.onnx`. From the rtmpose zip only the `end2end.onnx` member is kept (copied out by basename, never extracted by path).
- `rfdetr_nano` is a benchmark candidate only; it is a third-party ONNX export of Roboflow RF-DETR (Apache-2.0), recorded in the manifest note.
- `to_jsonable` maps NaN and infinity to `null`, because they are not valid JSON and `JSON.parse` rejects them.

## Open problems

- README licence table (CLAUDE.md rule) does not exist yet: rows for yolox_nano/tiny/s, rtmpose_s, dinov2_small and rfdetr_nano are owed in M7. The data is in `models/MANIFEST.json`.
- `write_manifest()` lists only files present on disk, so run `fetch-models` with no names (all models) before committing the manifest.
