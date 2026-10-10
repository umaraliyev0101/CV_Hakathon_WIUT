# Showroom Coach: progress log (Jira YAK-44)

Design: `docs/superpowers/specs/2026-10-11-showroom-coach-design.md`. Plan: `docs/superpowers/plans/2026-10-11-showroom-coach.md`.

## Log

- 2026-10-11 M0 / Task 0.1: package skeleton, `pyproject.toml` and pytest harness. Environment: Python 3.12.6 venv at `.venv` (not the default 3.14), Windows 11, i5-10210U, 16 GB, no GPU. Free disk on C: 21G; `.venv` is 708M. `pip install -e ".[dev]"` succeeded and `pytest -q` passes.

## Decisions

- Python 3.12 venv, because the machine default (3.14) is outside the `requires-python` range `>=3.11,<3.14`.

## Open problems

- None yet.
