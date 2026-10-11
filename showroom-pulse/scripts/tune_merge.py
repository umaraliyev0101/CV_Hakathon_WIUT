#!/usr/bin/env python
"""Dev-only grid over the two post-cache identity-merge knobs (`merge_sim_thr`, `merge_max_speed`).

Reads ONLY the dev clips of labels/split.json (test clips are never loaded here). Perception is read
from runs/cache, so a trial is the cheap re-run of identity.merge + scoring. Selection rule, fixed before
looking at any number:

  J = mean over the four spec targets of error / target, all pooled over the dev clips:
      entry/exit count error / 10 %, occupancy MAE / 0.5, zone dwell median % error / 10 %,
      (1 - journey accuracy) / 0.20.   Lower is better.
  Keep the current defaults unless a trial improves J by >= 5 % (relative) AND the best trial is an interior
  point of the grid in `merge_sim_thr` (an optimum on the grid edge only says "keep going", it is not an
  optimum). Ties go to the defaults. (The interior-point condition was added after the first full grid, see
  PROGRESS.md: an earlier run with a buggy count-error formula looked monotone down to sim 0.2, which
  merges nearly everyone; the bug was fixed and the grid re-run.)
  (A first version capped each ratio at 3; the cap hid a 64 % -> 92 % entry/exit degradation behind a
  small occupancy gain, so the cap was removed. Both runs are logged in PROGRESS.md.)

usage: tune_merge.py [--sims 0.6 0.65 ...] [--speeds 0.1 0.25 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pulse.config import Config, get_config  # noqa: E402
from pulse.eval import run_eval as R  # noqa: E402

SIMS = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)   # 1.01 = merge disabled
SPEEDS = (0.1, 0.25, 0.5)
TARGETS = (10.0, 0.5, 10.0, 0.20)


def objective(a: dict) -> tuple[float, list[float | None]]:
    errs = [a["entry_exit_error"], a["occupancy_mae"], a["dwell"]["median_abs_pct"],
            None if a["journey_mean"] is None else 1.0 - a["journey_mean"]]
    ratios = [3.0 if e is None else e / t for e, t in zip(errs, TARGETS)]   # None (no pairs) counts as 3x target
    return sum(ratios) / len(ratios), errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="*", type=float, default=list(SIMS))
    ap.add_argument("--speeds", nargs="*", type=float, default=list(SPEEDS))
    args = ap.parse_args()
    base = get_config("dealership")
    split = json.loads((ROOT / "labels" / "split.json").read_text())
    dev = split["dev"]                                         # never touches split["test"]
    inputs = [R.load_inputs(e, "dev", base) for e in dev]
    print("dev clips:", ", ".join(R._label(e["clip"]) for e in dev))
    grid = [(s, v) for s in args.sims for v in args.speeds]
    if (base.merge_sim_thr, base.merge_max_speed) not in grid:
        grid.insert(0, (base.merge_sim_thr, base.merge_max_speed))
    print(f"defaults: sim {base.merge_sim_thr} speed {base.merge_max_speed}\n")
    print("| trial | sim_thr | max_speed | persons | entry/exit err % | occ MAE | dwell med % | journey acc | "
          "IDF1 merged | entries ours/GT | J |\n|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for n, (sim, spd) in enumerate(grid, 1):
        cfg: Config = get_config("dealership", merge_sim_thr=sim, merge_max_speed=spd)
        rs = [R.evaluate(i, cfg) for i in inputs]
        a = R.aggregate(rs)
        j, errs = objective(a)
        rows.append((j, sim, spd))
        print(f"| {n} | {sim} | {spd} | {a['n_pred_persons']} | {R.f1(errs[0])} | {R.f2(errs[1])} | "
              f"{R.f1(errs[2])} | {R.share(a['journey_mean'])} | {R.share(a['mot_merged']['idf1'])} | "
              f"{a['entries']['pred']} / {a['entries']['gt']} | {j:.3f} |", flush=True)
    j0 = next(j for j, s, v in rows if (s, v) == (base.merge_sim_thr, base.merge_max_speed))
    best = min(rows, key=lambda r: (round(r[0], 6), abs(r[1] - base.merge_sim_thr), abs(r[2] - base.merge_max_speed)))   # ties: closest to the defaults
    sims = sorted({s for _, s, _ in rows})
    interior = sims[0] < best[1] < sims[-1]
    print(f"\ndefault J = {j0:.3f}; best J = {best[0]:.3f} at sim {best[1]}, speed {best[2]}")
    if best[0] > 0.95 * j0:
        print("RULE: improvement < 5% -> keep the defaults")
    elif not interior:
        print(f"RULE: best trial is on the grid edge (sim {best[1]}); J is monotone in merge_sim_thr, "
              "so there is no optimum -> keep the defaults")
    else:
        print(f"RULE: improvement {(1 - best[0] / j0) * 100:.1f}% >= 5% at an interior point -> switch to "
              f"sim {best[1]}, speed {best[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
