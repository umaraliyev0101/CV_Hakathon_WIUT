"""Tuning on the development clips only (spec section 6): which identity-merge threshold and which pair-detector
settings reproduce the MEVA ``person_talks_to_person`` annotations best.

    python -m showroom_coach.evaluation.tune --split-file data/meva/split.json --out runs/eval/tune_dev.json

Only the clips listed under ``dev`` in the split file are read; the ``test`` clips are never opened. Perception comes from
the cache of an earlier evaluation run (a missing entry is an error, never a silent re-run of the detector). Each dev
clip's perception and ground truth are loaded once. For every identity ``merge_thr`` the analysis runs once per clip;
for every engagement ``dist`` x ``min_dur`` (``gap_tol`` fixed) the roles-ignored pair intervals are scored against the
talks with ``meva_eval.talk_eval``, strict and twin-aware, pooled over the dev clips (micro-average). The same
``merge_thr`` rows also carry the persons' IDF1 / MOTA and the enter/exit recall. With ``--oracle`` the clips are
loaded as ground-truth tracks (``meva_eval.oracle_result``), only ``merge_thr`` 0.30 is scored (there are no embeddings
to merge) and the result is the dist / min_dur grid of the engagement rule alone.

Selection rule (``choose``): the highest twin-aware talk F1, the harmonic mean of ``recall_all`` and
``precision_mapped``; ties are broken by the persons' IDF1, then by the settings nearest the untuned defaults.
``merge_beats_default`` says whether the chosen ``merge_thr`` earns replacing the product's 0.30.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .. import analysis, config
from ..perception import pipeline
from . import meva_eval as me
from .kpf import Activity, GeomBox, load_activities, load_geom, load_types, person_ids

GRID_MERGE = (0.20, 0.30, 0.40, 0.50)
GRID_DIST = (1.5, 2.0, 2.5, 3.0)
GRID_MIN_DUR = (2.0, 4.0, 6.0, 8.0)
GAP_TOL = 1.5
DEFAULTS = {"merge_thr": 0.30, "dist": 2.0, "min_dur": 8.0}   # the untuned settings (the product's merge_thr, the CLI's)
DEFAULT_LIMIT_S = 150.0
TIE = 1e-9
SELECTION_RULE = ("maximise the twin-aware talk F1 (harmonic mean of recall_all and precision_mapped, micro over the dev "
                  "clips); ties by persons IDF1, then by the settings nearest merge_thr 0.30 / dist 2.0 / min_dur 8")


# --------------------------------------------------------------------------------------------- selection

def f1(recall, precision) -> float:
    """Harmonic mean of two ratios; 0.0 when either is missing (nothing to score) or both are 0."""
    if not recall or not precision:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def _score(r: dict) -> tuple[float, float]:
    return round(float(r["twin_aware"]["f1"]), 9), round(float(r.get("persons_idf1") or 0.0), 9)


def choose(rows: list[dict], defaults: dict = DEFAULTS) -> dict:
    """The best row: highest twin-aware talk F1, then highest persons IDF1, then nearest the untuned ``defaults``
    (merge_thr first, then dist, then min_dur). Independent of the order of ``rows``. ValueError when there are none."""
    if not rows:
        raise ValueError("no rows to choose from")

    def key(r: dict):
        f, idf1 = _score(r)
        near = tuple(round(abs(float(r[k]) - float(defaults[k])), 9) for k in ("merge_thr", "dist", "min_dur"))
        return (-f, -idf1) + near

    return min(rows, key=key)


def merge_beats_default(chosen: dict, default: dict) -> bool:
    """Does ``chosen`` (a row with another ``merge_thr``) beat ``default`` (the row with merge 0.30 at the same ``dist`` and
    ``min_dur``)? It must be better on one of twin-aware talk F1 and persons IDF1 and not worse on the other."""
    f_c, i_c = _score(chosen)
    f_d, i_d = _score(default)
    return (f_c >= f_d and i_c >= i_d) and (f_c > f_d or i_c > i_d)


def find(rows: list[dict], merge_thr: float, dist: float, min_dur: float) -> dict | None:
    """The row with exactly these settings, or None."""
    return next((r for r in rows if abs(r["merge_thr"] - merge_thr) < TIE and abs(r["dist"] - dist) < TIE
                 and abs(r["min_dur"] - min_dur) < TIE), None)


# ------------------------------------------------------------------------------------------------ one clip

@dataclass(eq=False)
class DevClip:
    """One dev clip, loaded once: perception, ground truth and everything that does not depend on the settings."""
    name: str
    res: object
    gt_boxes: dict[int, list[GeomBox]]
    acts: list[Activity]
    persons_gt: set[int]
    fps: float
    t_end: float
    clusters: dict[int, int]
    oracle: bool = False


def load_dev_clip(video, ann_prefix, stack, cache_root, limit_seconds: float | None, oracle: bool = False) -> DevClip:
    """Load one clip from the perception cache. RuntimeError (and no work done) when the cache has no entry for it: tuning
    never runs the detector. With ``oracle`` the observations are the ground-truth boxes (``meva_eval.oracle_result``;
    the cache only supplies the sampled frames) and the clip's ids map to themselves."""
    kw = {} if limit_seconds is None else {"max_seconds": float(limit_seconds)}
    entry = Path(cache_root) / pipeline.cache_key(video, pipeline.perception_params(stack, **kw))
    if not entry.is_dir():
        raise RuntimeError(f"no cached perception for {Path(str(video)).name} (looked for {entry}); run the evaluation "
                           "once to fill the cache - tuning does not run the detector")
    prefix = str(ann_prefix)
    persons_gt = person_ids(load_types(prefix + ".types.yml"))
    gt_boxes = load_geom(prefix + ".geom.yml", persons_gt)
    acts = load_activities(prefix + ".activities.yml")
    res, cached = pipeline.cached_perception(video, stack, Path(cache_root), **kw)
    if not cached:                                  # lost a race with a sweep: do not carry on with a fresh detector run
        raise RuntimeError(f"the cached perception of {Path(str(video)).name} was rebuilt while loading it")
    clusters = me.twin_clusters(res, gt_boxes)
    if oracle:
        res = me.oracle_result(res, gt_boxes, clusters)
    return DevClip(Path(prefix).name, res, gt_boxes, acts, persons_gt, float(res.video.src_fps),
                   float(res.video.duration), clusters, oracle)


# ------------------------------------------------------------------------------------------------- the grid

def _slim(obj):
    """``obj`` without the per-match error lists (they only matter for pooling, and bloat the JSON)."""
    if isinstance(obj, dict):
        return {k: _slim(v) for k, v in obj.items() if k not in ("start_errs_s", "errs_s")}
    return obj


def evaluate(clips: list[DevClip], merges=GRID_MERGE, dists=GRID_DIST, min_durs=GRID_MIN_DUR, gap_tol: float = GAP_TOL,
             preset: str = "short", log=None) -> dict:
    """Score every setting of the grid on ``clips``, pooled over them.

    Returns ``{"rows": [...], "per_merge": {...}}``. A row has ``merge_thr``, ``dist``, ``min_dur``, ``gap_tol``,
    ``strict`` and ``twin_aware`` (each ``{"talk": pooled talk_eval summary, "f1": ...}``), ``persons_idf1``,
    ``persons_mota`` and ``enter_exit_recall`` per variant. ``per_merge`` (keyed by the text of the threshold) holds what
    does not depend on ``dist`` / ``min_dur``: the persons' pooled tracking metrics, persons per clip and enter/exit
    summaries. Rows are ordered by merge_thr, dist, min_dur as given.
    """
    rows, per_merge = [], {}
    for merge in merges:
        started = time.perf_counter()
        ans = [analysis.analyze(c.res, preset=preset, merge_thr=merge) for c in clips]
        maps = {"strict": [me.oracle_id_map(an) if c.oracle else
                           me.match_ids(c.res, an, c.gt_boxes, True, min_actor_share=None) for c, an in zip(clips, ans)],
                "twin_aware": [me.oracle_id_map(an) if c.oracle else me.match_ids(c.res, an, c.gt_boxes, True)
                               for c, an in zip(clips, ans)]}
        tracking = me._sum_tracking([me.tracking_metrics(c.res, an, c.gt_boxes, True) for c, an in zip(clips, ans)])
        enter_exit = {v: me._sum_ee_blocks([
            me.enter_exit_eval(an, c.acts, m, c.fps, t_end=c.t_end, persons=c.persons_gt,
                               clusters=c.clusters if v == "twin_aware" else None)
            for c, an, m in zip(clips, ans, maps[v])]) for v in me.VARIANTS}
        per_merge[f"{merge:g}"] = {
            "merge_thr": merge, "n_persons": [len(an.persons) for an in ans], "persons_tracking": _slim(tracking),
            "enter_exit": _slim(enter_exit),
        }
        for dist in dists:
            for min_dur in min_durs:
                intervals = [me.pair_intervals(an, c.res, dist, min_dur, gap_tol) for c, an in zip(clips, ans)]
                row = {"merge_thr": merge, "dist": dist, "min_dur": min_dur, "gap_tol": gap_tol}
                for v in me.VARIANTS:
                    talk = me._sum_talk([
                        me.talk_eval(iv, c.acts, m, c.fps, t_max=c.t_end, persons=c.persons_gt,
                                     clusters=c.clusters if v == "twin_aware" else None)
                        for iv, c, m in zip(intervals, clips, maps[v])])
                    row[v] = {"talk": _slim(talk), "f1": f1(talk["recall_all"], talk["precision_mapped"])}
                row["persons_idf1"] = tracking["idf1"]
                row["persons_mota"] = tracking["mota"]
                row["enter_exit_recall"] = {v: enter_exit[v]["all"]["recall_all"] for v in me.VARIANTS}
                rows.append(row)
        if log:
            log(f"merge_thr {merge:g}: {len(dists) * len(min_durs)} settings scored in {time.perf_counter() - started:.1f} s")
    return {"rows": rows, "per_merge": per_merge}


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def format_grid(rows: list[dict], per_merge: dict, chosen: dict | None = None) -> str:
    """The grid as a text table (one line per setting, ``*`` marks the chosen one) and the per-merge summary."""
    lines = ["", "Dev grid, twin-aware talks pooled over the dev clips (R_all = recall of all ground-truth pair intervals,"
             " P_map = precision of the mapped detections, F1 = harmonic mean of the two); % values",
             f"  {'merge':>5} {'dist':>4} {'min_dur':>7} | {'n_gt':>4} {'n_det':>5} {'map':>4} {'match':>5} | {'R_all':>5} "
             f"{'R_trk':>5} {'P_map':>5} {'F1':>5} | {'strictF1':>8} | {'IDF1':>5} {'MOTA':>5} {'E/E R':>5}"]
    for r in rows:
        t = r["twin_aware"]["talk"]
        mark = "*" if chosen is not None and r is chosen else " "
        lines.append(f"{mark} {r['merge_thr']:5.2f} {r['dist']:4.1f} {r['min_dur']:7.1f} | {t['n_gt']:4d} {t['n_det']:5d} "
                     f"{t['n_det_mapped']:4d} {t['n_matched']:5d} | {_pct(t['recall_all'])} {_pct(t['recall_tracked'])} "
                     f"{_pct(t['precision_mapped'])} {_pct(r['twin_aware']['f1'])} | {_pct(r['strict']['f1']):>8} | "
                     f"{_pct(r['persons_idf1'])} {_pct(r['persons_mota'])} {_pct(r['enter_exit_recall']['twin_aware'])}")
    lines += ["", "Per identity merge threshold (independent of dist / min_dur):",
              f"  {'merge':>5} {'persons per clip':>18} {'IDF1':>5} {'MOTA':>5} {'prec':>5} {'recall':>6} {'IDSW':>5} "
              f"{'E/E R twin':>10} {'E/E R strict':>12}"]
    for p in per_merge.values():
        trk, ee = p["persons_tracking"], p["enter_exit"]
        lines.append(f"  {p['merge_thr']:5.2f} {str(p['n_persons']):>18} {_pct(trk['idf1'])} {_pct(trk['mota'])} "
                     f"{_pct(trk['precision'])} {_pct(trk['recall']):>6} {trk['num_switches']:5d} "
                     f"{_pct(ee['twin_aware']['all']['recall_all']):>10} {_pct(ee['strict']['all']['recall_all']):>12}")
    return "\n".join(lines)


def _describe(r: dict) -> str:
    t = r["twin_aware"]["talk"]
    return (f"merge_thr {r['merge_thr']:g}, dist {r['dist']:g}, min_dur {r['min_dur']:g}: twin-aware talk F1 "
            f"{_pct(r['twin_aware']['f1']).strip()} % (recall_all {_pct(t['recall_all']).strip()} %, precision_mapped "
            f"{_pct(t['precision_mapped']).strip()} %), persons IDF1 {_pct(r['persons_idf1']).strip()} %")


# ------------------------------------------------------------------------------------------- command line

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m showroom_coach.evaluation.tune",
                                description="Tune the identity merge and the pair detector on the MEVA dev clips only.")
    p.add_argument("--clips", default="data/meva/clips.csv", help="the clips table (default: %(default)s)")
    p.add_argument("--split-file", default="data/meva/split.json",
                   help='JSON {"dev": [...], "test": [...]}; only "dev" is read (default: %(default)s)')
    p.add_argument("--out", default=None,
                   help="results JSON (default: runs/eval/tune_dev.json, or tune_dev_oracle.json with --oracle)")
    p.add_argument("--preset", default="short", help="threshold preset for the analysis (default: %(default)s)")
    p.add_argument("--limit-seconds", type=float, default=None,
                   help=f"seconds of every clip that were evaluated (default: the split file's limit_seconds, else "
                        f"{DEFAULT_LIMIT_S:g})")
    p.add_argument("--cache", default=None, help="perception cache directory (default: the app's)")
    p.add_argument("--oracle", action="store_true",
                   help="ablation: tune dist and min_dur on ground-truth boxes instead of detections (perfect detection "
                        "and tracking); merge_thr has no effect there, so only 0.30 is scored")
    return p


def main(argv=None) -> int:
    """Exit codes: 0 done, 1 a dev clip could not be loaded (for instance no cached perception), 2 usage error."""
    args = _parser().parse_args(argv)
    out_path = args.out or ("runs/eval/tune_dev_oracle.json" if args.oracle else "runs/eval/tune_dev.json")
    merges = (DEFAULTS["merge_thr"],) if args.oracle else GRID_MERGE
    clips_path, split_path = Path(args.clips), Path(args.split_file)
    try:
        rows = me.select_clips(me.read_clips(clips_path), "dev", split_path)
        limit = args.limit_seconds
        if limit is None:
            limit = me._load_split(split_path).get("limit_seconds", DEFAULT_LIMIT_S)
        limit = float(limit)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("error: the 'dev' split selects no clips", file=sys.stderr)
        return 2
    try:
        me._motmetrics()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stack = me._default_stack()
    cache_root = Path(args.cache) if args.cache else config.cache_dir()
    clips: list[DevClip] = []
    for row in rows:
        video = clips_path.parent / "videos" / f"{row['clip']}.r13.avi"
        prefix = me._annotation_prefix(row, clips_path)
        needed = [video] + [Path(str(prefix) + s) for s in (".geom.yml", ".activities.yml", ".types.yml")]
        if not me._files_exist(*needed):
            print(f"error: {row['clip']}: video or annotation files missing", file=sys.stderr)
            return 1
        print(f"loading {row['clip']} ...", file=sys.stderr, flush=True)
        try:
            clips.append(load_dev_clip(video, prefix, stack, cache_root, limit, oracle=args.oracle))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    out = evaluate(clips, merges, GRID_DIST, GRID_MIN_DUR, GAP_TOL, args.preset,
                   log=lambda m: print(m, file=sys.stderr, flush=True))
    all_rows = out["rows"]
    chosen = choose(all_rows)
    untuned = find(all_rows, **DEFAULTS)
    merge_default = find(all_rows, DEFAULTS["merge_thr"], chosen["dist"], chosen["min_dur"])
    adopt = (not args.oracle and abs(chosen["merge_thr"] - DEFAULTS["merge_thr"]) > TIE and merge_default is not None
             and merge_beats_default(chosen, merge_default))
    decision = {"adopt": bool(adopt), "chosen_merge_thr": chosen["merge_thr"], "default_merge_thr": DEFAULTS["merge_thr"],
                "rule": "adopt a merge_thr other than 0.30 only if, at the chosen dist and min_dur, it is better than 0.30 "
                        "on one of twin-aware talk F1 and persons IDF1 and not worse on the other"}
    me._write_results(Path(out_path), {
        "mode": "oracle" if args.oracle else "system", "split_file": str(split_path), "clips": [c.name for c in clips], "limit_seconds": limit, "preset": args.preset,
        "grid": {"merge_thr": list(merges), "dist": list(GRID_DIST), "min_dur": list(GRID_MIN_DUR), "gap_tol": GAP_TOL},
        "selection_rule": SELECTION_RULE, "chosen": chosen, "default_untuned": untuned,
        "merge_default_at_chosen": merge_default, "merge_decision": decision, "per_merge": out["per_merge"],
        "rows": all_rows})
    if args.oracle:
        print("\nORACLE MODE: ground-truth boxes instead of detections (perfect detection and tracking); an ablation of "
              "the geometry and the engagement rule, not the system")
    print(format_grid(all_rows, out["per_merge"], chosen))
    print("")
    print("chosen   " + _describe(chosen))
    if untuned is not None:
        print("untuned  " + _describe(untuned))
    if merge_default is not None and merge_default is not chosen:
        print("merge 0.30 at the chosen dist/min_dur  " + _describe(merge_default))
    if not args.oracle:
        print(f"merge_thr {chosen['merge_thr']:g} {'beats' if adopt else 'is not adopted over'} the product default 0.30")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
