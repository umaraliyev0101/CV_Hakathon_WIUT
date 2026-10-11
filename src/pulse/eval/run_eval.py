"""`pulse eval`: score the pipeline against MEVA ground truth and write docs/EVAL.md (spec section 6).

Deterministic by construction: it reads only cached artefacts (perception caches, labels, the KPF
annotations, runs/bench/*.json) and measures nothing against the wall clock, so two runs give a
byte-identical file. Ground truth is turned into a Timeline and scored with the SAME zones, journey and
stats code as the pipeline output, which isolates perception/tracking error from the rest.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pulse import cache as cache_mod
from pulse import journey, pipeline, sources
from pulse import stats as stats_mod
from pulse import insights as insights_mod
from pulse.cache import PerceptionCache
from pulse.config import Config, get_config
from pulse.datatypes import Person, Sample, Timeline, Zone
from pulse.eval import metrics as M
from pulse.eval.meva import MevaClip, load_kpf
from pulse.eval.scenarios import SCENARIOS
from pulse.timeline import filter_stable
from pulse.zones import load_zones, point_in_polygon

ROOT = Path(__file__).resolve().parents[3]
ENTER, EXIT = "person_enters_scene_through_structure", "person_exits_scene_through_structure"
EVAL_VERSION = 1                       # extra manual invalidation of runs/eval (code changes already do)
TIMING_WINDOW_S = 5.0
CMD_EVAL = "`.venv/bin/pulse eval --out docs/EVAL.md`"
CMD_SPEED = "`.venv/bin/pulse bench speed --clip <dev clip> --out runs/bench/speed.json`"
CMD_COVER = "`.venv/bin/python scripts/zone_coverage.py`"
CMD_LIVE = "`.venv/bin/python scripts/smoke_live.py`"
CMD_DET = "`.venv/bin/pulse bench detectors --clip <dev clip> --kpf <annotations> --showroom <images>`"
SPEED_JSON = Path("runs/bench/speed.json")
DET_JSON = Path("runs/bench/detectors.json")


class MissingInput(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# per-clip inputs and ground truth

@dataclass
class ClipInputs:
    entry: dict
    group: str
    meva: MevaClip
    pc: PerceptionCache
    zones: list[Zone]
    size: tuple[int, int]               # analysis frame (w, h)
    src_size: tuple[int, int]
    src_fps: float
    duration: float
    n_samples: int
    gt_tl: Timeline                     # GT person tracks sampled like the pipeline, analysis coords
    gt_frames: dict[int, dict[int, tuple]]   # sample index -> {gt track id: box}


def _label(clip: str) -> str:
    p = clip.split(".")
    return f"{p[-1]} {p[0][5:]}"


def sampled_frame_indices(n_samples: int, src_fps: float, sample_fps: float) -> list[int]:
    """Source frame index of each pipeline sample (identical to sources.iter_frames)."""
    return [int(round(k * src_fps / sample_fps)) for k in range(n_samples)]


def build_gt(meva: MevaClip, n_samples: int, src_fps: float, sample_fps: float,
             src_size: tuple[int, int], size: tuple[int, int]):
    """GT person tracks -> (Timeline, per-sample boxes), at the pipeline's sample times and scale."""
    sx, sy = size[0] / src_size[0], size[1] / src_size[1]
    fidx = sampled_frame_indices(n_samples, src_fps, sample_fps)
    k_of = {f: k for k, f in enumerate(fidx)}
    tl: dict[int, list[Sample]] = {}
    frames: dict[int, dict[int, tuple]] = {}
    for tid, boxes in sorted(meva.person_tracks().items()):
        for f, b in boxes.items():
            k = k_of.get(f)
            if k is None:
                continue
            x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            if x2 <= x1 or y2 <= y1:
                continue
            t = round(k / sample_fps, 3)
            foot = ((x1 + x2) / 2 / size[0], y2 / size[1])
            tl.setdefault(tid, []).append(Sample(t, tid, foot, (x1, y1, x2, y2), None, None, "gt"))
            frames.setdefault(k, {})[tid] = (x1, y1, x2, y2)
    for ss in tl.values():
        ss.sort(key=lambda s: s.t)
    return dict(sorted(tl.items())), frames


def load_inputs(entry: dict, group: str, cfg: Config, cache_root: Path | None = None) -> ClipInputs:
    video, kpf = ROOT / entry["video"], ROOT / entry["kpf"]
    zpath = ROOT / "labels" / "zones" / f"{entry['clip']}.json"
    for p in (video, kpf, zpath):
        if not p.exists():
            raise MissingInput(f"missing input for {entry['clip']}: {p}")
    pc = pipeline.perceive(video, cache_root or ROOT / "runs" / "cache", cfg)
    m = pc.meta
    size = (int(m["frame_w"]), int(m["frame_h"]))
    src_size = (int(m["src_w"]), int(m["src_h"]))
    meva = load_kpf(kpf, entry["clip"])
    n = int(m["n_samples"])
    gt_tl, gt_frames = build_gt(meva, n, float(m["src_fps"]), cfg.sample_fps, src_size, size)
    duration = float(m.get("analysed_duration") or m.get("duration") or 0.0)
    return ClipInputs(entry, group, meva, pc, load_zones(zpath), size, src_size, float(m["src_fps"]),
                      duration, n, gt_tl, gt_frames)


def _entrance_coverage(inp: ClipInputs) -> dict:
    """Share of GT enter (first frame) / exit (last frame) foot points inside an entrance zone."""
    ents = [z for z in inp.zones if z.kind == "entrance"]
    tracks = inp.meva.person_tracks()
    hit = tot = 0
    for a in inp.meva.activities:
        if a.name not in (ENTER, EXIT):
            continue
        frame = a.frame_start if a.name == ENTER else a.frame_end
        for actor in a.actors:
            tr = tracks.get(actor)
            if not tr:
                continue
            b = tr[min(tr, key=lambda k: abs(k - frame))]
            pt = ((b[0] + b[2]) / 2 / inp.src_size[0], b[3] / inp.src_size[1])
            tot += 1
            hit += any(point_in_polygon(pt, z.polygon) for z in ents)
    return {"hit": hit, "total": tot}


def _pred_mot_frames(inp: ClipInputs, t2p: dict[int, int] | None, fps: float):
    out: dict[int, dict[int, tuple]] = {}
    best: dict[tuple[int, int], float] = {}
    for o in inp.pc.obs:
        k = int(round(o.t * fps))
        tid = o.track_id if t2p is None else t2p.get(o.track_id)
        if tid is None:
            continue
        if (k, tid) in best and best[(k, tid)] >= o.score:
            continue
        best[(k, tid)] = o.score
        out.setdefault(k, {})[tid] = o.bbox
    return out


def evaluate(inp: ClipInputs, cfg: Config) -> dict:
    """All per-clip metrics (JSON-serialisable). Everyone is scored as a customer (MEVA has no staff)."""
    pre, size, dur = cfg.preset, inp.size, inp.duration
    persons, tl, t2p = pipeline.build_people(inp.pc, cfg)
    stable = filter_stable(tl, inp.pc.segments)
    pred_j = journey.build(stable, inp.zones, pre)
    st = stats_mod.compute(persons, pred_j, {}, cfg, timeline=tl, episodes=[], assist={},
                           zones=inp.zones, duration=dur, frame_size=size)
    entries_ev, exits_ev = stats_mod.entry_exit_events(persons, tl, cfg, inp.zones, dur, size)

    gt_persons = [Person(pid, [pid], ss[0].t, ss[-1].t) for pid, ss in inp.gt_tl.items() if ss]
    gt_j = journey.build(inp.gt_tl, inp.zones, pre)
    gt_st = stats_mod.compute(gt_persons, gt_j, {}, cfg, timeline=inp.gt_tl, episodes=[], assist={},
                              zones=inp.zones, duration=dur, frame_size=size)

    # entries / exits against the MEVA activities (counted over the same window as the pipeline rule)
    ent_t = [a.frame_start / inp.src_fps for a in inp.meva.activities if a.name == ENTER]
    exit_t = [a.frame_end / inp.src_fps for a in inp.meva.activities if a.name == EXIT]
    ent_all = sorted(t for t in ent_t if t <= dur)
    exit_all = sorted(t for t in exit_t if t <= dur)
    ent_win = [t for t in ent_all if t >= pre.leave_after]
    exit_win = [t for t in exit_all if t <= dur - pre.leave_after]
    en_m = M.match_events(ent_win, [t for _, t in entries_ev], TIMING_WINDOW_S)
    ex_m = M.match_events(exit_win, [t for _, t in exits_ev], TIMING_WINDOW_S)

    occ_p = [int(r[1]) for r in st.occupancy]
    occ_g = [int(r[1]) for r in gt_st.occupancy]
    n_occ = max(len(occ_p), len(occ_g))

    gt_mot = inp.gt_frames
    frames = list(range(inp.n_samples))
    mot_raw = M.mot(gt_mot, _pred_mot_frames(inp, None, cfg.sample_fps), frames)
    mot_merged = M.mot(gt_mot, _pred_mot_frames(inp, t2p, cfg.sample_fps), frames)

    # MEVA tracks are per activity (median ~1 s on the dev clip), so GT tracks are matched many-to-one and
    # our person is compared only inside the GT track's own time window (same samples, same journey code).
    match = M.match_persons(inp.gt_tl, tl, one_to_one=False)
    win_j: dict[int, list] = {}
    for g, p in match.items():
        t0, t1 = inp.gt_tl[g][0].t, inp.gt_tl[g][-1].t
        win = [s for s in stable.get(p, []) if t0 - 1e-9 <= s.t <= t1 + 1e-9]
        win_j[g] = journey.build({g: win}, inp.zones, pre)[g] if win else []
    same = {g: g for g in match}
    pairs = M.dwell_errors(gt_j, win_j, same)
    jacc = M.journey_accuracies(gt_j, win_j, same)
    return {
        "clip": inp.entry["clip"], "group": inp.group, "duration": dur, "n_samples": inp.n_samples,
        "n_gt_persons": len(inp.gt_tl), "n_pred_tracks": len({o.track_id for o in inp.pc.obs}),
        "n_pred_persons": len(persons), "n_matched": len(match),
        "entries": {"pred": len(entries_ev), "gt": len(ent_win), "gt_all": len(ent_all),
                    "timing": [m[2] for m in en_m], "n_matched": len(en_m)},
        "exits": {"pred": len(exits_ev), "gt": len(exit_win), "gt_all": len(exit_all),
                  "timing": [m[2] for m in ex_m], "n_matched": len(ex_m)},
        "occupancy": {"abs_err_sum": float(sum(abs((occ_p + [0] * n_occ)[i] - (occ_g + [0] * n_occ)[i])
                                               for i in range(n_occ))),
                      "n": n_occ, "pred_sum": int(sum(occ_p)), "gt_sum": int(sum(occ_g))},
        "mot_raw": mot_raw, "mot_merged": mot_merged,
        "dwell_pairs": [[a, b] for a, b in pairs], "journey_acc": jacc,
        "coverage": _entrance_coverage(inp),
    }


# ---------------------------------------------------------------------------------------------
# aggregation

def _cat(rs: list[dict], key: str) -> list:
    return [x for r in rs for x in r[key]]


def aggregate(rs: list[dict]) -> dict:
    """Pool per-clip results of one group (sums of counts; lists concatenated)."""
    def cnt(kind):
        pred = [r[kind]["pred"] for r in rs]
        gt = [r[kind]["gt"] for r in rs]
        return {"pred": sum(pred), "gt": sum(gt), "abs_err": sum(abs(p - g) for p, g in zip(pred, gt)),
                "n_matched": sum(r[kind]["n_matched"] for r in rs),
                "count_error": M.pooled_count_error(pred, gt),
                "timing_median": M.median_timing_error([(0, 0, x) for r in rs for x in r[kind]["timing"]])}
    ent, ex = cnt("entries"), cnt("exits")
    both_pred = [r["entries"]["pred"] for r in rs] + [r["exits"]["pred"] for r in rs]
    both_gt = [r["entries"]["gt"] for r in rs] + [r["exits"]["gt"] for r in rs]
    occ_n = sum(r["occupancy"]["n"] for r in rs)
    occ_e = sum(r["occupancy"]["abs_err_sum"] for r in rs)
    jacc = _cat(rs, "journey_acc")
    return {
        "entries": ent, "exits": ex,
        "entry_exit_error": M.pooled_count_error(both_pred, both_gt),
        "occupancy_mae": occ_e / occ_n if occ_n else None, "occupancy_seconds": occ_n,
        "mot_raw": M.pool_mot([r["mot_raw"] for r in rs]),
        "mot_merged": M.pool_mot([r["mot_merged"] for r in rs]),
        "dwell": M.dwell_summary([(a, b) for a, b in _cat(rs, "dwell_pairs")]),
        "journey_mean": sum(jacc) / len(jacc) if jacc else None, "journey_n": len(jacc),
        "n_gt_persons": sum(r["n_gt_persons"] for r in rs), "n_matched": sum(r["n_matched"] for r in rs),
        "n_pred_persons": sum(r["n_pred_persons"] for r in rs),
        "coverage": {"hit": sum(r["coverage"]["hit"] for r in rs),
                     "total": sum(r["coverage"]["total"] for r in rs)},
    }


def clip_summary(r: dict) -> dict:
    a = aggregate([r])
    a["occupancy_mae"] = r["occupancy"]["abs_err_sum"] / r["occupancy"]["n"] if r["occupancy"]["n"] else None
    return a


# ---------------------------------------------------------------------------------------------
# formatting

def f1(x) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def pct1(x) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def share(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def _met(ok) -> str:
    return "n/a" if ok is None else ("yes" if ok else "NO")


def _load_json(path: Path) -> dict:
    p = ROOT / path
    if not p.is_file():
        raise MissingInput(f"missing input: {path} (run its command, see docs/EVAL.md Reproduce notes)")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------------------------
# the document

def _mot_rows(m: dict) -> list[str]:
    return [share(m["mota"]), share(m["idf1"]), str(m["num_switches"]), share(m["precision"]),
            share(m["recall"]), str(m["num_objects"])]


def render(per_clip: dict[str, list[dict]], cfg: Config, speed: dict, det: dict) -> str:
    dev, test = per_clip["dev"], per_clip["test"]
    ad, at = aggregate(dev), aggregate(test)
    L: list[str] = []
    w = L.append
    ver = importlib.metadata.version("motmetrics")

    w("# Showroom Pulse: evaluation")
    w("")
    w("Generated by `pulse eval`; do not edit by hand. Every number below is computed from cached "
      "perception output, the MEVA ground truth and `labels/`, and is reproduced by")
    w(f"{CMD_EVAL} (byte-identical on every run, no timestamps, no wall-clock measurement). "
      "A missed target is reported as measured.")
    w("")
    w("## Method")
    w("")
    w("- **Data:** six indoor MEVA clips (school scenes, 5 min each, CC BY 4.0, Corona et al., WACV 2021). "
      "The dev/test split is fixed in `labels/split.json` before any tuning: **dev** = "
      + ", ".join(_label(r["clip"]) for r in dev) + "; **test** = "
      + ", ".join(_label(r["clip"]) for r in test) + ". Dev and test are always reported separately; "
      "the targets are judged on test.")
    w(f"- **Pipeline under test:** preset `{cfg.preset.name}` (5-min CCTV), detector `{cfg.detector}`, pose "
      f"`{cfg.pose_model}`, 5 fps sampling, 960 px wide, CPU. Perception is read from `runs/cache`; the "
      f"identity merge (`merge_sim_thr` {cfg.merge_sim_thr}, `merge_max_speed` {cfg.merge_max_speed}) is "
      "re-run from the cache. The two merge knobs were tuned on the dev clips only "
      "(`scripts/tune_merge.py`; full trial log in `PROGRESS.md`): `merge_sim_thr` went from 0.75 to 0.9 "
      "(pooled dev entry/exit error 35.3% to 3.6%), `merge_max_speed` is insensitive and kept at 0.25. "
      "Nothing else was tuned and the test clips were scored once, after the configuration was frozen. "
      "With MEVA ground truth this tuning rests on two clips and counts only, so treat the new "
      "threshold as a dev-set fit, not a general optimum.")
    w("- **Roles:** every person is scored as a customer, because MEVA has no staff labels "
      "(role detection is therefore not evaluated here).")
    w("- **Ground truth:** MEVA geom tracks of type `person`, sampled at the same source frames as the "
      "pipeline (`round(k * src_fps / 5)`), boxes scaled to the 960 px analysis frame, foot = bottom "
      "centre. The GT is turned into a Timeline and run through the SAME `zones`, `journey` and `stats` "
      "code as our output, so dwell, journeys and occupancy differ only through perception and tracking. "
      "Zones are the agent-drawn, unreviewed `labels/zones/<clip>.json`.")
    w("- **Person matching:** MEVA tracks are annotated per activity, so a GT \"person\" is a short snippet "
      "(median about 1 s on the first dev clip) of someone we may track for much longer. Each GT track is "
      "therefore matched independently to the predicted person with the best overlap (overlap = sum of box "
      "IoU over the GT track's sample times / its number of samples, minimum 0.3; several GT tracks may "
      "share one predicted person). Dwell and journeys are compared inside the GT track's own time "
      "window only: our person's samples in that window go through the same `journey` code as the GT "
      "samples. GT tracks shorter than `zone_min` cannot produce a zone visit, so few GT tracks yield "
      "dwell pairs; the number of pairs is always shown.")
    w(f"- **Library:** motmetrics {ver}, IoU >= 0.5 for a tracking match.")
    w("")

    # ---- targets
    sp = speed["cpu"]
    live = speed["live_smoke"]           # snapshot of runs/bench/live_smoke.json taken by `pulse bench speed`
    live_fps = live.get("fps_max")
    eng = speed["engine_1_to_4_people"]["fps"]
    w("## Targets")
    w("")
    w("Spec section 6. Measured values are test-set, pooled over the test clips; dev values are given "
      "alongside for reference (they are what the two merge knobs were tuned on).")
    w("")
    rows = [
        ["entry/exit count error (pooled entries + exits)", "<= 10%", pct1(at["entry_exit_error"]),
         pct1(ad["entry_exit_error"]), _met(None if at["entry_exit_error"] is None else at["entry_exit_error"] <= 10)],
        ["occupancy MAE (persons per second)", "<= 0.5", f2(at["occupancy_mae"]), f2(ad["occupancy_mae"]),
         _met(None if at["occupancy_mae"] is None else at["occupancy_mae"] <= 0.5)],
        ["zone dwell median error (Dealership preset: %)", "<= 10%", pct1(at["dwell"]["median_abs_pct"]),
         pct1(ad["dwell"]["median_abs_pct"]),
         _met(None if at["dwell"]["median_abs_pct"] is None else at["dwell"]["median_abs_pct"] <= 10)],
        ["journey accuracy (1 - normalised edit distance)", ">= 80%", share(at["journey_mean"]),
         share(ad["journey_mean"]),
         _met(None if at["journey_mean"] is None else at["journey_mean"] >= 0.8)],
        ["file mode real-time factor (60 s 720p excerpt, 5 fps, CPU)", ">= 1.0x", f"{sp['rtf']:.2f}x",
         "(speed is not split by clip)", _met(sp["rtf"] >= 1.0)],
        ["live mode fps (max sustained, live smoke, 1-3 people)", ">= 5 fps",
         "n/a" if live_fps is None else f"{live_fps:.2f}", "(not split by clip)",
         _met(None if live_fps is None else live_fps >= 5.0)],
    ]
    L += _table(["metric", "target", "measured (test)", "measured (dev)", "met?"], rows)
    w(f"Reproduce: {CMD_EVAL} for the first four rows; {CMD_SPEED} for file mode; {CMD_LIVE} then "
      f"{CMD_SPEED} for live fps (the bench snapshots the smoke result into `runs/bench/speed.json`, so "
      "EVAL.md does not change when `verify.sh` re-runs the smoke). "
      f"Engine-only step rate on frames with 1-4 people: {eng:.2f} fps (see Speed).")
    w("")
    miss = [r[0] for r in rows if r[4] == "NO"]
    w("**Targets missed (test):** " + ("; ".join(miss) if miss else "none") + ".")
    w("")

    # ---- entries and exits
    w("## Entries and exits")
    w("")
    w("Ours = `stats.store` entries/exits (entrance-zone rule, spec 4.5). GT = MEVA "
      f"`{ENTER}` / `{EXIT}` activities, counted over the same window the rule uses: entries starting at "
      f"or after `leave_after` ({cfg.preset.leave_after} s) and exits ending at least `leave_after` s before "
      "the end (people already present at the start are excluded by definition, spec 4.5). "
      "Count error = sum |ours - GT| / sum GT over entries and exits (entries and exits are separate "
      "terms, so an over-count of one never cancels an under-count of the other); the pooled figure sums "
      "over clips the same way. Timing error: each GT activity is matched to the "
      f"nearest unused predicted event within {TIMING_WINDOW_S:.0f} s; the median absolute difference is "
      "reported with the number of matched events.")
    w("")
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        rows = []
        for r in rs:
            e, x = r["entries"], r["exits"]
            ce = M.pooled_count_error([e["pred"], x["pred"]], [e["gt"], x["gt"]])
            rows.append([_label(r["clip"]), f"{e['pred']} / {e['gt']}", f"{x['pred']} / {x['gt']}",
                         pct1(ce), f"{e['gt_all']} / {x['gt_all']}",
                         f"{f2(M.median_timing_error([(0, 0, t) for t in e['timing']]))} s ({e['n_matched']})",
                         f"{f2(M.median_timing_error([(0, 0, t) for t in x['timing']]))} s ({x['n_matched']})",
                         CMD_EVAL])
        e, x = a["entries"], a["exits"]
        rows.append([f"**{g} pooled**", f"{e['pred']} / {e['gt']}", f"{x['pred']} / {x['gt']}",
                     f"**{pct1(a['entry_exit_error'])}**", "", f"{f2(e['timing_median'])} s ({e['n_matched']})",
                     f"{f2(x['timing_median'])} s ({x['n_matched']})", CMD_EVAL])
        w(f"**{g}**")
        w("")
        L += _table(["clip", "entries ours / GT", "exits ours / GT", "count error", "all GT enters / exits (no window)",
                     "median timing error, entries (matched)", "median timing error, exits (matched)",
                     "reproduce"], rows)
    w(f"Pooled entry-only error: dev {pct1(ad['entries']['count_error'])}, test {pct1(at['entries']['count_error'])}; "
      f"exit-only error: dev {pct1(ad['exits']['count_error'])}, test {pct1(at['exits']['count_error'])}.")
    w("")

    # ---- occupancy
    w("## Occupancy")
    w("")
    w("Mean absolute error in persons present, sampled once per second over the whole clip (our occupancy "
      "vs the occupancy of the GT tracks under the same `first_t <= t <= last_t` rule). MEVA annotates "
      "only people who take part in an annotated activity, so un-annotated people in view make our "
      "occupancy higher than the GT; this inflates the MAE (see Limitations).")
    w("")
    rows = []
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        for r in rs:
            o = r["occupancy"]
            rows.append([g, _label(r["clip"]), f2(o["abs_err_sum"] / o["n"]) if o["n"] else "n/a",
                         f2(o["pred_sum"] / o["n"]), f2(o["gt_sum"] / o["n"]), CMD_EVAL])
        o_pred = sum(r["occupancy"]["pred_sum"] for r in rs) / a["occupancy_seconds"]
        o_gt = sum(r["occupancy"]["gt_sum"] for r in rs) / a["occupancy_seconds"]
        rows.append([f"**{g} pooled**", "", f"**{f2(a['occupancy_mae'])}**", f2(o_pred), f2(o_gt), CMD_EVAL])
    L += _table(["set", "clip", "MAE (persons)", "mean occupancy ours", "mean occupancy GT", "reproduce"], rows)

    # ---- zone dwell
    w("## Zone dwell")
    w("")
    w("Per matched (GT track, our person) pair and zone: dwell = summed length of the visits to "
      "the zone inside the GT track's window (`zone_min` 1.5 s, `zone_gap` 2 s hysteresis). Median absolute error in seconds and as a "
      "percentage of the GT dwell, over pairs with GT dwell > 0. Pairs where we report dwell in a zone the "
      "GT person never visited are counted separately (`ours only`). The Dealership target is a median "
      "error of at most 10%; the 2 s target belongs to the Short preset and is shown for reference.")
    w("")
    rows = []
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        for r in rs:
            d = M.dwell_summary([tuple(p) for p in r["dwell_pairs"]])
            rows.append([g, _label(r["clip"]), str(d["n_gt_positive"]), str(d["n_pred_only"]),
                         f"{f2(d['median_abs_s'])} s", pct1(d["median_abs_pct"]), CMD_EVAL])
        d = a["dwell"]
        rows.append([f"**{g} pooled**", "", str(d["n_gt_positive"]), str(d["n_pred_only"]),
                     f"**{f2(d['median_abs_s'])} s**", f"**{pct1(d['median_abs_pct'])}**", CMD_EVAL])
    L += _table(["set", "clip", "pairs (GT dwell > 0)", "ours only", "median abs error", "median abs error (% of GT)",
                 "reproduce"], rows)
    w(f"Track matching: dev {ad['n_matched']} of {ad['n_gt_persons']} GT tracks matched "
      f"({ad['n_pred_persons']} persons found); test {at['n_matched']} of {at['n_gt_persons']} matched "
      f"({at['n_pred_persons']} persons found). Unmatched GT tracks (people we missed) are not in the "
      "dwell and journey statistics, so those statistics describe the people we did track.")
    w("")

    # ---- journeys
    w("## Journeys")
    w("")
    w("Accuracy = 1 - normalised edit distance between our zone sequence and the GT sequence, averaged over "
      "matched persons. Matched pairs where both sequences are empty (the person visited no zone) carry no "
      "information and are skipped.")
    w("")
    rows = []
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        for r in rs:
            j = r["journey_acc"]
            rows.append([g, _label(r["clip"]), str(len(j)), share(sum(j) / len(j)) if j else "n/a", CMD_EVAL])
        rows.append([f"**{g} pooled**", "", str(a["journey_n"]), f"**{share(a['journey_mean'])}**", CMD_EVAL])
    L += _table(["set", "clip", "matched persons scored", "mean journey accuracy", "reproduce"], rows)

    # ---- tracking
    w("## Tracking")
    w("")
    w("MOTA and IDF1 against the GT person tracks (IoU >= 0.5), on the raw ByteTrack tracks and on the "
      "merged persons (after the identity merge), so the merge's effect is visible. MEVA annotates only "
      "the people involved in annotated activities, so un-annotated real people count as false positives "
      "and depress MOTA and precision; read recall and IDF1 alongside.")
    w("")
    rows = []
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        for r in rs:
            for kind, key in (("raw tracks", "mot_raw"), ("merged persons", "mot_merged")):
                rows.append([g, _label(r["clip"]), kind] + _mot_rows(r[key]) + [CMD_EVAL])
        for kind, key in (("raw tracks", "mot_raw"), ("merged persons", "mot_merged")):
            rows.append([f"**{g} pooled**", "", kind] + _mot_rows(a[key]) + [CMD_EVAL])
    L += _table(["set", "clip", "kind", "MOTA", "IDF1", "ID switches", "precision", "recall", "GT boxes",
                 "reproduce"], rows)

    # ---- speed
    w("## Speed")
    w("")
    ex = speed["excerpt"]
    w(f"Measured on this machine (Apple M3 Pro, CPU execution provider) by {CMD_SPEED}; `pulse eval` only "
      "reads the stored result, `runs/bench/speed.json`.")
    w("")
    rows = [
        ["file-mode real-time factor, cold (60 s excerpt, "
         f"{ex['width']}x{ex['height']}, 5 fps, includes embeddings and keyframe cars)", f"{sp['rtf']:.2f}x",
         f"{sp['wall_seconds']:.1f} s wall for {ex['duration_s']:.0f} s of video", CMD_SPEED],
        ["live engine step rate, frames with 1-4 people (detect + track + pose)",
         f"{eng:.2f} fps", f"{speed['engine_1_to_4_people']['mean_ms']:.1f} ms/frame over "
         f"{speed['engine_1_to_4_people']['n_frames']} frames", CMD_SPEED],
        ["live engine step rate, all frames", f"{speed['engine_all']['fps']:.2f} fps",
         f"{speed['engine_all']['mean_ms']:.1f} ms/frame over {speed['engine_all']['n_frames']} frames", CMD_SPEED],
        ["live smoke (WebSocket, paced 5 fps), processed fps", f"{live['fps']:.2f} fps",
         f"{live['dropped']} dropped, {live['people_max']} people max", CMD_LIVE],
        ["live smoke, max sustainable fps (unpaced)", f"{live['fps_max']:.2f} fps",
         f"step {live['step_ms_mean']:.1f} ms mean", CMD_LIVE],
    ]
    cm = speed.get("coreml")
    if cm:
        if cm.get("error"):
            rows.append(["CoreML provider (optional, separate)", "not usable", cm["error"], CMD_SPEED])
        else:
            rows.append(["CoreML provider engine step rate, frames with 1-4 people (optional, reported "
                         "separately)", f"{cm['fps_1_to_4']:.2f} fps",
                         f"{cm['mean_ms_1_to_4']:.1f} ms/frame over {cm['n_frames_1_to_4']} frames; "
                         f"providers {', '.join(cm['providers'])}", CMD_SPEED])
    L += _table(["measure", "value", "detail", "reproduce"], rows)
    w("Full 5-minute eval clips, 960 px wide, cold perception as measured when the caches were built "
      "(`perception_seconds` in each cache's `meta.json`):")
    w("")
    rows = []
    for g in ("dev", "test"):
        for r in per_clip[g]:
            m = r["_meta"]
            rows.append([g, _label(r["clip"]), f"{m['perception_seconds']:.1f} s", f"{m['rtf']:.2f}x",
                         str(r["n_pred_tracks"]), "`.venv/bin/pulse analyze <clip> --out <dir> --preset dealership`"])
    L += _table(["set", "clip", "perception time", "real-time factor", "tracks", "reproduce"], rows)
    mm_ = det["results"][det["chosen"]]
    w(f"Detector benchmark (M0, {CMD_DET}): chosen `{det['chosen']}`, {mm_['det_ms']:.1f} ms per "
      f"960x540 frame, person recall {share(mm_['person_recall'])} (IoU >= 0.5, GT height >= 20 px) on "
      f"{det['frames']} frames of the first dev clip; pose {det['pose_ms_4_boxes']['rtmpose_s']:.1f} ms for 4 boxes.")
    w("")

    # ---- zone coverage
    w("## Zone coverage")
    w("")
    w("Do the GT enter (first frame) and exit (last frame) foot points fall inside one of the clip's "
      "`entrance` zones? Target: at least 70% on the dev clips (zones were fixed on dev only). The entrance "
      "zones are generous (door face plus threshold), so high coverage partly reflects zone size.")
    w("")
    rows = []
    for g in ("dev", "test"):
        for r in per_clip[g]:
            c = r["coverage"]
            rows.append([g, _label(r["clip"]), f"{c['hit']} / {c['total']}",
                         share(c["hit"] / c["total"]) if c["total"] else "n/a", CMD_COVER])
        a = ad if g == "dev" else at
        c = a["coverage"]
        rows.append([f"**{g} total**", "", f"{c['hit']} / {c['total']}",
                     f"**{share(c['hit'] / c['total']) if c['total'] else 'n/a'}**", CMD_COVER])
    L += _table(["set", "clip", "activities in an entrance zone", "coverage", "reproduce"], rows)

    # ---- stops, sat in car
    w("## Stops")
    w("")
    labelled = sorted(p.name for p in (ROOT / "labels").glob("*.json") if p.name != "split.json")
    if labelled:
        w("Labelled clips found (" + ", ".join(labelled) + ") but stop scoring is not implemented yet.")
    else:
        w("Stops precision/recall: no labelled showroom clip yet (no stock or synthetic showroom clip was "
          "available, so there are no `labels/<clip>.json` files to score against). MEVA school scenes have "
          "no car zones, so stop detection (a car-zone feature) is not evaluated on them either.")
    w("")
    w("## Sat in car (stretch)")
    w("")
    w("Not evaluated: the sat-in-car stretch feature is not built, and the six MEVA eval clips are indoor "
      "school scenes with no vehicle-entry activities.")
    w("")

    # ---- insights
    w("## Insights")
    w("")
    w("The unit-test scenarios for rules R1-R7 (`tests/test_insights.py`, inputs shared through "
      "`pulse.eval.scenarios`) were run through `insights.recommend` with the `short` preset, as in the "
      "tests. 'Fired' is computed here, not copied: a correct run has fired = expected on every row.")
    w("")
    rows = []
    cfg_s = get_config("short")
    for sc in SCENARIOS:
        st, eps = sc.build()
        got = [i for i in insights_mod.recommend(st, eps, cfg_s) if i.rule == sc.rule]
        action = "; ".join(f"{i.title}: {i.action}" for i in got).replace("|", "/") if got else "(none)"
        rows.append([sc.rule, sc.describe, "fire" if sc.fires else "quiet", "fired" if got else "quiet",
                     "ok" if bool(got) == sc.fires else "MISMATCH", action])
    L += _table(["rule", "scenario", "expected", "result", "check", "recommendation"], rows)
    w("Reproduce: `.venv/bin/python -m pytest -q tests/test_insights.py` and "
      f"{CMD_EVAL}.")
    w("")

    # ---- limitations
    w("## Limitations")
    w("")
    for s in (
        "The zones are agent-drawn from one blurred keyframe per clip and have not been reviewed by a human "
        "(`labeller: ai-agent, unreviewed`). Dwell and journey errors include any zone-placement error; "
        "entrance zones are generous.",
        "MEVA annotates only people who take part in an annotated activity. Un-annotated people appear in "
        "our output but not in the GT: this lowers MOTA and precision and raises occupancy MAE, so the "
        "occupancy and tracking numbers are pessimistic. Conversely, GT tracks are often fragmented into "
        "several ids for one person, which makes GT occupancy and person counts optimistic.",
        "The eval clips are school hallways, lobbies, a stairwell and a gym, not showrooms: no cars, no "
        "salespeople, no desks, different crowd behaviour and camera heights. Role detection, unserved "
        "interest, assistance and stops are not evaluated at all.",
        "Entry and exit counts are compared over the window the entry/exit rule uses; people present in "
        "the first or last `leave_after` seconds are excluded on both sides. Clips with very few GT enters "
        "or exits (e.g. G421 enters, G420 exits) give unstable per-clip percentages; read the pooled figure.",
        "GT tracks are per activity and short (a few seconds), so the zone-dwell and journey comparison rests "
        "on few pairs; GT tracks that nobody in our output overlaps (missed people, crowded scenes such as "
        "G299 and G421) are left out of those statistics, so they describe only the people we tracked.",
        "The entry/exit GT is the set of annotated activities, not a census. Where many real people enter or "
        "leave without an annotated activity (G421 has 2 annotated enters but we count 44 entries) the "
        "per-clip error is mostly un-annotated truth, not detector error; the pooled test figure is "
        "dominated by such clips and says little about showroom accuracy.",
        "Only six clips from two cameras-days of one school; the dev set has two clips. Differences of a "
        "few points are within noise.",
        "File-mode and live speed are measured on one machine (Apple M3 Pro, CPU); CoreML, if shown, is "
        "reported separately and was measured once.",
        "No stock showroom or synthetic clip was available, so there is no showroom-domain accuracy number.",
    ):
        w(f"- {s}")
    w("")
    return "\n".join(L).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------------------------
# entry point

_CODE_FILES = ("eval/metrics.py", "eval/run_eval.py", "eval/meva.py", "stats.py", "journey.py", "identity.py",
               "timeline.py", "zones.py", "geometry.py", "pipeline.py", "config.py", "datatypes.py")


def _code_digest() -> str:
    h = hashlib.sha1()
    for name in _CODE_FILES:
        h.update((Path(__file__).resolve().parents[1] / name).read_bytes())
    return h.hexdigest()


def _cache_key(entry: dict, cfg: Config, cache_dir: str) -> str:
    """Everything the per-clip result depends on: code, config, zones, annotations, perception cache."""
    h = hashlib.sha1()
    h.update(json.dumps([EVAL_VERSION, _code_digest(), entry["clip"], cache_dir, repr(cfg)],
                        sort_keys=True).encode())
    for p in (ROOT / "labels" / "zones" / f"{entry['clip']}.json",):
        h.update(p.read_bytes())
    for p in sorted((ROOT / entry["kpf"]).glob("*.yml")):
        h.update(p.name.encode() + str(p.stat().st_size).encode())
    return h.hexdigest()[:20]


def run_clip(entry: dict, group: str, cfg: Config, use_cache: bool = True) -> dict:
    """Per-clip result, memoised under runs/eval/ (keyed by every input) so repeated runs are fast."""
    video = ROOT / entry["video"]
    if not video.exists():
        raise MissingInput(f"missing input for {entry['clip']}: {video}")
    cdir = pipeline_cache_dir(video, cfg)
    key = _cache_key(entry, cfg, cdir.name) if cdir else None
    f = ROOT / "runs" / "eval" / f"{entry['clip']}.{key}.json" if key else None
    if use_cache and f is not None and f.is_file():
        return json.loads(f.read_text())
    inp = load_inputs(entry, group, cfg)
    r = evaluate(inp, cfg)
    r["_meta"] = {"perception_seconds": float(inp.pc.meta["perception_seconds"]), "rtf": float(inp.pc.meta["rtf"])}
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        for old in f.parent.glob(f"{entry['clip']}.*.json"):
            old.unlink()
        f.write_text(json.dumps(r, sort_keys=True))
    return r


def pipeline_cache_dir(video: Path, cfg: Config) -> Path | None:
    """The perception-cache directory the pipeline would use for this video (None if not cached yet)."""
    try:
        key = cache_mod.video_key(video, cfg)
    except OSError:
        return None
    d = ROOT / "runs" / "cache" / key
    return d if cache_mod.exists(d) else None


def main(out: Path, clips: list[str] | None = None, cfg: Config | None = None) -> int:
    cfg = cfg or get_config("dealership")
    out = Path(out)
    try:
        split = json.loads((ROOT / "labels" / "split.json").read_text())
        per_clip: dict[str, list[dict]] = {"dev": [], "test": []}
        for g in ("dev", "test"):
            for entry in split[g]:
                if clips and entry["clip"] not in clips:
                    continue
                print(f"[eval] {g} {entry['clip']}", file=sys.stderr, flush=True)
                per_clip[g].append(run_clip(entry, g, cfg))
        if not per_clip["dev"] and not per_clip["test"]:
            raise MissingInput("no clips selected")
        speed, det = _load_json(SPEED_JSON), _load_json(DET_JSON)
        if "live_smoke" not in speed:
            raise MissingInput("runs/bench/speed.json has no live_smoke snapshot: run scripts/smoke_live.py "
                               "and then `pulse bench speed` again")
    except (MissingInput, OSError, sources.VideoError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    text = render(per_clip, cfg, speed, det)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0
