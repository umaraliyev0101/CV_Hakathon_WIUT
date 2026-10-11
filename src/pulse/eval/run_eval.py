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
from pulse import journey, pipeline, sources, zones as zones_mod
from pulse import stats as stats_mod
from pulse import insights as insights_mod
from pulse.cache import PerceptionCache
from pulse.config import PRESETS, Config, get_config
from pulse.datatypes import Person, Sample, Timeline, Zone
from pulse.eval import metrics as M
from pulse.eval.meva import MevaClip, load_kpf
from pulse.eval.scenarios import SCENARIOS
from pulse.eval import stops as stops_mod
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
STOPS_PRESET = "short"                 # labelled stock clips are short showroom clips
CMD_STOPS = "`.venv/bin/pulse eval --out docs/EVAL.md`"
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
        "obs_total": len(inp.pc.obs), "obs_not_in_person": sum(1 for o in inp.pc.obs if o.track_id not in t2p),
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
# stops on hand-labelled stock clips

def labelled_clips() -> list[Path]:
    out = []
    for p in sorted((ROOT / "labels").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and "labeller" in d and "stops" in d:
            out.append(p)
    return out


def _stops_video(clip: str) -> Path:
    for d in ("data/stock", "data/synthetic/share"):
        p = ROOT / d / f"{clip}.mp4"
        if p.is_file():
            return p
    raise MissingInput(f"missing video for labelled clip {clip} (looked in data/stock, data/synthetic/share)")


def evaluate_stops(label_path: Path, cfg: Config) -> dict:
    """Predicted stopped ZoneVisits (everyone a customer, hand-drawn zones, stable-camera samples only) vs
    the hand-labelled stops. Reads the perception cache only; it never runs a model."""
    lab = json.loads(Path(label_path).read_text())
    clip = lab["clip"]
    video = _stops_video(clip)
    cdir = pipeline_cache_dir(video, cfg)
    if cdir is None:
        raise MissingInput(f"no perception cache for {clip}: run `.venv/bin/pulse analyze {video.relative_to(ROOT)} "
                           f"--preset {cfg.preset.name} --zones {lab['zones_file']} --out runs/tmp/{clip}` first")
    pc = cache_mod.load(cdir)
    zones = load_zones(ROOT / lab["zones_file"])
    _, tl, _ = pipeline.build_people(pc, cfg)
    stable = filter_stable(tl, pc.segments)
    journeys = journey.build(stable, zones, cfg.preset)
    visits = [v for vs in journeys.values() for v in vs]
    dur = float(pc.meta.get("analysed_duration") or pc.meta.get("duration") or 0.0)
    tol = 2.0
    ivs = [(s.t0, min(s.t1, dur)) for s in pc.segments if s.stable and s.same_view and s.t0 < dur]
    gt_u = stops_mod.usable_gt(lab["stops"], ivs, cfg.preset.stop_time)
    pr = stops_mod.precision_recall(visits, lab["stops"], tol)
    pr1 = stops_mod.precision_recall(visits, lab["stops"], tol, one_to_one=True)
    pr_u = stops_mod.precision_recall(visits, gt_u, tol)
    pr1_u = stops_mod.precision_recall(visits, gt_u, tol, one_to_one=True)
    usable = sum(max(0.0, min(s.t1, dur) - s.t0) for s in pc.segments if s.stable and s.same_view)
    return {
        "clip": clip, "labeller": lab["labeller"], "preset": cfg.preset.name, "duration": dur,
        "usable_seconds": usable, "n_gt_visits": len(lab["visits"]), "n_pred_persons": len(journeys),
        "n_pred_zone_visits": len(visits),
        "pred_stops": [[v.person_id, v.zone_id, round(v.t_in, 1), round(v.t_out, 1)]
                       for v in sorted(visits, key=lambda v: (v.t_in, v.person_id)) if v.stopped],
        "gt_stops": [[g["visit_id"], g["zone_id"], g["t_start"], g["t_end"]] for g in lab["stops"]],
        "pr": pr, "pr_1to1": pr1, "pr_usable": pr_u, "pr_1to1_usable": pr1_u,
        "n_gt_usable": len(gt_u),
    }


def run_stops(label_path: Path, cfg: Config, use_cache: bool = True) -> dict:
    """evaluate_stops memoised under runs/eval/ (keyed by code, config, labels, zones, perception cache)."""
    lab = json.loads(Path(label_path).read_text())
    cdir = pipeline_cache_dir(_stops_video(lab["clip"]), cfg)
    key = None
    if cdir is not None:
        h = hashlib.sha1(json.dumps([EVAL_VERSION, _code_digest(), cdir.name, repr(cfg)]).encode())
        h.update(Path(label_path).read_bytes())
        h.update((ROOT / lab["zones_file"]).read_bytes())
        key = h.hexdigest()[:20]
    f = ROOT / "runs" / "eval" / f"stops-{lab['clip']}.{key}.json" if key else None
    if use_cache and f is not None and f.is_file():
        return json.loads(f.read_text())
    r = evaluate_stops(label_path, cfg)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        for old in f.parent.glob(f"stops-{lab['clip']}.*.json"):
            old.unlink()
        f.write_text(json.dumps(r, sort_keys=True))
    return r


def _iv(rows: list, tag: str) -> str:
    return "; ".join(f"{tag}{r[0]} at {r[1]} {r[2]:g}-{r[3]:g} s" for r in rows) or "none"


def _stops_section(res: list[dict]) -> list[str]:
    L: list[str] = []
    w = L.append
    if not res:
        w("Stops precision/recall: no labelled showroom clip found (no `labels/<clip>.json` besides "
          "`split.json`), so nothing is scored. MEVA school scenes have no car zones, so stop detection (a "
          "car-zone feature) is not evaluated on them either.")
        w("")
        return L
    tol = res[0]["pr"]["tol_s"]
    pre = res[0]["preset"]
    w(f"Stop = a person at a car zone with foot-point speed below `stop_speed` for at least `stop_time` "
      f"(preset `{pre}`: {PRESETS[pre].stop_time:g} s). Ground truth: `labels/<clip>.json` stops, made by "
      "viewing face-blurred frames sampled at 1 s, with hand-drawn car zones in `labels/zones/<clip>.json`; "
      "both carry `labeller: ai-agent, unreviewed`. The labels were written before the pipeline's stop "
      "output for these clips was looked at, and nothing was tuned on them. Ours = `ZoneVisit.stopped` from "
      "the same `journey` code as everywhere else, run on the cached perception output (stable-camera "
      "samples only, every person scored as a customer, no role detection). A predicted stop matches a GT "
      f"stop at the same zone when the intervals overlap within +-{tol:g} s. Two matchings are shown: "
      "many-to-many (precision = predicted stops matching any GT stop / predicted stops; recall = GT stops "
      "matched by any predicted stop / GT stops) and one-to-one (each stop used once, greedy by overlap). "
      "`n/a` = zero denominator.")
    w("")
    def cell(k, n):
        if not n:
            return "n/a"
        lo, hi = stops_mod.wilson(k, n)
        return f"{k}/{n} = {k / n * 100:.0f}% (95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"

    def pooled(key, a, b):
        return cell(sum(r[key][a] for r in res), sum(r[key][b] for r in res))

    def table(prec_key, rec_key, rec_u_key, title):
        w(title)
        w("")
        rows = []
        for r in res:
            q, qu = r[prec_key], r[rec_u_key]
            rows.append([r["clip"], f"{r['duration']:.1f} s", f"{r['usable_seconds']:.1f} s",
                         cell(q["pred_matched"], q["n_pred"]), cell(r[rec_key]["gt_matched"], r[rec_key]["n_gt"]),
                         cell(qu["gt_matched"], qu["n_gt"])])
        rows.append(["pooled", "", "", pooled(prec_key, "pred_matched", "n_pred"),
                     pooled(rec_key, "gt_matched", "n_gt"), pooled(rec_u_key, "gt_matched", "n_gt")])
        L.extend(_table(["clip", "length", "usable (stable, same view)", "precision",
                         "recall, all GT stops", "recall, GT stops in usable time"], rows))

    table("pr", "pr", "pr_usable", "**Many-to-many matching**")
    table("pr_1to1", "pr_1to1", "pr_1to1_usable", "**One-to-one matching** (greedy by overlap)")
    w(f"Reproduce: {CMD_STOPS}. Counts come first; intervals are Wilson 95% intervals and are wide because the "
      "sample is tiny. **Recall over all GT stops mostly measures the camera-motion filter on "
      "`wm-bangalore-04`**: the pipeline only scores samples whose view matches the keyframe (the `usable` "
      "column), so GT stops in the panning part of the clip cannot be found. The last column keeps only GT "
      f"stops with at least `stop_time` ({PRESETS[pre].stop_time:g} s) inside usable time; that is the number "
      "that says whether stop detection works when the view is valid. Many-to-many lets one prediction "
      "satisfy several GT stops: on `wm-bangalore-04` GT v2 (0-61 s) and v5 (41-61 s) share predictions, "
      "which is why the one-to-one table is shown beside it.")
    w("")
    w("Stops listed (GT `v<visit id>`, ours `p<predicted person id>`, each with zone and start-end):")
    w("")
    for r in res:
        w(f"- `{r['clip']}` GT: {_iv(r['gt_stops'], 'v')}. Ours: {_iv(r['pred_stops'], 'p')}.")
    w("")
    w("Caveats:")
    w("")
    for s in (
        "The labeller is an AI agent working from 1 s frames of a hand-held camera, unreviewed; people ids "
        "and stop intervals are approximate (a few seconds), and several labels are low confidence (see "
        "`notes` in each label file and `PROGRESS.md`).",
        "Tiny sample: a few stops on two clips. One stop more or less moves a ratio by tens of points. "
        "Read these as an indication of whether stops work on showroom-like footage, not as an accuracy.",
        "`wm-bangalore-04` is a crowded car display; roughly 15 spectators stand behind a barrier and are "
        "neither labelled nor inside the zone, so any stop the pipeline predicts for them counts against "
        "precision. The camera is hand-held and pans during the first 36 s; "
        "zones are only valid while the view matches the keyframe, so the pipeline drops those samples "
        "(the `usable` column) and every GT stop inside them counts against recall.",
        "`wm-bangalore-07` has no GT stop: every predicted stop there is a false positive and its recall is "
        "`n/a`; only the pooled recall carries information.",
        "Everyone is scored as a customer and no synthetic (teammate C, `data/synthetic/share/`) clips "
        "existed when this was run, so no dwell, journey or walk-out errors are reported for showroom clips.",
    ):
        w(f"- {s}")
    w("")
    return L


# ---------------------------------------------------------------------------------------------
# the document

def _mot_rows(m: dict) -> list[str]:
    return [share(m["mota"]), share(m["idf1"]), str(m["num_switches"]), share(m["precision"]),
            share(m["recall"]), str(m["num_objects"])]


# ---------------------------------------------------------------------------------------------
# sat in car (stretch): recall against MEVA person_enters_vehicle on one outdoor clip

SAT_CLIP = {"clip": "2018-03-15.15-45-07.15-50-07.hospital.G436",
            "video": "data/meva/videos/2018-03-15.15-45-07.15-50-07.hospital.G436.r13.avi",
            "kpf": "data/meva/annotations/outdoor_2018-03-15.15-45-07.15-50-07.hospital.G436"}
SAT_ACT = "person_enters_vehicle"
SAT_WINDOW_S = 5.0


def _overlap_min(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / m if iw > 0 and ih > 0 and m > 0 else 0.0


def evaluate_sat_in_car(cfg: Config) -> dict | None:
    """Recall of GT `person_enters_vehicle` by detected sat-in-car events (either rule), from the cache.

    A GT activity counts as found when a detected event's time span meets [start - 5 s, end + 5 s] AND its
    keyframe car box overlaps the GT vehicle's box (at the activity's last frame) by at least half of the
    smaller box ("same vehicle"). The `any_car` figure drops the spatial condition (upper bound). None if
    the clip or its cache is missing."""
    e = SAT_CLIP
    video = ROOT / e["video"]
    cdir = pipeline_cache_dir(video, cfg) if video.is_file() else None
    if cdir is None or not (ROOT / e["kpf"]).is_dir():
        return None
    pc = cache_mod.load(cdir)
    persons, tl, t2p = pipeline.build_people(pc, cfg)
    size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
    zones = zones_mod.auto_zones(pc.cars, (size[1], size[0]))
    ev = pipeline.sat_in_car_events(pc, filter_stable(tl, pc.segments), t2p, zones, cfg)
    meva = load_kpf(ROOT / e["kpf"], e["clip"])
    fps = float(pc.meta["src_fps"])
    sx = size[0] / int(pc.meta["src_w"])
    gt = []
    for a in sorted((a for a in meva.activities if a.name == SAT_ACT), key=lambda a: (a.frame_start, a.act_id)):
        t0, t1 = a.frame_start / fps, a.frame_end / fps
        vbox = None
        for i in a.actors:
            if meva.track_types.get(i) == "vehicle" and meva.tracks.get(i):
                fr = min(meva.tracks[i], key=lambda f: abs(f - a.frame_end))
                vbox = tuple(v * sx for v in meva.tracks[i][fr])
        near = [x for x in ev if x.t_start <= t1 + SAT_WINDOW_S and (x.t_end or x.t_start) >= t0 - SAT_WINDOW_S]
        same = [x for x in near if vbox and _overlap_min(x.data["car_bbox"], vbox) >= 0.5]
        gt.append({"act_id": a.act_id, "t_start": round(t0, 1), "t_end": round(t1, 1),
                   "vehicle_in_keyframe_car": bool(vbox and any(
                       _overlap_min(d.bbox, vbox) >= 0.5 for d in pc.cars)),
                   "found_same_vehicle": bool(same), "found_any_car": bool(near)})
    n = len(gt)
    return {
        "clip": e["clip"], "n_gt": n, "found_same": sum(g["found_same_vehicle"] for g in gt),
        "found_any": sum(g["found_any_car"] for g in gt),
        "n_events": len(ev), "n_events_a": sum(x.data["rule"] == "a" for x in ev),
        "n_events_b": sum(x.data["rule"] == "b" for x in ev), "n_cars": len(pc.cars),
        "n_gt_on_keyframe_car": sum(g["vehicle_in_keyframe_car"] for g in gt),
        "n_persons": len(persons), "n_exit_gt": sum(a.name == "person_exits_vehicle" for a in meva.activities),
        "gt": gt,
        "events": [[x.data["rule"], x.person_ids[0], round(x.t_start, 1), round(x.t_end or x.t_start, 1)]
                   for x in ev],
    }


def run_sat_in_car(cfg: Config, use_cache: bool = True) -> dict | None:
    """evaluate_sat_in_car memoised under runs/eval/ (keyed by code, config, annotations, perception cache)."""
    video = ROOT / SAT_CLIP["video"]
    cdir = pipeline_cache_dir(video, cfg) if video.is_file() else None
    if cdir is None:
        return None
    h = hashlib.sha1(json.dumps([EVAL_VERSION, _code_digest(), cdir.name, repr(cfg)]).encode())
    for p in sorted((ROOT / SAT_CLIP["kpf"]).glob("*.yml")):
        h.update(p.name.encode() + str(p.stat().st_size).encode())
    f = ROOT / "runs" / "eval" / f"sat-{SAT_CLIP['clip']}.{h.hexdigest()[:20]}.json"
    if use_cache and f.is_file():
        return json.loads(f.read_text())
    r = evaluate_sat_in_car(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    for old in f.parent.glob(f"sat-{SAT_CLIP['clip']}.*.json"):
        old.unlink()
    f.write_text(json.dumps(r, sort_keys=True))
    return r


def _sat_section(r: dict | None) -> list[str]:
    L: list[str] = []
    if r is None:
        return ["The outdoor clip " + SAT_CLIP["clip"] + " or its perception cache is not present on this "
                "machine, so sat-in-car was not evaluated.", ""]
    n = r["n_gt"]
    L.append("Recall of the MEVA `person_enters_vehicle` activities on one **outdoor parking-lot / drive-loop "
             f"clip** (`{r['clip']}`, hospital camera G436, CC BY 4.0). This is a proxy only, not a "
             "showroom: the vehicles are driving in and out, the people are small and far away, so pose "
             "keypoints are unreliable, and the rule's assumption that cars are fixed (one keyframe box per "
             "car) is broken. Numbers describe the proxy and say little about showroom accuracy. "
             f"Reproduce: {CMD_EVAL}.")
    L.append("")
    L.append("A GT activity is *found* when a detected sat-in-car event (rule a or b) overlaps "
             f"[start - {SAT_WINDOW_S:.0f} s, end + {SAT_WINDOW_S:.0f} s] and its keyframe car box overlaps "
             "the GT vehicle box (at the activity's last frame) by at least half of the smaller box. "
             "`any car` drops the spatial condition (an upper bound that ignores which vehicle).")
    L.append("")
    rows = [["GT `person_enters_vehicle` activities", str(n)],
            ["GT vehicle overlaps a keyframe car box (detectable at all)", f"{r['n_gt_on_keyframe_car']} / {n}"],
            ["found, same vehicle (recall)", f"{r['found_same']} / {n} = {share(r['found_same'] / n) if n else 'n/a'}"],
            ["found, any car (upper bound)", f"{r['found_any']} / {n} = {share(r['found_any'] / n) if n else 'n/a'}"],
            ["keyframe vehicle detections", str(r["n_cars"])],
            ["detected sat-in-car events (a / b)", f"{r['n_events']} ({r['n_events_a']} / {r['n_events_b']})"],
            ["GT `person_exits_vehicle` activities (not scored)", str(r["n_exit_gt"])]]
    L += _table(["measure", "value"], rows)
    if r["n_gt_on_keyframe_car"] == 0:
        L.append("**Reading:** none of the GT vehicles sits on a keyframe car box (the keyframe shows the "
                 "vehicles of one instant; the ones that carry passengers drive in and park elsewhere), so the "
                 "static-car rules cannot match any activity by construction. This clip is effectively "
                 "unusable for scoring the rules: the 0% is a limitation of the proxy, not a measurement of "
                 "sat-in-car quality, and no showroom accuracy claim follows from it.")
        L.append("")
    L.append("Precision is not reported: MEVA annotates only people in annotated activities, so most detected "
             "events cannot be judged against ground truth. The recall figure rests on "
             f"{n} activities and one clip.")
    L.append("")
    return L


def render(per_clip: dict[str, list[dict]], cfg: Config, speed: dict, det: dict,
           stops: list[dict] | None = None, sat: dict | None = None) -> str:
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
      "The selection objective was written down before the first run and revised twice after seeing dev "
      "results (the per-term cap of 3 was removed, and an interior-optimum condition was added); a first "
      "run also used a wrong count-error formula and was discarded. The optimum (0.9) is the same under "
      "every version of the rule. **Test numbers were visible during tuning:** draft full evaluations of "
      "all six clips at the then-default 0.75 (including the buggy formula) were run and read before "
      "`merge_sim_thr` was changed, and the switch from one-to-one to many-to-one track matching (below) "
      "was made after seeing match rates on dev and test clips. The tuner itself never loads the test "
      "clips and its objective used dev numbers only; after the freeze only report rendering and caching "
      "code changed (no metric or configuration change). With MEVA ground truth this tuning rests on two "
      "clips and counts only, so treat the new threshold as a dev-set fit, not a general optimum; at 0.9 "
      "the merge joins few tracks and, on test, does not improve IDF1 (see Tracking).")
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
    eng = speed["engine_1_to_4_people"]["fps"]
    full = [(r["clip"], r["_meta"]["rtf"]) for g in ("dev", "test") for r in per_clip[g]]
    below = [f"{_label(c)} {x:.2f}x" for c, x in full if x < 1.0]
    rtf_met = (sp["rtf"] >= 1.0) and not below
    rtf_text = (f"{sp['rtf']:.2f}x on the 60 s excerpt; full 5-min clips {min(x for _, x in full):.2f}x-"
                f"{max(x for _, x in full):.2f}x" + (f" (below 1x: {', '.join(below)})" if below else ""))
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
        ["file mode real-time factor (CPU, 5 fps; 60 s 720p excerpt, then the six full 5-min clips)", ">= 1.0x",
         rtf_text, "(speed is not split by clip)",
         "yes" if rtf_met else ("partly" if sp["rtf"] >= 1.0 else "NO")],
        ["live mode fps (engine step on frames with 1-4 people; live smoke paced / max, "
         f"at most {live['people_max']} people in view)", ">= 5 fps",
         f"{eng:.2f} engine; smoke {live['fps']:.2f} paced, {live['fps_max']:.2f} max",
         "(not split by clip)", _met(eng >= 5.0 and live["fps"] >= 5.0)],
    ]
    L += _table(["metric", "target", "measured (test)", "measured (dev)", "met?"], rows)
    w(f"Reproduce: {CMD_EVAL} for the first four rows; {CMD_SPEED} for file mode; {CMD_LIVE} then "
      f"{CMD_SPEED} for live fps (the bench snapshots the smoke result into `runs/bench/speed.json`, so "
      "EVAL.md does not change when `verify.sh` re-runs the smoke). "
      f"Engine-only step rate on frames with 1-4 people: {eng:.2f} fps (see Speed).")
    w("")
    miss = [r[0] for r in rows if r[4] == "NO"]
    part = [r[0] for r in rows if r[4] == "partly"]
    w("**Targets missed (test):** " + ("; ".join(miss) if miss else "none") + ".")
    if part:
        w("")
        w("**Targets only partly met:** " + "; ".join(part) + ".")
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
    w("**What this measures:** MEVA tracks are annotated per activity and are short (median about 1.2 s on "
      "the first dev clip; many are under `zone_min`), and each GT track is matched many-to-one inside its "
      "own time window. These dwell figures therefore measure zone assignment inside short GT snippets, not "
      "a person's dwell over a whole visit, and they are invariant to the identity merge by construction "
      "(identical across all 42 tuning trials). They cannot detect merge errors.")
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
    w("**What this measures:** the same windowing and many-to-one matching as Zone dwell. The sequences "
      "compared are the zones visited inside one short GT activity track, not a person's end-to-end "
      "journey through a visit, so the 80% target is judged on a metric that is invariant to identity "
      "merging (unchanged across all 42 tuning trials) and says little about whole-visit journeys.")
    w("")
    rows = []
    for g, rs, a in (("dev", dev, ad), ("test", test, at)):
        for r in rs:
            j = r["journey_acc"]
            rows.append([g, _label(r["clip"]), str(len(j)), share(sum(j) / len(j)) if j else "n/a", CMD_EVAL])
        rows.append([f"**{g} pooled**", "", str(a["journey_n"]), f"**{share(a['journey_mean'])}**", CMD_EVAL])
    L += _table(["set", "clip", "matched persons scored", "mean journey accuracy", "reproduce"], rows)

    # ---- tracking
    tot_obs = sum(r["obs_total"] for g in ("dev", "test") for r in per_clip[g])
    tot_drop = sum(r["obs_not_in_person"] for g in ("dev", "test") for r in per_clip[g])
    w("## Tracking")
    w("")
    w("MOTA and IDF1 against the GT person tracks (IoU >= 0.5), on the raw ByteTrack tracks and on the "
      "merged persons (after the identity merge), so the merge's effect is visible. MEVA annotates only "
      "the people involved in annotated activities, so un-annotated real people count as false positives "
      "and depress MOTA and precision; read recall and IDF1 alongside. Raw and merged are scored on the same "
      f"observations ({tot_obs - tot_drop} of {tot_obs} observations are in a person; "
      f"{tot_drop} tracked observations belong to no person and would be missing from the merged set). "
      f"At `merge_sim_thr` {cfg.merge_sim_thr} the merge does not help IDF1: pooled test IDF1 "
      f"{share(at['mot_raw']['idf1'])} raw vs {share(at['mot_merged']['idf1'])} merged, ID switches "
      f"{at['mot_raw']['num_switches']} vs {at['mot_merged']['num_switches']}; pooled dev IDF1 "
      f"{share(ad['mot_raw']['idf1'])} vs {share(ad['mot_merged']['idf1'])}. The merge was tuned for "
      "person counts, not for IDF1 (GT ids are per activity, so IDF1 penalises merging).")
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
    w(f"Measured on `{speed['machine']}` (CPU execution provider) by {CMD_SPEED}; `pulse eval` only "
      "reads the stored result, `runs/bench/speed.json`, recorded at git revision "
      f"`{speed.get('git_rev', 'unknown')}` with config hash `{speed.get('config_hash', 'unknown')}` "
      "(a stale file shows up as a different revision).")
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
      "zones are generous (door face plus threshold), so high coverage partly reflects zone size. The "
      "G419 test clip (stairwell) is below 70% (not tuned).")
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
    L += _stops_section(stops or [])
    w("## Sat in car (stretch)")
    w("")
    L += _sat_section(sat)

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
        "The file-mode real-time factor of the Targets table is measured on a quiet 60 s excerpt; on the "
        "full 5-minute clips the crowded scenes (G421 lobby, G299 gym, 320+ tracks) run below real time "
        "(see Speed). Real-time file mode is not met on crowded footage with this CPU configuration.",
        "Only six clips from two cameras-days of one school; the dev set has two clips. Differences of a "
        "few points are within noise.",
        "File-mode and live speed are measured on one machine (see Speed), CPU; CoreML, if shown, is "
        "reported separately and was measured once.",
        "The only showroom-domain numbers are the stop precision/recall on two stock auto-show clips "
        "(Stops section): agent labels, unreviewed, a handful of stops; they are an indication, not an "
        "accuracy estimate. Entries, exits, occupancy, dwell and journeys have no showroom ground truth.",
    ):
        w(f"- {s}")
    w("")
    return "\n".join(L).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------------------------
# entry point

_CODE_FILES = ("eval/metrics.py", "eval/run_eval.py", "eval/meva.py", "stats.py", "journey.py", "identity.py",
               "eval/stops.py", "sat_in_car.py", "timeline.py", "zones.py", "geometry.py", "pipeline.py", "config.py", "datatypes.py")


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
        stops_cfg = get_config(STOPS_PRESET)
        stops_res = [run_stops(p, stops_cfg) for p in labelled_clips()]
    except (MissingInput, OSError, sources.VideoError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    text = render(per_clip, cfg, speed, det, stops_res, run_sat_in_car(cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0
