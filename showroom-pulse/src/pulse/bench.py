"""Detector benchmark (spec §5): person recall on MEVA, cars on showroom images, CPU speed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from pulse.config import get_config
from pulse.eval.meva import load_kpf
from pulse.perception.detect import VEHICLE_CLASSES, Detector
from pulse.perception.pose import PoseEstimator

CANDIDATES = ("yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "rfdetr_n", "rfdetr_s")
CAR_CLASSES = ("car", "truck", "motorcycle")
RULE_TEXT = ("Fastest detector whose person recall (IoU>=0.5, GT height>=20px) is within 5 "
             "points of the best AND which detects >=1 car|truck|motorcycle (score>=0.3) on "
             "every showroom image. If none qualifies: among detectors within 5 points of best "
             "recall, the one with the most vehicle detections (ties: fastest).")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _resize(img, max_w=960):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / w
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA), s


def sample_frames(video: Path, n: int, max_w: int = 960):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(i) for i in np.linspace(total * 0.02, total * 0.98, n)]
    want, out, i = set(idxs), {}, 0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while i <= max(idxs):
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                out[i] = _resize(f, max_w)[0]
        i += 1
    cap.release()
    return out, src_w


def match_recall(gt: list, dets: list, thr=0.5) -> tuple[int, int]:
    used, hit = set(), 0
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    for g in gt:
        best, bj = 0.0, -1
        for j in order:
            if j in used:
                continue
            v = _iou(g, dets[j].bbox)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            hit += 1
    return hit, len(gt)


def _time_ms(fn, runs=30, warm=3) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def choose(results: dict) -> tuple[str, str]:
    names = list(results)
    best_rec = max(results[n]["person_recall"] for n in names)
    near = [n for n in names if results[n]["person_recall"] >= best_rec - 0.05]
    ok = [n for n in near if results[n]["cars_all_images"]]
    if ok:
        return min(ok, key=lambda n: results[n]["det_ms"]), "rule met"
    pick = max(near, key=lambda n: (results[n]["cars_total"], -results[n]["det_ms"]))
    return pick, "no candidate met the car rule on every image; fallback rule applied"


def run_detector_bench(clip: Path, kpf: Path, showroom_imgs: list[Path], out: Path,
                       n_frames: int = 60, candidates=CANDIDATES) -> dict:
    clip = Path(clip)
    name = clip.name.removesuffix(".r13.avi").removesuffix(".avi")
    frames, src_w = sample_frames(clip, n_frames)
    mc = load_kpf(Path(kpf), name)
    persons = mc.person_tracks()
    gt_by_frame: dict[int, list] = {}
    for f, img in frames.items():
        s = img.shape[1] / src_w
        boxes = []
        for tr in persons.values():
            b = tr.get(f)
            if b and (b[3] - b[1]) * s >= 20:
                boxes.append(tuple(v * s for v in b))
        gt_by_frame[f] = boxes
    n_gt = sum(len(v) for v in gt_by_frame.values())
    showroom = [(p, _resize(cv2.imread(str(p)))[0]) for p in showroom_imgs]
    mid = frames[sorted(frames)[len(frames) // 2]]
    speed_img = cv2.resize(mid, (960, 540))
    pose_boxes = [(100, 100, 220, 400), (300, 120, 420, 420), (500, 100, 620, 400),
                  (700, 120, 820, 420)]
    cfg = get_config("short")
    res: dict[str, dict] = {}
    for cand in candidates:
        det = Detector(cand, cfg)
        hit = tot = 0
        for f, img in frames.items():
            h, t = match_recall(gt_by_frame[f], det.detect(img, ("person",)))
            hit, tot = hit + h, tot + t
        cars = {}
        for p, im in showroom:
            cars[p.name] = sum(1 for d in det.detect(im, CAR_CLASSES) if d.score >= 0.3)
        ms = _time_ms(lambda: det.detect(speed_img, ("person",)))
        res[cand] = {"person_recall": hit / tot if tot else 0.0, "gt_boxes": tot,
                     "cars_per_image": cars, "cars_total": sum(cars.values()),
                     "cars_all_images": bool(cars) and all(v >= 1 for v in cars.values()),
                     "det_ms": ms, "actual_model": det.name}
        print(f"{cand:11s} recall={res[cand]['person_recall']:.3f} cars={cars} {ms:.1f} ms",
              flush=True)
    pose_ms = {}
    for pm in ("rtmpose_s", "rtmpose_m"):
        pe = PoseEstimator(pm)
        pose_ms[pm] = _time_ms(lambda: pe.pose(speed_img, pose_boxes))
    chosen, why = choose(res)
    out_d = {"clip": name, "frames": len(frames), "gt_boxes": n_gt, "providers": "CPUExecutionProvider",
             "showroom_images": [p.name for p, _ in showroom], "results": res,
             "pose_ms_4_boxes": pose_ms, "chosen": chosen, "chosen_why": why,
             "rule": RULE_TEXT}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d, indent=2))
    print("\n| detector | person recall | cars (per image) | ms/frame 960x540 |\n|---|---|---|---|")
    for n, r in res.items():
        print(f"| {n} | {r['person_recall']:.3f} | {list(r['cars_per_image'].values())} | {r['det_ms']:.1f} |")
    print(f"pose ms (4 boxes): {pose_ms}\nchosen: {chosen} ({why})")
    return out_d


# ---- speed benchmark (spec section 6: file-mode RTF at 5 fps / 720p, live fps with 1-4 people) ----

_ROOT = Path(__file__).resolve().parents[2]
EXCERPT_START_S = 40.0      # people are in view from ~44 s on the first dev clip
EXCERPT_DUR_S = 60.0
EXCERPT_WIDTH = 1280        # 720p-class input (the MEVA source is 1920x1072, so the excerpt is 1280x714)
_WARMUP = 3


def _git_rev() -> str:
    """Short git revision of the code that ran the benchmark ('+dirty' if the tree had changes)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("+dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_clip():
    spec = importlib.util.spec_from_file_location("make_clip", _ROOT / "scripts" / "make_clip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_clip


def _step_stats(engine, frames, cfg) -> dict:
    """Time engine.process on already-decoded frames; bucket by detected-person count."""
    engine.reset()
    all_ms, few_ms = [], []
    for k, (t, frame) in enumerate(frames):
        t0 = time.perf_counter()
        engine.process(t, frame, k)
        ms = (time.perf_counter() - t0) * 1000.0
        if k < _WARMUP:
            continue
        n = sum(1 for d in engine.last_person_dets if d.score > cfg.det_score_thr)
        all_ms.append(ms)
        if 1 <= n <= 4:
            few_ms.append(ms)

    def agg(xs):
        m = statistics.fmean(xs) if xs else None
        return {"n_frames": len(xs), "mean_ms": m, "fps": 1000.0 / m if m else None}
    return {"all": agg(all_ms), "1_to_4": agg(few_ms)}


def run_speed_bench(clip: Path, out: Path, start: float = EXCERPT_START_S, dur: float = EXCERPT_DUR_S,
                    width: int = EXCERPT_WIDTH, coreml: bool = True, coreml_frames: int = 60) -> dict:
    """File-mode RTF (cold perceive of a 60 s 720p excerpt, temporary cache, CPU EP) and the engine's
    per-frame step rate (all frames / frames with 1-4 detected persons). CoreML, if available, is
    measured once on the first `coreml_frames` frames and labelled separately."""
    import onnxruntime as ort

    from pulse import sources
    from pulse.config import get_config
    from pulse.engine import PerceptionEngine
    from pulse.pipeline import perceive

    cfg = get_config("dealership")
    clip, out = Path(clip), Path(out)
    fx = _ROOT / "runs" / "fixtures" / f"speed_{int(start)}s_{int(dur)}s_{width}w.mp4"
    fx.parent.mkdir(parents=True, exist_ok=True)
    if not fx.is_file():
        _make_clip()(clip, fx, start=start, dur=dur, width=width)
    info = sources.probe(fx)
    print(f"excerpt {fx.name}: {info.width}x{info.height}, {info.duration:.1f} s", flush=True)

    with tempfile.TemporaryDirectory(prefix="pulse-speed-") as tmp:
        t0 = time.perf_counter()
        pc = perceive(fx, Path(tmp), cfg)
        wall = time.perf_counter() - t0
    analysed = float(pc.meta["analysed_duration"])
    cpu = {"wall_seconds": wall, "analysed_seconds": analysed, "rtf": analysed / wall,
           "perception_seconds": float(pc.meta["perception_seconds"]), "providers": list(pc.meta["providers"]),
           "tracks": len({o.track_id for o in pc.obs})}
    print(f"cold perceive: {wall:.1f} s for {analysed:.1f} s -> RTF {cpu['rtf']:.2f}x", flush=True)

    frames = list(sources.iter_frames(fx, cfg.sample_fps, cfg.max_width, cfg.max_duration_s))
    st = _step_stats(PerceptionEngine(cfg), frames, cfg)
    print(f"engine step: all {st['all']['fps']:.2f} fps, 1-4 people {st['1_to_4']['fps']:.2f} fps "
          f"({st['1_to_4']['n_frames']} frames)", flush=True)

    res = {"clip": clip.name, "machine": platform.platform(), "git_rev": _git_rev(),
           "config_hash": hashlib.sha1(repr(cfg).encode()).hexdigest()[:12], "provider": "CPUExecutionProvider",
           "excerpt": {"start_s": start, "duration_s": float(info.duration), "width": info.width,
                       "height": info.height, "sample_fps": cfg.sample_fps, "analysis_width": cfg.max_width},
           "cpu": cpu, "engine_all": st["all"], "engine_1_to_4_people": st["1_to_4"]}
    live = _ROOT / "runs" / "bench" / "live_smoke.json"
    if live.is_file():       # snapshot, so docs/EVAL.md does not change when verify.sh re-runs the live smoke
        d = json.loads(live.read_text())
        res["live_smoke"] = {k: d[k] for k in ("fps", "fps_max", "frames", "dropped", "people_max",
                                               "step_ms_mean", "seconds") if k in d}
    if coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
        try:
            eng = PerceptionEngine(cfg, providers=("CoreMLExecutionProvider", "CPUExecutionProvider"))
            cst = _step_stats(eng, frames[:coreml_frames], cfg)
            res["coreml"] = {"providers": list(eng.detector.session.get_providers()),
                             "frames": min(coreml_frames, len(frames)),
                             "fps_all": cst["all"]["fps"], "mean_ms_all": cst["all"]["mean_ms"],
                             "fps_1_to_4": cst["1_to_4"]["fps"], "mean_ms_1_to_4": cst["1_to_4"]["mean_ms"],
                             "n_frames_1_to_4": cst["1_to_4"]["n_frames"]}
            print(f"coreml engine step: {res['coreml']}", flush=True)
        except Exception as e:  # noqa: BLE001 - optional path, record why it was unusable
            res["coreml"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"coreml unusable: {res['coreml']['error']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    return res
