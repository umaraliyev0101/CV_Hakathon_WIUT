"""Result files for one analysis: JSON, CSV, charts, heatmap, thumbnails, contact sheets, video.

PRIVACY: every image written here is face-blurred using ALL detections at that sampled time
(`observations_by_t`), not only the persons that made it into the timeline. The raw keyframe stays
inside the perception cache."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import render as render_mod  # noqa: E402
from . import sources  # noqa: E402
from .config import Config  # noqa: E402
from .roles import export_roles_json  # noqa: E402
from .zones import save_zones  # noqa: E402

BANNER = "Anonymous, aggregated analytics; decision support only."
VISIT_COLS = ["person_id", "role", "first_t", "last_t", "already_present", "visit_s", "zones",
              "stopped_cars", "assisted", "t_first_assist", "group"]
CAR_COLS = ["zone_id", "name", "visitors", "stops", "stop_rate", "hold_rate", "median_dwell",
            "total_dwell", "repeat_visits", "deep_interest", "unserved", "unassisted_share", "rank", "sat_in_car", "sold", "attention_to_sales"]
OCC_COLS = ["t", "customers", "waiting", "customers_per_staff"]


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def observations_by_t(pc) -> dict[float, list]:
    """Everything to blur per sampled time: tracked observations at t and at the neighbouring samples
    (covers one-frame track gaps) plus every raw person detection near t, tracked or not.
    Over-blurring is fine; under-blurring is not."""
    dt = 1.0 / float(pc.meta.get("sample_fps") or 5.0)
    out: dict[float, list] = {}

    def add(t: float, box, kp) -> None:
        for k in (round(t - dt, 3), round(t, 3), round(t + dt, 3)):
            out.setdefault(k, []).append((box, kp))

    for o in pc.obs:
        add(o.t, o.bbox, o.keypoints)
    for t, box, _ in pc.raw_dets:
        add(t, box, None)
    return out


def build_overlays(analysis, pc, cfg: Config) -> dict[float, render_mod.FrameOverlay]:
    sample_fps = cfg.sample_fps
    n = int(pc.meta.get("n_samples", 0))
    kp = {(round(o.t, 3), tuple(o.bbox)): o.keypoints for o in pc.obs}
    by_t: dict[float, list] = {}
    for pid, ss in analysis.timeline.items():
        for s in ss:
            by_t.setdefault(round(s.t, 3), []).append(s)
    visits = [v for vs in analysis.journeys.values() for v in vs]
    unserved = [e for e in analysis.episodes if e.unserved]
    cars = {z.zone_id for z in analysis.zones if z.kind == "car"}
    overlays: dict[float, render_mod.FrameOverlay] = {}
    for k in range(n):
        t = round(k / sample_fps, 3)
        persons = []
        for s in by_t.get(t, []):
            role = analysis.roles.get(s.person_id, "customer")
            v = next((v for v in analysis.journeys.get(s.person_id, []) if v.t_in <= t <= v.t_out), None)
            dwell = t - v.t_in if v else None
            alert = bool(
                v and role != "staff" and v.zone_id in cars and dwell >= cfg.preset.alert_after
                and any(e.person_id == s.person_id and e.zone_id == v.zone_id
                        and e.t_start <= t <= e.t_end for e in unserved))
            persons.append({"person_id": s.person_id, "bbox": s.bbox,
                            "keypoints": kp.get((t, tuple(s.bbox))), "role": role,
                            "zone_id": v.zone_id if v else None, "dwell": dwell, "alert": alert})
        zones = []
        for z in analysis.zones:
            zv = [v for v in visits if v.zone_id == z.zone_id]
            zones.append({"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon,
                          "now": len({v.person_id for v in zv if v.t_in <= t <= v.t_out}),
                          "total": len({v.person_id for v in zv if v.t_in <= t})})
        overlays[t] = render_mod.FrameOverlay(t, persons, zones)
    return overlays


def _keyframe(pc, video, obs_t, cfg: Config, warnings: list[str]) -> np.ndarray:
    """Face-blurred keyframe in analysis-frame size. Raw frame from the cache dir, else re-read from
    the video at the analysis width. An image that cannot be reconciled with the analysis frame (so
    the boxes would not line up) is replaced by a neutral grey background, never shown unblurred."""
    fw, fh = int(pc.meta["frame_w"]), int(pc.meta["frame_h"])
    img = cv2.imread(str(pc.keyframe_path)) if pc.keyframe_path.is_file() else None
    if img is None and video is not None:
        for t, frame in sources.iter_frames(video, cfg.sample_fps, cfg.max_width,
                                            pc.meta.get("analysed_duration")):
            if t >= pc.keyframe_t - 1e-6:
                img = frame
                break
    if img is not None and img.shape[:2] != (fh, fw):
        h, w = img.shape[:2]
        if abs(w / h - fw / fh) <= 0.02 * fw / fh:
            img = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
        else:
            img = None
            warnings.append("Keyframe does not match the analysis frame; grey background used")
    if img is None:
        return np.full((fh, fw, 3), 128, np.uint8)
    boxes = obs_t.get(round(pc.keyframe_t, 3), [])
    return render_mod.blur_faces(img, boxes)


def _heatmap(key: np.ndarray, stats) -> np.ndarray:
    h, w = key.shape[:2]
    counts = np.asarray(stats.heatmap["counts"], np.float32)
    out = key.copy()
    if counts.size and counts.max() > 0:
        heat = cv2.GaussianBlur(cv2.resize(counts, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
        norm = (255 * heat / max(float(heat.max()), 1e-6)).astype(np.uint8)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        alpha = (0.55 * norm.astype(np.float32) / 255.0)[..., None]
        out = (key * (1 - alpha) + colour * alpha).astype(np.uint8)
    for p in stats.paths[:10]:
        pts = np.array([[x * w, y * h] for x, y in p["points"]], np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _topdown_heatmap(path: Path, analysis, pc) -> None:
    """Customer foot positions on the floor plane, in metres.

    PRIVACY: this is a pure matplotlib plot of coordinates; it contains no video pixels, so there is
    nothing to face-blur."""
    from .timeline import filter_stable
    world = np.asarray(analysis.floor["world"], float)
    lo, hi = world.min(axis=0), world.max(axis=0)
    margin = 0.1 * max(float((hi - lo).max()), 1.0)
    lo, hi = lo - margin, hi + margin
    stable = filter_stable(analysis.timeline, pc.segments)
    pts = [s.pos_m for pid, ss in stable.items() if analysis.roles.get(pid) != "staff"
           for s in ss if s.dist_method == "floor" and s.pos_m is not None]
    fig, ax = plt.subplots(figsize=(6, 6 * max(hi[1] - lo[1], 1e-6) / max(hi[0] - lo[0], 1e-6)
                                     if hi[0] > lo[0] else 6))
    if pts:
        a = np.asarray(pts, float)
        h, xe, ye = np.histogram2d(a[:, 0], a[:, 1], bins=40, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
        ax.imshow(np.ma.masked_equal(h.T, 0), origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
                  cmap="inferno", aspect="equal")
    closed = np.vstack([world, world[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "c--", lw=1, label="calibration")
    ax.plot(world[:, 0], world[:, 1], "co", ms=4)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Customer positions on the floor (top-down)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="png", metadata={"Software": None})
    plt.close(fig)


def _line_chart(path: Path, rows: list, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    if rows:
        ax.step([r[0] for r in rows], [r[1] for r in rows], where="post", color="#d9650b")
    ax.set_title(title)
    ax.set_xlabel("seconds")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _journey_flow(analysis) -> dict:
    names = {z.zone_id: z.name for z in analysis.zones}
    paths = [["Start", *v["zones"], "End"] for v in analysis.stats.visits if v["role"] != "staff"]
    links = Counter((a, b) for p in paths for a, b in zip(p, p[1:]))
    used = {n for p in paths for n in p} if paths else {"Start", "End"}
    nodes = [{"id": n, "name": names.get(n, n)} for n in sorted(used, key=lambda n: (n != "Start", n == "End", n))]
    top = Counter(tuple(p) for p in paths).most_common(5)
    return {"nodes": nodes,
            "links": [{"source": a, "target": b, "count": c} for (a, b), c in sorted(links.items())],
            "top_paths": [{"path": [names.get(n, n) for n in p], "count": c} for p, c in top]}


def _frames_at(video, wanted: dict[float, list], cfg: Config):
    """One sequential pass; yields (t, frame) only for sampled times in `wanted`."""
    if not wanted:
        return
    last = max(wanted)
    for t, frame in sources.iter_frames(video, cfg.sample_fps, cfg.max_width, cfg.max_duration_s):
        if t in wanted:
            yield t, frame
        if t >= last:
            return


def _thumbnails(analysis, pc, video, cfg, obs_t, out_dir: Path) -> dict[int, str]:
    t2p = {tid: p.person_id for p in analysis.persons for tid in p.track_ids}
    best: dict[int, object] = {}
    for o in pc.obs:
        pid = t2p.get(o.track_id)
        if pid is None:
            continue
        key = o.score * max(o.bbox[3] - o.bbox[1], 1.0)
        if pid not in best or key > best[pid][0]:
            best[pid] = (key, o)
    wanted: dict[float, list] = {}
    for pid, (_, o) in best.items():
        wanted.setdefault(round(o.t, 3), []).append((pid, o))
    paths: dict[int, str] = {}
    (out_dir / "thumbs").mkdir(exist_ok=True)
    for t, frame in _frames_at(video, wanted, cfg):
        safe = render_mod.blur_faces(frame, obs_t.get(t, []))
        for pid, o in wanted[t]:
            rel = f"thumbs/person_{pid}.jpg"
            cv2.imwrite(str(out_dir / rel), render_mod.thumbnail(safe, o.bbox, o.keypoints),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[pid] = rel
    return paths


_OWNED = ["report.json", "stats.json", "insights.json", "zones.json", "roles.json", "people.json",
          "visits.csv", "per_car.csv", "occupancy.csv", "heatmap.png", "heatmap_topdown.png", "occupancy.svg", "waiting.svg",
          "journey_flow.json", "keyframe.jpg", "report.html", "annotated.mp4", "annotated.part.mp4"]


def _clean(out_dir: Path) -> None:
    """Remove stale outputs of an earlier run; touches only the names this module writes."""
    for name in _OWNED:
        (out_dir / name).unlink(missing_ok=True)
    for sub, pattern in (("thumbs", "person_*.jpg"), ("contact_sheets", "insight_*.jpg")):
        d = out_dir / sub
        if d.is_dir():
            for f in d.glob(pattern):
                f.unlink()
            try:
                d.rmdir()
            except OSError:
                pass                                 # not empty: holds someone else's files


def build_report(analysis, pc, out_dir, cfg: Config, video=None, render_video: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = analysis.stats
    size = (int(pc.meta["frame_w"]), int(pc.meta["frame_h"]))
    obs_t = observations_by_t(pc)
    files: dict[str, str] = {}

    def put(name: str, rel: str) -> Path:
        files[name] = rel
        return out_dir / rel

    warnings = list(st.warnings)
    _clean(out_dir)
    key = _keyframe(pc, video, obs_t, cfg, warnings)
    cv2.imwrite(str(put("keyframe", "keyframe.jpg")), key, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(put("heatmap", "heatmap.png")), _heatmap(key, st))
    if getattr(analysis, "floor", None):
        _topdown_heatmap(put("heatmap_topdown", "heatmap_topdown.png"), analysis, pc)
    _line_chart(put("occupancy_chart", "occupancy.svg"), st.occupancy, "Customers present", "customers")
    _line_chart(put("waiting_chart", "waiting.svg"), st.waiting, "Customers waiting unattended", "customers")

    _dump(put("stats", "stats.json"), st.to_dict())
    _dump(put("insights", "insights.json"), [i.to_dict() for i in analysis.insights])
    save_zones(put("zones", "zones.json"), analysis.zones, size, floor=getattr(analysis, "floor", None))
    _dump(put("roles", "roles.json"), export_roles_json(analysis.roles))
    flow = _journey_flow(analysis)
    _dump(put("journey_flow", "journey_flow.json"), flow)

    with open(put("visits", "visits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(VISIT_COLS)
        for v in st.visits:
            w.writerow([" > ".join(v[c]) if c in ("zones", "stopped_cars") else v.get(c) for c in VISIT_COLS])
    with open(put("per_car", "per_car.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CAR_COLS)
        for c in st.per_car:
            w.writerow([c.get(k) for k in CAR_COLS])
    with open(put("occupancy", "occupancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OCC_COLS)
        for (t, n), (_, wt), (_, cps) in zip(st.occupancy, st.waiting, st.crowding):
            w.writerow([t, n, wt, cps])

    thumbs = _thumbnails(analysis, pc, video, cfg, obs_t, out_dir) if video is not None else {}
    for pid, rel in thumbs.items():
        files[f"thumb_{pid}"] = rel
    sug = {s.person_id: s for s in analysis.suggestions}
    people = [{"person_id": p.person_id, "role": analysis.roles.get(p.person_id, "customer"),
               "suggested_role": sug[p.person_id].role if p.person_id in sug else None,
               "confidence": sug[p.person_id].confidence if p.person_id in sug else None,
               "reasons": sug[p.person_id].reasons if p.person_id in sug else [],
               "features": sug[p.person_id].features if p.person_id in sug else {},
               "first_t": p.first_t, "last_t": p.last_t, "thumbnail": thumbs.get(p.person_id)}
              for p in analysis.persons]
    _dump(put("people", "people.json"), {"persons": people})

    overlays = None
    if video is not None and (render_video or any(i.moments for i in analysis.insights)):
        overlays = build_overlays(analysis, pc, cfg)
    if video is not None:
        for k, ins in enumerate(analysis.insights):
            if ins.moments:
                rel = f"contact_sheets/insight_{k}.jpg"
                try:
                    render_mod.contact_sheet(video, ins.moments, overlays, put(f"contact_sheet_{k}", rel),
                                             cfg, obs_t)
                except ValueError as e:
                    files.pop(f"contact_sheet_{k}", None)
                    warnings.append(f"Contact sheet for insight {k} failed: {e}")
    if render_video and video is not None:
        render_mod.render(video, overlays, put("annotated_video", "annotated.mp4"), cfg, obs_t)

    report = {"banner": BANNER, "meta": analysis.meta, "store": st.store, "per_car": st.per_car,
              "per_zone": st.per_zone, "zones": [z.to_dict() for z in analysis.zones],
              "people": people, "insights": [i.to_dict() for i in analysis.insights],
              "journey_flow": flow,
              "charts": {"occupancy": st.occupancy, "waiting": st.waiting, "crowding": st.crowding,
                         "time_buckets": st.time_buckets},
              "files": files, "warnings": warnings}
    from .report_html import render_html
    put("report_html", "report.html").write_text(
        render_html(report, (out_dir / "heatmap.png").read_bytes()), encoding="utf-8")
    files["report"] = "report.json"
    _dump(out_dir / "report.json", report)
    return report
