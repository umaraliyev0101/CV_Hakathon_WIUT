"""Render: privacy-first annotated video, thumbnails and contact sheets.
PRIVACY: every frame leaving this module has faces pixelated + blurred (staff included).
`render_timeline` is the Coach-compatible `render(video, timeline, events, out)` wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from pulse.config import Config
from pulse.sources import iter_frames

CUSTOMER = (0, 140, 255)
STAFF = (180, 120, 60)
ALERT = (0, 0, 255)
TILE_W = 320
Obs = tuple  # (bbox, keypoints | None)


def face_box(obs_bbox, keypoints, kp_thr: float = 0.3) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = obs_bbox
    pts = [k for k in (keypoints[:5] if keypoints else []) if k[2] >= kp_thr]
    if not pts:
        return int(bx1), int(by1), int(bx2), int(by1 + 0.2 * (by2 - by1))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    min_side = 0.12 * (bx2 - bx1)
    if x2 - x1 < min_side:
        x1, x2 = (x1 + x2) / 2 - min_side / 2, (x1 + x2) / 2 + min_side / 2
    if y2 - y1 < min_side:
        y1, y2 = (y1 + y2) / 2 - min_side / 2, (y1 + y2) / 2 + min_side / 2
    pad = 0.6 * max(x2 - x1, y2 - y1)
    return int(x1 - pad), int(y1 - pad), int(round(x2 + pad)), int(round(y2 + pad))


def blur_faces(frame: np.ndarray, observations) -> np.ndarray:
    """Copy of `frame` with each observation's face box pixelated (<= 6x6 blocks) then Gaussian-blurred."""
    out = frame.copy()
    h, w = out.shape[:2]
    for bbox, kps in observations:
        x1, y1, x2, y2 = face_box(bbox, kps)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        roi = out[y1:y2, x1:x2]
        small = cv2.resize(roi, (min(6, x2 - x1), min(6, y2 - y1)), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        k = max(3, (max(x2 - x1, y2 - y1) // 3) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(pix, (k, k), 0)
    return out


@dataclass
class FrameOverlay:
    t: float
    persons: list[dict] = field(default_factory=list)  # person_id, bbox, keypoints, role, zone_id, dwell, alert
    zones: list[dict] = field(default_factory=list)    # zone_id, name, polygon (normalised), now, total


def _text(img, s, org, color, scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, overlay: FrameOverlay, frame_size, extra_faces=()) -> np.ndarray:
    """Blur first (on a copy), then draw. `extra_faces`: more (bbox, keypoints) to blur."""
    faces = [(p["bbox"], p.get("keypoints")) for p in overlay.persons] + list(extra_faces)
    img = blur_faces(frame, faces)
    w, h = frame_size
    for z in overlay.zones:
        poly = np.array([[x * w, y * h] for x, y in z["polygon"]], np.int32)
        cv2.polylines(img, [poly], True, (200, 200, 200), 2, cv2.LINE_AA)
        _text(img, f"{z['name']}  now {z['now']} · total {z['total']}".replace("·", "|"),
              (int(poly[:, 0].min()) + 3, int(poly[:, 1].min()) + 14), (255, 255, 255), 0.45)
    for p in overlay.persons:
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        staff = p.get("role") == "staff"
        color = ALERT if p.get("alert") else (STAFF if staff else CUSTOMER)
        if staff and not p.get("alert"):
            layer = img.copy()
            cv2.rectangle(layer, (x1, y1), (x2, y2), color, 2)
            img = cv2.addWeighted(layer, 0.5, img, 0.5, 0)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if p.get("alert") else 2)
        label = f"{'S' if staff else 'C'}{p['person_id']}"
        if not staff and p.get("zone_id") and p.get("dwell") is not None:
            label += f" {p['dwell']:.0f}s"
        _text(img, label, (x1, max(12, y1 - 5)), color, 0.5)
    return img


def thumbnail(frame, bbox, keypoints, size=(128, 256)) -> np.ndarray:
    safe = blur_faces(frame, [(bbox, keypoints)])
    h, w = safe.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    crop = safe[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size[1], size[0], 3), np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _even(img):
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def _annotated(frame, t, overlays, observations_by_t, default_zones):
    ov = overlays.get(t) or FrameOverlay(t, [], default_zones)
    h, w = frame.shape[:2]
    return draw_overlay(frame, ov, (w, h), (observations_by_t or {}).get(t, ()))


def render(video, overlays: dict[float, FrameOverlay], out_path, cfg: Config,
           observations_by_t: dict | None = None) -> Path:
    """Annotated H.264 yuv420p mp4 at cfg.sample_fps. Frames without an overlay are still written
    (zone outlines from the first overlay; faces blurred from `observations_by_t` when supplied)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    default_zones = next(iter(overlays.values())).zones if overlays else []
    writer = None
    try:
        for t, frame in iter_frames(video, cfg.sample_fps, cfg.max_width, cfg.max_duration_s):
            img = _even(_annotated(frame, t, overlays, observations_by_t, default_zones))
            if writer is None:
                h, w = img.shape[:2]
                writer = imageio_ffmpeg.write_frames(
                    str(out_path), (w, h), fps=cfg.sample_fps, codec="libx264", pix_fmt_in="bgr24",
                    pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-loglevel", "error"])
                writer.send(None)
            writer.send(np.ascontiguousarray(img))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no frames read from {video}")
    return out_path


def render_timeline(video, timeline, events, out, cfg: Config) -> Path:
    """Coach-compatible wrapper (Coach calls it `render(video, timeline, events, out)`).
    Persons come from timeline samples (role unknown -> customer); a person in any event with
    kind containing 'unserved' or 'alert' is outlined red while the event is open."""
    overlays: dict[float, FrameOverlay] = {}
    for pid, samples in timeline.items():
        for s in samples:
            alert = any(pid in e.person_ids and ("unserved" in e.kind or "alert" in e.kind)
                        and e.t_start <= s.t <= (e.t_end if e.t_end is not None else s.t)
                        for e in events)
            ov = overlays.setdefault(s.t, FrameOverlay(s.t))
            ov.persons.append({"person_id": pid, "bbox": s.bbox, "keypoints": None, "role": "customer",
                               "zone_id": None, "dwell": 0.0, "alert": alert})
    return render(video, overlays, out, cfg)


def contact_sheet(video, moments: list[dict], overlays: dict[float, FrameOverlay], out_path, cfg: Config,
                  cols: int = 2, observations_by_t: dict | None = None) -> Path:
    """Up to 4 tiles (frame nearest each moment.t), blurred + annotated, caption `t=..s`; JPEG."""
    moments = moments[:4]
    best: list[tuple[float, float, np.ndarray] | None] = [None] * len(moments)
    default_zones = next(iter(overlays.values())).zones if overlays else []
    for t, frame in iter_frames(video, cfg.sample_fps, cfg.max_width, cfg.max_duration_s):
        for i, m in enumerate(moments):
            d = abs(t - m["t"])
            if best[i] is None or d < best[i][0]:
                best[i] = (d, t, _annotated(frame, t, overlays, observations_by_t, default_zones))
    tiles = []
    for b in best:
        if b is None:
            continue
        img = b[2]
        h, w = img.shape[:2]
        tile = cv2.resize(img, (TILE_W, int(round(h * TILE_W / w))), interpolation=cv2.INTER_AREA)
        _text(tile, f"t={b[1]:.1f}s", (6, tile.shape[0] - 8), (255, 255, 255), 0.5)
        tiles.append(tile)
    if not tiles:
        raise ValueError(f"no frames read from {video}")
    th = tiles[0].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * TILE_W, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * TILE_W:(c + 1) * TILE_W] = tile[:th]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
