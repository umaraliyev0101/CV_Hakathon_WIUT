"""Rendering: the face blur, the annotated H.264 video, the timeline image, event contact sheets and thumbnails.

Privacy comes first. Every rendered frame, contact-sheet panel and saved thumbnail goes through `blur_faces` BEFORE
anything is drawn on it, and nothing here recognises, matches or stores a face. Per output frame the blur covers
  - the top-20 % fallback region of every raw person detection of the sampled perception frames just before and after
    it (so people the tracker dropped as flicker are covered too),
  - the keypoint face region of every person that is drawn (their box and keypoints interpolated to this frame), and
  - the last known face region of a tracked person who is not drawn because their observations are too far apart.

Everything is drawn in the PROCESSED frame size (`VideoInfo.width x height`, even), the size the perception boxes are
in, so boxes never need rescaling. The video is sampled with `sources.iter_frames` (the same rule as perception, at
`fps_out`), so output frame times line up with the observations.

Text is drawn with OpenCV's Hershey font, which has ASCII only: names are folded to ASCII (accents dropped, anything
else becomes "?"), and the separators the spec shows as "·" and "—" are written "|" and "-".
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import unicodedata
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import cv2
import numpy as np

from .export import atomic_replace
from .sources import VideoError, iter_frames

if TYPE_CHECKING:
    from .analysis import Analysis
    from .perception.pipeline import PerceptionResult

log = logging.getLogger(__name__)

KPT_THR = 0.3                # a face keypoint must be at least this confident to place the face
FACE_KPTS = 5                 # COCO order: nose, left eye, right eye, left ear, right ear
FACE_EXPAND = 1.8             # the keypoint box is grown by this factor around its centre ...
FACE_MIN_SIDE = 0.5           # ... and is never narrower / shorter than this fraction of the person box width
FALLBACK_HEAD_FRACTION = 0.2  # without usable face keypoints: the top 20 % of the person box, full width
PIXELATE_DIV = 6              # blur: the region is first averaged down to 1/6 of its size (blocks of about 6 px)
_EPS = 1e-6                   # slack so exact box edges survive float arithmetic when rounded outward


def _floor(v: float) -> int:
    return int(math.floor(v + _EPS))


def _ceil(v: float) -> int:
    return int(math.ceil(v - _EPS))


def _face_points(kpts, kpt_thr: float) -> np.ndarray | None:
    """(K, 2) coordinates of the confident nose / eye / ear keypoints, or None when there are none."""
    if kpts is None:
        return None
    try:
        a = np.asarray(kpts, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] < FACE_KPTS or a.shape[1] < 3:
        return None
    a = a[:FACE_KPTS]
    ok = np.isfinite(a[:, :3]).all(axis=1) & (a[:, 2] >= kpt_thr)
    return a[ok, :2] if ok.any() else None


def face_region(bbox, kpts, kpt_thr: float = KPT_THR) -> tuple[int, int, int, int]:
    """Where to blur one person's face, as integer pixel corners (x1, y1, x2, y2) in the frame of `bbox`.

    With confident nose / eye / ear keypoints: their bounding box grown x1.8 about its centre, at least 0.5 x the
    person box width on each side. Without them, or when that region's centre falls outside the person's box (a
    stray keypoint): the top 20 % of the box, full width. The result may extend past the frame; `blur_faces` clips it.
    A box with non-finite coordinates yields an empty region (0, 0, 0, 0).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return 0, 0, 0, 0
    w = x2 - x1
    pts = _face_points(kpts, kpt_thr)
    if pts is not None:
        fx1, fy1 = pts.min(axis=0)
        fx2, fy2 = pts.max(axis=0)
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            half_w = max((fx2 - fx1) * FACE_EXPAND, FACE_MIN_SIDE * w) / 2.0
            half_h = max((fy2 - fy1) * FACE_EXPAND, FACE_MIN_SIDE * w) / 2.0
            return _floor(cx - half_w), _floor(cy - half_h), _ceil(cx + half_w), _ceil(cy + half_h)
    return _floor(x1), _floor(y1), _ceil(x2), _ceil(y1 + FALLBACK_HEAD_FRACTION * (y2 - y1))


def _obscure(roi: np.ndarray) -> np.ndarray:
    """Pixelate-then-blur: block averaging throws the detail away for good, the Gaussian smooths the blocks."""
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // PIXELATE_DIV), max(1, h // PIXELATE_DIV)), interpolation=cv2.INTER_AREA)
    blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(max(w, h) * 0.5) | 1)
    return cv2.GaussianBlur(blocks, (k, k), 0)


def blur_faces(frame: np.ndarray, bboxes: Sequence, kpts_list: Sequence | None = None) -> np.ndarray:
    """A copy of `frame` (BGR) with every person's face region blurred hard.

    `bboxes` are person boxes in the frame's pixels; `kpts_list[i]` is that person's (17, 3) keypoints or None
    (people past the end of a short `kpts_list` count as None, so nobody is left unblurred). Regions are clipped
    to the frame; the rest of the image is untouched and `frame` itself is never modified.
    """
    out = frame.copy()
    fh, fw = out.shape[:2]
    kpts_list = kpts_list if kpts_list is not None else ()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = face_region(bbox, kpts_list[i] if i < len(kpts_list) else None)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, fw), min(y2, fh)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out[y1:y2, x1:x2] = _obscure(out[y1:y2, x1:x2])
    return out


# =============================================================================================== the video writer

STAFF_BGR = (230, 120, 20)      # blue
CUSTOMER_BGR = (0, 140, 255)    # orange
ALERT_BGR = (30, 30, 200)       # red: banners and alert markers
TICK_BGR = (80, 175, 60)        # green: a greeted customer
GREET_BGR = TICK_BGR
PRESENT_BGR = (175, 175, 175)   # grey: on screen (timeline)
WHITE, DARK = (255, 255, 255), (25, 25, 25)
TIMER_BG, TIMER_FG = (30, 30, 30), (0, 190, 255)
HUD_BG = (35, 35, 35)

FFMPEG_PARAMS = ["-preset", "veryfast", "-crf", "26", "-movflags", "+faststart"]
JPEG_QUALITY = 88
_EPS = 1e-6


def _probe_frame_count(path) -> int:
    """How many frames a fresh decoder finds in the video at `path` (0 when it cannot be opened). The container's own
    count is used; if it reports none the frames are counted by decoding."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            while cap.grab():
                n += 1
        return n
    finally:
        cap.release()


class H264Writer:
    """Context manager writing BGR frames to an H.264 mp4 (libx264, yuv420p, faststart) that browsers can play.

    The video is encoded under a temporary name next to `path` and moved into place when the `with` block ends
    normally, so a half-written file is never served; on an error the temporary file is deleted and `path` is left as
    it was. Before that move the file is opened with a decoder and must hold at least `frames written - 1` frames,
    otherwise OSError (a truncated encode is never put in place). `size` is (width, height) and must match every frame. A block that wrote no frames raises ValueError.
    """

    def __init__(self, path, size: tuple[int, int], fps: float):
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        if min(self.size) < 2:
            raise ValueError(f"bad video size {self.size}")
        if not (self.fps > 0 and math.isfinite(self.fps)):
            raise ValueError(f"bad fps {fps!r}")
        suffix = self.path.suffix or ".mp4"                      # ffmpeg picks the container from the extension
        self._tmp = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex[:8]}.part{suffix}")
        self._gen = None
        self.frames = 0

    def __enter__(self) -> "H264Writer":
        import imageio_ffmpeg

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # quality=None: imageio would add its own -crf, and the crf in FFMPEG_PARAMS is the one we mean
        self._gen = imageio_ffmpeg.write_frames(
            str(self._tmp), self.size, pix_fmt_in="bgr24", codec="libx264", fps=self.fps, quality=None,
            macro_block_size=2, ffmpeg_log_level="error", output_params=list(FFMPEG_PARAMS))
        self._gen.send(None)                                      # prime the generator
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._gen is None:
            raise ValueError("H264Writer is not open")
        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"expected a {self.size[1]}x{self.size[0]}x3 uint8 BGR frame, got "
                             f"{frame.shape} {frame.dtype}")
        self._gen.send(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        gen, self._gen = self._gen, None
        try:
            if gen is not None:
                gen.close()                                       # lets ffmpeg finish the file
            if exc_type is None:
                if self.frames == 0:
                    raise ValueError("no frames were written")
                if not self._tmp.is_file() or self._tmp.stat().st_size == 0:
                    raise OSError("ffmpeg did not produce a video")
                got = _probe_frame_count(self._tmp)
                if got < self.frames - 1:
                    raise OSError(f"the encoded video has only {got} of {self.frames} frames")
                atomic_replace(self._tmp, self.path)
        finally:
            self._tmp.unlink(missing_ok=True)
        return False


# ============================================================================================== drawing the overlay

_FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
BANNER_S = 3.0                          # an alert banner stays up this long after its event
GREETED_STATES = ("greeted", "engaged", "browsing")


def _ascii(text) -> str:
    """`text` for Hershey fonts: accents dropped (Zoë -> Zoe), every other non-ASCII character becomes "?"."""
    folded = unicodedata.normalize("NFKD", "" if text is None else str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in folded)


def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _font_scale(width: int) -> float:
    return float(min(0.7, max(0.42, width / 1600.0)))


def _chip_height(text: str, fs: float, pad: int = 3) -> int:
    (_, th), base = cv2.getTextSize(text, _FONT, fs, 1)
    return th + base + 2 * pad


def _chip(img: np.ndarray, text: str, x: int, top: int, bg, fg, fs: float, pad: int = 3) -> tuple[int, int, int, int]:
    """A filled label with text whose top-left corner is (x, top); kept inside the image sideways."""
    (tw, th), base = cv2.getTextSize(text, _FONT, fs, 1)
    w = tw + 2 * pad
    x = int(min(max(x, 0), max(img.shape[1] - w, 0)))
    x2, y2 = x + w, top + th + base + 2 * pad
    cv2.rectangle(img, (x, top), (x2, y2), bg, -1)
    cv2.putText(img, text, (x + pad, top + pad + th), _FONT, fs, fg, 1, cv2.LINE_AA)
    return x, top, x2, y2


def _int_box(bbox) -> tuple[int, int, int, int] | None:
    try:
        v = [float(c) for c in bbox]
    except (TypeError, ValueError):
        return None
    if len(v) != 4 or not all(math.isfinite(c) for c in v):
        return None
    x1, y1, x2, y2 = (int(round(min(max(c, -1e5), 1e5))) for c in v)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _pair_centres(people: list[dict], pairs: Sequence) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Centre points of the boxes joined by each "engaged" pair. Pairs name people by their "pid" when every
    person dict has one (like the live overlay), otherwise by their index in `people`; a third item other than
    "engaged" (e.g. "close") means no line; a pair that points at nobody is ignored."""
    by_pid = {p["pid"]: p for p in people} if people and all("pid" in p for p in people) else None
    out = []
    for pair in pairs:
        if len(pair) > 2 and pair[2] != "engaged":
            continue
        try:
            if by_pid is not None:
                a, b = by_pid.get(pair[0]), by_pid.get(pair[1])
            else:
                a = people[int(pair[0])] if 0 <= int(pair[0]) < len(people) else None
                b = people[int(pair[1])] if 0 <= int(pair[1]) < len(people) else None
        except (TypeError, ValueError):
            continue
        boxes = [_int_box(p["bbox"]) if p is not None else None for p in (a, b)]
        if all(boxes):
            out.append(tuple(((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes))
    return out


def _draw_tick(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    cv2.circle(img, (cx, cy), r, TICK_BGR, -1, cv2.LINE_8)
    pts = np.array([(cx - 0.5 * r, cy + 0.0 * r), (cx - 0.1 * r, cy + 0.45 * r), (cx + 0.55 * r, cy - 0.4 * r)])
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, WHITE, 2, cv2.LINE_AA)


def _draw_person(img: np.ndarray, p: dict, fs: float, floor_y: int) -> None:
    box = _int_box(p.get("bbox"))
    if box is None:
        return
    x1, y1, x2, y2 = box
    staff = p.get("role") == "staff"
    colour = STAFF_BGR if staff else CUSTOMER_BGR
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, BOX_THICKNESS, cv2.LINE_8)

    chips = []                                           # top to bottom: the wait timer sits above the label
    wait = p.get("wait_s")
    if not staff and p.get("state") == "waiting" and wait is not None and math.isfinite(float(wait)):
        chips.append((f"wait {_mmss(float(wait))}", TIMER_BG, TIMER_FG))
    label = _ascii(p.get("label"))
    if label:
        chips.append((label, colour, WHITE if staff else DARK))
    last = None
    if chips:
        heights = [_chip_height(c[0], fs) for c in chips]
        top = y1 - BOX_THICKNESS - sum(heights)
        if top < floor_y:                                # no room above the box: stack the chips inside its top
            top = y1 + BOX_THICKNESS + 1
        for (text, bg, fg), h in zip(chips, heights):
            last = _chip(img, text, x1, top, bg, fg, fs)
            top += h
    if not staff and p.get("state") in GREETED_STATES:
        r = max(5, int(round(9 * fs / 0.55)))
        if last is not None and label:
            cx, cy = last[2] + r + 4, (last[1] + last[3]) // 2
        else:
            cx, cy = x1 + r + 2, max(r + 1, y1 - r - 3)
        _draw_tick(img, cx, cy, r)


def _draw_banner(img: np.ndarray, text: str, fs: float) -> int:
    """A full-width alert banner along the top; returns its height."""
    width = img.shape[1]
    text = _ascii(text)
    fs_b = fs * 1.2
    while fs_b > 0.3 and cv2.getTextSize(text, _FONT, fs_b, 1)[0][0] > width - 16:
        fs_b -= 0.05
    (_, th), base = cv2.getTextSize(text, _FONT, fs_b, 1)
    h = th + base + 14
    cv2.rectangle(img, (0, 0), (width - 1, h), ALERT_BGR, -1)
    cv2.putText(img, text, (8, 7 + th), _FONT, fs_b, WHITE, 1, cv2.LINE_AA)
    return h


def _draw(img: np.ndarray, people: list[dict], pairs: Sequence, hud: str, alert: str | None) -> None:
    """The overlay, drawn onto `img` in place (callers pass a private copy)."""
    fs = _font_scale(img.shape[1])
    banner_h = _draw_banner(img, alert, fs) if alert else 0
    for a, b in _pair_centres(people, pairs):            # under the boxes: dark outline, white core
        cv2.line(img, a, b, DARK, 5, cv2.LINE_AA)
        cv2.line(img, a, b, WHITE, 2, cv2.LINE_AA)
    for p in people:
        _draw_person(img, p, fs, banner_h)
    if hud:
        _chip(img, _ascii(hud), 6, banner_h + 6, HUD_BG, WHITE, fs)


def draw_overlay(frame: np.ndarray, people: list[dict], pairs: Sequence, hud: str, alert: str | None) -> np.ndarray:
    """A copy of `frame` (BGR) with boxes, labels, engaged lines, timers, ticks, a HUD and an optional banner.

    `people` are dicts with `bbox` (x1, y1, x2, y2), `role` ("staff" | "customer"), `label`, `state`
    (customers: waiting | greeted | engaged | browsing; staff: busy | free) and `wait_s`, and optionally `pid`.
    Staff boxes are blue (STAFF_BGR), customers orange (CUSTOMER_BGR); the label sits above the box; waiting customers
    with a known `wait_s` get a "wait m:ss" timer above it; greeted, engaged and browsing customers get a green tick
    beside the label. `pairs` are (a, b[, "engaged"]) tuples joined by a white line with a dark outline (see
    `_pair_centres` for how a and b name people). `hud` is a small text top-left; `alert` is a red banner across the
    top (the HUD moves below it). Nothing here blurs: callers blur the frame first.
    """
    out = frame.copy()
    _draw(out, people, pairs, hud, alert)
    return out


# ======================================================================================== who is where, and doing what

MAX_INTERP_GAP_S = 0.5     # a person is interpolated between two observations at most this far apart ...
NEAREST_S = 0.25           # ... otherwise shown at the nearest observation if it is within this long, else not at all
HOLD_BLUR_S = 1.0          # an undrawn tracked person's last known face region stays blurred this long


class _EventIndex:
    """The analysis events arranged for "what is person X doing at time t" questions (states, engaged pairs, banners)."""

    def __init__(self, events, duration: float, names: dict | None):
        self.names = dict(names or {})
        end = float(duration)
        self._visits: dict[int, list[tuple[float, bool]]] = {}     # customer -> [(enter time, entry known)]
        self._greets: dict[int, list[float]] = {}
        self._cust_eng: dict[int, list[tuple[float, float]]] = {}
        self._staff_eng: dict[int, list[tuple[float, float]]] = {}
        self._engs: list[tuple[float, float, int, int]] = []       # (start, end, staff, customer)
        self._banners: list[tuple[float, str]] = []
        for e in sorted(events, key=lambda e: (e.t, e.type)):
            c = e.customer
            if e.type == "enter" and c is not None:
                self._visits.setdefault(c, []).append((e.t, bool(e.data.get("entry_known"))))
            elif e.type == "greet" and c is not None:
                self._greets.setdefault(c, []).append(e.t)
            elif e.type == "engagement":
                t1 = end if e.t_end is None else e.t_end
                if c is not None:
                    self._cust_eng.setdefault(c, []).append((e.t, t1))
                if e.staff is not None:
                    self._staff_eng.setdefault(e.staff, []).append((e.t, t1))
                if c is not None and e.staff is not None:
                    self._engs.append((e.t, t1, e.staff, c))
            elif e.type == "wait_alert" and c is not None:
                free = ", ".join(self.staff_name(s) for s in e.data.get("free_staff") or []) or "none"
                self._banners.append((e.t, f"Long wait: C{c} - free: {free}"))
            elif e.type == "walkout" and c is not None:
                self._banners.append((e.t, f"Walk-out: C{c}"))

    def staff_name(self, pid: int) -> str:
        return self.names.get(pid) or f"S{pid}"

    def state_of(self, pid: int, role: str, t: float) -> tuple[str, float | None]:
        """(state, wait_s). Staff: busy while engaged, else free. Customers: engaged during an engagement; browsing
        after one has ended in this visit; greeted after a greet; otherwise waiting, with wait_s counted from a known
        entry (None when the entry is unknown, or when there is no visit on record)."""
        if role == "staff":
            busy = any(a - _EPS <= t <= b + _EPS for a, b in self._staff_eng.get(pid, ()))
            return ("busy" if busy else "free"), None
        spans = self._cust_eng.get(pid, ())
        if any(a - _EPS <= t <= b + _EPS for a, b in spans):
            return "engaged", None
        visit = None
        for enter in self._visits.get(pid, ()):
            if enter[0] > t + _EPS:
                break
            visit = enter
        if visit is None:
            return "waiting", None
        t_enter, known = visit
        if any(t_enter - _EPS <= a and b < t for a, b in spans):
            return "browsing", None
        if any(t_enter - _EPS <= g <= t + _EPS for g in self._greets.get(pid, ())):
            return "greeted", None
        return "waiting", (t - t_enter if known else None)

    def pairs(self, t: float, drawn: set[int]) -> list[tuple[int, int, str]]:
        """(staff, customer, "engaged") for engagements active at t where both people are on screen."""
        return sorted({(s, c, "engaged") for a, b, s, c in self._engs
                       if a - _EPS <= t <= b + _EPS and s in drawn and c in drawn})

    def alert(self, t: float) -> str | None:
        """The banner text for the latest wait_alert / walkout that happened in the last BANNER_S seconds."""
        text = None
        for t0, msg in self._banners:
            if t0 - _EPS <= t < t0 + BANNER_S:
                text = msg
        return text


class _Scene:
    """One analysed clip, ready to answer "who is where at time t": interpolated boxes and keypoints of the tracked
    people, the raw detections around t, their states, and the face blur for a frame at t."""

    def __init__(self, res: PerceptionResult, an: Analysis):
        self.res, self.an = res, an
        obs_pid = np.asarray(an.obs_pid)
        if len(obs_pid) != len(res.obs_frame):
            raise ValueError("the analysis does not belong to this perception result")
        self.frame_times = np.asarray(res.frame_times, np.float64)
        n_frames = len(self.frame_times)
        raw_frame = np.asarray(res.raw_frame, np.int64)
        order = np.argsort(raw_frame, kind="stable")
        self._raw = np.asarray(res.raw_bbox, np.float64).reshape(-1, 4)[order]
        self._raw_start = np.searchsorted(raw_frame[order], np.arange(n_frames + 1))
        kept = np.flatnonzero(obs_pid > 0)
        kept = kept[np.lexsort((res.obs_frame[kept], obs_pid[kept]))]
        obs_bbox, obs_kpts = np.asarray(res.obs_bbox, np.float64), np.asarray(res.obs_kpts, np.float32)
        self._tracks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group in np.split(kept, np.flatnonzero(np.diff(obs_pid[kept])) + 1):
            if len(group):
                self._tracks[int(obs_pid[group[0]])] = (
                    self.frame_times[res.obs_frame[group]], obs_bbox[group], obs_kpts[group])
        self.events = _EventIndex(an.events, an.duration, an.names)

    # ---- people

    def _at(self, pid: int, t: float):
        """(bbox, keypoints) of a tracked person at t by the module's gap rules, or None."""
        ts, boxes, kpts = self._tracks[pid]
        i = int(np.searchsorted(ts, t + _EPS, side="right"))     # ts[:i] are at or before t
        a = i - 1 if i > 0 else None
        b = i if i < len(ts) else None
        if a is not None and t - ts[a] <= _EPS:
            return boxes[a], kpts[a]
        if a is not None and b is not None and ts[b] - ts[a] <= MAX_INTERP_GAP_S + _EPS:
            w = (t - ts[a]) / (ts[b] - ts[a])
            k = kpts[a].copy()
            k[:, :2] = kpts[a][:, :2] * (1.0 - w) + kpts[b][:, :2] * w
            k[:, 2] = np.minimum(kpts[a][:, 2], kpts[b][:, 2])
            return boxes[a] * (1.0 - w) + boxes[b] * w, k
        near = [j for j in (a, b) if j is not None]
        if not near:
            return None
        j = min(near, key=lambda j: abs(ts[j] - t))
        return (boxes[j], kpts[j]) if abs(ts[j] - t) <= NEAREST_S + _EPS else None

    def label(self, pid: int, role: str) -> str:
        name = self.an.names.get(pid)
        return f"{'S' if role == 'staff' else 'C'}{pid}" + (f" {name}" if name else "")

    def people(self, t: float) -> list[dict]:
        """The people to draw at t, by pid: bbox, kpts, role, label, state, wait_s (and pid)."""
        out = []
        for pid in self._tracks:
            hit = self._at(pid, t)
            if hit is None:
                continue
            bbox, kpts = hit
            role = self.an.roles.get(pid, "customer")
            state, wait_s = self.events.state_of(pid, role, t)
            out.append({"pid": pid, "bbox": tuple(float(v) for v in bbox), "kpts": kpts, "role": role,
                        "label": self.label(pid, role), "state": state, "wait_s": wait_s})
        return out

    # ---- the blur

    def raw_near(self, t: float) -> np.ndarray:
        """Boxes of every person detection of the sampled perception frames just before and after t (just the one
        when t is on a sampled frame)."""
        n = len(self.frame_times)
        if not n:
            return np.zeros((0, 4))
        j = int(np.searchsorted(self.frame_times, t + _EPS, side="right")) - 1
        exact = j >= 0 and t - self.frame_times[j] <= _EPS
        frames = {max(j, 0)} if exact else {max(j, 0), min(j + 1, n - 1)}
        return np.concatenate([self._raw[self._raw_start[f]:self._raw_start[f + 1]] for f in sorted(frames)])

    def held(self, t: float, drawn: set[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(bbox, keypoints) of the nearest observation of each tracked person who is NOT drawn at t, when it is
        within HOLD_BLUR_S: their face stays blurred where it was last seen."""
        out = []
        for pid, (ts, boxes, kpts) in self._tracks.items():
            if pid in drawn:
                continue
            j = int(np.argmin(np.abs(ts - t)))
            if abs(ts[j] - t) <= HOLD_BLUR_S + _EPS:
                out.append((boxes[j], kpts[j]))
        return out

    def blur(self, frame: np.ndarray, t: float, people: list[dict]) -> np.ndarray:
        """A blurred copy of `frame` (the unmodified video frame at time t): every raw detection nearby, every drawn
        person (by keypoints) and every undrawn tracked person's last known position."""
        boxes = [tuple(b) for b in self.raw_near(t)]
        kpts: list = [None] * len(boxes)
        for p in people:
            boxes.append(p["bbox"])
            kpts.append(p["kpts"])
        for bbox, k in self.held(t, {p["pid"] for p in people}):
            boxes.append(tuple(bbox))
            kpts.append(k)
        return blur_faces(frame, boxes, kpts)

    # ---- the whole overlay

    def annotate(self, blurred: np.ndarray, t: float, people: list[dict], hud: str, scale: float = 1.0,
                 banner: bool = True) -> np.ndarray:
        """This scene's overlay drawn IN PLACE onto `blurred` (a private, already face-blurred image `scale` x the
        processed size) and returned. Boxes are scaled by `scale`; `banner` adds the alert banner active at t."""
        if scale != 1.0:
            people = [{**p, "bbox": tuple(v * scale for v in p["bbox"])} for p in people]
        pairs = self.events.pairs(t, {p["pid"] for p in people})
        _draw(blurred, people, pairs, hud, self.events.alert(t) if banner else None)
        return blurred


# =============================================================================================== the annotated video

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        atomic_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_jpeg(path: Path, img: np.ndarray) -> Path:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"cannot encode {path.name}")
    _atomic_write_bytes(path, buf.tobytes())
    return path


def render_video(video_path, res: PerceptionResult, an: Analysis, out_path, fps_out: float = 10.0,
                 progress: Callable[[float, str], None] | None = None) -> Path:
    """Write the annotated, face-blurred H.264 video of `video_path` to `out_path` and return its path.

    The source is decoded with `iter_frames` at `fps_out` (capped at the source's own frame rate, so a slow source
    is not sped up) in the processed size and trimmed like the perception run. Each frame is blurred first (see the
    module docstring), then boxes, labels, engaged lines, waiting timers, greeted ticks, the HUD ("m:ss | preset")
    and the 3 s alert banners are drawn. `progress(fraction, message)` is called about every 10 frames and finally
    with 1.0. The file appears at `out_path` only when it is complete; a failure (VideoError for an unreadable source)
    leaves nothing behind. A source that stops decoding early is drawn as far as it decodes.
    """
    video = res.video
    scene = _Scene(res, an)
    size = (int(video.width), int(video.height))
    fps = min(float(fps_out), float(video.src_fps))
    duration = float(video.duration) or 1.0
    out_path = Path(out_path)
    n = 0
    with H264Writer(out_path, size, fps) as writer:
        for _, t, frame in iter_frames(video_path, sample_fps=fps_out, max_width=size[0],
                                       max_seconds=video.max_seconds):
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            people = scene.people(t)
            blurred = scene.blur(frame, t, people)                 # blur FIRST, then draw
            writer.write(scene.annotate(blurred, t, people, f"{_mmss(t)} | {an.preset}"))
            if progress is not None and n % 10 == 0:
                progress(min(1.0, max(0.0, t / duration)), f"Rendering the annotated video: {t:.0f} of {duration:.0f} s")
            n += 1
    if progress is not None:
        progress(1.0, "Annotated video ready")
    return out_path


# ==================================================================================================== timeline image

TIMELINE_ROW_H, TIMELINE_GAP = 22, 8
TIMELINE_LEFT, TIMELINE_RIGHT, TIMELINE_TOP = 150, 24, 40
TIMELINE_AXIS_H, TIMELINE_LEGEND_H = 30, 40
GRID_BGR, ROW_BG_BGR = (228, 228, 228), (245, 245, 245)
LONG_CLIP_S = 120.0       # above this the time axis is labelled every minute instead of every 10 s


def _tick_step(duration: float) -> float:
    return 10.0 if duration <= LONG_CLIP_S else 60.0


def _timeline_order(an: Analysis) -> list:
    """The people in order of first appearance."""
    return sorted(an.persons, key=lambda p: (p.t_first, p.pid))


def _fit_text(text: str, fs: float, max_w: int) -> str:
    text = _ascii(text)
    while len(text) > 2 and cv2.getTextSize(text, _FONT, fs, 1)[0][0] > max_w:
        text = text[:-2] + "."
    return text


def _alert_marker(img: np.ndarray, kind: str, x: int, y: int) -> None:
    """A red marker centred on x, with y the top of the row: a triangle for a long wait, a cross for a walk-out."""
    if kind == "wait_alert":
        cv2.fillPoly(img, [np.array([(x - 6, y - 5), (x + 6, y - 5), (x, y + 7)], np.int32)], ALERT_BGR, cv2.LINE_8)
    else:
        h = TIMELINE_ROW_H
        cv2.line(img, (x - 5, y + 4), (x + 5, y + h - 4), ALERT_BGR, 3, cv2.LINE_8)
        cv2.line(img, (x - 5, y + h - 4), (x + 5, y + 4), ALERT_BGR, 3, cv2.LINE_8)


def timeline_png(an: Analysis, out_path, width: int = 1200) -> Path:
    """One row per person (in order of first appearance) over the clip's time axis, drawn with OpenCV: present in
    grey, waiting for a greet in orange, engaged in blue, a green tick at the greet, red markers for long-wait alerts
    (triangle) and walk-outs (cross), time labels every 10 s (every 60 s for clips over 2 minutes) and a legend.
    No frames are involved, so there is nothing to blur. Returns `out_path`."""
    width = max(int(width), 480)
    people = _timeline_order(an)
    dur = float(an.duration) if an.duration and math.isfinite(an.duration) and an.duration > 0 else 1.0
    left, top = TIMELINE_LEFT, TIMELINE_TOP
    plot_w = width - left - TIMELINE_RIGHT
    row_h, pitch = TIMELINE_ROW_H, TIMELINE_ROW_H + TIMELINE_GAP
    plot_h = max(len(people), 1) * pitch
    axis_h, legend_h = TIMELINE_AXIS_H, TIMELINE_LEGEND_H
    img = np.full((top + plot_h + axis_h + legend_h, width, 3), 255, np.uint8)

    def x_of(t) -> int:
        return int(round(left + min(max(float(t), 0.0), dur) / dur * plot_w))

    cv2.putText(img, _ascii(f"Timeline - {len(people)} people, {_mmss(dur)}"), (10, 26), _FONT, 0.6, DARK, 1,
                cv2.LINE_AA)
    y_axis = top + plot_h
    for i in range(len(people)):
        y0 = top + i * pitch + TIMELINE_GAP // 2
        cv2.rectangle(img, (left, y0 - 1), (left + plot_w, y0 + row_h + 1), ROW_BG_BGR, -1)
    step = _tick_step(dur)
    k = 0
    while k * step <= dur + _EPS:
        x = x_of(k * step)
        label = _mmss(k * step)
        tw = cv2.getTextSize(label, _FONT, 0.45, 1)[0][0]
        cv2.line(img, (x, top), (x, y_axis), GRID_BGR, 1)
        cv2.line(img, (x, y_axis), (x, y_axis + 5), DARK, 1)
        cv2.putText(img, label, (min(max(x - tw // 2, 0), width - tw - 2), y_axis + 20), _FONT, 0.45, DARK, 1,
                    cv2.LINE_AA)
        k += 1
    cv2.line(img, (left, y_axis), (left + plot_w, y_axis), DARK, 1)

    markers: dict[int, list[tuple[str, float]]] = {}
    for e in an.events:
        if e.type in ("wait_alert", "walkout") and e.customer is not None:
            markers.setdefault(e.customer, []).append((e.type, e.t))
    if not people:
        cv2.putText(img, "nobody detected", (left + 10, top + row_h), _FONT, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
    for i, p in enumerate(people):
        y0 = top + i * pitch + TIMELINE_GAP // 2
        tl = an.timelines.get(p.pid) or {}

        def bars(spans, colour, inset):
            for a, b in spans or ():
                xa, xb = x_of(a), x_of(b)
                cv2.rectangle(img, (xa, y0 + inset), (max(xb, xa + 1), y0 + row_h - inset), colour, -1)

        bars(tl.get("present"), PRESENT_BGR, 7)
        bars(tl.get("waiting"), CUSTOMER_BGR, 2)
        bars(tl.get("engaged"), STAFF_BGR, 2)
        if tl.get("greeted") is not None:
            xg = x_of(tl["greeted"])
            cv2.line(img, (xg, y0 - 2), (xg, y0 + row_h + 2), GREET_BGR, 3, cv2.LINE_8)
            cv2.circle(img, (xg, y0 - 3), 4, GREET_BGR, -1, cv2.LINE_8)
        for kind, t in markers.get(p.pid, ()):
            _alert_marker(img, kind, x_of(t), y0)
        role = an.roles.get(p.pid, "customer")
        name = an.names.get(p.pid)
        text = f"{'S' if role == 'staff' else 'C'}{p.pid}" + (f" {name}" if name else "")
        swatch = STAFF_BGR if role == "staff" else CUSTOMER_BGR
        cv2.rectangle(img, (8, y0 + row_h // 2 - 5), (18, y0 + row_h // 2 + 5), swatch, -1)
        cv2.putText(img, _fit_text(text, 0.45, left - 30), (24, y0 + row_h // 2 + 5), _FONT, 0.45, DARK, 1, cv2.LINE_AA)

    x, y = left, y_axis + axis_h + 6
    for kind, text in (("present", "present"), ("waiting", "waiting to be greeted"), ("engaged", "engaged"),
                       ("greet", "greeted"), ("wait_alert", "long-wait alert"), ("walkout", "walk-out")):
        if kind in ("present", "waiting", "engaged"):
            colour = {"present": PRESENT_BGR, "waiting": CUSTOMER_BGR, "engaged": STAFF_BGR}[kind]
            cv2.rectangle(img, (x, y), (x + 16, y + 12), colour, -1)
        elif kind == "greet":
            cv2.line(img, (x + 8, y - 2), (x + 8, y + 14), GREET_BGR, 3, cv2.LINE_8)
            cv2.circle(img, (x + 8, y - 3), 4, GREET_BGR, -1, cv2.LINE_8)
        else:
            _alert_marker(img, kind, x + 8, y - 4)
        cv2.putText(img, text, (x + 24, y + 11), _FONT, 0.45, DARK, 1, cv2.LINE_AA)
        x += 24 + cv2.getTextSize(text, _FONT, 0.45, 1)[0][0] + 22

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_png(out_path, img)


def _write_png(path: Path, img: np.ndarray) -> Path:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"cannot encode {path.name}")
    _atomic_write_bytes(path, buf.tobytes())
    return path


# =================================================================================================== contact sheets

SHEET_TYPES = ("wait_alert", "walkout", "greet", "engagement")     # also the order of the balanced pick
SHEET_PANEL_H = 240
SHEET_SAMPLE_FPS = 10.0      # contact-sheet frames are the nearest decoded frame at this rate (within 0.05 s)
SHEET_OFFSET_S = 2.0
SHEET_CAPTION_H = 30
_SHEET_NAME = re.compile(r"(\d{2})_(greet|engagement|wait_alert|walkout)_(\d+)s\.jpg")
SHEET_LABELS = {"greet": "Greet", "engagement": "Engagement", "wait_alert": "Long wait", "walkout": "Walk-out"}


def contact_sheet_files(out_dir) -> list[dict]:
    """The contact sheets `contact_sheets` left in `out_dir`, in their number (= time) order:
    `[{"name", "type", "t", "caption"}]`, with type and t (whole seconds) read from the `{n:02d}_{type}_{t:.0f}s.jpg`
    file name and a short caption such as "Long wait at 0:42". An absent directory gives []."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found = []
    for p in out_dir.iterdir():
        m = _SHEET_NAME.fullmatch(p.name)
        if m and p.is_file():
            n, kind, t = int(m.group(1)), m.group(2), float(m.group(3))
            found.append((n, p.name, {"name": p.name, "type": kind, "t": t,
                                      "caption": f"{SHEET_LABELS[kind]} at {_mmss(t)}"}))
    return [item for _, _, item in sorted(found)]


def _pick_events(events, limit: int) -> list:
    """Up to `limit` events of the sheet types, taken round-robin over the types (long waits and walk-outs first) and
    each type in time order, then put back in time order: a clip with 40 greets still shows its alerts."""
    groups = {k: sorted((e for e in events if e.type == k), key=lambda e: e.t) for k in SHEET_TYPES}
    picked: list = []
    while len(picked) < limit and any(groups.values()):
        for k in SHEET_TYPES:
            if groups[k] and len(picked) < limit:
                picked.append(groups[k].pop(0))
    return sorted(picked, key=lambda e: (e.t, SHEET_TYPES.index(e.type)))


def _caption(e, idx: _EventIndex) -> str:
    who = f"C{e.customer}" if e.customer is not None else "a customer"
    staff = idx.staff_name(e.staff) if e.staff is not None else "staff"
    if e.type == "greet":
        ttg = e.data.get("time_to_greet")
        after = f" after {ttg:.0f} s" if isinstance(ttg, (int, float)) and math.isfinite(ttg) else ""
        return f"Greet at {_mmss(e.t)}: {who} greeted by {staff}{after}"
    if e.type == "engagement":
        end = e.t if e.t_end is None else e.t_end
        return f"Engagement {_mmss(e.t)}-{_mmss(end)}: {who} with {staff} ({max(0.0, end - e.t):.0f} s)"
    free = ", ".join(idx.staff_name(s) for s in e.data.get("free_staff") or []) or "none"
    if e.type == "wait_alert":
        wait = e.data.get("wait_s")
        waited = f" waiting {wait:.0f} s," if isinstance(wait, (int, float)) and math.isfinite(wait) else ""
        return f"Long wait at {_mmss(e.t)}: {who}{waited} free staff: {free}"
    return f"Walk-out at {_mmss(e.t)}: {who} left without a greet, free staff: {free}"


def _caption_bar(text: str, width: int) -> np.ndarray:
    bar = np.full((SHEET_CAPTION_H, width, 3), HUD_BG, np.uint8)
    text, fs = _ascii(text), 0.55
    while fs > 0.35 and cv2.getTextSize(text, _FONT, fs, 1)[0][0] > width - 16:
        fs -= 0.05
    text = _fit_text(text, fs, width - 16)
    cv2.putText(bar, text, (8, SHEET_CAPTION_H // 2 + 6), _FONT, fs, WHITE, 1, cv2.LINE_AA)
    return bar


def _grab_frames(video_path, res: PerceptionResult, targets: list[float]) -> dict[float, tuple[float, np.ndarray]]:
    """The decoded frame nearest in time to each target, in one pass over the source: {target: (actual time, frame)}.

    Frames arrive in time order, so once a frame at or after a target has been seen the nearest one is known (it is
    that frame or the one before it) and the target is finished; the pass ends when every target is. There is no
    distance limit: on a 2 fps source a target 0.2 s from a frame still gets that frame, and a target past the end of
    the clip gets the last frame, which is then the nearest. Targets may come in any order."""
    video = res.video
    best: dict[float, tuple[float, float, np.ndarray]] = {}        # target -> (distance, time, frame) so far
    open_targets = sorted(set(targets))                              # targets still waiting for a frame at or after them
    with contextlib.closing(iter_frames(video_path, SHEET_SAMPLE_FPS, video.width, video.max_seconds)) as frames:
        for _, t, frame in frames:
            for target in open_targets:
                d = abs(t - target)
                if target not in best or d < best[target][0] - _EPS:   # a tie keeps the earlier frame
                    best[target] = (d, t, frame)
            open_targets = [tg for tg in open_targets if t < tg - _EPS]
            if not open_targets:
                break
    if not best:
        raise ValueError("the video has no frames")
    return {tg: (best[tg][1], best[tg][2]) for tg in targets}


def contact_sheets(video_path, res: PerceptionResult, an: Analysis, out_dir, max_events: int = 12) -> list[Path]:
    """One jpg per greet / engagement / wait_alert / walkout event (at most `max_events`, balanced over the types):
    three face-blurred frames at t-2 s, t and t+2 s (clipped to the clip) side by side, each 240 px tall with the
    boxes drawn, under a caption bar describing the event. Files are `{n:02d}_{type}_{t:.0f}s.jpg`, numbered in time
    order; older sheets of that naming left in `out_dir` by a previous run are removed. Returns the new paths.
    Raises VideoError when the source cannot be decoded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = _pick_events(an.events, max(0, int(max_events)))
    paths: list[Path] = []
    if events:
        scene = _Scene(res, an)
        hi = max(0.0, float(res.video.duration) - _EPS)
        times = [[min(max(e.t + d, 0.0), hi) for d in (-SHEET_OFFSET_S, 0.0, SHEET_OFFSET_S)] for e in events]
        grabbed = _grab_frames(video_path, res, sorted({t for row in times for t in row}))
        panels: dict[float, np.ndarray] = {}
        for n, (e, row) in enumerate(zip(events, times), start=1):
            for target in row:
                if target not in panels:
                    t, frame = grabbed[target]
                    people = scene.people(t)
                    blurred = scene.blur(frame, t, people)                # blur FIRST, then draw
                    h, w = blurred.shape[:2]
                    small = cv2.resize(blurred, (max(1, round(w * SHEET_PANEL_H / h)), SHEET_PANEL_H),
                                       interpolation=cv2.INTER_AREA)
                    panels[target] = scene.annotate(small, t, people, _mmss(t), scale=SHEET_PANEL_H / h, banner=False)
            strip = np.hstack([panels[t] for t in row])
            sheet = np.vstack([strip, _caption_bar(_caption(e, scene.events), strip.shape[1])])
            paths.append(_write_jpeg(out_dir / f"{n:02d}_{e.type}_{e.t:.0f}s.jpg", sheet))
    keep = {p.name for p in paths}
    for old in out_dir.iterdir():
        if _SHEET_NAME.fullmatch(old.name) and old.name not in keep:
            old.unlink(missing_ok=True)
    return paths


# ====================================================================================================== thumbnails

THUMB_MAX_H = 160


def _shrink(img: np.ndarray, max_h: int = THUMB_MAX_H) -> np.ndarray:
    if img.shape[0] <= max_h:
        return img
    s = max_h / img.shape[0]
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), max_h), interpolation=cv2.INTER_AREA)


def save_thumbnails(res: PerceptionResult, an: Analysis, out_dir) -> dict[int, str]:
    """Put a face-blurred picture of each person at `out_dir/<pid>.jpg` and return {pid: file name}.

    The pipeline already saved a blurred best crop per track (`crops_dir/<Person.thumb>`): that file is copied. For a
    person whose crop is missing, their best observation (largest box x score) is cut from the source video, with every
    person in that frame blurred first. People who still cannot be pictured (source gone, frame unreadable) are left
    out of the result with a logged warning."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[int, str] = {}
    missing = []
    for p in an.persons:
        src = Path(res.crops_dir) / Path(p.thumb).name if res.crops_dir and p.thumb else None
        if src is not None and src.is_file():
            _atomic_write_bytes(out_dir / f"{p.pid}.jpg", src.read_bytes())
            done[p.pid] = f"{p.pid}.jpg"
        else:
            missing.append(p)
    if missing:
        done.update(_cut_thumbnails(res, an, missing, out_dir))
    return dict(sorted(done.items()))


def _cut_thumbnails(res: PerceptionResult, an: Analysis, people: list, out_dir: Path) -> dict[int, str]:
    from .perception.embed import person_crop

    obs_pid = np.asarray(an.obs_pid)
    best: dict[int, int] = {}                                    # pid -> index of the best observation
    for p in people:
        idx = np.flatnonzero(obs_pid == p.pid)
        if len(idx):
            b = res.obs_bbox[idx]
            best[p.pid] = int(idx[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * res.obs_score[idx])])
    by_frame: dict[int, list[int]] = {}
    for pid, i in best.items():
        by_frame.setdefault(int(res.obs_frame[i]), []).append(pid)
    out: dict[int, str] = {}
    wanted = {int(res.src_frame_idx[fi]): fi for fi in by_frame}     # source frame index -> sampled frame index
    todo = set(wanted)
    try:
        # The same decode as the perception pass (same sampling rule), matched on the source frame index, so the blur
        # boxes always belong to the picture they are applied to. There is no seeking.
        video = res.video
        with contextlib.closing(iter_frames(video.path, video.sample_fps, video.width, video.max_seconds)) as frames:
            for src_i, _, frame in frames:
                if src_i not in todo:
                    continue
                todo.discard(src_i)
                fi = wanted[src_i]
                raw_boxes = np.asarray(res.raw_bbox, np.float64).reshape(-1, 4)[res.raw_frame == fi]
                here = np.flatnonzero(res.obs_frame == fi)
                boxes = [tuple(b) for b in raw_boxes] + [tuple(res.obs_bbox[i]) for i in here]
                kpts = [None] * len(raw_boxes) + [res.obs_kpts[i] for i in here]
                blurred = blur_faces(frame, boxes, kpts)         # everyone in the frame, not just this person
                for pid in by_frame[fi]:
                    crop = person_crop(blurred, res.obs_bbox[best[pid]])
                    if crop is not None and crop.size:
                        out[pid] = f"{pid}.jpg"
                        _write_jpeg(out_dir / out[pid], _shrink(crop))
                if not todo:
                    break
    except VideoError as exc:
        log.warning("cannot read %s to cut thumbnails: %s", res.video.path, exc)
    for p in people:
        if p.pid not in out:
            log.warning("no thumbnail for person %d", p.pid)
    return out
