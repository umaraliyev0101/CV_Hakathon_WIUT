"""Timeline states -> which mocap clip(s) to sample at which clip frame, with blend weights (pure).

Blender (sim/blender/animate.py) only samples the clips at the planned frames and writes keys.

* walk: the clip phase advances with distance walked (cycle length scaled by body height), so the
  stride matches the ground speed and the planted foot does not slide.
* turn_left / turn_right (in-place turns): the 90-degree turn clip is played by accumulated turn
  angle (90 deg of yaw = the whole clip), so the feet step in sync with the body rotation; the
  clip's own root rotation is discarded (yaw comes from the timeline).
* idle / look_around / talk / listen: looped by time with a crossfade at the loop seam and a
  per-person start offset, so people are not in sync.
* any clip change: the previous clip keeps advancing and fades out over blend_s.
Customers in conversation use the listening clip; staff use the talking clips.
"""
import json
import math
from pathlib import Path

ANIMS = Path(__file__).resolve().parent / "configs" / "assets_anims.json"


def load_lib(path=ANIMS):
    return json.loads(Path(path).read_text())


def clip_name(lib, anim, gender, role):
    key = anim
    if anim.startswith("talk") and role == "customer":
        key = "listen"
    return f"{gender.lower()}_{lib['map'][key]}"


def _loop_frames(lib, clip, x):
    """Clip time x (frames, >= 0) -> [(frame, weight)] for a looped clip with a seam crossfade."""
    n = lib["clips"][clip]["frames"]
    W = lib["loop_crossfade_frames"]
    P = (n - 1) - W
    u = x % P
    main = 1.0 + W + u
    if u >= P - W:
        w2 = (u - (P - W)) / W
        return [(main, 1.0 - w2), (1.0 + (u - (P - W)), w2)]
    return [(main, 1.0)]


class _Track:
    """Running clip-time state for one clip instance."""

    def __init__(self, lib, clip, kind, offset, scale):
        self.lib, self.clip, self.kind = lib, clip, kind
        self.x = offset          # loop: frames; walk: distance (m); turn: degrees
        self.scale = scale

    def advance(self, st, prev, fps):
        if self.kind == "walk":
            if prev is not None:
                self.x += math.dist(st["pos_m"], prev["pos_m"])
        elif self.kind == "turn":
            if prev is not None:
                self.x += abs((st["yaw_deg"] - prev["yaw_deg"] + 180.0) % 360.0 - 180.0)
        else:
            self.x += self.lib["fps"] / fps

    def frames(self):
        c = self.lib["clips"][self.clip]
        if self.kind == "walk":
            L = c["cycle_m"] * self.scale
            return [(1.0 + ((self.x / L) % 1.0) * (c["frames"] - 1), 1.0)]
        if self.kind == "turn":
            return [(1.0 + ((self.x / 90.0) % 1.0) * (c["frames"] - 1), 1.0)]
        return _loop_frames(self.lib, self.clip, self.x)


def plan(frames, gender, role, height_m, src_height_m, fps, pid, lib=None):
    """frames: the agent's timeline states at scene fps. Returns one list per frame of
    (clip_name, clip_frame, weight) with weights summing to 1."""
    lib = lib or load_lib()
    scale = height_m / src_height_m
    nb = max(1, int(round(lib["blend_s"] * fps)))
    out = []
    cur = None          # (anim_key, track)
    fading = []         # [(track, start_index)]
    walk_track = None
    prev = None
    for i, st in enumerate(frames):
        anim = st["anim_clip"]
        clip = clip_name(lib, anim, gender, role)
        kind = lib["clips"][clip]["kind"]
        if cur is None or cur[0] != clip:
            if cur is not None:
                fading.append((cur[1], i))
            if kind == "walk":
                walk_track = walk_track or _Track(lib, clip, "walk", 0.0, scale)
                tr = walk_track
            else:
                off = (pid * 211.0) % 300.0 if kind == "loop" else 0.0
                tr = _Track(lib, clip, kind, off, scale)
            cur = (clip, tr)
            fading = [(t, s) for t, s in fading if t is not tr]
        else:
            cur[1].advance(st, prev, fps)
        for t, s in fading:
            t.advance(st, prev, fps)
        fading = [(t, s) for t, s in fading if i - s < nb]
        entries = []
        w_new = 1.0
        if fading:
            t, s = fading[-1]
            w_new = (i - s + 1) / (nb + 1)
            entries += [(t.clip, f, w * (1.0 - w_new)) for f, w in t.frames()]
        entries += [(cur[1].clip, f, w * w_new) for f, w in cur[1].frames()]
        out.append(entries)
        prev = st
    return out
