import math

from sim import anim_plan, behaviour, schema


def _plans(cfg, tl):
    lib = anim_plan.load_lib()
    out = {}
    for pid, ag in tl["agents"].items():
        p = schema.person(cfg, int(pid))
        out[pid] = anim_plan.plan(ag["frames"], "M", ag["role"], p["height_m"], 1.8, tl["fps"], int(pid), lib)
    return lib, out


def test_weights_sum_to_one_and_frames_in_range(cfg, timeline):
    lib, plans = _plans(cfg, timeline)
    for pid, pl in plans.items():
        assert len(pl) == len(timeline["agents"][pid]["frames"])
        for entries in pl:
            assert abs(sum(w for _, _, w in entries) - 1.0) < 1e-9
            for clip, f, w in entries:
                assert 1.0 <= f <= lib["clips"][clip]["frames"] + 1e-9
                assert w >= 0.0


def test_walk_phase_matches_distance(cfg, timeline):
    """While walking (no blend), the walk clip advances one full cycle per scaled cycle length."""
    lib, plans = _plans(cfg, timeline)
    for pid, pl in plans.items():
        fr = timeline["agents"][pid]["frames"]
        h = schema.person(cfg, int(pid))["height_m"]
        L = lib["clips"]["m_walk_neutral_01"]["cycle_m"] * h / 1.8
        for i in range(1, len(fr)):
            if len(pl[i]) == 1 and len(pl[i - 1]) == 1 and pl[i][0][0].endswith("walk_neutral_01") \
                    and pl[i - 1][0][0] == pl[i][0][0]:
                d = math.dist(fr[i]["pos_m"], fr[i - 1]["pos_m"])
                df = (pl[i][0][1] - pl[i - 1][0][1]) % 36.0
                assert abs(df - d / L * 36.0) < 1e-6


def test_customer_listens_staff_talks(cfg, timeline):
    lib, plans = _plans(cfg, timeline)
    for pid, pl in plans.items():
        role = timeline["agents"][pid]["role"]
        for st, entries in zip(timeline["agents"][pid]["frames"], pl):
            if st["anim_clip"].startswith("talk") and len(entries) == 1:
                assert ("listen" in entries[0][0]) == (role == "customer")


def test_loop_crossfade_is_continuous():
    lib = anim_plan.load_lib()
    clip = "m_idle_look_around_01"
    prev = None
    for k in range(0, 2000):
        x = k * 0.5
        ent = anim_plan._loop_frames(lib, clip, x)
        main = max(ent, key=lambda e: e[1])
        if prev is not None and len(ent) == 1 and len(prev) == 1:
            assert abs(ent[0][0] - prev[0][0]) <= 0.5 + 1e-9 or abs(ent[0][0] - prev[0][0]) < 1e-9
        prev = ent
