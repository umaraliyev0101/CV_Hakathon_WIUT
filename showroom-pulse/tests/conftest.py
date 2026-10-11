import pytest
from pulse.datatypes import Sample, Zone

def rect(zid, name, kind, x0, y0, x1, y1):
    return Zone(zid, name, kind, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "user")

def track(pid, points, t0=0.0, dt=0.2, speed=None, pos=None):
    """points: list of normalised (x, y) feet, one per sample. pos: list of (X, Z) or None."""
    out = []
    for i, (x, y) in enumerate(points):
        p = pos[i] if pos is not None else None
        out.append(Sample(round(t0 + i * dt, 3), pid, (x, y), (0, 0, 10, 30), p,
                          speed if speed is not None else 0.0, "full"))
    return out

def stay(pid, x, y, t0, seconds, dt=0.2, speed=0.0, pos=None):
    n = int(round(seconds / dt)) + 1
    return track(pid, [(x, y)] * n, t0=t0, dt=dt, speed=speed,
                 pos=[pos] * n if pos is not None else None)

@pytest.fixture
def zones4():
    return [rect("car-a", "Car A", "car", 0.05, 0.5, 0.3, 0.9),
            rect("car-b", "Car B", "car", 0.6, 0.5, 0.85, 0.9),
            rect("desk", "Desk", "desk", 0.4, 0.1, 0.5, 0.2),
            rect("entrance", "Entrance", "entrance", 0.9, 0.0, 1.0, 1.0)]
