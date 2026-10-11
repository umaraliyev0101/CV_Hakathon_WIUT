"""Car footprints for synthetic clips.

gt.json stores car *zones* (footprint + `zone_apron_m` on every side). Distances should be measured
to the car body, so rebuild the footprints from the scene config with the simulator's own layout code.
In a real deployment the footprints are marked once per camera at install time instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.layout import build_layout

CONFIGS = Path(__file__).resolve().parent.parent / "sim" / "configs"


def car_footprints(gt: dict) -> dict[str, list]:
    """{"car_<id>": footprint polygon in floor metres} for a synthetic clip's gt.json.

    Raises if the scene config no longer matches the zones the clip was rendered with.
    """
    cfg = json.loads((CONFIGS / f"scene_{int(gt['scene_id'][1:]):03d}.json").read_text())
    layout = build_layout(cfg)
    out = {}
    for car in layout["cars"]:
        name = f"car_{car['id']}"
        if name in gt["zones"] and not np.allclose(car["zone_m"], gt["zones"][name]["floor_m"], atol=1e-3):
            raise ValueError(f"{gt['clip_id']}: {name} config zone differs from rendered gt.json zone")
        out[name] = car["footprint_m"]
    return out
