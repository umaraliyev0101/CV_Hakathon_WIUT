import json
import numpy as np
from showroom_coach.types import Event, Det, to_jsonable


def test_event_roundtrips_to_json():
    e = Event("greet", 9.0, customer=2, staff=1, data={"time_to_greet": np.float32(4.0)})
    d = json.loads(json.dumps(to_jsonable(e)))
    assert d == {"type": "greet", "t": 9.0, "customer": 2, "staff": 1, "t_end": None,
                 "data": {"time_to_greet": 4.0}}


def test_to_jsonable_handles_numpy_and_int_keys():
    d = to_jsonable({1: np.array([1.5, 2.0]), "b": (np.int64(3), Det((0, 0, 1, 1), 0.5))})
    assert d == {"1": [1.5, 2.0], "b": [3, {"bbox": [0, 0, 1, 1], "score": 0.5, "cls": 0}]}


def test_to_jsonable_non_finite_floats_become_null():
    d = to_jsonable({"a": float("nan"), "b": np.float64("inf"), "c": [np.float32("nan"), 1.0]})
    assert d == {"a": None, "b": None, "c": [None, 1.0]}
    json.dumps(d, allow_nan=False)  # strict JSON must accept the result
