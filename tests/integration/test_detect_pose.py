import cv2
import numpy as np
import pytest

from pulse.config import get_config
from pulse.perception.detect import Detector
from pulse.perception.pose import PoseEstimator

pytestmark = pytest.mark.slow


def _frame(video, idx=300, width=960):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    assert ok, f"cannot read frame {idx}"
    h, w = f.shape[:2]
    return cv2.resize(f, (width, int(round(h * width / w))))


def test_detect_and_pose_on_meva(meva_dev_clip):
    cfg = get_config("short")
    # Frame 300 has no GT person in this clip (first person appears at ~621); use frame 700,
    # where CLIPS.md verified a GT person box.
    frame = _frame(meva_dev_clip, 700)
    det = Detector(cfg.detector, cfg)
    persons = det.detect(frame, ("person",))
    assert len(persons) >= 1
    pe = PoseEstimator(cfg.pose_model)
    kps = pe.pose(frame, [d.bbox for d in persons])
    assert kps.shape == (len(persons), 17, 3)
    assert np.isfinite(kps).all()
    assert pe.pose(frame, []).shape == (0, 17, 3)
