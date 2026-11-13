import numpy as np

from infrastructure.ai_detection.detectors.motion_detector import MotionFilter

H, W = 1440, 2560
ROI = [[200, 300], [1200, 300], [1200, 900], [200, 900]]


def frames_with_square(x, y, size=300):
    still = np.full((H, W, 3), 90, dtype=np.uint8)
    moved = still.copy()
    moved[y:y + size, x:x + size] = 250
    return still, moved


def test_motion_outside_roi_is_ignored():
    motion = MotionFilter()
    still, moved = frames_with_square(1900, 1000)
    motion.has_significant_motion("cam", still, roi_polygon=ROI)
    assert not motion.has_significant_motion("cam", moved, roi_polygon=ROI)


def test_motion_inside_roi_counts():
    motion = MotionFilter()
    still, moved = frames_with_square(500, 450)
    motion.has_significant_motion("cam", still, roi_polygon=ROI)
    assert motion.has_significant_motion("cam", moved, roi_polygon=ROI)


def test_without_roi_the_whole_frame_counts():
    motion = MotionFilter()
    still, moved = frames_with_square(1900, 1000)
    motion.has_significant_motion("cam", still)
    assert motion.has_significant_motion("cam", moved)
