import datetime
import numpy as np

from domain.detection_result import DetectionResult


class FrameContext:
    def __init__(self, camera_id: str, frame_id: int, frame_data, detections=None):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.frame_data = frame_data
        self.detections = detections if detections is not None else []

    @property
    def is_empty(self):
        return (
            self.frame_data is None
            or self.frame_data.size == 0
        )