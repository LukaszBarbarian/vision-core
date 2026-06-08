import datetime
import numpy as np
from typing import List
from config.models.detection_event import DetectionEvent

class FrameContext:
    """
    Carries the raw video frame matrix along with its structural metadata 
    and appended AI detection results.
    """
    def __init__(self, camera_id: str, frame_id: int, frame_data: np.ndarray):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.timestamp = datetime.datetime.now()
        self.frame_data = frame_data  # NumPy matrix (BGR format)
        
        # Pojemnik na wyniki analizy AI dla tej konkretnej klatki
        self.detections: List[DetectionEvent] = []

    @property
    def is_empty(self) -> bool:
        return self.frame_data is None or self.frame_data.size == 0