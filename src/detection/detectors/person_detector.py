import numpy as np
from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from domain.detection_result import DetectionResult

class PersonDetector(BaseDetector):
    def __init__(self, model_provider):
        self._model = model_provider.get_model(DetectionType.PERSON)

    def execute(self, frame_data: np.ndarray, context: dict) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        # YOLO: klasa 0 to 'person'
        results = self._model(frame_data, classes=[0], conf=0.5, verbose=False)
        
        person_boxes = results[0].boxes
        context["detected_persons"] = person_boxes  # Przekazujemy do kontekstu

        detections = []
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(DetectionResult(
                detection_type="person_detection",
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(box.conf[0]),
                label="Person"
            ))
        return detections