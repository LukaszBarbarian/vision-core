import numpy as np

from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from domain.detection_result import DetectionResult


class PersonDetector(BaseDetector):
    def __init__(self, model_provider):
        self._model = model_provider.get_model(DetectionType.PERSON)

    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (255, 0, 0)

    @classmethod
    def get_display_name(cls) -> str:
        return "Osoba"

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict,
    ) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        results = self._model(
            frame_data,
            classes=[0],
            conf=0.5,
            verbose=False,
        )

        detections: list[DetectionResult] = []

        if not results or len(results[0].boxes) == 0:
            context["detected_persons"] = []
            return detections

        person_boxes = results[0].boxes
        context["detected_persons"] = person_boxes

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                DetectionResult(
                    detection_type=DetectionType.PERSON.value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    label="Osoba",
                )
            )

        return detections
