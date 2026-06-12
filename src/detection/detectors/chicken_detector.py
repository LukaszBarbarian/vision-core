import numpy as np

from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from detection.model_provider import DetectionModelProvider
from domain.detection_result import DetectionResult


class ChickenDetector(BaseDetector):
    def __init__(
        self,
        model_provider: DetectionModelProvider,
    ):
        self._model = model_provider.get_model(
            detection_type=DetectionType.CHICKEN,
        )
        self._chicken_classes = [14]

    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (0, 165, 255)

    @classmethod
    def get_display_name(cls) -> str:
        return "Kurka"

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict,
    ) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        results = self._model.track(
            frame_data,
            persist=True,
            classes=self._chicken_classes,
            conf=0.15,
            verbose=False,
        )

        detections: list[DetectionResult] = []

        if not results or len(results[0].boxes) == 0:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append(
                DetectionResult(
                    detection_type=DetectionType.CHICKEN.value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    label="Kurka",
                )
            )

        return detections
