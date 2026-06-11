import numpy as np

from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from detection.model_provider import DetectionModelProvider

from domain.detection_result import DetectionResult


class ChickenDetector(BaseDetector):

    def __init__(
        self,
        model_provider: DetectionModelProvider
    ):
        self._model = model_provider.get_model(
            detection_type=DetectionType.CHICKEN
        )

        self._chicken_classes = [14]

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict
    ) -> list[DetectionResult]:

        if frame_data is None or frame_data.size == 0:
            return []

        results = self._model.track(
            frame_data,
            persist=True,
            classes=self._chicken_classes,
            conf=0.15,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        if len(results[0].boxes) == 0:
            return detections

        for box in results[0].boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(box.conf[0])

            detections.append(
                DetectionResult(
                    detection_type="chicken_detection",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    label="Kurka"
                )
            )

        return detections