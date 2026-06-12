import numpy as np

from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from detection.model_provider import DetectionModelProvider

from domain.detection_result import DetectionResult


class CarDetector(BaseDetector):

    def __init__(
        self,
        model_provider: DetectionModelProvider
    ):
        self._model = model_provider.get_model(
            detection_type=DetectionType.CAR
        )

        self._vehicle_classes = [
            2,  # car
            3,  # motorcycle
            5,  # bus
            7   # truck
        ]

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict
    ) -> list[DetectionResult]:

        if frame_data is None or frame_data.size == 0:
            return []

        if "clean_frame" not in context:
            context["clean_frame"] = frame_data.copy()

        results = self._model(
            frame_data,
            classes=self._vehicle_classes,
            conf=0.25,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        if len(results[0].boxes) == 0:
            return detections

        context["detected_cars"] = results[0].boxes

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            if cls_id not in self._vehicle_classes:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            class_name = self._model.names.get(
                cls_id,
                str(cls_id)
            )

            detections.append(
                DetectionResult(
                    detection_type="car_detection",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    label=class_name
                )
            )


        return detections
    



    def get_color(cls) -> tuple[int, int, int]:
        return (0, 255, 255)  # żółty BGR

    @classmethod
    def get_display_name(cls) -> str:
        return "Pojazd"