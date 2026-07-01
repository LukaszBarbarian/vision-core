import numpy as np

from domain.models.detection_type import DetectionType
from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.providers.model_provider import DetectionModelProvider
from domain.models.detection_result import DetectionResult


class ChickenDetector(BaseDetector):
    def __init__(
        self,
        model_provider: DetectionModelProvider,
        use_tracking: bool = False,
        imgsz: int = 768,
        conf: float = 0.18,
        max_det: int = 50,
    ):
        self._model = model_provider.get_model(
            detection_type=DetectionType.CHICKEN,
        )

        # Jeśli to custom model, upewnij się, że 14 faktycznie oznacza kurę.
        # W COCO klasa 14 to zwykle "bird".
        self._chicken_classes = [14]

        self.use_tracking = bool(use_tracking)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_det = int(max_det)

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

        height, width = frame_data.shape[:2]

        if self.use_tracking:
            results = self._model.track(
                frame_data,
                persist=True,
                classes=self._chicken_classes,
                conf=self.conf,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        else:
            results = self._model(
                frame_data,
                classes=self._chicken_classes,
                conf=self.conf,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )

        detections: list[DetectionResult] = []

        if not results or len(results[0].boxes) == 0:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detection = DetectionResult(
                detection_type=DetectionType.CHICKEN.value,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=conf,
                label="Kurka",
            )

            track_id = getattr(box, "id", None)
            if track_id is not None:
                try:
                    detection.track_id = int(track_id[0])
                except Exception:
                    pass

            detections.append(detection)

        return detections