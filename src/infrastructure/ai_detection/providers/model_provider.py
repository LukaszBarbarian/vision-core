from ultralytics import YOLO

from domain.models.detection_type import DetectionType


class DetectionModelProvider:
    def __init__(self):
        self._models_by_name: dict[str, YOLO] = {}

    def get_model(self, detection_type: DetectionType):
        model_name = self._resolve_model(detection_type)

        if model_name not in self._models_by_name:
            self._models_by_name[model_name] = YOLO(model_name)

        return self._models_by_name[model_name]

    def _resolve_model(self, detection_type):
        mapping = {
            DetectionType.CHICKEN: "yolov8m.pt",
            DetectionType.CAR: "yolov8n.pt",
            DetectionType.PERSON: "yolov8n.pt",
            DetectionType.LICENSE_PLATE: "D:/projects/cv-demo7/vision-core/license_plate_detector.pt",
        }

        return mapping[detection_type]