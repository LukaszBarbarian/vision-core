from ultralytics import YOLO

from domain.models.detection_type import DetectionType

class DetectionModelProvider:
    def __init__(self):
        self._models = {}

    def get_model(self, detection_type: DetectionType):

        if detection_type not in self._models:

            model_name = self._resolve_model(detection_type)

            self._models[detection_type] = YOLO(model_name)

        return self._models[detection_type]

    def _resolve_model(self, detection_type):

        mapping = {
            DetectionType.CHICKEN: "yolov8m.pt",
            DetectionType.CAR: "yolov8n.pt",
            DetectionType.PERSON: "yolov8n.pt",
            DetectionType.LICENSE_PLATE: "D:/projects/cv-demo7/vision-core/license_plate_detector.pt",

        }

        return mapping[detection_type]