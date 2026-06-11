import numpy as np
from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from detection.model_provider import DetectionModelProvider




class DetectorProvider:
    def __init__(self, model_provider: DetectionModelProvider):
        self._model_provider = model_provider
        self._detectors = {}

    def get_detector(self, detection_type: str) -> BaseDetector:
        try:
            d_type = DetectionType(detection_type)
        except ValueError:
            raise ValueError(f"Unknown detection type configured: {detection_type}")

        if d_type not in self._detectors:
            if d_type == DetectionType.CAR:
                from detection.detectors.car_detector import CarDetector
                self._detectors[d_type] = CarDetector(self._model_provider)

            elif d_type == DetectionType.LICENSE_PLATE:
                from detection.detectors.anpr_detector import AnprDetector
                self._detectors[d_type] = AnprDetector(self._model_provider)

            elif d_type == DetectionType.CHICKEN:
                from detection.detectors.chicken_detector import ChickenDetector
                self._detectors[d_type] = ChickenDetector(self._model_provider)

            elif d_type == DetectionType.FACE:
                from detection.detectors.face_detector import FaceDetector
                self._detectors[d_type] = FaceDetector(self._model_provider)

            elif d_type == DetectionType.PERSON:
                from detection.detectors.person_detector import PersonDetector
                self._detectors[d_type] = PersonDetector(self._model_provider)                
                
            else:
                raise ValueError(f"No implementation for: {d_type}")
                
        return self._detectors[d_type]