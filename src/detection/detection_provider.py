import numpy as np
from detection.detectors.base import BaseDetector
from detection.detectors.car_detector import CarDetector
from detection.detectors.poultry_detector import PoultryDetector
from src.detection.model_provider import ModelProvider



class DetectorProvider:
    """Fabryka zarządzająca cyklem życia detektorów biznesowych."""
    def __init__(self, model_provider: ModelProvider):
        self._model_provider = model_provider
        self._detectors = {}

    def get_detector(self, detection_type: str) -> BaseDetector:
        if detection_type not in self._detectors:
            if detection_type == "car_detection":
                self._detectors[detection_type] = CarDetector(self._model_provider)
            elif detection_type == "poultry_detection":
                self._detectors[detection_type] = PoultryDetector(self._model_provider)
            elif detection_type == "anpr_detection":
                from detection.detectors.anpr_detector import AnprDetector
                self._detectors[detection_type] = AnprDetector(self._model_provider)
            elif detection_type == "chicken_detection":
                from detection.detectors.chicken_detector import ChickenDetector
                self._detectors[detection_type] = ChickenDetector(self._model_provider)
            else:
                raise ValueError(f"Unknown detection type configured: {detection_type}")
                
        return self._detectors[detection_type]