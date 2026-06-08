from abc import ABC, abstractmethod
from typing import List
from config.models.detection_event import DetectionEvent
from src.streaming.frame_context import FrameContext

class Detector(ABC):
    @abstractmethod
    async def detect(self, frame: FrameContext) -> List[DetectionEvent]:
        """
        Processes a frame and returns a list of detected objects.
        """
        pass

# Na razie to "zaślepki", które w przyszłości podepniemy pod YOLO
class PoultryDetector(Detector):
    async def detect(self, frame: FrameContext) -> List[DetectionEvent]:
        # Tu wjedzie logika YOLO dla kur
        return []

class FaceDetector(Detector):
    async def detect(self, frame: FrameContext) -> List[DetectionEvent]:
        # Tu wjedzie logika np. z biblioteki face_recognition
        return []

class DetectorFactory:
    def __init__(self):
        # Rejestrujemy dostępne instancje detektorów (Singletony na poziom aplikacji)
        self._detectors = {
            "poultry": PoultryDetector(),
            "face": FaceDetector(),
        }

    def create(self, detection_type: str) -> Detector:
        detector = self._detectors.get(detection_type)
        if not detector:
            raise ValueError(f"Detector for type '{detection_type}' is not registered in the system.")
        return detector

# Globalna instancja fabryki do użycia w aplikacji
detector_factory = DetectorFactory()