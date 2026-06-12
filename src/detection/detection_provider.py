from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from detection.model_provider import DetectionModelProvider


class DetectorProvider:
    def __init__(self, model_provider: DetectionModelProvider):
        self._model_provider = model_provider
        self._detectors: dict[DetectionType, BaseDetector] = {}

    def get_detector(
        self,
        detection_type: str | DetectionType,
    ) -> BaseDetector:
        d_type = self._parse_type(detection_type)

        if d_type not in self._detectors:
            detector_cls = self._resolve_detector_class(d_type)
            self._detectors[d_type] = detector_cls(self._model_provider)

        return self._detectors[d_type]

    def get_color(
        self,
        detection_type: str | DetectionType,
    ) -> tuple[int, int, int]:
        d_type = self._parse_type(detection_type)
        detector_cls = self._resolve_detector_class(d_type)
        return detector_cls.get_color()

    def get_display_name(
        self,
        detection_type: str | DetectionType,
    ) -> str:
        d_type = self._parse_type(detection_type)
        detector_cls = self._resolve_detector_class(d_type)
        return detector_cls.get_display_name()

    def _parse_type(
        self,
        detection_type: str | DetectionType,
    ) -> DetectionType:
        try:
            return DetectionType(detection_type)
        except ValueError:
            raise ValueError(
                f"Unknown detection type configured: {detection_type}"
            )

    def _resolve_detector_class(
        self,
        d_type: DetectionType,
    ) -> type[BaseDetector]:
        if d_type == DetectionType.CAR:
            from detection.detectors.car_detector import CarDetector
            return CarDetector

        if d_type == DetectionType.LICENSE_PLATE:
            from detection.detectors.anpr_detector import AnprDetector
            return AnprDetector

        if d_type == DetectionType.CHICKEN:
            from detection.detectors.chicken_detector import ChickenDetector
            return ChickenDetector

        if d_type == DetectionType.FACE:
            from detection.detectors.face_detector import FaceDetector
            return FaceDetector

        if d_type == DetectionType.PERSON:
            from detection.detectors.person_detector import PersonDetector
            return PersonDetector

        raise ValueError(f"No implementation for: {d_type}")
