from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain.models.detection_type import DetectionType
from domain.models.detector_definiton import DetectorDefinition
from domain.models.face_detection_options import FaceDetectorOptions
from infrastructure.ai_detection.detectors.anpr_detector import AnprDetector
from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.detectors.car_detector import CarDetector
from infrastructure.ai_detection.detectors.chicken_detector import ChickenDetector
from infrastructure.ai_detection.detectors.face_detector import FaceDetector
from infrastructure.ai_detection.detectors.person_detector import PersonDetector
from infrastructure.ai_detection.face_identity_store import FaceIdentityStore
from infrastructure.ai_detection.providers.model_provider import DetectionModelProvider





class DetectorProvider:
    def __init__(
        self,
        model_provider: DetectionModelProvider,
        face_identity_store: FaceIdentityStore | None = None,
        face_detector_options: FaceDetectorOptions | None = None,
    ):
        self._model_provider = model_provider
        self._face_identity_store = face_identity_store
        self._face_detector_options = face_detector_options or FaceDetectorOptions()

        self._detectors: dict[DetectionType, BaseDetector] = {}
        self._definitions = self._build_definitions()

    def get_detector(
        self,
        detection_type: str | DetectionType,
    ) -> BaseDetector:
        d_type = self._parse_type(detection_type)

        if d_type not in self._detectors:
            self._detectors[d_type] = self._create_detector(d_type)

        return self._detectors[d_type]

    def get_color(
        self,
        detection_type: str | DetectionType,
    ) -> tuple[int, int, int]:
        d_type = self._parse_type(detection_type)
        definition = self._get_definition(d_type)

        return definition.detector_cls.get_color()

    def get_display_name(
        self,
        detection_type: str | DetectionType,
    ) -> str:
        d_type = self._parse_type(detection_type)
        definition = self._get_definition(d_type)

        return definition.detector_cls.get_display_name()

    def set_face_identity_store(
        self,
        face_identity_store: FaceIdentityStore | None,
    ) -> None:
        """
        Pozwala podmienić store, jeśli provider powstał wcześniej niż AppOrchestrator
        w pełni zainicjalizował identity store.

        Po zmianie czyścimy tylko FaceDetector z cache, bo stary egzemplarz miałby
        starą referencję do identity_store.
        """
        self._face_identity_store = face_identity_store
        self._definitions = self._build_definitions()
        self._detectors.pop(DetectionType.FACE, None)

    def _create_detector(
        self,
        d_type: DetectionType,
    ) -> BaseDetector:
        definition = self._get_definition(d_type)
        return definition.factory()

    def _get_definition(
        self,
        d_type: DetectionType,
    ) -> DetectorDefinition:
        definition = self._definitions.get(d_type)

        if definition is None:
            raise ValueError(f"No implementation for: {d_type}")

        return definition

    def _build_definitions(self) -> dict[DetectionType, DetectorDefinition]:
        return {
            DetectionType.CAR: DetectorDefinition(
                detector_cls=CarDetector,
                factory=lambda: CarDetector(
                    self._model_provider,
                ),
            ),
            DetectionType.LICENSE_PLATE: DetectorDefinition(
                detector_cls=AnprDetector,
                factory=lambda: AnprDetector(
                    self._model_provider,
                ),
            ),
            DetectionType.CHICKEN: DetectorDefinition(
                detector_cls=ChickenDetector,
                factory=lambda: ChickenDetector(
                    self._model_provider,
                ),
            ),
            DetectionType.PERSON: DetectorDefinition(
                detector_cls=PersonDetector,
                factory=lambda: PersonDetector(
                    self._model_provider,
                ),
            ),
            DetectionType.FACE: DetectorDefinition(
                detector_cls=FaceDetector,
                factory=self._create_face_detector,
            ),
        }

    def _create_face_detector(self) -> FaceDetector:
        options = self._face_detector_options

        return FaceDetector(
            model_provider=None,
            identity_store=self._face_identity_store,
            require_person=options.require_person,
            fallback_to_full_frame=options.fallback_to_full_frame,
            use_upper_body_roi=options.use_upper_body_roi,
            model=options.model,
            number_of_times_to_upsample=options.number_of_times_to_upsample,
            crop_upscale=options.crop_upscale,
            full_frame_scale=options.full_frame_scale,
            min_face_size=options.min_face_size,
            num_jitters=options.num_jitters,
            debug=options.debug,
        )

    def _parse_type(
        self,
        detection_type: str | DetectionType,
    ) -> DetectionType:
        if isinstance(detection_type, DetectionType):
            return detection_type

        try:
            return DetectionType(str(detection_type))
        except ValueError:
            raise ValueError(
                f"Unknown detection type configured: {detection_type}"
            )