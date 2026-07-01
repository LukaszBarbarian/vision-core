
from dataclasses import dataclass
from typing import Callable

from infrastructure.ai_detection.detectors.base import BaseDetector


@dataclass(frozen=True, slots=True)
class DetectorDefinition:
    detector_cls: type[BaseDetector]
    factory: Callable[[], BaseDetector]