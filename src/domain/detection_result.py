# domain/detection_result.py

from dataclasses import dataclass


@dataclass
class DetectionResult:
    detection_type: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    label: str | None = None