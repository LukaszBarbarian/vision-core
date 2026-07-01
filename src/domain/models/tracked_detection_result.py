from dataclasses import dataclass
from domain.models.detection_result import DetectionResult


@dataclass
class TrackedDetectionResult(DetectionResult):
    track_id: int = -1