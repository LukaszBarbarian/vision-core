from domain.models.detection_result import DetectionResult
from domain.models.tracked_detection_result import TrackedDetectionResult


class MockTracker:
    def update(self, detections: list[DetectionResult]) -> list[TrackedDetectionResult]:
        # Tu w przyszłości będzie logika ByteTrackera
        # Na razie konwertujemy surowe wyniki na wersję z track_id
        return [TrackedDetectionResult(**d.__dict__, track_id=0) for d in detections]