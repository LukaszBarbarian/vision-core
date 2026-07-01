from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.tracking.mock_tracker import MockTracker


class TrackingDetectorDecorator(BaseDetector):
    def __init__(self, wrapped: BaseDetector):
        self._wrapped = wrapped
        self._tracker = MockTracker()

    def execute(self, frame_data, context):
        results = self._wrapped.execute(frame_data, context)
        return self._tracker.update(results)

    def get_color(self) -> tuple[int, int, int]:
        # Delegacja: nie zgaduję koloru, pytam wnętrze
        return self._wrapped.get_color()

    def get_display_name(self) -> str:
        # Dekoracja: wzbogacam nazwę wnętrza
        return f"Tracked({self._wrapped.get_display_name()})"