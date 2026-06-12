import numpy as np

from domain.detection_result import DetectionResult


class BaseDetector:
    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (0, 255, 0)

    @classmethod
    def get_display_name(cls) -> str:
        return cls.__name__

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict,
    ) -> list[DetectionResult]:
        raise NotImplementedError
