import numpy as np

from domain.detection_result import DetectionResult


class BaseDetector:

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict
    ) -> list[DetectionResult]:
        raise NotImplementedError