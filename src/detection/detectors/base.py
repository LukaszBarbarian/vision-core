import numpy as np


class BaseDetector:
    def execute(self, frame_data: np.ndarray) -> list:
        raise NotImplementedError