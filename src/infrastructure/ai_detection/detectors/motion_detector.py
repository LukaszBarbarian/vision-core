# infrastructure/ai_detection/filters/motion_filter.py
import cv2
import numpy as np

class MotionFilter:
    def __init__(self, threshold: int = 25, min_area_ratio: float = 0.002, learning_rate: float = 0.02):
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.lr = learning_rate
        self._backgrounds: dict[str, np.ndarray] = {}

    def has_significant_motion(self, camera_id: str, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if camera_id not in self._backgrounds:
            self._backgrounds[camera_id] = gray.astype(np.float32)
            return True # pierwsza klatka – traktuj jako ruch

        cv2.accumulateWeighted(gray, self._backgrounds[camera_id], self.lr)
        background = cv2.convertScaleAbs(self._backgrounds[camera_id])

        delta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = int(frame.shape[0] * frame.shape[1] * self.min_area_ratio)
        return any(cv2.contourArea(c) >= min_area for c in contours)

    def reset(self, camera_id: str):
        self._backgrounds.pop(camera_id, None)