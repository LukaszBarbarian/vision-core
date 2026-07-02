import cv2
import numpy as np


class MotionFilter:
    def __init__(
        self,
        threshold: int = 25,
        min_area_ratio: float = 0.003,
        learning_rate: float = 0.02,
        analysis_width: int = 480,
        debug: bool = False,
    ):
        self.threshold = int(threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.lr = float(learning_rate)
        self.analysis_width = int(analysis_width)
        self.debug = bool(debug)

        self._backgrounds: dict[str, np.ndarray] = {}

    def has_significant_motion(
        self,
        camera_id: str,
        frame: np.ndarray,
    ) -> bool:
        if frame is None or frame.size == 0:
            return False

        original_h, original_w = frame.shape[:2]

        if self.debug:
            size_mb = frame.nbytes / (1024 * 1024)
            print(
                f"[MotionFilter] {camera_id}: "
                f"{original_w}x{original_h}, {size_mb:.2f} MB"
            )

        small = self._resize_for_analysis(frame)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if camera_id not in self._backgrounds:
            self._backgrounds[camera_id] = gray.astype(np.float32)
            return True

        cv2.accumulateWeighted(
            gray,
            self._backgrounds[camera_id],
            self.lr,
        )

        background = cv2.convertScaleAbs(self._backgrounds[camera_id])
        delta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(
            delta,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = gray.shape[:2]
        min_area = int(h * w * self.min_area_ratio)

        return any(
            cv2.contourArea(contour) >= min_area
            for contour in contours
        )

    def reset(self, camera_id: str):
        self._backgrounds.pop(camera_id, None)

    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]

        if width <= self.analysis_width:
            return frame

        scale = self.analysis_width / float(width)
        target_width = self.analysis_width
        target_height = max(1, int(height * scale))

        return cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )