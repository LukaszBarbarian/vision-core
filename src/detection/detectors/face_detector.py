import cv2
import numpy as np
import face_recognition

from detection.detection_type import DetectionType
from detection.detectors.base import BaseDetector
from domain.detection_result import DetectionResult


class FaceDetector(BaseDetector):
    def __init__(self, model_provider=None):
        # face_recognition is used directly. There is no YOLO model for FACE in
        # DetectionModelProvider, so this detector must not call get_model(FACE).
        pass

    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (255, 0, 255)

    @classmethod
    def get_display_name(cls) -> str:
        return "Twarz"

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict,
    ) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        person_boxes = context.get("detected_persons")

        detections: list[DetectionResult] = []

        if person_boxes is not None and len(person_boxes) > 0:
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = rgb_frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                face_locations = face_recognition.face_locations(person_crop)

                for top, right, bottom, left in face_locations:
                    detections.append(
                        DetectionResult(
                            detection_type=DetectionType.FACE.value,
                            x1=x1 + left,
                            y1=y1 + top,
                            x2=x1 + right,
                            y2=y1 + bottom,
                            confidence=1.0,
                            label="Twarz",
                        )
                    )
        else:
            face_locations = face_recognition.face_locations(rgb_frame)

            for top, right, bottom, left in face_locations:
                detections.append(
                    DetectionResult(
                        detection_type=DetectionType.FACE.value,
                        x1=left,
                        y1=top,
                        x2=right,
                        y2=bottom,
                        confidence=1.0,
                        label="Twarz",
                    )
                )

        return detections
