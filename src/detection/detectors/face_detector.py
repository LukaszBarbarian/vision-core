import cv2
import numpy as np
import face_recognition
from detection.detectors.base import BaseDetector
from domain.detection_result import DetectionResult

class FaceDetector(BaseDetector):
    def execute(self, frame_data: np.ndarray, context: dict) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        person_boxes = context.get("detected_persons")
        
        detections = []

        # SCENARIUSZ A: Mamy wyniki z PersonDetector
        if person_boxes is not None and len(person_boxes) > 0:
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Wycinek osoby
                person_crop = rgb_frame[y1:y2, x1:x2]
                if person_crop.size == 0: continue
                
                # Szukaj twarzy tylko w wycinku osoby
                face_locs = face_recognition.face_locations(person_crop)
                for (top, right, bottom, left) in face_locs:
                    # Przelicz współrzędne z powrotem na cały obraz
                    detections.append(DetectionResult(
                        detection_type="face_detection",
                        x1=x1 + left, y1=y1 + top,
                        x2=x1 + right, y2=y1 + bottom,
                        confidence=1.0, label="Face"
                    ))
        
        # SCENARIUSZ B: Brak danych o osobach (działa jak wcześniej - tryb awaryjny)
        else:
            face_locs = face_recognition.face_locations(rgb_frame)
            for (top, right, bottom, left) in face_locs:
                detections.append(DetectionResult(
                    detection_type="face_detection",
                    x1=left, y1=top, x2=right, y2=bottom,
                    confidence=1.0, label="Face"
                ))

        return detections