import cv2
import numpy as np
from detection.detectors.base import BaseDetector
from detection.model_provider import ModelProvider


class PoultryDetector(BaseDetector):
    def __init__(self, model_provider: ModelProvider):
        self._model = model_provider.get_model("yolov8n.pt")
        self._poultry_classes = [16]  # W zbiorze COCO 16 to 'bird' (testowo jako drób)

    def execute(self, frame_data: np.ndarray) -> tuple[list, np.ndarray]:
        results = self._model(frame_data, classes=self._poultry_classes, verbose=False)
        annotated_frame = frame_data.copy()
        
        # ✔ FIX: Teraz również zwracamy narysowaną klatkę (np. z niebieskimi ramkami)
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Niebieska ramka
                cv2.putText(annotated_frame, f"Drob: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return results, annotated_frame