import cv2
import numpy as np

from detection.detectors.base import BaseDetector
from detection.model_provider import ModelProvider


class ChickenDetector(BaseDetector):
    def __init__(self, model_provider: ModelProvider):
        self._model = model_provider.get_model("yolov8n.pt")
        self._chicken_classes = [14] 

    def execute(self, frame_data: np.ndarray) -> tuple[list, np.ndarray]:
        if frame_data is None or frame_data.size == 0:
            print("⚠️ [ChickenDetector] Otrzymano PUSTĄ klatkę!")
            return [], frame_data


        results = self._model(
                frame_data, 
                classes=self._chicken_classes, 
                conf=0.15, 
                imgsz=1280, 
                verbose=False)
        
        annotated_frame = frame_data.copy()
        
        if results and len(results[0].boxes) > 0:
            print(f"🐓 [CHICKEN DETECTOR] Wykryto ruch na grzędzie! Liczba obiektów: {len(results[0].boxes)}")
            
            for box in results[0].boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Zamiast pisać 'bird', robimy przyjacielski rebranding na 'Kurka'
                display_label = f"Kurka: {conf:.2f}"
                
                # Rysujemy ładną żółto-pomarańczową ramkę (kolorystycznie pod kury)
                # (0, 165, 255) to kolor pomarańczowy w formacie BGR
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated_frame, display_label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            # Logowanie zakomentowane, żeby nie śmiecić w konsoli, gdy kury śpią
            pass

        return results, annotated_frame