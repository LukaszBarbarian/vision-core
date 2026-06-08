import cv2
import numpy as np

from detection.detectors.base import BaseDetector
from detection.model_provider import ModelProvider


class CarDetector(BaseDetector):
    def __init__(self, model_provider: ModelProvider):
        self._model = model_provider.get_model("yolov8n.pt")
        self._vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def execute(self, frame_data: np.ndarray) -> tuple[list, np.ndarray]:
        if frame_data is None or frame_data.size == 0:
            print("⚠️ [CarDetector] Otrzymano PUSTĄ klatkę!")
            return [], frame_data

        # 📊 TA LINIA POWIE NAM WSZYSTKO:
        print(f"📊 [DIAGNOSTYKA MATRYCY] Shape: {frame_data.shape} | Typ: {frame_data.dtype} | Min piksel: {frame_data.min()} | Max piksel: {frame_data.max()} | Średnia: {frame_data.mean():.2f}")

        results = self._model(frame_data, classes=self._vehicle_classes, conf=0.25, verbose=False)        
        annotated_frame = frame_data.copy()
        
        # 3. Logujemy do konsoli każdą, nawet najgłupszą detekcję modelu
        if results and len(results[0].boxes) > 0:
            print(f"🔍 [YOLO RAW] Na klatce znaleziono {len(results[0].boxes)} obiektów:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Pobieramy nazwę klasy z modelu (np. 'car', 'person', 'bench')
                class_name = self._model.names.get(cls_id, f"ID-{cls_id}")
                print(f"   -> Wykryto: '{class_name}' (Klasa ID: {cls_id}) z pewnością: {conf:.2f}")
                
                # Jeśli to jedna z naszych klas pojazdów, rysujemy ramkę
                if cls_id in self._vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("🔍 [YOLO RAW] Model nie wykrył absolutnie nic, nawet z progiem 10%.")

        return results, annotated_frame