import cv2
import numpy as np
import easyocr

from detection.detectors.base import BaseDetector
from detection.model_provider import ModelProvider

# 🎭 Klasy imitujące YOLO, żeby generyczny worker poprawnie zliczał wyniki
class MockBox:
    def __init__(self, cls_id: int):
        self.cls = [cls_id]

class MockResult:
    def __init__(self, boxes_list: list):
        self.boxes = boxes_list


class AnprDetector(BaseDetector):
    def __init__(self, model_provider: ModelProvider):
        print("⏳ [AnprDetector] Inicjalizacja potoku hybrydowego (YOLOv8 + EasyOCR)...")
        
        # 1. Pobieramy ten sam model, którego używa CarDetector
        self._model = model_provider.get_model("yolov8n.pt")
        self._vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # 2. Odpalamy OCR dla polskich i angielskich znaków
        self._reader = easyocr.Reader(['pl', 'en'], gpu=True)

    def execute(self, frame_data: np.ndarray) -> tuple[list, np.ndarray]:
        if frame_data is None or frame_data.size == 0:
            return [], frame_data

        annotated_frame = frame_data.copy()
        detected_plates_count = 0

        # 🔥 KROK 1: Lokalizujemy pojazdy za pomocą YOLO (wyszukujemy tylko strefy z autami)
        yolo_results = self._model(frame_data, classes=self._vehicle_classes, conf=0.25, verbose=False)        
        
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                # Pobieramy współrzędne wykrytego samochodu na pełnym kadrze 2K
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Zabezpieczenie, żeby indeksy nie wyszły poza matrycę np.ndarray
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_data.shape[1], x2), min(frame_data.shape[0], y2)

                # 🔥 KROK 2: Wycinamy prostokąt z SAMYM samochodem (Crop)
                # Dzięki temu odcinamy resztę kadru, w tym nieszczęsną datę i godzinę!
                car_crop = frame_data[y1:y2, x1:x2]
                if car_crop.size == 0:
                    continue

                # OPCJONALNIE: Rysujemy bardzo cienką zieloną linię wokół auta, żeby widzieć strefę analizy
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # 🔥 KROK 3: Skupiamy wzrok OCR wyłącznie na wyciętym samochodzie
                ocr_results = self._reader.readtext(car_crop)

                for (bbox, text, prob) in ocr_results:
                    # Czyszczenie i standaryzacja tekstu
                    clean_text = text.upper().replace(" ", "").replace("-", "").strip()
                    
                    # Heurystyka: tablica ma od 4 do 8 znaków i musi zawierać przynajmniej jedną cyfrę
                    if 4 <= len(clean_text) <= 8 and any(char.isdigit() for char in clean_text):
                        detected_plates_count += 1
                        
                        # 🔥 KROK 4: Mapowanie współrzędnych wstecznych (z wycinka na pełny kadr 2K)
                        # Punkty w 'bbox' są relatywne do lewego górnego rogu wycinka (który w 2K ma współrzędne x1, y1)
                        p0 = (int(bbox[0][0] + x1), int(bbox[0][1] + y1))
                        p1 = (int(bbox[1][0] + x1), int(bbox[1][1] + y1))
                        p2 = (int(bbox[2][0] + x1), int(bbox[2][1] + y1))
                        p3 = (int(bbox[3][0] + x1), int(bbox[3][1] + y1))
                        
                        # Rysujemy grubą, drapieżną czerwoną ramkę dokładnie wokół tablicy na głównym obrazie
                        pts = np.array([p0, p1, p2, p3], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], True, (0, 0, 255), 3)
                        
                        # Wypisujemy rejestrację bezpośrednio nad znalezioną tablicą
                        cv2.putText(annotated_frame, f"REJ: {clean_text} ({int(prob*100)}%)", (p0[0], p0[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        print(f"🎯 [ANPR SUCCESS] Kamera wykryła tablicę: {clean_text} (Pewność: {prob:.2f})")

        # Pakujemy wynik w atrapę struktury YOLO dla zachowania kompatybilności z Workerem
        mocked_boxes = [MockBox(cls_id=99) for _ in range(detected_plates_count)]
        mocked_results_list = [MockResult(boxes_list=mocked_boxes)]

        return mocked_results_list, annotated_frame