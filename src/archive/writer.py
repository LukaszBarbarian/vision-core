# archive/writer.py
import asyncio
import os
import json  # 🎯 Potrzebne do spakowania listy ramek do tekstu
from datetime import datetime

class ArchiveWriter:
    def __init__(self, event_bus, log_dir="archive_logs"):
        self.event_bus = event_bus
        self.log_dir = log_dir
        self.running = True
        os.makedirs(self.log_dir, exist_ok=True)

    async def run(self):
        """Niezależna pętla zapisująca dane CSV wraz ze współrzędnymi ramek."""
        queue = self.event_bus.subscribe("detections")
        print("💾 [ArchiveWriter] Uruchomiono zapis CSV z pełnymi współrzędnymi ramek...")
        
        try:
            while self.running:
                event = await queue.get()
                count = event.get("count", 0)
                
                if count <= 0:
                    continue
                
                camera_id = event.get("camera_id")
                det_type = event.get("detection_type")
                timestamp = event.get("timestamp")
                boxes = event.get("boxes", [])  # 🎯 Odbieramy ramki
                
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                
                # Konwertujemy listę ramek na bezpieczny dla CSV string (w cudzysłowie)
                boxes_json = json.dumps(boxes)
                
                filename = os.path.join(self.log_dir, f"{camera_id}_{det_type}.csv")
                file_exists = os.path.exists(filename)
                
                with open(filename, "a", encoding="utf-8") as f:
                    if not file_exists:
                        # 🎯 Dodajemy nową kolumnę do nagłówka
                        f.write("timestamp,count,bounding_boxes\n")
                    
                    # Zapisujemy wiersz. boxes_json opakowujemy w cudzysłów, na wypadek gdyby zawierał przecinki
                    f.write(f'{readable_time},{count},"{boxes_json}"\n')
                    
        except asyncio.CancelledError:
            print("💾 [ArchiveWriter] Pomyślnie zatrzymano zapisywacz plików.")
        finally:
            self.event_bus.unsubscribe("detections", queue)