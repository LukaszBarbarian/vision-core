import asyncio
import os
import json
from datetime import datetime

class ArchiveWriter:
    def __init__(self, event_bus, log_dir="archive_logs"):
        self.event_bus = event_bus
        self.log_dir = log_dir
        self.running = True
        os.makedirs(self.log_dir, exist_ok=True)

    async def run(self):
        queue = self.event_bus.subscribe("detections")
        print("💾 [ArchiveWriter] Uruchomiono zapis...")
        
        try:
            while self.running:
                event = await queue.get()
                count = event.get("count", 0)
                
                # Zmieniono z event.get("boxes") na event.get("bounding_boxes")
                boxes = event.get("bounding_boxes", [])
                
                camera_id = event.get("camera_id")
                det_type = event.get("detection_type")
                timestamp = event.get("timestamp")
                
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                boxes_json = json.dumps(boxes)
                
                filename = os.path.join(self.log_dir, f"{camera_id}_{det_type}.csv")
                file_exists = os.path.exists(filename)
                
                with open(filename, "a", encoding="utf-8") as f:
                    if not file_exists:
                        f.write("timestamp,count,bounding_boxes\n")
                    f.write(f'{readable_time},{count},"{boxes_json.replace("\"", "\"\"")}"\n')
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.event_bus.unsubscribe("detections", queue)