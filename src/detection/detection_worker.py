import time
import asyncio
import traceback

import copy as py_copy

from detection.detection_provider import DetectorProvider

class DetectionWorker:
    def __init__(self, camera_id, frame_bus, event_bus, detector_provider: DetectorProvider, specs):
        self.camera_id = camera_id
        self.frame_bus = frame_bus
        self.event_bus = event_bus
        self.detector_provider = detector_provider
        self.specs = specs
        self.running = False


    async def run(self):
        self.running = True
        queue = self.frame_bus.subscribe(self.camera_id)
        last_execution_times = {spec.detection_type: 0.0 for spec in self.specs}
        
        print(f"🚀 [Worker-{self.camera_id}] Uruchomiono pętlę detekcji dla: {[s.detection_type for s in self.specs]}")
        
        # 🕵️‍♂️ LINIA DIAGNOSTYCZNA 1: Sprawdzamy stan flagi
        print(f"🔍 [DEBUG-{self.camera_id}] Czy pętla ma pozwolenie na start? self.running = {self.running}")
        
        try:
            while self.running:
                # 🕵️‍♂️ LINIA DIAGNOSTYCZNA 2: Sprawdzamy czy utknął na pobieraniu
                print(f"📥 [DEBUG-{self.camera_id}] Oczekuję na klatkę z szyny '{self.camera_id}'...")
                
                frame = await queue.get()
                
                # 🕵️‍♂️ LINIA DIAGNOSTYCZNA 3: Sprawdzamy czy klatka przeszła
                print(f"📦 [DEBUG-{self.camera_id}] Pomyślnie odebrano klatkę! ID: {frame.frame_id}")
                
                now = time.time()
                for spec in self.specs:
                    interval = 1.0 / spec.ai_fps
                    if now - last_execution_times[spec.detection_type] >= interval:
                        last_execution_times[spec.detection_type] = now
                        
                        print(f"🧠 [Worker-{self.camera_id}] Wywołuję detektor: {spec.detection_type}")
                        
                        detector = self.detector_provider.get_detector(spec.detection_type)
                        results, annotated_frame = await asyncio.to_thread(detector.execute, frame.frame_data)
                        
                        detected_count = len(results[0].boxes) if results else 0
                        
                        ai_frame_ctx = py_copy.copy(frame)
                        ai_frame_ctx.frame_data = annotated_frame
                        await self.frame_bus.publish(f"{self.camera_id}_ai", ai_frame_ctx)
                        
                        await self.event_bus.publish("detections", {
                            "camera_id": self.camera_id,
                            "detection_type": spec.detection_type,
                            "count": detected_count,
                            "timestamp": now
                        })
                        
        except Exception as critical_err:
            print(f"💥 [Worker-{self.camera_id}] KRYTYCZNY błąd w pętli: {critical_err}")