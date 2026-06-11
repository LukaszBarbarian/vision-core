import time
import asyncio
import traceback
import copy
from cameras.frame_context import FrameContext
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

        try:
            while self.running:
                frame_ctx = await queue.get()
                all_detections = []
                now = time.time()

                for spec in self.specs:
                    interval = 1.0 / spec.ai_fps
                    if (now - last_execution_times[spec.detection_type]) < interval:
                        continue
                    last_execution_times[spec.detection_type] = now

                    detector = self.detector_provider.get_detector(spec.detection_type)
                    detections = await asyncio.to_thread(detector.execute, frame_ctx.frame_data, {})
                    
                    # Publikuj wynik dla TEGO KONKRETNEGO detektora
                    if detections:
                        await self.event_bus.publish("detections", {
                            "camera_id": self.camera_id,
                            "detection_type": spec.detection_type,
                            "count": len(detections),
                            "timestamp": now,
                            "bounding_boxes": [
                                {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, 
                                 "confidence": d.confidence, "label": d.label, "type": d.detection_type}
                                for d in detections
                            ]
                        })
                    
                    all_detections.extend(detections)

                # Publikacja pełnej klatki ze wszystkimi wykrytymi obiektami dla Renderer'a
                new_ctx = FrameContext(
                    camera_id=frame_ctx.camera_id,
                    frame_id=frame_ctx.frame_id,
                    frame_data=frame_ctx.frame_data.copy(),
                    detections=all_detections
                )
                await self.frame_bus.publish(f"{self.camera_id}_ai", new_ctx)

        except Exception as err:
            traceback.print_exc()