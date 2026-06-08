import asyncio
from detection.detection_provider import DetectorProvider
from detection.detection_worker import DetectionWorker

class DetectionManager:
    def __init__(self, frame_bus, event_bus, model_provider, settings):
        self.frame_bus = frame_bus
        self.event_bus = event_bus
        self.settings = settings
        
        # Tworzymy centralnego dostawcę detektorów na bazie dostawcy modeli
        self.detector_provider = DetectorProvider(model_provider)
        self.tasks = []

    async def start(self):
        for camera_id, config in self.settings.registry.items():
            if not config.active or not config.detections:
                continue

            worker = DetectionWorker(
                camera_id=camera_id,
                frame_bus=self.frame_bus,
                event_bus=self.event_bus,
                detector_provider=self.detector_provider,
                specs=config.detections
            )

            task = asyncio.create_task(
                worker.run(), 
                name=f"detection-worker-{camera_id}"
            )
            self.tasks.append(task)