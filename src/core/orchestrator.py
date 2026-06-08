import asyncio
from cameras.camera_manager import CameraManager
from detection.detection_manager import DetectionManager
from detection.model_provider import ModelProvider
from messaging.event_bus import EventBus
from messaging.frame_bus import FrameBus


class AppOrchestrator:

    def __init__(self, settings):
        self.settings = settings

        self.frame_bus = FrameBus()
        self.event_bus = EventBus()

        self.model_provider = ModelProvider()

        self.camera_manager = CameraManager(
            frame_bus=self.frame_bus,
            event_bus=self.event_bus,
            settings=settings
        )

        self.detection_manager = DetectionManager(
            frame_bus=self.frame_bus,
            event_bus=self.event_bus,
            model_provider=self.model_provider,
            settings=settings
        )

        self.running = False

    async def start(self):
        if self.running:
            return
        self.running = True

        print("[Orchestrator] Starting systems...")
        await self.camera_manager.start_all()
        await self.detection_manager.start()

        # NOWOŚĆ: Odpalamy asynchronicznego konsumenta zdarzeń
        self._event_consumer_task = asyncio.create_task(self._listen_to_events())

        print("[Orchestrator] Startup complete.")

    async def stop(self):
        if not self.running:
            return
        print("[Orchestrator] Stopping systems...")
        
        # Anulujemy zadanie nasłuchiwania zdarzeń
        if hasattr(self, '_event_consumer_task'):
            self._event_consumer_task.cancel()

        await self.camera_manager.stop_all()
        await self.detection_manager.stop()
        self.running = False
        print("[Orchestrator] Shutdown complete.")

    async def _listen_to_events(self):
        """Nasłuchuje szyny zdarzeń i reaguje na wykryte obiekty."""
        # Subskrybujemy temat "detections", na który publikuje DetectionWorker
        queue = self.event_bus.subscribe("detections")
        
        try:
            while self.running:
                event = await queue.get()
                camera_id = event.get("camera_id")
                det_type = event.get("detection_type")
                count = event.get("count")

                if count > 0:
                    print(f"🚨 [AI ALERT] Kamera: {camera_id} | Wykryto: {det_type} | Ilość: {count}")
                else:
                    # Log techniczny, żebyś widział, że system żyje i analizuje
                    print(f"🔍 [AI Spokój] Kamera: {camera_id} | Brak obiektów typu {det_type}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.event_bus.unsubscribe("detections", queue)