import asyncio
from cameras.camera_manager import CameraManager
from detection.detection_manager import DetectionManager
from detection.model_provider import DetectionModelProvider
from messaging.event_bus import EventBus
from messaging.frame_bus import FrameBus
from archive.writer import ArchiveWriter 


class AppOrchestrator:

    def __init__(self, settings):
        self.settings = settings

        self.frame_bus = FrameBus()
        self.event_bus = EventBus()

        self.model_provider = DetectionModelProvider()

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
        
        self.archive_writer = ArchiveWriter(event_bus=self.event_bus)

        self.running = False

    async def start(self):
        if self.running:
            return
        self.running = True

        print("[Orchestrator] Starting systems...")
        await self.camera_manager.start_all()
        await self.detection_manager.start()

        # Odpalamy asynchronicznego konsumenta zdarzeń (Twoje printy w konsoli)
        self._event_consumer_task = asyncio.create_task(self._listen_to_events())
        
        # 💾 NOWOŚĆ: Odpalamy niezależne zadanie zapisu do plików CSV
        self._archive_writer_task = asyncio.create_task(self.archive_writer.run())

        print("[Orchestrator] Startup complete.")

    async def stop(self):
        if not self.running:
            return
        print("[Orchestrator] Stopping systems...")
        
        # Anulujemy zadanie nasłuchiwania zdarzeń w konsoli
        if hasattr(self, '_event_consumer_task'):
            self._event_consumer_task.cancel()
            
        # 💾 NOWOŚĆ: Grzecznie anulujemy zadanie zapisu plików na dysku
        if hasattr(self, '_archive_writer_task'):
            self._archive_writer_task.cancel()

        await self.camera_manager.stop_all()
        await self.detection_manager.stop()
        self.running = False
        print("[Orchestrator] Shutdown complete.")

    async def _listen_to_events(self):
        """Nasłuchuje szyny zdarzeń i reaguje na wykryte obiekty."""
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
                    print(f"🔍 [AI Spokój] Kamera: {camera_id} | Brak obiektów typu {det_type}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.event_bus.unsubscribe("detections", queue)