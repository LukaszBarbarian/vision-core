import asyncio
import cv2 # Użyjemy tu PyAV w finalnej wersji
from src.streaming.frame_context import FrameContext
from src.streaming.stream_handler import frame_bus


class CameraWorker:
    def __init__(self, camera_config):
        self.config = camera_config
        self.is_running = False

    async def run(self):
        self.is_running = True
        # Tutaj PyAV zamiast cv2.VideoCapture
        cap = cv2.VideoCapture(self.config.url) 
        
        frame_id = 0
        while self.is_running:
            success, frame = cap.read()
            if success:
                frame_id += 1
                ctx = FrameContext(self.config.id, frame_id, frame)
                # Wrzucamy na szynę - to jest jedyne miejsce gdzie "pobieramy" obraz
                await frame_bus.publish(self.config.id, ctx)
            
            # Kontrola FPS (np. 30 FPS)
            await asyncio.sleep(1/30)