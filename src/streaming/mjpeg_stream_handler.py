import cv2
import asyncio

class LiveStreamHandler:
    def __init__(self, camera_id: str, frame_bus):
        self.camera_id = camera_id
        self.frame_bus = frame_bus

    async def generate_mjpeg_stream(self):
        queue = self.frame_bus.subscribe(self.camera_id)
        
        try:
            while True:
                # 1. Czekamy na klatkę z szyny
                frame_ctx = await queue.get()
                
                # 2. Kompresujemy do JPEG
                ret, buffer = cv2.imencode(".jpg", frame_ctx.frame_data, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ret:
                    continue

                # 3. Wysyłamy do przeglądarki
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                
                # 4. CRITICAL: Oddajemy kontrolę do loopa, żeby Uvicorn faktycznie WYPCHNĄŁ te bajty do sieci
                await asyncio.sleep(0.03) # ~30 FPS, absolutnie wystarczy do płynnego podglądu
                
        finally:
            self.frame_bus.unsubscribe(self.camera_id, queue)