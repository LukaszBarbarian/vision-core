import asyncio
import cv2

from rendering.frame_render import FrameRenderer




class LiveStreamHandler:

    def __init__(
        self,
        camera_id: str,
        frame_bus,
        visible_types: list[str] | None = None
    ):
        self.camera_id = f"{camera_id}_ai"
        self.frame_bus = frame_bus

        self.renderer = FrameRenderer()

        self.visible_types = (
            set(visible_types)
            if visible_types
            else set()
        )

    async def generate_mjpeg_stream(self):

        queue = self.frame_bus.subscribe(
            self.camera_id
        )

        try:

            while True:

                frame_ctx = await queue.get()

                image = self.renderer.render(
                    frame_ctx,
                    self.visible_types
                )

                ret, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        75
                    ]
                )

                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

                await asyncio.sleep(0.03)

        finally:

            self.frame_bus.unsubscribe(
                self.camera_id,
                queue
            )