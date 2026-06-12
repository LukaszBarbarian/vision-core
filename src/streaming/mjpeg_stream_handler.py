import asyncio

import cv2

from rendering.frame_render import FrameRenderer


class LiveStreamHandler:
    def __init__(
        self,
        topic: str,
        frame_bus,
        detector_provider=None,
        visible_types: list[str] | None = None,
    ):
        self.topic = topic
        self.frame_bus = frame_bus
        self.renderer = FrameRenderer(detector_provider)
        self.visible_types = set(visible_types or [])

    async def generate_mjpeg_stream(self):
        queue = self.frame_bus.subscribe(self.topic)
        print(
            f"[LiveStreamHandler] subscribed topic='{self.topic}' "
            f"visible_types={sorted(self.visible_types)}"
        )

        try:
            while True:
                frame_ctx = await queue.get()

                image = self.renderer.render(
                    frame_ctx,
                    self.visible_types,
                )

                ret, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75],
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
            self.frame_bus.unsubscribe(self.topic, queue)
            print(f"[LiveStreamHandler] unsubscribed topic='{self.topic}'")
