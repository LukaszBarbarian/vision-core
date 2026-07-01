from __future__ import annotations

import asyncio
import time

import cv2

from infrastructure.rendering.frame_render import FrameRenderer


class LiveStreamHandler:
    def __init__(
        self,
        topic: str,
        frame_bus,
        detector_provider=None,
        visible_types: list[str] | set[str] | None = None,
        camera_config=None,
        audio_overlay_state=None,
        jpeg_quality: int = 75,
        max_fps: float = 12.0,
        max_width: int | None = None,
    ):
        self.topic = topic
        self.frame_bus = frame_bus

        self.renderer = FrameRenderer(
            detector_provider,
            audio_overlay_state,
        )

        self.visible_types = set(visible_types or [])
        self.camera_config = camera_config

        self.jpeg_quality = max(25, min(95, int(jpeg_quality)))
        self.max_fps = max(0.5, float(max_fps))
        self.max_width = int(max_width) if max_width else None

    async def generate_mjpeg_stream(self):
        queue = self._subscribe()

        print(
            f"[LiveStreamHandler] subscribed topic='{self.topic}' "
            f"visible_types={sorted(self.visible_types)} "
            f"quality={self.jpeg_quality} fps={self.max_fps} "
            f"width={self.max_width}"
        )

        frame_interval = 1.0 / self.max_fps
        last_emit_at = 0.0

        try:
            while True:
                frame_ctx = await queue.get()
                frame_ctx = self._drain_to_latest_frame(queue, frame_ctx)

                now = time.monotonic()
                elapsed = now - last_emit_at

                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
                    frame_ctx = self._drain_to_latest_frame(queue, frame_ctx)

                chunk = await asyncio.to_thread(
                    self._render_and_encode,
                    frame_ctx,
                )

                if chunk is None:
                    continue

                last_emit_at = time.monotonic()

                yield chunk

                await asyncio.sleep(0)

        finally:
            self.frame_bus.unsubscribe(self.topic, queue)
            print(f"[LiveStreamHandler] unsubscribed topic='{self.topic}'")

    def _subscribe(self):
        try:
            return self.frame_bus.subscribe(self.topic, maxsize=1)
        except TypeError:
            return self.frame_bus.subscribe(self.topic)

    def _drain_to_latest_frame(self, queue, current_frame_ctx):
        latest = current_frame_ctx

        while True:
            try:
                latest = queue.get_nowait()
            except asyncio.QueueEmpty:
                return latest
            except Exception:
                return latest

    def _render_and_encode(self, frame_ctx):
        image = self.renderer.render(
            frame_ctx,
            self.visible_types,
            config=self.camera_config,
        )

        image = self._resize_for_stream(image)

        ret, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )

        if not ret:
            return None

        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    def _resize_for_stream(self, image):
        if self.max_width is None:
            return image

        height, width = image.shape[:2]

        if width <= self.max_width:
            return image

        scale = self.max_width / float(width)
        target_width = int(width * scale)
        target_height = int(height * scale)

        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )