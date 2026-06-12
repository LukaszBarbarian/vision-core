import asyncio
import time
from datetime import datetime

import av

from cameras.frame_context import FrameContext


class CameraDriver:
    def __init__(self, config, frame_bus):
        self.config = config
        self.frame_bus = frame_bus
        self.running = False

    async def run(self):
        self.running = True
        camera_id = self.config.camera.id

        while self.running:
            container = None

            try:
                print(f"[{camera_id}] Opening RTSP stream...")

                container = av.open(
                    self.config.camera.url,
                    options={
                        "rtsp_transport": "tcp",
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                    }
                )

                stream = container.streams.video[0]

                stream_session_id = (
                    f"{camera_id}-"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )

                print(
                    f"[{camera_id}] Stream opened. "
                    f"session={stream_session_id}"
                )

                frame_id = 0

                for frame in container.decode(stream):
                    if not self.running:
                        break

                    img = frame.to_ndarray(format="bgr24")

                    ctx = FrameContext(
                        camera_id=camera_id,
                        frame_id=frame_id,
                        frame_data=img,
                        timestamp=time.time(),
                        stream_session_id=stream_session_id,
                    )

                    await self.frame_bus.publish(camera_id, ctx)

                    frame_id += 1
                    await asyncio.sleep(0)

            except Exception as ex:
                print(f"[{camera_id}] Driver error: {ex}")
                print(f"[{camera_id}] Reconnecting in 2s...")
                await asyncio.sleep(2)

            finally:
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass

            await asyncio.sleep(0.5)

        print(f"[{camera_id}] Driver stopped.")

    def stop(self):
        self.running = False
