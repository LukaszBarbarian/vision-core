from infrastructure.cameras.camera_driver import redact_credentials


def test_stream_credentials_are_hidden():
    message = "Immediate exit requested: 'rtsp://admin:secret%23@10.0.0.4:554/Streaming/Channels/101'"
    redacted = redact_credentials(message)

    assert "secret" not in redacted
    assert "admin" not in redacted
    assert "rtsp://***@10.0.0.4:554/Streaming/Channels/101" in redacted


def test_text_without_credentials_is_unchanged():
    assert redact_credentials("rtsp://10.0.0.4:554/live") == "rtsp://10.0.0.4:554/live"


def test_frames_from_reader_thread_reach_the_bus():
    import asyncio
    import threading
    from types import SimpleNamespace

    from domain.models.frame_context import FrameContext
    from infrastructure.cameras.camera_driver import CameraDriver

    class Bus:
        def __init__(self):
            self.frames = []

        async def publish(self, topic, frame):
            self.frames.append((topic, frame.frame_id))

    async def scenario():
        bus = Bus()
        config = SimpleNamespace(camera=SimpleNamespace(id="cam", url="rtsp://x", max_fps=10.0))
        driver = CameraDriver(config=config, frame_bus=bus)
        driver._loop = asyncio.get_running_loop()

        reader = threading.Thread(
            target=lambda: [driver._post_frame(FrameContext("cam", i, None)) for i in range(3)]
        )
        reader.start()
        reader.join()
        await asyncio.sleep(0.05)
        return bus.frames

    frames = asyncio.run(scenario())

    # latest-wins hand-over: at least the newest frame always arrives
    assert frames and frames[-1] == ("cam", 2)
