import asyncio
from types import SimpleNamespace

import numpy as np

from application.messaging.event_bus import EventBus
from application.messaging.frame_bus import FrameBus
from application.orchestration.detection_worker import DetectionWorker
from domain.models.detection_result import DetectionResult
from domain.models.frame_context import FrameContext


class ParkedCarDetector:
    def execute(self, frame, context):
        return [DetectionResult("car_detection", 0.9, 300, 400, 800, 750, label="car")]


class Provider:
    def get_detector(self, detection_type):
        return ParkedCarDetector()


class ScriptedMotion:
    """Motion filter stand-in that replays a fixed sequence."""

    def __init__(self, sequence):
        self.sequence = list(sequence)

    def has_significant_motion(self, camera_id, frame, roi_polygon=None):
        return self.sequence.pop(0)

    def reset(self, camera_id):
        pass


def test_still_scene_does_not_re_archive_a_parked_car():
    async def scenario():
        frame_bus, event_bus = FrameBus(), EventBus()
        events = event_bus.subscribe("detections")

        worker = DetectionWorker(
            camera_id="dom_c1",
            frame_bus=frame_bus,
            event_bus=event_bus,
            detector_provider=Provider(),
            specs=[SimpleNamespace(detection_type="car_detection", ai_fps=1000)],
        )
        worker.absence_reset_sec = 0.0
        # motion -> long still period -> motion again (e.g. wind in the trees)
        motion = [True] + [False] * 5 + [True]
        worker.motion_filter = ScriptedMotion(motion)

        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.05)

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for frame_id in range(len(motion)):
            await frame_bus.publish("dom_c1", FrameContext("dom_c1", frame_id, image))
            await asyncio.sleep(0.05)

        worker.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        published = []
        while not events.empty():
            published.append(events.get_nowait())
        return published

    published = asyncio.run(scenario())

    assert [e["event_reason"] for e in published] == ["new_detection"]
