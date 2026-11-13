import asyncio
from types import SimpleNamespace

from domain.models.detection_result import DetectionResult
from infrastructure.automation.automation_orchestrator import AutomationRuleEngine


class RecordingBus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, event):
        self.published.append((topic, event))


def make_engine():
    rule = {
        "name": "wjazd_brama",
        "type": "tripwire",
        "target_classes": ["car_detection"],
        "line": [[200, 500], [1000, 500]],
        "trigger_event": "GATE_APPROACH",
    }
    config = SimpleNamespace(camera=SimpleNamespace(id="dom_c1", rules=[rule]))
    bus = RecordingBus()
    return AutomationRuleEngine(frame_bus=None, event_bus=bus, camera_config=config), bus


def frame(frame_id, ts, *ground_points):
    detections = [
        DetectionResult("car_detection", 0.9, x - 150, y - 200, x + 150, y, label="car")
        for x, y in ground_points
    ]
    return SimpleNamespace(detections=detections, timestamp=ts, frame_id=frame_id, stream_session_id="s", scene_id=None)


def run(engine, frames):
    async def go():
        for f in frames:
            await engine._process_frame(f)

    asyncio.run(go())


def test_crossing_publishes_one_gate_approach_event():
    engine, bus = make_engine()
    run(engine, [frame(1, 0.0, (600, 600)), frame(2, 1.0, (600, 450)), frame(3, 2.0, (600, 300))])

    assert [topic for topic, _ in bus.published] == ["GATE_APPROACH"]
    event = bus.published[0][1]
    assert event["type"] == "tripwire.crossed"
    assert event["previous_point"] == {"x": 600, "y": 600}
    assert event["point"] == {"x": 600, "y": 450}
    assert event["detection"].triggered is True


def test_car_waiting_on_gate_side_publishes_nothing():
    engine, bus = make_engine()
    run(engine, [frame(i, float(i), (605, 274)) for i in range(60)])
    assert bus.published == []


def test_other_classes_are_ignored():
    engine, bus = make_engine()
    f1, f2 = frame(1, 0.0, (600, 600)), frame(2, 1.0, (600, 450))
    for f in (f1, f2):
        for d in f.detections:
            d.detection_type = "person_detection"
    run(engine, [f1, f2])
    assert bus.published == []
