# application/automation/automation_rule_engine.py

from __future__ import annotations

import asyncio
import time
import traceback

from infrastructure.automation.tripwire import TripwireCrossingDetector, side_of_line


class AutomationRuleEngine:
    def __init__(
        self,
        frame_bus,
        event_bus,
        camera_config,
        cooldown_seconds: float = 10.0,
    ):
        self.frame_bus = frame_bus
        self.event_bus = event_bus
        self.config = camera_config

        self.last_triggered: dict[str, float] = {}
        self.cooldown_seconds = float(cooldown_seconds)

        self.crossing_detectors: dict[str, TripwireCrossingDetector] = {}

        self.running = False

    async def start_processing(self, topic: str):
        self.running = True
        queue = self.frame_bus.subscribe(topic)

        print(
            f"[RuleEngine] Subskrybuję {topic} "
            f"dla kamery {self.config.camera.id}"
        )

        try:
            while self.running:
                frame_ctx = await queue.get()

                if frame_ctx is None:
                    continue

                await self._process_frame(frame_ctx)

        except asyncio.CancelledError:
            pass

        except Exception:
            print(
                f"[RuleEngine] Worker crashed for camera "
                f"{self.config.camera.id}"
            )
            traceback.print_exc()

        finally:
            self.frame_bus.unsubscribe(topic, queue)
            print(
                f"[RuleEngine] Stopped for camera "
                f"{self.config.camera.id}"
            )

    def stop(self):
        self.running = False

    async def _process_frame(self, frame_ctx):
        rules = self.config.camera.rules
        detections = getattr(frame_ctx, "detections", []) or []

        for rule in rules:
            if rule.get("type") != "tripwire":
                continue

            await self._process_tripwire_rule(
                rule=rule,
                frame_ctx=frame_ctx,
                detections=detections,
            )

    async def _process_tripwire_rule(
        self,
        rule: dict,
        frame_ctx,
        detections: list,
    ):
        rule_name = rule.get("name", "default")
        line = rule.get("line")
        target_classes = rule.get("target_classes", [])
        trigger_event = rule.get("trigger_event")

        if not line or len(line) != 2:
            return

        if not trigger_event:
            return

        crossing_detector = self._get_crossing_detector(rule_name, rule)

        candidates = {}

        for detection in detections:
            if getattr(detection, "detection_type", None) not in target_classes:
                continue

            point = (int((detection.x1 + detection.x2) / 2), int(detection.y2))
            candidates[(float(point[0]), float(point[1]))] = detection

        frame_ts = float(getattr(frame_ctx, "timestamp", None) or time.time())
        crossings = crossing_detector.update(list(candidates), frame_ts)

        for previous_point, current_point in crossings:
            detection = candidates[current_point]
            detection.triggered = True

            now = time.time()
            cooldown_key = f"{self.config.camera.id}:{rule_name}"
            last_time = self.last_triggered.get(cooldown_key, 0.0)

            if now - last_time <= self.cooldown_seconds:
                continue

            self.last_triggered[cooldown_key] = now

            cx, bottom_y = int(current_point[0]), int(current_point[1])

            event = {
                "type": "tripwire.crossed",
                "camera_id": self.config.camera.id,
                "rule_name": rule_name,
                "trigger_event": trigger_event,
                "timestamp": now,
                "stream_session_id": getattr(frame_ctx, "stream_session_id", None),
                "frame_id": getattr(frame_ctx, "frame_id", None),
                "scene_id": getattr(frame_ctx, "scene_id", None),
                "detection": detection,
                "point": {
                    "x": cx,
                    "y": bottom_y,
                },
                "previous_point": {
                    "x": int(previous_point[0]),
                    "y": int(previous_point[1]),
                },
                "direction": crossing_detector.direction,
                "line": line,
            }

            print(
                f"[RuleEngine] 🔥 WYZWALAM ZDARZENIE: "
                f"{trigger_event}, camera={self.config.camera.id}, "
                f"rule={rule_name}"
            )

            await self.event_bus.publish(trigger_event, event)

    def _get_crossing_detector(self, rule_name: str, rule: dict) -> TripwireCrossingDetector:
        detector = self.crossing_detectors.get(rule_name)

        if detector is None:
            detector = TripwireCrossingDetector(
                line=rule["line"],
                direction=rule.get("direction", "to_trigger_side"),
                max_gap_seconds=float(rule.get("max_gap_seconds", 3.0)),
                max_jump_px=float(rule.get("max_jump_px", 600.0)),
            )
            self.crossing_detectors[rule_name] = detector

        return detector

    def _is_point_on_trigger_side(
        self,
        point: tuple[int, int],
        line: list,
    ) -> bool:
        return side_of_line(point, line) > 0
