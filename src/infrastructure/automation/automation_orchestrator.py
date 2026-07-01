# application/automation/automation_rule_engine.py

from __future__ import annotations

import asyncio
import time
import traceback


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

        for detection in detections:
            detection_type = getattr(detection, "detection_type", None)

            if detection_type not in target_classes:
                continue

            cx = int((detection.x1 + detection.x2) / 2)
            bottom_y = int(detection.y2)

            crossed = self._is_point_on_trigger_side(
                point=(cx, bottom_y),
                line=line,
            )

            if not crossed:
                continue

            detection.triggered = True

            now = time.time()
            cooldown_key = f"{self.config.camera.id}:{rule_name}"
            last_time = self.last_triggered.get(cooldown_key, 0.0)

            if now - last_time <= self.cooldown_seconds:
                continue

            self.last_triggered[cooldown_key] = now

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
                "line": line,
            }

            print(
                f"[RuleEngine] 🔥 WYZWALAM ZDARZENIE: "
                f"{trigger_event}, camera={self.config.camera.id}, "
                f"rule={rule_name}"
            )

            await self.event_bus.publish(trigger_event, event)

    def _is_point_on_trigger_side(
        self,
        point: tuple[int, int],
        line: list,
    ) -> bool:
        px, py = point

        x1 = int(line[0][0])
        y1 = int(line[0][1])
        x2 = int(line[1][0])
        y2 = int(line[1][1])

        cross_val = (
            (px - x1) * (y2 - y1)
            - (py - y1) * (x2 - x1)
        )

        return cross_val > 0